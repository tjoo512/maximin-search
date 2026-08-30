import time
import argparse

from datasets import load_dataset

from qwen_eval import math_equal, strip_string
from utils import seed_everything, extract_answer_og
from save_utils import (
    make_run_id, make_run_tag, find_latest_matching_run,
    load_progress, aggregate_shards, BenchmarkLogger,
)
from generator import StepDecode, BeamSearch
from distributed import setup_distributed, broadcast_object, barrier, cleanup


SUPPORTED_MODELS = {
    "microsoft/Phi-3.5-mini-instruct": "phi3mini",
    "Qwen/Qwen2.5-Math-1.5B": "qwen25_math_15b",
}

REWARD_MODEL = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"


def load_benchmark_dataset(dataset_name, output_dir):
    """Load a benchmark dataset and return (dataset_split, output_directory)."""
    configs = {
        "gsm8k":    (lambda: load_dataset("openai/gsm8k", "main")["test"],         f"{output_dir}_gsm8k"),
        "math500":  (lambda: load_dataset("HuggingFaceH4/MATH-500")["test"],        f"{output_dir}_math500"),
        "aime2024": (lambda: load_dataset("HuggingFaceH4/aime_2024")["train"],      f"{output_dir}_aime2024"),
        "aime2025": (lambda: load_dataset("MathArena/aime_2025")["train"],          f"{output_dir}_aime2025"),
    }
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    loader, directory = configs[dataset_name]
    return loader(), directory


def get_problem_and_gt(item, dataset_name):
    """Extract (prompt, ground_truth) from a dataset item."""
    if dataset_name == "gsm8k":
        return item["question"], extract_answer_og(item["answer"])
    return item["problem"], item["answer"]


def check_correct(pred, gt, dataset_name):
    """Return correctness as int (0/1) for a given dataset's grading scheme."""
    if dataset_name == "gsm8k":
        try:
            return int(abs(float(pred) - float(gt)) < 1e-6)
        except (TypeError, ValueError):
            return 0
    else:
        return int(math_equal(str(pred), strip_string(str(gt))))


def benchmark_generator(generator, dataset, logger, args, dist_info):
    """Run generation and evaluation over the dataset, with resume support.

    Under data-parallel, each rank processes a strided shard of the dataset
    (`idx % world_size == rank`) and streams its results to a per-rank shard
    file, so that interrupted runs can be resumed without re-running completed
    examples.
    """
    rank, world_size = dist_info.rank, dist_info.world_size
    tag = f"[rank {rank}] " if dist_info.is_distributed else ""

    completed, correct, total = load_progress(logger.jsonl_path)
    if completed:
        print(f"{tag}[RESUME] Resuming from {len(completed)} completed examples.")

    for idx, item in enumerate(dataset):
        if args.max_examples and idx >= args.max_examples:
            break
        if world_size > 1 and idx % world_size != rank:
            continue
        if idx in completed:
            continue

        if idx % 10 == 0:
            print(f"{tag}[{idx}] evaluating...")

        prompt, gt = get_problem_and_gt(item, args.dataset)

        start = time.time()
        pred, solns, scores = generator.generate(prompt)
        latency = time.time() - start

        is_correct = check_correct(pred, gt, args.dataset)

        print(f"  RM pred: {pred} | gt: {gt}")

        correct += is_correct
        total += 1
        completed.add(idx)

        logger.log_example({
            "index": idx,
            "question": prompt,
            "ground_truth": gt,
            "prediction": pred,
            "correct": is_correct,
            "latency_sec": latency,
            "beams": solns,
            "scores": scores,
        })

    acc = correct / total if total > 0 else 0.0
    if dist_info.is_distributed:
        print(f"{tag}local accuracy: {acc:.4f} ({correct}/{total})")
    else:
        print(f"\nFinal Accuracy: {acc:.4f} ({correct}/{total})")


def parse_args():
    parser = argparse.ArgumentParser()

    # Model and dataset
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "math500", "aime2024", "aime2025"])

    # Search budget
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--branching_factor", type=float, default=4,
                        help="Number of children per node. Set to 1 for Best-of-N with PRM.")
    parser.add_argument("--max_depth", type=int, default=30)

    # Sampling
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--step_max_new_tokens", type=int, default=128)
    parser.add_argument("--parallel_chunk_budget", type=int, default=-1)

    # Reward model / pruning
    parser.add_argument("--prm", action="store_true",
                        help="Use PRM scores for beam pruning.")
    parser.add_argument("--repulsive_factor", type=float, default=0,
                        help="Diversity regularization strength (0 = disabled).")
    parser.add_argument("--prm_noise", type=float, default=0,
                        help="Gaussian noise added to PRM scores for exploration.")

    # Search mode overrides
    parser.add_argument("--use_beam", action="store_true",
                        help="Use standard beam search instead of step-level decode.")
    parser.add_argument("--greedy_search", action="store_true")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="ablation")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dist_info = setup_distributed()
    # Offset the seed by rank so ranks don't share an identical RNG stream;
    # for single-GPU (rank 0) this is exactly the original seed.
    seed_everything(args.seed + dist_info.rank)

    if args.model not in SUPPORTED_MODELS:
        raise NotImplementedError(f"Unsupported model: {args.model}. Add it to SUPPORTED_MODELS.")

    dataset, directory = load_benchmark_dataset(args.dataset, args.output_dir)
    directory += f"_{SUPPORTED_MODELS[args.model]}"

    # Data-parallel: pin a full model copy to this rank's GPU. Single-GPU keeps "auto".
    device_map = {"": dist_info.local_rank} if dist_info.is_distributed else "auto"

    generator_cls = BeamSearch if (args.use_beam or args.greedy_search) else StepDecode
    generator = generator_cls(args.model, args, reward_model_name=REWARD_MODEL, device_map=device_map)

    # Resume-aware run ID, decided once by rank 0 and broadcast so every rank
    # writes its shard into the same run directory.
    run_tag = make_run_tag(
        args.model, args.batch_size, args.temperature,
        args.branching_factor, args.repulsive_factor,
        args.use_beam, args.greedy_search,
    )
    run_id, is_new_run = None, False
    if dist_info.is_main:
        existing_run = find_latest_matching_run(directory, run_tag)
        if existing_run is not None:
            run_id = existing_run
            print(f"[RESUME] run_id: {run_id}")
        else:
            run_id = make_run_id(
                args.model, args.batch_size, args.temperature,
                args.branching_factor, args.repulsive_factor,
                args.use_beam, args.greedy_search,
            )
            is_new_run = True
            print(f"[NEW RUN] run_id: {run_id}")
    run_id = broadcast_object(run_id, dist_info)

    shard_suffix = f".rank{dist_info.rank}" if dist_info.is_distributed else ""
    logger = BenchmarkLogger(run_id, directory, shard_suffix=shard_suffix)
    if dist_info.is_main and is_new_run:
        logger.save_config(vars(args))

    benchmark_generator(generator, dataset, logger, args, dist_info)

    # Aggregate all per-rank shards into a single global accuracy on rank 0.
    barrier(dist_info)
    if dist_info.is_distributed and dist_info.is_main:
        correct, total = aggregate_shards(logger.dir)
        acc = correct / total if total > 0 else 0.0
        print(f"\n[GLOBAL] Accuracy: {acc:.4f} ({correct}/{total}) "
              f"across {dist_info.world_size} ranks")
    cleanup(dist_info)