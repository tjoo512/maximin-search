import os
import numpy as np

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM

from greedy_solver import solve_kernel_subset_problem
from skywork_prm import prepare_input, prepare_batch_input_for_model, derive_step_rewards, PRM_MODEL
from utils import rbf_kernel
from qwen_eval import extract_answer


def load_model_and_tokenizer(model_name, auth_token, device_map="auto"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=True,
        token=auth_token,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=auth_token,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
    )
    return model, tokenizer


class Generator:
    """Base class for LLM-based solution generators.

    Handles model loading, prompt templating, and PRM-based scoring.
    Subclasses implement specific search strategies (StepDecode, BeamSearch).
    """

    @torch.inference_mode()
    def __init__(self, model_name, args, reward_model_name=None, device_map="auto"):
        self.model_name = model_name
        self.auth_token = os.environ.get("HF_TOKEN")
        # Under data-parallel (torchrun), `device_map` pins a full model copy to
        # this rank's GPU, e.g. {"": local_rank}. Single-GPU runs keep "auto".
        self.device_map = device_map

        model, tokenizer = load_model_and_tokenizer(model_name, self.auth_token, device_map)
        self.model = model
        self.tokenizer = tokenizer


        self.parallel_chunk_budget = args.parallel_chunk_budget
        if self.parallel_chunk_budget < 0:
            self.parallel_chunk_budget = args.batch_size
        self.max_depth = args.max_depth

        self.step_stop_ids = self.tokenizer("\n\n", add_special_tokens=False).input_ids
        self.model.eval()

        self.use_beam = args.use_beam
        self.prm_noise = args.prm_noise
        self.dataset = args.dataset
        self.prm = args.prm
        self.branching_factor = args.branching_factor

        # Load reward model (PRM) unless doing beam/greedy search
        if args.use_beam or args.greedy_search:
            self.rm = None
            self.rm_tokenizer = None
        elif reward_model_name is not None:
            if "Qwen" not in reward_model_name:
                raise NotImplementedError("Use Qwen-based PRMs")

            self.rm = PRM_MODEL.from_pretrained(
                reward_model_name, device_map=self.device_map
            ).eval()
            self.rm_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
            if self.rm_tokenizer.pad_token is None:
                self.rm_tokenizer.pad_token = self.rm_tokenizer.eos_token

        self.repulsive_factor = args.repulsive_factor
        self.embedding_model = None

        self.greedy_search = args.greedy_search
        self.batch_size = args.batch_size
        self.max_new_tokens = args.max_new_tokens
        self.top_p = args.top_p
        self.temperature = args.temperature
        self.do_sample = True

        self.step_max_new_tokens = getattr(
            args, "step_max_new_tokens", min(128, self.max_new_tokens)
        )
        self.terminators = self._get_terminators()

    def _get_terminators(self):
        """Return EOS token IDs, including model-specific special tokens."""
        if "phi" in self.model_name.lower():
            return [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|end|>"),
            ]
        return [self.tokenizer.eos_token_id]

    def get_templated_prompt(self, prompt, trajectory=""):
        """Wrap a prompt (and optional partial trajectory) in the model's chat template.

        Args:
            prompt: The math problem string.
            trajectory: Partial solution string, or list of partial solutions.

        Returns:
            Formatted string (or list of strings) ready for tokenization.
        """
        TEMPLATED_USER = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{question}\n\n### Response:"
        )
        BOXED_INSTRUCTION = (
            "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
        )

        def _is_phi():
            return "phi" in self.model_name.lower()

        def _phi_template(t):
            # Trajectory is a partial assistant turn; do not close with <|end|>
            # so generation continues directly after these tokens.
            user_content = TEMPLATED_USER.format(question=prompt) + " " + BOXED_INSTRUCTION
            return (
                "<|system|>You are a helpful assistant<|end|>"
                "<|user|>" + user_content + "<|end|>"
                "<|assistant|>" + (t or "")
            )

        def _plain(t):
            return (
                TEMPLATED_USER.format(question=prompt)
                + " " + BOXED_INSTRUCTION
                + (t or "")
            )

        fmt = _phi_template if _is_phi() else _plain

        if isinstance(trajectory, str):
            return fmt(trajectory)
        if isinstance(trajectory, list):
            return [fmt(t) for t in trajectory]
        raise ValueError("trajectory must be str or list[str].")

    @torch.inference_mode()
    def compute_prm_scores(self, prompt, generation_texts):
        """Score each candidate solution with the process reward model (PRM).

        Returns:
            scores:       Sigmoid-normalized PRM scores, shape (N,).
            token_states: Last-token hidden states from the PRM, shape (N, D).
            max_diffs:    Per-sample max step-score range (used for lambda estimation).
        """
        chunk = self.parallel_chunk_budget
        datas = [{"problem": prompt, "response": g} for g in generation_texts]
        convs = [
            prepare_input(d["problem"], d["response"], tokenizer=self.rm_tokenizer, step_token="\n\n")
            for d in datas
        ]

        scores_all, token_states_all, max_diffs = [], [], []

        for i in range(0, len(convs), chunk):
            input_ids, steps, reward_flags = zip(*convs[i : i + chunk])
            input_ids, attention_mask, reward_flags = prepare_batch_input_for_model(
                input_ids, reward_flags, self.rm_tokenizer.pad_token_id
            )
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            reward_flags = reward_flags.cuda()

            _, _, rewards, last_hidden_state = self.rm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_probs=True,
                return_repr=True,
            )

            # Extract last non-padding token representation
            last_token_indices = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            batch_indices = torch.arange(batch_size, device=last_hidden_state.device)
            last_token_states = last_hidden_state[batch_indices, last_token_indices]

            step_rewards = derive_step_rewards(rewards, reward_flags)
            sigm_step = [torch.sigmoid(torch.Tensor(s)) for s in step_rewards]
            max_diffs += [s.max() - s.min() for s in sigm_step]

            # Mean step reward per candidate (the only scoring path; previously
            # gated behind `if self.prm`, which left `s` undefined when PRM was off).
            s = torch.Tensor([np.mean(s_temp) for s_temp in step_rewards]).cuda()
            if s.dim() == 0:
                s = s.unsqueeze(0)

            scores_all.append(s)
            token_states_all.append(last_token_states)

        scores = torch.sigmoid(torch.cat(scores_all, dim=0))
        token_states = torch.cat(token_states_all, dim=0)
        return scores, token_states, max_diffs


class StepDecode(Generator):
    """Step-level decoding with PRM-guided pruning and optional repulsive sampling.

    At each depth, generates `branching_factor` continuations per live trajectory,
    scores them with the PRM, and prunes to `batch_size / branching_factor` survivors.
    When `repulsive_factor > 0`, diversity is encouraged via a kernel-regularized
    subset selection (see greedy_solver.py).

    Special case: branching_factor=1 reduces to Best-of-N sampling, where all
    `batch_size` solutions are generated independently and the PRM selects the best.
    """

    @torch.inference_mode()
    def generate(self, prompt, debug=False):
        branching_size = (
            max(1, int(self.batch_size ** self.branching_factor))
            if self.branching_factor < 1
            else int(self.branching_factor)
        )

        trajectories = [""]
        gen_token_counts = [0]
        finished_trajectories = []
        lambda_estimated = 0.0
        step_lengths = []

        def generate_one_step(trajectories, chunk_counts, use_branch=None):
            """Extend each trajectory by one reasoning step."""
            num_muls = use_branch if use_branch is not None else branching_size
            trajectories = trajectories * num_muls
            chunk_counts = chunk_counts * num_muls

            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            eos_id = self.tokenizer.eos_token_id
            device = self.model.device
            self.tokenizer.padding_side = "left"

            generated_texts, generated_token_counts = [], []

            for start in range(0, len(trajectories), self.parallel_chunk_budget):
                chunk_trajs = trajectories[start : start + self.parallel_chunk_budget]
                templated = self.get_templated_prompt(prompt, chunk_trajs)

                be = self.tokenizer(
                    templated,
                    padding=True,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                input_ids = be.input_ids.to(device, non_blocking=True)
                attention_mask = be.attention_mask.to(device, non_blocking=True)
                input_width = input_ids.shape[1]

                eos_token_id = [x for x in self.terminators if x is not None] or eos_id

                seqs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.step_max_new_tokens,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_id,
                    do_sample=self.do_sample,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    return_dict_in_generate=False,
                    output_scores=False,
                    renormalize_logits=True,
                    remove_invalid_values=True,
                    stop_strings=["\n\n"],
                    tokenizer=self.tokenizer,
                )
                seqs_cpu = seqs.detach().cpu()
                del be, input_ids, attention_mask, seqs

                for i in range(seqs_cpu.shape[0]):
                    gen_ids = seqs_cpu[i, input_width:]

                    # Count generated tokens (up to the first pad/eos)
                    step_new_tokens = gen_ids.numel()
                    for sentinel in (pad_id, eos_id):
                        if sentinel is not None:
                            pos = (gen_ids == sentinel).nonzero(as_tuple=False)
                            if pos.numel() > 0:
                                step_new_tokens = int(pos[0].item())
                                break

                    text = self.tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
                    generated_texts.append(chunk_trajs[i] + text)
                    generated_token_counts.append(chunk_counts[i] + step_new_tokens)

                del seqs_cpu

            return generated_texts, -1, generated_token_counts

        # ── Main search loop ──────────────────────────────────────────────────
        for step_i in range(self.max_depth):
            # First step fans out from a single trajectory to batch_size candidates
            if step_i == 0:
                trajectories, _, gen_token_counts = generate_one_step(
                    trajectories, gen_token_counts, int(self.batch_size)
                )
            else:
                trajectories, _, gen_token_counts = generate_one_step(
                    trajectories, gen_token_counts
                )

            # ── Pruning ───────────────────────────────────────────────────────
            width = max(1, int(self.batch_size / branching_size))

            if len(trajectories) > width and self.prm:
                scores, token_states, max_diff_err = self.compute_prm_scores(prompt, trajectories)

                if self.prm_noise != 0:
                    scores = scores + self.prm_noise * torch.randn_like(scores)

                if self.repulsive_factor != 0:
                    # Kernel-regularized subset selection for diversity-aware pruning
                    normalized_embedding = F.normalize(token_states.cuda(), dim=1, eps=1e-8)
                    K = rbf_kernel(normalized_embedding, sigma=None)

                    # Estimate repulsion strength λ via kernel regression on score spread
                    estt = torch.Tensor(max_diff_err).cuda()
                    A = K + 1e-6 * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
                    y = torch.linalg.solve(A, estt)
                    lambda_estimated = self.repulsive_factor * float((estt @ y) ** 0.5)

                    K_np = K.detach().float().cpu().numpy()
                    K_np = (K_np + K_np.T) / 2.0

                    result = solve_kernel_subset_problem(
                        scores.detach().float().cpu().numpy(),
                        K_np,
                        lambda_estimated,
                        width,
                        epsilon=0.05,
                        n_restarts=4,
                        max_passes=30,
                        random_state=123,
                    )
                    selected = set(result["indices"])
                    assert len(selected) == width
                    trajectories = [t for i, t in enumerate(trajectories) if i in selected]

                else:
                    # Standard top-k beam pruning
                    _, topk_indices = torch.topk(scores, width)
                    topk_indices = set(topk_indices.tolist())
                    trajectories = [t for i, t in enumerate(trajectories) if i in topk_indices]

            if debug:
                print(f"Depth {step_i} | trajectories: {trajectories}")

            # ── Split finished / unfinished ───────────────────────────────────
            unfinished = []
            for i, traj in enumerate(trajectories):
                if "boxed" not in traj and gen_token_counts[i] < 2048:
                    unfinished.append(traj)
                else:
                    step_lengths.append(step_i)
                    finished_trajectories.append(traj)

            if not unfinished:
                break
            trajectories = unfinished

        # ── Final answer selection ────────────────────────────────────────────
        if not finished_trajectories:
            return "N/A", ["N/A"], ["N/A"]

        valid_trajectories, valid_answers = [], []
        for traj in finished_trajectories:
            ext = extract_answer(traj, "math500")
            if ext:
                valid_trajectories.append(traj)
                valid_answers.append(ext)

        if not valid_answers:
            return "N/A", ["N/A"], ["N/A"]

        # Re-score valid solutions and pick the PRM top-1
        scores, _, _ = self.compute_prm_scores(prompt, valid_trajectories)
        top_idx = int(scores.argmax())
        out = valid_answers[top_idx]

        return out, valid_answers, scores.cpu().tolist()


class BeamSearch(Generator):
    """Standard beam search baseline.

    Supports greedy decoding (num_beams=1), beam search with batch_size=16,
    and a chunked multi-repeat variant for batch_size=64.
    """

    @torch.inference_mode()
    def generate(self, prompt, debug=True):
        templated_prompt = self.get_templated_prompt(prompt)
        batch_encoding = self.tokenizer(
            [templated_prompt],
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.model.device)

        # ── Greedy decoding ───────────────────────────────────────────────────
        if self.greedy_search:
            full_generation = self.model.generate(
                input_ids=batch_encoding.input_ids,
                attention_mask=batch_encoding.attention_mask,
                max_new_tokens=2048,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=1,
                do_sample=False,
                top_p=0.0,
                temperature=False,
                output_scores=False,
                num_return_sequences=self.batch_size,
                return_dict_in_generate=True,
            )
            generation_text = self.tokenizer.batch_decode(
                full_generation.sequences, skip_special_tokens=False
            )[0]
            e = extract_answer(generation_text, "math500")
            return (e, [e], ["deterministic search"]) if e else ("N/A", ["N/A"], ["N/A"])

        # ── Beam search (batch_size=16) ───────────────────────────────────────
        elif self.use_beam and self.batch_size == 16:
            full_generation = self.model.generate(
                input_ids=batch_encoding.input_ids,
                attention_mask=batch_encoding.attention_mask,
                max_new_tokens=2048,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=self.batch_size,
                do_sample=self.do_sample,
                top_p=self.top_p,
                temperature=self.temperature,
                output_scores=False,
                num_return_sequences=self.batch_size,
                return_dict_in_generate=True,
            )
            generation_texts = self.tokenizer.batch_decode(
                full_generation.sequences, skip_special_tokens=False
            )

            exts = [
                extract_answer(t, "math500")
                for t in generation_texts
                if extract_answer(t, "math500")
            ]
            if not exts:
                return "N/A", ["N/A"], ["N/A"]

            out = exts[0]  # beam-score top-1

            if debug:
                print(f"answers: {exts} | beam top: {out}")

            return out, exts, ["n/a"]

        # ── Chunked beam search (batch_size=64, 8×8 beams) ───────────────────
        elif self.use_beam and self.batch_size == 64:
            beam_chunk_size = 8
            num_repeats = 8
            candidates = []

            for repeat_idx in range(num_repeats):
                full_generation = self.model.generate(
                    input_ids=batch_encoding.input_ids,
                    attention_mask=batch_encoding.attention_mask,
                    max_new_tokens=2048,
                    eos_token_id=self.terminators,
                    pad_token_id=self.tokenizer.pad_token_id,
                    num_beams=beam_chunk_size,
                    num_return_sequences=beam_chunk_size,
                    do_sample=self.do_sample,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                generation_texts = self.tokenizer.batch_decode(
                    full_generation.sequences, skip_special_tokens=False
                )

                # Use length-normalized beam scores for global ranking
                sequence_scores = (
                    full_generation.sequences_scores.detach().cpu().tolist()
                    if hasattr(full_generation, "sequences_scores")
                    and full_generation.sequences_scores is not None
                    else [float("-inf")] * len(generation_texts)
                )

                for beam_idx, text in enumerate(generation_texts):
                    e = extract_answer(text, "math500")
                    if e:
                        candidates.append(
                            {"answer": e, "score": sequence_scores[beam_idx],
                             "repeat_idx": repeat_idx, "beam_idx": beam_idx}
                        )

            if not candidates:
                return "N/A", ["N/A"], ["N/A"]

            candidates.sort(key=lambda x: x["score"], reverse=True)
            exts = [c["answer"] for c in candidates]
            scores = [c["score"] for c in candidates]
            out = candidates[0]["answer"]

            if debug:
                print(f"answers: {exts} | best: {out}")

            return out, exts, scores

        else:
            raise NotImplementedError
