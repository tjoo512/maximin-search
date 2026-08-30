import os
import json
import glob
from datetime import datetime



class BenchmarkLogger:
    def __init__(self, run_id, log_dir="results", shard_suffix=""):
        self.run_id = run_id
        self.dir = os.path.join(log_dir, run_id)
        os.makedirs(self.dir, exist_ok=True)

        # Under data-parallel, each rank writes its own shard (e.g. ".rank1") so
        # ranks never contend on the same file; shards are aggregated at the end.
        self.jsonl_path = os.path.join(self.dir, f"outputs{shard_suffix}.jsonl")
        self.config_path = os.path.join(self.dir, "config.json")

    def save_config(self, config_dict):
        with open(self.config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def log_example(self, record: dict):
        record["timestamp"] = datetime.utcnow().isoformat()

        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")


def make_run_tag(model_name, batch_size, temp, branch, repulsive_factor, use_beam, greedy_search):
    model_short = model_name.split("/")[-1]
    name = f"{model_short}_bs{batch_size}_temp{temp}_branch{branch}"
    if repulsive_factor > 0:
       name += f"_repf{repulsive_factor}"

    if use_beam:
        name += '_beam_search'
    elif greedy_search:
        name += '_greedy_search'

    return name


def make_run_id(model_name, batch_size, temp, branch, repulsive_factor, use_beam, greedy_search):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tag = make_run_tag(model_name, batch_size, temp, branch, repulsive_factor, use_beam, greedy_search)
    return f"{ts}_{tag}"


def find_latest_matching_run(log_dir, run_tag):
    """
    Finds the newest run directory in log_dir whose folder name ends with run_tag.
    Assumes folder names start with a sortable UTC timestamp, like YYYYMMDD_HHMMSS_...
    """
    if not os.path.isdir(log_dir):
        return None

    candidates = []
    for name in os.listdir(log_dir):
        path = os.path.join(log_dir, name)
        if os.path.isdir(path) and name.endswith(run_tag):
            candidates.append(name)

    if not candidates:
        return None

    # newest by lexicographic sort if timestamp prefix is used
    candidates.sort()
    return candidates[-1]


def aggregate_shards(run_dir):
    """Aggregate every ``outputs*.jsonl`` shard in a run directory.

    Deduplicates by example ``index`` (so overlapping resumes are counted once)
    and returns ``(correct, total)`` across all ranks.
    """
    seen = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "outputs*.jsonl"))):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                idx = rec.get("index")
                if idx is not None:
                    seen[idx] = rec

    total = len(seen)
    correct = sum(int(bool(r.get("correct", False))) for r in seen.values())
    return correct, total


def load_progress(jsonl_path):
    """
    Returns:
      completed: set of indices already logged
      correct: number correct already logged
      total: number total already logged
    Robust to partially-written last line.
    """
    completed = set()
    correct = 0
    total = 0

    if not os.path.exists(jsonl_path):
        return completed, correct, total

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # likely a partially-written line at crash time; ignore
                continue

            idx = rec.get("index", None)
            if idx is None:
                continue

            completed.add(idx)
            total += 1
            correct += int(bool(rec.get("correct", False)))

    return completed, correct, total
