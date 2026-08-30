"""Data-parallel helpers for distributing benchmark examples across GPUs.

The benchmark is embarrassingly parallel: every GPU holds a full copy of the
generator and PRM and evaluates a disjoint shard of the dataset, with no
gradient synchronization. Launch with one process per GPU via torchrun:

    torchrun --nproc_per_node=<N> src/main.py [args...]

When the script is started with plain ``python`` (no torchrun env vars), all
helpers fall back to a single rank, preserving the original single-GPU
behavior bit-for-bit.
"""
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistInfo:
    """Read torchrun env vars, pin this process to its GPU, and init NCCL.

    Falls back to ``(rank=0, world_size=1, local_rank=0)`` when the env vars are
    absent (plain ``python`` launch), in which case no process group is created.
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Pin the default CUDA device so bare ``.cuda()`` calls (and the NCCL
    # backend) land on this rank's GPU rather than always on cuda:0.
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    return DistInfo(rank=rank, world_size=world_size, local_rank=local_rank)


def broadcast_object(obj, info: DistInfo):
    """Broadcast a picklable object from rank 0 to every rank."""
    if not info.is_distributed:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def barrier(info: DistInfo) -> None:
    if info.is_distributed:
        dist.barrier()


def cleanup(info: DistInfo) -> None:
    if info.is_distributed and dist.is_initialized():
        dist.destroy_process_group()
