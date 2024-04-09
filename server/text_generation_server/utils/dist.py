import os
import torch

from datetime import timedelta
from loguru import logger

# Tensor Parallelism settings
RANK = int(os.getenv("RANK", "0"))
RANK_LOCAL = int(os.getenv("RANK_LOCAL", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))

# CUDA memory fraction
MEMORY_FRACTION = float(os.getenv("CUDA_MEMORY_FRACTION", "1.0"))


class FakeBarrier:
    def wait(self):
        pass


class FakeGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def allreduce(self, *args, **kwargs):
        return FakeBarrier()

    def allgather(self, inputs, local_tensor, **kwargs):
        assert (
            len(inputs[0]) == len(local_tensor) == 1
        ), f"{len(inputs[0])} != {len(local_tensor)} != 1, and the FakeGroup is supposed to join on simple tensors"
        for input_ in inputs:
            input_[0].data = local_tensor[0].data
        return FakeBarrier()

    def barrier(self, *args, **kwargs):
        return FakeBarrier()

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def initialize_torch_distributed():
    if torch.cuda.is_available():
        from torch.distributed import ProcessGroupNCCL

        # Set the device id.
        assert RANK_LOCAL <= torch.cuda.device_count(), "Each local rank process is one gpu"
        device = RANK_LOCAL % torch.cuda.device_count()
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device)
        backend = "nccl"
        options = ProcessGroupNCCL.Options()
        options.is_high_priority_stream = True
        options._timeout = timedelta(seconds=60)
    else:
        backend = "gloo"
        options = None
        device = None

    if WORLD_SIZE == 1:
        return FakeGroup(RANK, WORLD_SIZE), RANK, WORLD_SIZE, device
    else:
        if os.getenv("DEBUG", None) == "1":
            return FakeGroup(RANK, WORLD_SIZE), RANK, WORLD_SIZE, device

        if not torch.distributed.is_initialized():
            # Call the init process.
            torch.distributed.init_process_group(
                init_method=os.getenv("INIT_METHOD", None),
                backend=backend,
                world_size=WORLD_SIZE,
                rank=RANK,
                timeout=timedelta(seconds=60),
                pg_options=options,
            )
        else:
            logger.warning("torch.distributed is already initialized.")

        return torch.distributed.group.WORLD, RANK, WORLD_SIZE, device
