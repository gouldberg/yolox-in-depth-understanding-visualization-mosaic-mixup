
import os

import itertools
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.sampler import Sampler


############################################################################################
# ------------------------------------------------------------------------------------------
# yolox.data.datasets  samplers.py
# InfiniteSampler
# ------------------------------------------------------------------------------------------

class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed: Optional[int] = 0,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size


# ------------------------------------------------------------------------------------------
# itertools.islice
# ------------------------------------------------------------------------------------------

data = [i for i in range(1000)]

# this is same as data[:10]
for line in itertools.islice(data, 10):
    print(line)


# ------------------------------------------------------------------------------------------
# check InfiniteSampler
# ------------------------------------------------------------------------------------------

_size = 10
_seed = 0
g = torch.Generator()
g.manual_seed(_seed)

data = torch.randperm(_size, generator=g)
print(data)

start = 3
_world_size = 3
print(data)
for line in itertools.islice(data, start, None, _world_size):
    print(line)

start = 3
_world_size = 10
print(data)
for line in itertools.islice(data, start, None, _world_size):
    print(line)


# ---------
_size = 100
_rank = 3
_world_size = 1
sampler = InfiniteSampler(
    size=_size,
    shuffle=True,
    seed=0,
    rank=_rank,
    world_size=_world_size
)

print(next(iter(sampler)))
