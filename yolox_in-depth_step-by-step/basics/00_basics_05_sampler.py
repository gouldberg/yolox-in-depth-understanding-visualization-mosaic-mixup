
import os
import torch

from torch.utils.data.sampler import Sampler, BatchSampler, SequentialSampler, WeightedRandomSampler


############################################################################################
# ------------------------------------------------------------------------------------------
# SequentialSampler
# ------------------------------------------------------------------------------------------

dataset = [i for i in range(10)]

print(list(SequentialSampler(dataset)))


# ------------------------------------------------------------------------------------------
# BatchSampler, SequentialSampler
# ------------------------------------------------------------------------------------------

print(list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False)))

print(list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True)))



# ------------------------------------------------------------------------------------------
# WeightedRandomSampler
# ------------------------------------------------------------------------------------------

batch_size = 20

# dataset has 10 class-1 samples, 1 class-2 samples, etc.
class_sample_count = [10, 1, 20, 3, 4]

weights = 1 / (torch.Tensor(class_sample_count)*1e-5)

# convert to float64
weights = weights.double()
print(weights)


# ----------
sampler = WeightedRandomSampler(weights, batch_size)


# for example to use
trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, sampler=sampler)


