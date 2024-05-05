
import os
import torch

import torch.nn.functional as F


# ------------------------------------------------------------------------------------------
# torch.no_grad
# ------------------------------------------------------------------------------------------

# pool of square window of size=3, stride=2
input = torch.tensor([[[1, 2, 3, 4, 5, 6, 7]]], dtype=torch.float32)

F.avg_pool1d(input, kernel_size=3, stride=2)

F.avg_pool1d(input, kernel_size=3, stride=2, padding=1)



