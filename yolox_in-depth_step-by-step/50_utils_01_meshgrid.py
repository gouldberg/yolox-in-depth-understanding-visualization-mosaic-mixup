
import os

import torch
import torch.nn as nn


############################################################################################
# ------------------------------------------------------------------------------------------
# yolox.utils
# compat.py:  meshgrid --> meshgrid basics
# ------------------------------------------------------------------------------------------

x = torch.tensor([1, 2, 3])

y = torch.tensor([4, 5, 6])

grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

print(grid_x)
print(grid_y)

print(grid_x.shape)
print(grid_y.shape)


# ----------
print(torch.dstack([grid_x, grid_y]))
print(torch.hstack([grid_x, grid_y]))
print(torch.vstack([grid_x, grid_y]))


# this is equal
print(torch.cat(tuple(torch.dstack([grid_x, grid_y]))))
print(torch.cartesian_prod(x, y))


# ----------
import matplotlib.pyplot as plt
xs = torch.linspace(-5, 5, steps=100)
ys = torch.linspace(-5, 5, steps=100)
x, y = torch.meshgrid(xs, ys, indexing='xy')
z = torch.sin(torch.sqrt(x * x + y * y))
ax = plt.axes(projection='3d')
ax.plot_surface(x.numpy(), y.numpy(), z.numpy())
plt.show()


