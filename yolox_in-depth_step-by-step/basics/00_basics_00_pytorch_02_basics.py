
import os
import torch


# ------------------------------------------------------------------------------------------
# torch.no_grad
# ------------------------------------------------------------------------------------------

x = torch.tensor([1.], requires_grad=True)

print(x)

# ----------
with torch.no_grad():
    y = x * 2

y.requires_grad


# ----------
@torch.no_grad()
def doubler(x):
    return x * 2

z = doubler(x)

z.requires_grad


# ------------------------------------------------------------------------------------------
# torch.enable_grad
# ------------------------------------------------------------------------------------------

x = torch.tensor([1.], requires_grad=True)

with torch.no_grad():
    with torch.enable_grad():
        y = x * 2

y.requires_grad

y.backward()

x.grad


# ----------
@torch.enable_grad()
def doubler(x):
    return x * 2

with torch.no_grad():
    z = doubler(x)

z.requires_grad
