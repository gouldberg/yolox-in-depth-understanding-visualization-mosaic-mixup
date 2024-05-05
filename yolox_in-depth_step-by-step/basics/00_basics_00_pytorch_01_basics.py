
import os
import torch


# ------------------------------------------------------------------------------------------
# torch.views:
# PyTorch allows a tensor to be a View of an existing tensor.
# View tensor shares the same underlying data with its base tensor.
# Supporting View avoids explicit data copy,
# thus allows us to do fast and memory efficient reshaping,
# slicing and element-wise operations.
# ------------------------------------------------------------------------------------------

t = torch.rand(4, 4)

b = t.view(2, 8)

# `t` and `b` share the same underlying data.
print(t.storage().data_ptr() == b.storage().data_ptr())


# ----------
# Modifying view tensor changes base tensor as well.
b[0][0] = 3.14

print(t[0][0])


# ------------------------------------------------------------------------------------------
# torch.tensor.view:
# Returns a new tensor with the same data as the self tensor but of a different shape.
# ------------------------------------------------------------------------------------------

x = torch.randn(4, 4)
x.size()
print(x)


# ---------
y = x.view(16)
y.size()
print(y)


# ---------
# the size -1 is inferred from other dimensions
z = x.view(-1, 8)
z.size()
print(z)


# ----------
a = torch.randn(1, 2, 3, 4)
a.size()
print(a)


# Swaps 2nd and 3rd dimension
b = a.transpose(1, 2)
b.size()
print(b)


# Does not change tensor layout in memory
c = a.view(1, 3, 2, 4)
c.size()

print(b.size())
print(c.size())

torch.equal(b, c)


# ------------------------------------------------------------------------------------------
# torch.reshape
# ------------------------------------------------------------------------------------------

a = torch.arange(4.)
print(a)
print(torch.reshape(a, (2, 2)))

b = torch.tensor([[0, 1], [2, 3]])
print(b)
print(torch.reshape(b, (-1,)))


# ------------------------------------------------------------------------------------------
# torch.tensor.expand:
# Returns a new view of the self tensor with singleton dimensions expanded to a larger size.
# ------------------------------------------------------------------------------------------

x = torch.tensor([[1], [2], [3]])
x.size()
print(x)

x.expand(3, 4)

x.expand(-1, 4)


# ------------------------------------------------------------------------------------------
# torch.squeeze:
# Returns a tensor with all specified dimensions of input of size 1 removed.
# The returned tensor shares the storage with the input tensor,
# so changing the contents of one will change the contents of the other.
# ------------------------------------------------------------------------------------------

x = torch.zeros(2, 1, 2, 1, 2)
x.size()
print(x)

y = torch.squeeze(x)
y.size()
print(y)


# ----------
y = torch.squeeze(x, 0)
y.size()
print(y)

y = torch.squeeze(x, 1)
y.size()
print(y)

y = torch.squeeze(x, (1, 2, 3))
x.size()
y.size()
print(y)


# ------------------------------------------------------------------------------------------
# torch.unsqueeze:
# Returns a new tensor with a dimension of size one inserted at the specified position.
# The returned tensor shares the same underlying data with this tensor.
# ------------------------------------------------------------------------------------------

x = torch.tensor([1, 2, 3, 4])

torch.unsqueeze(x, 0)

torch.unsqueeze(x, 1)


# ------------------------------------------------------------------------------------------
# torch.tensor.unfold:
# Returns a view of the original tensor which contains all slices of size size from self tensor
# in the dimension dimension.
# Step between two slices is given by step.
# ------------------------------------------------------------------------------------------

x = torch.arange(1., 8)
print(x)

# ----------
# dimension: dimension in which unfolding happens
# size: the size of each slice that is unfolded
# step: the step between each slice

x.unfold(0, 2, 1)

x.unfold(0, 2, 2)


# ------------------------------------------------------------------------------------------
# torch.take:
# Returns a new tensor with the elements of input at the given indices.
# The input tensor is treated as if it were viewed as a 1-D tensor.
# The result takes the same shape as the indices.
# ------------------------------------------------------------------------------------------

src = torch.tensor([[4, 3, 5],[6, 7, 8]])

torch.take(src, torch.tensor([0, 2, 5]))


# ------------------------------------------------------------------------------------------
# torch.tile:
# Constructs a tensor by repeating the elements of input.
# The dims argument specifies the number of repetitions in each dimension.
# ------------------------------------------------------------------------------------------

x = torch.tensor([1, 2, 3])
x.tile((2,))

y = torch.tensor([[1, 2], [3, 4]])
torch.tile(y, (2, 2))


# ------------------------------------------------------------------------------------------
# torch.unbind:
# Removes a tensor dimension.
# Returns a tuple of all slices along a given dimension, already without it.
# ------------------------------------------------------------------------------------------

torch.unbind(torch.tensor([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]]))


torch.unbind(torch.tensor([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]]), dim=1)


# ------------------------------------------------------------------------------------------
# torch.where:
# Return a tensor of elements selected from either input or other, depending on condition.
# ------------------------------------------------------------------------------------------

x = torch.randn(3, 2)

print(x)

# condition, value if true, value if false
torch.where(x > 0, 1.0, 2.0)


# ---------
y = torch.ones(3, 2)

print(x)
print(y)

# this replace
torch.where(x > 0, x, y)


# ------------------------------------------------------------------------------------------
# torch.hstack:
#   - Stack tensors in sequence horizontally (column wise).
#   - This is equivalent to concatenation along the first axis for 1-D tensors,
#     and along the second axis for all other tensors.
# torch.vstack
#   - Stack tensors in sequence vertically (row wise).
#   - This is equivalent to concatenation along the first axis after all 1-D tensors
#     have been reshaped by torch.atleast_2d().
# torch.dstack:
#   - Stack tensors in sequence depthwise (along third axis).
#   - This is equivalent to concatenation along the third axis after 1-D and 2-D tensors
#     have been reshaped by torch.atleast_3d().
# ------------------------------------------------------------------------------------------

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

torch.hstack((a,b))
torch.vstack((a,b))
torch.dstack((a,b))

c = torch.tensor([[1],[2],[3]])
d = torch.tensor([[4],[5],[6]])

torch.hstack((c,d))
torch.vstack((c,d))
torch.dstack((c,d))


# ------------------------------------------------------------------------------------------
# torch.flip:
# Reverse the order of an n-D tensor along given axis in dims.
# ------------------------------------------------------------------------------------------

x = torch.arange(16).view(2, 2, 2, 2)

print(x)

torch.flip(x, [0, 1, 2, 3])

torch.flip(x, [0, 1, 2])

torch.flip(x, [0, 1])

torch.flip(x, [0])

torch.flip(x, [2])


# ------------------------------------------------------------------------------------------
# torch.fliplr:
# Flip tensor in the left/right direction, returning a new tensor.
# Flip the entries in each row in the left/right direction. 
# Columns are preserved, but appear in a different order than before.
# ------------------------------------------------------------------------------------------

x = torch.arange(4).view(2, 2)

print(x)

torch.fliplr(x)


y = torch.arange(12).view(2, 2, 3)

print(y)

torch.fliplr(y)



# ------------------------------------------------------------------------------------------
# torch.flipud:
# Flip tensor in the up/down direction, returning a new tensor.
# Flip the entries in each row in the up/down direction. 
# Columns are preserved, but appear in a different order than before.
# ------------------------------------------------------------------------------------------

x = torch.arange(4).view(2, 2)

print(x)

torch.flipud(x)


y = torch.arange(12).view(2, 2, 3)

print(y)

torch.flipud(y)


# ------------------------------------------------------------------------------------------
# torch.flatten:
# Flattens input by reshaping it into a one-dimensional tensor.
# If start_dim or end_dim are passed, only dimensions starting with start_dim and
# ending with end_dim are flattened. The order of elements in input is unchanged.
# ------------------------------------------------------------------------------------------

t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(t.shape)

torch.flatten(t)

# start_dim (int) – the first dim to flatten
torch.flatten(t, start_dim=1)

# end_dim (int) – the last dim to flatten
torch.flatten(t, end_dim=1)


# ------------------------------------------------------------------------------------------
# torch.ravel:
# Return a contiguous flattened tensor. A copy is made only if needed.
# ------------------------------------------------------------------------------------------

t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

torch.ravel(t)


# ------------------------------------------------------------------------------------------
# torch.roll:
# Roll the tensor input along the given dimension(s).
# Elements that are shifted beyond the last position are re-introduced at the first position.
# If dims is None, the tensor will be flattened before rolling and then restored to the original
# shape.
# ------------------------------------------------------------------------------------------

x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)

print(x)


torch.roll(x, shifts=1)

torch.roll(x, shifts=1, dims=0)

torch.roll(x, shifts=1, dims=1)

torch.roll(x, shifts=-1, dims=0)

torch.roll(x, shifts=(2, 1), dims=(0, 1))


# ------------------------------------------------------------------------------------------
# torch.sort:
# Sorts the elements of the input tensor along a given dimension in ascending order by value.
# ------------------------------------------------------------------------------------------

x = torch.randn(3, 4)

print(x)

# ----------
sorted, indices = torch.sort(x)
print(sorted)
print(indices)


# ----------
sorted, indices = torch.sort(x, 0, descending=True)
print(sorted)
print(indices)


# ----------
x = torch.tensor([0, 1] * 9)

x.sort()

# makes the sorting routine stable,
# which guarantees that the order of equivalent elements is preserved.
x.sort(stable=True)


# ------------------------------------------------------------------------------------------
# torch.fmax:
# Computes the element-wise maximum of input and other.
# torch.fmin:
# Computes the element-wise minimum of input and other.
# ------------------------------------------------------------------------------------------

a = torch.tensor([9.7, float('nan'), 3.1, float('nan')])

b = torch.tensor([-2.2, 0.5, float('nan'), float('nan')])

torch.fmax(a, b)

torch.fmin(a, b)


# ------------------------------------------------------------------------------------------
# torch.tensor.new_full:
# torch.tensor.new_empty:
# torch.tensor.new_zeros:
# ------------------------------------------------------------------------------------------

tensor = torch.ones((2,), dtype=torch.float64)

tensor.new_full((3, 4), 3.141592)


# ----------
tensor = torch.ones(())

print(tensor)

tensor.new_empty((2, 3))


# ----------
tensor = torch.tensor((), dtype=torch.float64)

tensor.new_zeros((2, 3))


