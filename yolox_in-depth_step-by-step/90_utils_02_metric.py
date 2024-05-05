
import os
import torch
import psutil


# ------------------------------------------------------------------------------------------
# yolox.utils  metric.py
# get_total_and_free_memory_in_Mb
# ------------------------------------------------------------------------------------------
def get_total_and_free_memory_in_Mb(cuda_device):
    devices_info_str = os.popen(
        "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader"
    )
    devices_info = devices_info_str.read().strip().split("\n")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
        cuda_device = int(visible_devices[cuda_device])
    total, used = devices_info[int(cuda_device)].split(",")
    return int(total), int(used)


# ----------
devices_info_str = os.popen(
    "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader"
)

devices_info = devices_info_str.read().strip().split("\n")

print(devices_info_str)
print(devices_info)

if "CUDA_VISIBLE_DEVICES" in os.environ:
    visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
    cuda_device = int(visible_devices[cuda_device])

total, used = devices_info[int(cuda_device)].split(",")


# ------------------------------------------------------------------------------------------
# yolox.utils  metric.py
# occupy_mem
# ------------------------------------------------------------------------------------------

def occupy_mem(cuda_device, mem_ratio=0.9):
    """
    pre-allocate gpu memory for training to avoid memory Fragmentation.
    """
    total, used = get_total_and_free_memory_in_Mb(cuda_device)
    max_mem = int(total * mem_ratio)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x
    time.sleep(5)

# ----------
cuda_device = 0

total, used = get_total_and_free_memory_in_Mb(cuda_device)

print(f'total: {total}  used: {used}')

mem_ratio = 0.9
max_mem = int(total * mem_ratio)
block_mem = max_mem - used

print(f'max mem: {max_mem}  block_mem: {block_mem}')

x = torch.cuda.FloatTensor(256, 1024, block_mem)
del x


# ------------------------------------------------------------------------------------------
# yolox.utils  metric.py
# gpu_mem_usage
# ------------------------------------------------------------------------------------------

def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


# ----------
mem_usage_bytes = torch.cuda.max_memory_allocated()

print(f'memory usage bytes: {mem_usage_bytes / (1024 * 1024)}')



# ------------------------------------------------------------------------------------------
# yolox.utils  metric.py
# mem_usage
# ------------------------------------------------------------------------------------------

def mem_usage():
    """
    Compute the memory usage for the current machine (GB).
    """
    gb = 1 << 30
    mem = psutil.virtual_memory()
    return mem.used / gb

gb = 1 << 30

mem = psutil.virtual_memory()

print(mem.used / gb)


