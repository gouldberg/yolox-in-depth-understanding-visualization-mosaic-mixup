
import os

import numpy as np

import torch
from torch import distributed as dist


# ------------------------------------------------------------------------------------------
# yolox.utils : dist.py
# get_num_devices()
# ------------------------------------------------------------------------------------------

def get_num_devices():
    gpu_list = os.getenv('CUDA_VISIBLE_DEVICES', None)
    if gpu_list is not None:
        return len(gpu_list.split(','))
    else:
        devices_list_info = os.popen("nvidia-smi -L")
        devices_list_info = devices_list_info.read().strip().split("\n")
        return len(devices_list_info)


# ----------
devices_list_info = os.popen("nvidia-smi -L")
devices_list_info = devices_list_info.read().strip().split("\n")

print(devices_list_info)
print(len(devices_list_info))


# ------------------------------------------------------------------------------------------
# yolox.utils : dist.py
# get_world_size()
# ------------------------------------------------------------------------------------------

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


# ----------
print(dist.is_available())

print(dist.is_initialized())

print(dist.get_world_size())



# ------------------------------------------------------------------------------------------
# yolox.utils : dist.py
# get_rank()
# ------------------------------------------------------------------------------------------

def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


# ------------------------------------------------------------------------------------------
# yolox.utils : dist.py
# get_local_rank()
# ------------------------------------------------------------------------------------------

_LOCAL_PROCESS_GROUP = None

def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if _LOCAL_PROCESS_GROUP is None:
        return get_rank()
    # ----------
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


# ----------
print(get_rank())



# ------------------------------------------------------------------------------------------
# yolox.utils : dist.py
# get_local_size()
# ------------------------------------------------------------------------------------------

_LOCAL_PROCESS_GROUP = None

def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group, i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


# ----------
print(get_local_size())


