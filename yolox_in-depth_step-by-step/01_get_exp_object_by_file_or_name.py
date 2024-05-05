import os
import sys

import importlib


############################################################################################
# ------------------------------------------------------------------------------------------
# yolox.exp.build : 
# get_exp_by_file()
# ------------------------------------------------------------------------------------------

def get_exp_by_file(exp_file):
    try:
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp()
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp


# ----------
base_dir = '/home/kswada/kw/yolox/YOLOX'

exp_file = os.path.join(base_dir, 'exps/default/yolox_s.py')


# ----------
sys.path.append(os.path.dirname(exp_file))


# ----------
print(os.path.basename(exp_file).split(".")[0])

current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])

print(current_exp)


# ----------
exp = current_exp.Exp()

print(exp)

print(dir(exp))

print(exp.exp_name)
print(f'width : {exp.width}')
print(f'depth : {exp.depth}')


# ----------
print(exp.data_dir)


############################################################################################
# ------------------------------------------------------------------------------------------
# yolox.exp.build : 
# get_exp_by_name()
# ------------------------------------------------------------------------------------------

def get_exp_by_name(exp_name):
    exp = exp_name.replace("-", "_")  # convert string like "yolox-s" to "yolox_s"
    module_name = ".".join(["yolox", "exp", "default", exp])
    exp_object = importlib.import_module(module_name).Exp()
    return exp_object


exp_name = 'yolox-s'

# convert string like "yolox-s" to "yolox_s"
exp = exp_name.replace("-", "_")

module_name = ".".join(["yolox", "exp", "default", exp])


print(exp)
print(module_name)

exp_object = importlib.import_module(module_name).Exp()

print(exp_object)

print(dir(exp_object))




