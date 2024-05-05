
import os
import torch


# ------------------------------------------------------------------------------------------
# load checkpoint
# ------------------------------------------------------------------------------------------

base_dir = '/home/kswada/kw/yolox/YOLOX'

# ckpt_file = os.path.join(base_dir, 'YOLOX_outputs/yolox_s/latest_ckpt.pth')
ckpt_file = os.path.join(base_dir, 'YOLOX_outputs/yolox_s/best_ckpt.pth')

device = 'cuda'
ckpt = torch.load(ckpt_file, map_location=device)


# ----------
for k, v in ckpt.items():
    print(k)


# ---------
print(f"start epoch: {ckpt['start_epoch']}")
print(f"best ap: {ckpt['best_ap']}")
print(f"current ap: {ckpt['curr_ap']}")


# ----------
for k, v in ckpt['model'].items():
    print(k)

for k, v in ckpt['optimizer'].items():
    print(k)

for k, v in ckpt['optimizer']['state'].items():
    print(k)

print(ckpt['optimizer']['param_groups'])


# ------------------------------------------------------------------------------------------
# yolox.utils : checkpoint.py
# load_ckpt()
# ------------------------------------------------------------------------------------------

def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


# ------------------------------------------------------------------------------------------
# load model
# ------------------------------------------------------------------------------------------

# model.load_state_dict(ckpt["model"])


# model = ckpt['model']

# for (module_name, module) in model.named_modules():
#     # print(module_name, module)
#     print(module_name)


# # ----------
# # (height, width)
# input_size = (640, 640)
# # input_size = (320, 320)

# batch_size = 1

# summary(
#     backbone,
#     input_size=(batch_size, 3, input_size[0], input_size[1]),
#     col_names=['input_size', 'output_size', 'num_params', 'kernel_size'],
# )

