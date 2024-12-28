from mml.model.model import Net
from mml.utils.config import ConfigS, ConfigQwen

import torch
import os

config = ConfigS()
ckpt_dir = "weights/small"
ckpt_name = "epoch_18.pt"
ckpt_path = os.path.join(ckpt_dir, ckpt_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net(
        clip_model=config.clip_model,
        text_model=config.text_model,
        ep_len=config.ep_len,
        num_layers=config.num_layers,
        n_heads=config.n_heads,
        forward_expansion=config.forward_expansion,
        dropout=config.dropout,
        max_len=config.max_len,
        device=device,
    )

ckpt = torch.load(ckpt_path, map_location=device)

model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.freeze_layers()

filtered_state_dict = {k: v for k, v in model.state_dict().items() if model.get_parameter(k).requires_grad}  

filtered_ckpt_path = os.path.join(ckpt_dir, "filtered_" + ckpt_name)
torch.save(filtered_state_dict, filtered_ckpt_path)

# filtered_ckpt = torch.load(filtered_ckpt_path, map_location=device)
# model = Net(
#         clip_model=config.clip_model,
#         text_model=config.text_model,
#         ep_len=config.ep_len,
#         num_layers=config.num_layers,
#         n_heads=config.n_heads,
#         forward_expansion=config.forward_expansion,
#         dropout=config.dropout,
#         max_len=config.max_len,
#         device=device,
#     )

# model.load_state_dict(filtered_ckpt, strict=False)


