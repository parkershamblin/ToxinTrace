"""Copy a GPT-2 checkpoint saved with Linear-swapped Conv1D layers back to standard HF Conv1D layout.

IF-Guide's construct_hf_model replaces each Conv1D with nn.Linear (weight transposed) for Kronfluence;
the fine-tuned checkpoint is saved in that layout. Here every Conv1D weight
(attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj) is transposed back so a plain GPT2LMHeadModel loads it.
"""
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=False)
for f in os.listdir(src):
    if f != "model.safetensors":
        p = os.path.join(src, f)
        if os.path.isfile(p):
            shutil.copy2(p, dst)

sd = load_file(os.path.join(src, "model.safetensors"))
suffixes = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
n = 0
for k in list(sd):
    if k.endswith(suffixes):
        sd[k] = sd[k].t().contiguous()
        n += 1
    # The swapped Linear layers trained in float32 while the rest was bfloat16; from_pretrained does not
    # recast them, so store everything in bfloat16 (the registry dtype the evals run in).
    sd[k] = sd[k].to(torch.bfloat16)
save_file(sd, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
print(f"transposed {n} tensors -> {dst}")

# Check: standard GPT2 load, then the IF-Guide swap must reproduce the saved Linear weights exactly
from transformers import AutoModelForCausalLM
sys.path.insert(0, os.path.expanduser("~/toxintrace/IF-Guide/src"))
from utils.models import replace_conv1d_modules

m = AutoModelForCausalLM.from_pretrained(dst, local_files_only=True, torch_dtype=torch.bfloat16)
replace_conv1d_modules(m)
msd = m.state_dict()
orig = load_file(os.path.join(src, "model.safetensors"))
bad = [k for k in orig if k in msd and not torch.equal(orig[k].to(torch.bfloat16), msd[k])]
print("mismatched tensors after round trip:", len(bad), bad[:3])
import collections
print("dtypes:", dict(collections.Counter(str(p.dtype) for p in m.parameters())))
with torch.no_grad():
    m.cuda().eval()(torch.tensor([[464, 3290]]).cuda())
print("forward ok")
