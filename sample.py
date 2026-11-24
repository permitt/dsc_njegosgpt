"""
Generate text from a trained NjegošGPT model.

uv run sample.py --start="Bura" --num_samples=3 --temperature=0.8 --top_k=20
"""

import argparse

import torch

from model import GPT, GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="out/ckpt.pt")
parser.add_argument("--start", default="\n")  # starting text (prompt)
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=500)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top_k", type=int, default=None)
parser.add_argument("--seed", type=int, default=1337)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the model and vocabulary from the checkpoint
ckpt = torch.load(args.ckpt, map_location=device)
model = GPT(GPTConfig(**ckpt["config"])).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
stoi, itos = ckpt["stoi"], ckpt["itos"]

start = args.start.replace("\\n", "\n")  # so that "\n" typed in the terminal becomes a newline
idx = torch.tensor([[stoi[c] for c in start]], dtype=torch.long, device=device)
for _ in range(args.num_samples):
    out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k)
    print("".join(itos[i] for i in out[0].tolist()))
    print("---------------")
