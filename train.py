"""
Training the character-level NjegošGPT model.

Full training:  uv run train.py
Live demo:      uv run train.py --preset=demo
Any config value can be changed with --key=value,
e.g. uv run train.py --max_iters=2000 --compile=False
"""

import math
import os
import sys
import time
from ast import literal_eval
from contextlib import nullcontext
from dataclasses import asdict

import torch

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# configuration
data_path = "data/njegos.txt"
out_dir = "out"
batch_size = 64
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.3
max_iters = 4000
eval_interval = 250
eval_iters = 200
log_interval = 10
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 100
weight_decay = 0.1
grad_clip = 1.0
seed = 1337
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
compile = device == "cuda"
preset = ""

PRESETS = {
    "demo": {"n_layer": 4, "n_head": 4, "n_embd": 256, "max_iters": 1000, "eval_interval": 100, "eval_iters": 50}
}

# simple "configurator": --key=value overrides the global variable
args = dict(a[2:].split("=", 1) for a in sys.argv[1:])
args = PRESETS.get(args.get("preset"), {}) | args
for key, val in args.items():
    assert key in globals(), f"unknown parameter: {key}"
    try:
        val = literal_eval(val)  # numbers, True/False...
    except (ValueError, SyntaxError):
        pass  # ...or a plain string
    print(f"{key} = {val}")
    globals()[key] = val
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else device
ctx = torch.autocast(device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

# data: text -> characters -> integers
with open(data_path, encoding="utf-8") as f:
    text = f.read()


chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


print(f"{len(text):,} characters, vocabulary: {vocab_size} -> {''.join(chars)!r}")
data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))

train_data, val_data = data[:n], data[n:]  # 90% training, 10% validation


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))  # random starts
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])  # shifted by 1
    if device_type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


# model
config = GPTConfig(vocab_size, block_size, n_layer, n_head, n_embd, dropout)
model = GPT(config).to(device)
raw_model = model  # uncompiled version for generation and saving

# AdamW: weight decay only on 2D tensors (matrices and embeddings), not on LayerNorm/bias
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
optimizer = torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.99), fused=device_type == "cuda")

if compile:
    try:
        print("compiling the model...")
        model = torch.compile(model)
        with ctx:
            model(*get_batch("train"))  # the first call actually triggers compilation
    except Exception as e:
        print(f"torch.compile failed ({e}), continuing without it")
        model = raw_model


def get_lr(it):
    # linear warmup, then cosine decay to min_lr
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    ratio = min(1.0, (it - warmup_iters) / (max_iters - warmup_iters))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            with ctx:
                _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def sample(n_chars, start="\n"):
    raw_model.eval()
    idx = torch.tensor([encode(start)], dtype=torch.long, device=device)
    with ctx:
        out = raw_model.generate(idx, n_chars)
    raw_model.train()
    return decode(out[0].tolist())


# training loop
log_path = os.path.join(out_dir, "log.csv")
with open(log_path, "w") as f:
    f.write("step,train_loss,val_loss\n")
ckpt_path = os.path.join(out_dir, "ckpt.pt")
best_val_loss = float("inf")
t0, t_it = time.time(), 0
for it in range(max_iters + 1):
    lr = get_lr(it)
    for g in optimizer.param_groups:
        g["lr"] = lr

    # periodically: loss estimate, a text sample and saving the best model
    if it % eval_interval == 0 or it == max_iters:
        losses = estimate_loss()
        print(f"--- step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}")
        with open(log_path, "a") as f:
            f.write(f"{it},{losses['train']:.4f},{losses['val']:.4f}\n")
        print(sample(200), "\n---")
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            ckpt = {"model": raw_model.state_dict(), "config": asdict(config), "stoi": stoi, "itos": itos}
            torch.save(ckpt | {"iter": it, "best_val_loss": best_val_loss}, ckpt_path)
        t0, t_it = time.time(), it  # don't time the evaluation
    if it == max_iters:
        break

    # one step: forward, backward, update
    x, y = get_batch("train")
    with ctx:
        logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    if it % log_interval == 0:
        loss_item = loss.item()  # syncs the GPU, so the timing is accurate
        dt = (time.time() - t0) / (it + 1 - t_it)  # average time per step
        tok_s = batch_size * block_size / dt
        print(f"step {it:5d} | loss {loss_item:.4f} | lr {lr:.2e} | {dt * 1000:.1f} ms | {tok_s:,.0f} tok/s")
        t0, t_it = time.time(), it + 1

# a long sample from the best model
raw_model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
with open(os.path.join(out_dir, "sample_final.txt"), "w", encoding="utf-8") as f:
    f.write(sample(3000))
print(f"done. best val loss: {best_val_loss:.4f}, sample in {out_dir}/sample_final.txt")
