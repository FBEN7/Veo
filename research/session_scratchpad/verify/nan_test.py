import sys, warnings
sys.path.insert(0, "/home/user/Veo")
import numpy as np, torch
import train_player_reid as T
torch.manual_seed(0)
model = T.build()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
rng = np.random.default_rng(0)
def step(nper):
    model.train()
    crops = [rng.integers(0,255,(80,40,3),dtype=np.uint8) for _ in range(nper*T.K)]
    labels = [k for k in range(nper) for _ in range(T.K)]
    e = model(T.to_tensor(crops))
    lab = torch.tensor(labels)
    d = torch.cdist(e, e)
    same = lab[:, None] == lab[None, :]
    other = ~same
    pos = same & ~torch.eye(len(lab), dtype=torch.bool)
    t = d[:, :, None] - d[:, None, :]
    valid = pos[:, :, None] & other[:, None, :]
    loss = torch.nn.functional.softplus(t[valid]).mean()
    opt.zero_grad()
    loss.backward()
    g_nan = any(p.grad is not None and torch.isnan(p.grad).any().item() for p in model.parameters())
    g_max = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
    opt.step()
    w_nan = any(torch.isnan(p).any().item() for p in model.parameters())
    print(f"nper={nper} loss={loss.item()} grad_nan={g_nan} grad_absmax={g_max} weights_nan={w_nan} valid={int(valid.sum())}")
step(3)
step(1)
step(1)
step(3)
# also check one-person grad from scratch (no prior Adam state)
