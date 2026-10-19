import sys, hashlib
sys.path.insert(0, "/home/user/Veo")
import torch
print("initial_seed", torch.initial_seed())
from train_player_reid import build
m = build()
w = m.head.weight.detach().numpy()
print("head hash", hashlib.md5(w.tobytes()).hexdigest(), float(w[0,0]))
s0 = torch.load("/home/user/Veo/.cache/player_reid_imagenet_s0.pt")
print("equal to s0 head:", torch.equal(s0["head.weight"], m.head.weight.detach()))
print("body equal to s0 body:", all(torch.equal(s0[k], v) for k, v in m.state_dict().items() if k.startswith("body.")))
