"""Two test v2 checkpoints from v1's weights: conv1_motion zero (must give
v1's candidates) and small random (the motion path must matter), each with
v2 training provenance (so training windows need --allow-seen)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/user/Veo")
from src import ball_heatmap as bh  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
sd = torch.load("/home/user/Veo/.cache/ball_heatmap.3000.pt")
for name, scale in (("v2zero.pt", 0.0), ("v2rand.pt", 0.02)):
    m = bh.build(motion=True)
    m.load_state_dict(sd, strict=False)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        m.conv1_motion.weight.copy_(scale * torch.randn(m.conv1_motion.weight.shape, generator=g))
    torch.save(m.state_dict(), out / name)
    bh.write_provenance(out / name, {"train": bh.V2_TRAIN, "exclude": bh.V2_EXCLUDE,
                                     "note": "smoke test, v1 weights"})
    print("wrote", out / name)
