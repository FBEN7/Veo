"""Two test v2 checkpoints: untrained (build(motion=True) as is, no
provenance) and v1 weights + small random motion stem (v2 provenance, so
a training window needs --allow-seen)."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/user/Veo")
from src import ball_heatmap as bh  # noqa: E402

out = Path(sys.argv[1])
torch.manual_seed(0)
m = bh.build(motion=True)
torch.save(m.state_dict(), out / "untrained_v2.pt")
sd = torch.load("/home/user/Veo/.cache/ball_heatmap.3000.pt")
m = bh.build(motion=True)
m.load_state_dict(sd, strict=False)
g = torch.Generator().manual_seed(0)
with torch.no_grad():
    m.conv1_motion.weight.copy_(0.02 * torch.randn(m.conv1_motion.weight.shape, generator=g))
torch.save(m.state_dict(), out / "v1rand_v2.pt")
bh.write_provenance(out / "v1rand_v2.pt", {"train": bh.V2_TRAIN, "exclude": bh.V2_EXCLUDE,
                                           "note": "smoke test, v1 weights"})
(out / "v1rand_v2.pt.calib.json").write_text(json.dumps({"a": 0.8, "b": -0.5, "tau": 0.4}))
print("ok")
