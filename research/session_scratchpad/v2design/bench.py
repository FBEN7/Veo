"""Micro-benchmark on a synthetic tensor (no video, no trained weights):
v1 BallNet forward vs the same with a 9-channel motion stem added."""
import time, torch, torchvision
nn = torch.nn
torch.set_num_threads(2)
class Net(nn.Module):
    def __init__(self, motion):
        super().__init__()
        r = torchvision.models.resnet18(weights=None)
        self.conv1, self.rest = r.conv1, nn.Sequential(r.bn1, r.relu, r.maxpool)
        self.motion = nn.Conv2d(9, 64, 7, 2, 3, bias=False) if motion else None
        self.layer1, self.layer2 = r.layer1, r.layer2
        self.lat = nn.Conv2d(128, 64, 1)
        self.head = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, 1))
    def forward(self, x):
        y = self.conv1(x[:, :3])
        if self.motion is not None:
            y = y + self.motion(x[:, 3:])
        c1 = self.layer1(self.rest(y))
        up = nn.functional.interpolate(self.lat(self.layer2(c1)), size=c1.shape[2:], mode="bilinear", align_corners=False)
        return self.head(c1 + up)[:, 0]
for motion in (False, True):
    m = Net(motion).eval()
    x = torch.randn(1, 12 if motion else 3, 656, 3904)
    with torch.inference_mode():
        m(x)
        t = time.time()
        for _ in range(2): m(x)
    print("motion" if motion else "v1", "%.2f s/frame (2 threads, 656x3904)" % ((time.time() - t) / 2))
m = Net(True).train()
x = torch.randn(16, 12, 256, 256)
opt = torch.optim.AdamW(m.parameters())
torch.set_num_threads(4)
t = time.time()
for _ in range(3):
    loss = m(x).mean(); opt.zero_grad(); loss.backward(); opt.step()
print("train step batch 16 (4 threads, motion): %.2f s" % ((time.time() - t) / 3))
m = Net(False).train(); x = torch.randn(16, 3, 256, 256); opt = torch.optim.AdamW(m.parameters())
t = time.time()
for _ in range(3):
    loss = m(x).mean(); opt.zero_grad(); loss.backward(); opt.step()
print("train step batch 16 (4 threads, v1): %.2f s" % ((time.time() - t) / 3))
