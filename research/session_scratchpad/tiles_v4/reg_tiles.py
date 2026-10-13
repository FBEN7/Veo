"""Run ball_tiles.main (a given file) with a fake video reader (a fixed
gradient image), so keys and sheets can be compared without decoding.
usage: reg_tiles.py MODULE_PATH -- ball_tiles args..."""
import importlib.util
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/user/Veo")
mod_path = sys.argv[1]
args = sys.argv[sys.argv.index("--") + 1:]

IMG = (np.indices((1080, 4096)).sum(axis=0) % 251).astype(np.uint8)
IMG = np.dstack([IMG, IMG[:, ::-1], IMG.T[:1080, :4096] if False else IMG])


class FakeCap:
    def __init__(self, path):
        self.n = 0

    def read(self):
        self.n += 1
        if self.n > 20000:
            return False, None
        return True, IMG

    def release(self):
        pass


cv2.VideoCapture = FakeCap
spec = importlib.util.spec_from_file_location("bt_under_test", mod_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.argv = ["ball_tiles.py"] + args
m.main()
