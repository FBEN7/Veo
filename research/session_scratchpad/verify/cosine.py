import sys, torch
sys.path.insert(0, "/home/user/Veo")
from train_player_reid import build, load_crops, evaluate
m3 = load_crops([".cache/soccertrack_v2/reid_val/st2_118575_1st_f015000.pkl"], pool=False)
m5 = load_crops([".cache/soccertrack_v2/reid_train/st2_118577_1st_f030000.pkl"], pool=False)
for w in [".cache/player_reid_st2.pt", ".cache/player_reid_imagenet_s0.pt", sys.argv[1]]:
    m = build(); m.load_state_dict(torch.load(w))
    evaluate(m, m3, f"M3 {w}")
    if "st2" not in w:
        evaluate(m, m5, f"M5 {w}")
