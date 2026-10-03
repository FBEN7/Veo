"""A panning camera, placed between the frames its poses were solved on.

Goal poses are solved every few frames. Between them the placer used to
reuse the nearest pose, freezing the pan and then jumping it; it now
interpolates. This checks both halves: a point placed through the
interpolated pose lands where the true camera puts it, and the old
nearest-pose reading is measurably worse, so the check would notice a
regression. Runs standalone and exits non-zero on failure.
"""

import sys

import cv2
import numpy as np

from src.goal_placer import GoalPlacer
from src.goal_pose import GoalPose, ground_point
from src.midfield_pose import look_at

EYE = np.array([-64.0, 18.8, 52.3])


def pose(target, focal):
    rot = look_at(EYE, target)
    rvec, _ = cv2.Rodrigues(rot)
    return GoalPose(rvec=rvec, tvec=(-rot @ EYE).reshape(3, 1),
                    focal_px=focal, cx=640.0, cy=360.0, reprojection_px=0.0,
                    n_corners=0)


def main() -> int:
    # The camera pans 3 m across the box and zooms in over 8 frames; poses
    # are known on frames 0, 4 and 8, as the placer's grid solves them.
    def truth(frame):
        return pose(np.array([10.0 + 0.375 * frame, 0.0, 12.0]),
                    2600.0 * (1.0 + 0.01 * frame))

    placer = GoalPlacer({f: truth(f) for f in (0, 4, 8)}, grid=4)
    worst_new = worst_old = 0.0
    for frame in (1, 2, 3, 5, 6, 7):
        real = truth(frame)
        for u, v in ((640.0, 500.0), (300.0, 600.0), (1000.0, 450.0)):
            want = np.array(ground_point(real, u, v))
            got = np.array(ground_point(placer.pose_at(frame), u, v))
            near = min((0, 4, 8), key=lambda f: abs(f - frame))
            old = np.array(ground_point(truth(near), u, v))
            worst_new = max(worst_new, float(np.linalg.norm(got - want)))
            worst_old = max(worst_old, float(np.linalg.norm(old - want)))
    ok = worst_new < 0.05 and worst_old > 5 * worst_new
    print(f"   between grid poses, worst placement error: interpolated "
          f"{worst_new:.3f} m, nearest pose {worst_old:.3f} m   "
          f"{'ok' if ok else 'WRONG'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
