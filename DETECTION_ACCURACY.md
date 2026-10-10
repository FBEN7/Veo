# Detection accuracy against human labels

Measured accuracy, not plausibility. Everything before this was counts per
frame, positions on the pitch, and speeds that look physical -- none of which
is accuracy, and the difference mattered.

Source: Roboflow football-players-detection (CC BY 4.0), 663 human-labelled
1920x1080 images, 114 from 08fd33 -- the same match as the broadcast validation
clip. Detector: COCO yolov8m at imgsz 1280. Goalkeeper, player and referee are
merged because COCO has one person class. One-to-one matching at IoU >= 0.3.

Reproduce with `python eval_detection_groundtruth.py`.

## Results

| split | class | truth | precision | recall | F1 |
|---|---|---|---|---|---|
| valid | person | 870 | 0.83 | 0.92 | 0.88 |
| test | person | 299 | 0.80 | 0.92 | 0.86 |
| our match | person | 1218 | 0.81 | 0.97 | 0.89 |
| valid | **ball** | 35 | **0.50** | **0.43** | 0.46 |
| test | **ball** | 11 | **0.30** | **0.27** | 0.29 |
| our match | **ball** | 48 | **0.66** | **0.56** | 0.61 |

These figures were first measured in August 2026 and reproduced exactly on
14 September after the working container was lost, on a later ultralytics
release of the yolov8m weights. Every cell above matches to two decimal
places across that gap.

## Ball detection is not solved

This corrects a claim repeated for weeks. "73.5% ball coverage" and "ball
detection is solid" rested on *coverage* -- whether any candidate exists in a
frame -- and on visually checking nine frames. Coverage is not accuracy. The
"100% ball detection" figure was an artefact of a cap of three candidates per
frame being hit in 750 frames out of 750: 2250 = 750 x 3.

The misses are real, not an artefact of scoring small boxes by IoU. Scoring by
centre distance instead:

| criterion | valid | test | our match |
|---|---|---|---|
| centre within 10 px | 0.43 | 0.27 | 0.56 |
| centre within 20 px | 0.43 | 0.27 | 0.56 |
| centre within 40 px | 0.43 | 0.27 | 0.56 |

Identical at every tolerance. The error distribution is bimodal: when the ball
is found the centre is essentially exact (median 1.0 px), and when it is missed
the nearest detection is hundreds of pixels away -- a different object
entirely. Loosening the criterion recovers nothing.

The test is fair. Images are full 1920x1080 and annotated balls are about 12 px
across, matching the boxes our detector produces on the broadcast clip.

## Confidence is a trade, and the ball has no good setting

Valid split, both classes at the same threshold:

| conf | person P | person R | ball P | ball R |
|---|---|---|---|---|
| 0.02 | 0.37 | 0.98 | 0.21 | 0.54 |
| 0.05 | 0.58 | 0.97 | 0.31 | 0.49 |
| 0.10 | 0.71 | 0.96 | 0.50 | 0.43 |
| 0.15 | 0.77 | 0.95 | 0.61 | 0.40 |
| 0.25 | 0.83 | 0.92 | 0.83 | 0.29 |
| 0.40 | 0.87 | 0.85 | 0.75 | 0.17 |
| 0.50 | 0.89 | 0.80 | 0.80 | 0.11 |

Person detection has a broad usable plateau. The ball does not: recall never
exceeds 0.54 at any threshold, and buying precision costs recall immediately.
No confidence setting rescues it, because the problem is not the threshold --
COCO's `sports ball` class was never trained to find a 12-pixel object in a
1920x1080 frame.

That is an argument for a different detector rather than a different
threshold. Purpose-built architectures take several consecutive frames and
predict a heatmap, learning trajectory rather than appearance; the TrackNet
family was built for exactly this -- small, fast, motion-blurred, intermittently
invisible balls.

## Scope

Stills only. Nothing here measures event detection or tracking continuity. The
114 images from 08fd33 are near-in-domain -- same match, camera and lighting --
so those figures flatter the detector relative to unseen footage, which is
visible in the gap between the 08fd33 ball recall (0.56) and the test split
(0.27).
