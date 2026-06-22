# handpose_v1

## Objective

Given egocentric video, extract accurate 3D hand pose. Make it good lol.

The end goal is a reliable pipeline that can take EgoVideo-style data and
produce 3D hand-pose labels good enough to train downstream models and evaluate
their quality.

## Core Questions

- How good is the existing EgoVerse hand-pose data when inspected visually?
- How accurate are the EgoVerse labels against ground truth, in millimeters?
- Can a model trained on EgoVerse generalize to our own egocentric video data?
- How do current hand-pose systems compare on the same inputs?
- Can multiple models be combined into a voting or consensus system to produce
  better labels than any single model?

## Evaluation First

Before training a model, we need a way to evaluate label quality.

If the source labels are already off by 5 mm, then a model trained on those
labels is unlikely to beat that error floor unless it has additional supervision
or a better correction signal. The dataset evaluation should estimate:

- 3D joint error against ground truth, reported in millimeters.
- Per-joint error, because fingertips, wrist, and occluded joints may fail in
  different ways.
- Per-frame and per-sequence visual quality.
- Failure modes such as occlusion, motion blur, hand-object interaction,
  left/right swaps, depth scale errors, and impossible hand geometry.
- Confidence calibration, meaning whether the system knows when it is likely
  wrong.

## Plan

1. Inspect EgoVerse data visually.
   - Load representative sequences.
   - Overlay 2D and projected 3D hand poses on the video.
   - Inspect easy, medium, and hard cases.
   - Identify obvious annotation artifacts and dataset biases.

2. Quantitatively evaluate EgoVerse labels.
   - Find available ground-truth or higher-confidence reference annotations.
   - Measure millimeter-level 3D error where possible.
   - Produce visual reports for the best, median, and worst examples.
   - Decide whether EgoVerse labels are good enough to train from directly.

3. Train a baseline model on EgoVerse.
   - Start with a simple reproducible training setup.
   - Track dataset version, train/val split, config, checkpoint, and metrics.
   - Validate on held-out EgoVerse data before testing on our data.

4. Run inference on our data.
   - Apply the trained model to our egocentric videos.
   - Visually inspect projected hand poses on our frames.
   - Record domain-shift failures such as camera angle, lighting, gloves,
     hand-object contact, and unusual motion.

5. Benchmark competing systems.
   - Benchmark the EgoVerse-trained baseline.
   - Benchmark Eddy's handpose system.
   - Benchmark [SAM 3D](https://ai.meta.com/research/sam3d/).
   - Benchmark [HGGT: Robust and Flexible 3D Hand Mesh Reconstruction from
     Uncalibrated Images](https://arxiv.org/html/2603.23997).
   - Compare accuracy, runtime, robustness, confidence quality, and visual
     failure modes.

6. Build a voting-style labeler.
   - Run multiple hand-pose systems on the same EgoVideo frames.
   - Align outputs into a shared 3D coordinate convention.
   - Use model agreement, confidence, temporal consistency, and hand-geometry
     constraints to select or fuse labels.
   - Flag low-agreement frames for manual review instead of silently accepting
     bad labels.

7. Produce a hand-pose dataset from EgoVideo.
   - Store labels with provenance: source video, frame id, model outputs,
     consensus method, confidence, and validation status.
   - Keep immutable dataset manifests so training runs can be reproduced.
   - Include visual QA artifacts alongside quantitative metrics.

## Benchmarks

Primary metrics:

- Mean per-joint position error (MPJPE), in millimeters.
- Procrustes-aligned MPJPE, to separate pose quality from global alignment
  errors.
- Percentage of correct keypoints (PCK) at thresholds such as 5 mm, 10 mm, and
  20 mm.
- Mesh or surface error if a method outputs a full hand mesh.
- Temporal stability across consecutive frames.

Operational metrics:

- Inference latency per frame.
- GPU memory usage.
- Failure rate on occluded or hand-object interaction frames.
- Percentage of frames requiring manual review.

Qualitative checks:

- Video overlays.
- 3D hand renderings from multiple camera views.
- Best/median/worst-case galleries.
- Side-by-side model comparisons.

## Dataset Quality Bar

The dataset is only useful if its label noise is below the accuracy target for
the model we want to train. The first milestone is therefore not "train the
model"; it is to estimate the label quality ceiling:

- If labels are consistently within the target error range, train directly.
- If labels are close but noisy, use consensus labeling and filtering.
- If labels are systematically wrong, fix the labeling pipeline before training.
- If only some frames are reliable, create a filtered high-confidence subset
  first and expand later.
