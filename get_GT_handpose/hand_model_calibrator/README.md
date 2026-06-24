# Hand Model Calibrator

Static browser tool for recording one personal 21-keypoint hand template.

Open:

```text
/Users/zikangjiang/dev/handpose_v1/get_GT_handpose/hand_model_calibrator/index.html
```

What it saves:

- subject metadata
- hand side
- 21 practical labels: wrist plus 4 labeled digit joints per finger
- 20 bone lengths in millimeters
- standard 21 hand keypoint names
- computed flat rest-pose keypoints in millimeters
- JSON schema: `personal_hand_model_v1`

Model note:

- This is a metric 21-landmark calibration template for early reprojection and
  triangulation checks.
- For this tool, treat every finger as having 4 labeled digit joints; the wrist
  is the shared root label.
- It is not a replacement for MANO or UmeTrack-style model fitting.
- Use this JSON as a simple personal scale prior first; later fitting should
  move to a parametric or kinematic hand model.

The browser autosaves to local storage. Use `Save File` or `Export` to create a
portable JSON file for later triangulation, reprojection, and hand-model fitting
scripts.
