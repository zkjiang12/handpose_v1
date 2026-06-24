# Getting GT 3D Handpose Data

How do we get a dataset of good GT 3D handpose to train from?

Seems like EgoExo is the best way to do this?

OR

Try it with a stereo camera. Have a charuco board on the ground, april tags or markers on the hand. Then just run a 2D handpose model and triangulate to get 3D.

## Local Tools

- [Hand Model Calibrator](hand_model_calibrator/index.html): enter personal
  hand bone lengths and export a `personal_hand_model_v1` JSON template.
- [Ego/Exo Visualizations](ego_exo_visualizations/): generated ChArUco
  calibration, marker triangulation, and left-hand 21-keypoint outputs.
- [triangulate_hand_labels.py](triangulate_hand_labels.py): triangulate
  stereo 21-keypoint labels and render 2D/3D diagnostics.
- [visualize_charuco_extrinsics.py](visualize_charuco_extrinsics.py):
  estimate and visualize ChArUco-based camera extrinsics.
