# Ego-Exo v2 calibration, sync, and triangulation outputs

These outputs come from one v2 recording session split into two MP4 file segments.

- `segment1/` mirrors the first MP4 segment from each camera.
- `segment2/` mirrors the second MP4 segment from each camera.
- `triangulation_eval/` compares ChArUco board-corner triangulation using 2, 3, and 4 camera combinations.
- `session_visualizer/` is a static app for inspecting synced video, audio onset envelopes, and IMU timelines.
- `handpose_labeler/` is a static app for synced left/right MediaPipe-21 hand labeling.

Raw videos and extracted WAV audio are intentionally not included. The manual ChArUco labeler is also intentionally excluded.

Key results:

- Audio sync uses `cam1` as the reference camera.
- Segment 1 audio offsets: `cam2=-6.310s`, `cam3=-12.280s`, `cam4=-3.835s`.
- Segment 2 audio offsets: `cam2=-6.305s`, `cam3=-12.280s`, `cam4=-3.835s`.
- Segment 1 best 3-camera ChArUco triangulation: `cam1+cam2+cam3`, median `1.08 mm`, p95 `2.51 mm`.
- Segment 1 all-4 ChArUco triangulation: median `1.70 mm`, p95 `2.69 mm`.
- Segment 2 best 3-camera ChArUco triangulation: `cam1+cam3+cam4`, median `1.28 mm`, p95 `2.44 mm`.
- Segment 2 all-4 ChArUco triangulation: median `1.35 mm`, p95 `2.46 mm`.
