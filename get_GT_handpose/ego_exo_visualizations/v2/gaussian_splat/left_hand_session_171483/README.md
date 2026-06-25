# 3D Gaussian Splat Starter Export

This folder is a calibrated starter package for a 3D Gaussian Splat experiment from the v2 four-camera setup.

## What is included

- `images/cam1.jpg` ... `images/cam4.jpg`: synchronized, undistorted still frames at session time `171.483s`.
- `raw_images/cam1.jpg` ... `raw_images/cam4.jpg`: original fisheye frames for reference.
- `transforms.json`: Nerfstudio-style camera transform export using pinhole undistorted intrinsics.
- `colmap_text/`: COLMAP text-format cameras/images export using the known ChArUco extrinsics.
- `board_plane_splat_points.ply`: colored point-splat proxy for the known ChArUco board plane.
- `full_scene_feature_splat_points.ply`: sparse full-scene feature points triangulated from SIFT matches across camera views.
- `dense_stereo_splat_points.ply`: denser, noisier full-scene points from rectified stereo depth across calibrated camera pairs.
- `full_scene_combined_splat_points.ply`: board-plane splat plus sparse feature splat plus dense stereo splat.
- `board_plane_splat_preview.html`: local browser preview of the board-plane splat proxy.
- `full_scene_splat_preview.html`: local browser preview of the combined full-scene splat proxy.

## Important limitation

This is not a trained 3DGS model yet. A real trained splat needs a trainer such as Nerfstudio/gsplat/GraphDECO 3DGS. Those are not installed in this environment, and this machine currently has no CUDA GPU. Also, this capture only has four unique viewpoints, so a trained model will mostly reproduce these views and will not hallucinate unseen scene sides reliably. The full-scene proxy here is a classical multiview/stereo point splat, not learned dense Gaussian optimization.

## Recommended next step

Train on a CUDA machine or cloud instance using `transforms.json` as the camera-pose source. Treat moving hands as dynamic foreground; mask or avoid those frames if you want the static room/table scene.
