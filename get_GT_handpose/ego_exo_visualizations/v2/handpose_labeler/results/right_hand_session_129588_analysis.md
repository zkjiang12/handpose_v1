# Right Hand Session 129.588s Analysis

## TLDR

- Label file: `../labels/right_hand_session_129588.json`
- Primary complete triangulation: `cam1+cam4`
- Reason: `cam1+cam4` is the only camera combination with all 21 labeled keypoints.
- Median reprojection error: `0.680 px`
- Median absolute bone-length error: `3.01 mm`
- Mean absolute bone-length error: `4.99 mm`
- Max absolute bone-length error: `21.76 mm`
- Joint-position diagnostic vs GT-length-corrected pose:
  - median `12.42 mm`
  - mean `10.69 mm`
  - max `21.76 mm`

## Visualizers

- [Clean 3D hand render](right_hand_session_129588_clean_3d_handpose_render_cam1_cam4.png)
- [Interactive 3D handpose](right_hand_session_129588_interactive_3d_handpose_cam1_cam4.html)
- [Full 3D scene, cameras, board, hand](right_hand_session_129588_full_3d_scene_z_up_cam1_cam4.png)
- [2D pose comparison](right_hand_session_129588_2d_pose_comparison_cam1_cam4.png)
- [Bone error heatmap](right_hand_session_129588_bone_error_heatmap.png)
- [Triangulated results JSON](right_hand_session_129588_triangulated_results.json)
- [Bone lengths CSV](right_hand_session_129588_bone_lengths.csv)

## Coordinate Frame

The render uses a handedness-preserving display transform:

```text
display_X = raw_world_X
display_Y = -raw_world_Y
display_Z = -raw_world_Z
```

This keeps the hand visually right-handed while making `display_Z` point upward from the ChArUco board.

## Main Error Pattern

The worst errors are mostly wrist-to-MCP palm bones:

- `wrist_pinky_mcp`: triangulated `60.24 mm`, GT `82 mm`, error `-21.76 mm`
- `wrist_ring_mcp`: triangulated `68.37 mm`, GT `82 mm`, error `-13.63 mm`
- `wrist_middle_mcp`: triangulated `69.58 mm`, GT `82 mm`, error `-12.42 mm`
- `wrist_index_mcp`: triangulated `73.47 mm`, GT `82 mm`, error `-8.53 mm`

Thumb and index finger segments are comparatively more plausible because their direction produces more lateral image-plane displacement in the current top-down camera layout.

## Interpretation

The current cameras are mostly top-down / high-oblique views. That means they are good at resolving board-plane position but weaker at resolving depth/height for hand segments that point along the shared viewing direction.

For thumb/index, the joints move more sideways in the image, so the 2D evidence gives triangulation stronger metric leverage.

For wrist-to-middle/ring/pinky MCP, the hand is slanted upward/away from the top-down cameras. In the images those segments are foreshortened, and because there is no strong side view, the multiview rays do not provide enough independent depth evidence. The triangulated 3D palm therefore comes out too short.

This does not mean triangulation is just measuring 2D length. It means the current camera geometry is close to degenerate for those directions, so foreshortened image evidence dominates.

## Practical Conclusion

This setup is useful for debugging, occlusions, and first-pass pseudo-GT. It is not enough for strong metric GT handpose for all articulations.

For the next setup, add side/low-oblique cameras so at least one camera sees the palm/finger depth direction as image-plane displacement. That should reduce wrist-to-MCP and palm-depth errors substantially.
