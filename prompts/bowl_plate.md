## Task Specific Tips

### How to pick up a bowl:

To pick up a bowl, we need to pinch it by its side wall with a straight top-down descent. Here we give example waypoints assuming we pick it from the right (+Y direction).

Example waypoints:

1. Locate the bowl, and click on the rightmost point on the rim (pinch point). Trick: The click point **must** land on the bowl not on the table; After `gripper_teleport_via_click`, read the salient point's Y from the output; nudge the click along the rim and re-check until Y stops increasing while the click point still lands on the bowl not on the table.
2. Descend the gripper low for a firm grasp.
3. Close the gripper.
4. Lift the gripper. After a lift, judge grasp success from the point cloud (frontview could be helpful), not the wrist view.

### How to place onto a plate:

This task requires the held object to be placed **exactly at the center** to claim a success. When holding a bowl, the held object's center is offset from the gripper, so we need to account for that when deciding the target location for the gripper.

Example waypoints:

1. `gripper_teleport_via_click` on the desired location of the plate so that the held object will roughly appear above the center of the plate. E.g. holding the bowl by its right (+Y) rim, click on the right (+Y) half of the plate. Adjust the height after clicking to avoid collision.
2. Observe the outcome of the first waypoint and adjust the gripper if needed. Do NOT solely trust the readout from `hover`. Check the visual from the camera and 3D point cloud carefully as well.
3. Descend until the object rests on the plate (contact stops the descent), with a final small XY nudge if needed.
4. Open the gripper to release the held object.
