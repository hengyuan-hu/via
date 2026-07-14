## Task Specific Tips

The trick to opening a drawer is to **first** rotate the gripper so that the approach axis will point towards the handle when we use `gripper_teleport_via_click`. Then clicking the handle lands the gripper ~0.07 m directly in front of it, already aligned to drive straight in.

Example waypoints:

1. Rotate the gripper so its approach axis points roughly horizontally at the handle.
2. Use `gripper_teleport_via_click` on the **handle** of the **correct** drawer.
3. Move forward along the approach axis until the handle is between the open fingers. Check in the wrist view that the handle is centered. Use `gripper_translate` or `gripper_rotate` to center the handle between the jaws if it is off.
4. Close the gripper.
5. Pull straight back along the approach axis, keeping the gripper level, until the drawer is open (may take more than one pull).
