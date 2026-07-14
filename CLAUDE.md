# Robot Control Guide

You operate a simulated robot arm through the `sphinx-robot` MCP server (`via/mcp_server.py`) to complete manipulation tasks.


## General Information

You control the robot via a 3D browser GUI, similar to 3D design software or a 3D game.
The scene contains a robot arm with a gripper and a tabletop environment where you perform manipulations as instructed.

You control the robot by setting and executing a sequence of "waypoints":
* There is a blue virtual gripper in the scene, called the "target gripper". You manipulate this target gripper via tools to set a "waypoint".
* Repeat the following to complete a task, one waypoint at a time:
  * Set: move the target gripper to a desired pose (waypoint) via the tools defined in the MCP server.
  * Execute: A controller will bring the real gripper to the target gripper via a linear interpolation while obeying physical constraints.
  * Check: After the waypoint is reached, the tool will return an image showing the updated UI (outcome). Decide the next pose based on the outcome. If the outcome isn't what you intended, diagnose why it failed and retry that step or any earlier one that caused it.

For example, to pick up a cube: move the gripper above it, descend until the cube sits between the fingers, close, then lift — each step is one waypoint that you set and then execute before moving on to the next.


## Tips

### Decomposition and Planning
* Decompose the task into **small subtasks** by breaking the goal into a short sequence of simple stages.
* Prefer closed-loop control over open-loop. Adjust the plan as you observe the outcome of previous waypoints.
* Give `gripper_toggle` its own `execute_waypoint` cycle. Do NOT combine it with another pose change (translate/rotate/advance...) in one waypoint. Moving and toggling the gripper at the same time could be unpredictable.

### Precision
* Be precise with `gripper_teleport_via_click` — it's the best way to set a good initial position.
  - The clicked point lands on the gripper's approach axis, so a precise click will make future approaching easy.
  - Use `hover` to double-check whether the predicted (u,v) would land where you intend to click.
* Placing an object onto another object is a precision-sensitive task. First reason about where the held object should land, and then reason about where to set the target gripper, accounting for the relation between the gripper and the held object.
* Always check the visual from the point cloud or the camera feeds. Do NOT solely rely on the coordinates from `hover`. When necessary, make fine adjustments with translate, rotate, advance_or_retreat, and similar operations.


### Viewing and Camera Movement:
* Two camera feeds on the left
  - They provide raw camera images from 3rd-person view and wrist view.
  - The 3rd-person view is good for understanding the scene and disambiguating objects.
  - The wrist view is good for close-up analysis and fine adjustments.
  - `gripper_teleport_via_click` also accepts clicking on a camera feed. Sometimes this could be more convenient than clicking on the point cloud.
* The 3D point cloud on the right
  - This is the main workspace. It constructs the 3D scene from multiple depth cameras.
  - The initial view (resettable via `camera_reset`) is framed to be good for most operations. It is a good default so prefer working in it.
  - A pure top-down view (elevation >= 80) may not be as useful as you think, because the gripper and robot arm may block most of the workspace.
  - Sideview (orbit by changing azimuth +/- 30~45) or frontview (elevation ~ 10) could be very helpful if object-gripper or object-object relation is hard to interpret in the initial default view.
  - Move the camera gently, and reset to the initial view via `camera_reset` to recover from a failed camera operation.

### Common failures
* Grasping too high is a common failure. Descend deep so that the fingers and the object to hold have enough contact surface.
