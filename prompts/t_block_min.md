## Instructions

The scene has three cubes of the **same size**: one blue and two green.

Goal: build a structure that is exactly **two cubes tall**:
- Bottom layer: the blue cube, resting on the table.
- Top layer: **both** green cubes, sitting side by side on top of the blue cube.

For the result to count as correct, **all** of the following must hold:
- Both green cubes rest **directly** on the blue cube — each green cube touches the blue cube's top face.
- The two green cubes are at the **same** height (same level), side by side.
- The final structure is **two cubes tall**, never three.
- The final structure is stable: each green cube has enough contact/support area on the blue cube that it will not tip or fall.

Each cube is the same size, so two greens side by side are twice as wide as the blue:

```
[green][green]   <- top layer: both greens, same level, both on the blue
   [blue ]       <- bottom layer: blue on the table (same size as one green)
==============      table
```

Incorrect (do NOT do this):
- A three-tall tower: one green on the blue, and the second green stacked on top of the first green.
- Either green cube resting on the table, or on the other green cube, instead of directly on the blue cube.
- The two green cubes ending up at different heights.

Tips:
Note that because each green cube is exactly as wide as the blue cube's top face, two greens placed side by side in the blue's default orientation cannot both get enough support area — at least one will overhang too far and tip off. Reason about how to reorient the blue cube so that both greens gain enough contact area to rest stably, side by side, at the same level.
