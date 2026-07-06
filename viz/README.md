# viz — interactive visualizers

Self-contained HTML pages that play back a real SkillData v1 capture so the pipeline is *visible*:
the arm moving, and the motion becoming sensor data. No build step, no dependencies, no server —
just open the file.

| File | What it shows |
|------|---------------|
| `skillsuit_2d.html` | The planar 2-DOF reach: the captured human arm (cyan) + the same motion **retargeted** onto a 2-link robot (amber), with live S2/S4 gyro + accel telemetry and the prep/active/settle phase strip. |
| `skillsuit_3d.html` | The 7-DOF `human_arm_7dof` reach in **3D** (drag to orbit the camera), with 3-axis S2/S4/S5 telemetry — the gyro is active on all three axes, confirming genuine 3D motion. |

The motion data is embedded directly in each page (a subsampled trial from the pipeline), so the
files are fully portable.

## Open it

Double-click **`start.command`** (macOS) — it opens both pages in your browser. Or open either
`.html` file directly.

## Regenerate the embedded data

The `gen_*_data.py` scripts recompute a capture from the simulation and print the JSON that gets
embedded into the pages (replacing the `const D = …` blob). Run from the repo root:

```bash
PYTHONPATH="$PWD" uv run python viz/gen_2d_data.py   # -> JSON for skillsuit_2d.html
PYTHONPATH="$PWD" uv run python viz/gen_3d_data.py   # -> JSON for skillsuit_3d.html
```

Then paste the output in place of the `const D = {…};` value near the top of the corresponding
`<script>`. (The pages are built to be edited this way — the data is the only thing that changes.)
