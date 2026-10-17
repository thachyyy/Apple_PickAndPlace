"""Save one image per pipeline step so each stage can be inspected after a run.

Files are numbered in the order they happen, e.g.
    00_instruction.json
    01_pick1_detect_red.png      camera | color mask | detections
    02_pick1_approach.png        frontview after each motion phase
    ...
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

KEPT_COLOR = (255, 230, 0)  # apples that will be picked
SKIPPED_COLOR = (150, 150, 150)  # blobs already inside the box


class StepRecorder:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.index = 0
        self.stage = ""  # prefix for the current part of the task, e.g. "pick2"
        print(f"Recorder: saving step images to {self.out_dir.resolve()}")

    def _path(self, name, suffix):
        stem = f"{self.stage}_{name}" if self.stage else name
        path = self.out_dir / f"{self.index:02d}_{stem}{suffix}"
        self.index += 1
        return path

    def save_json(self, name, data):
        path = self._path(name, ".json")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_image(self, name, rgb):
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(self._path(name, ".png"))

    def save_camera(self, name, obs, camera):
        """Save the current image of `camera` from the observation dict."""
        key = f"{camera}_image"
        if key in obs:
            self.save_image(name, obs[key])

    def save_detection(self, color, rgb, mask, detections):
        """Three panels: raw camera image | color mask | detected apples.

        `detections` is a list of (u, v, xyz, kept) tuples.
        """
        raw = Image.fromarray(rgb)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
        annotated = raw.copy()
        draw = ImageDraw.Draw(annotated)
        radius = max(6, rgb.shape[0] // 40)
        legend = []
        for n, (u, v, xyz, kept) in enumerate(detections, start=1):
            outline = KEPT_COLOR if kept else SKIPPED_COLOR
            draw.ellipse([u - radius, v - radius, u + radius, v + radius], outline=outline, width=2)
            draw.line([u - 3, v, u + 3, v], fill=outline)
            draw.line([u, v - 3, u, v + 3], fill=outline)
            draw.text((u + radius + 2, v - radius - 2), str(n), fill=outline)
            status = "target" if kept else "already in box"
            legend.append(
                (f"{n}: pixel ({u:.0f}, {v:.0f}) -> x={xyz[0]:+.3f} y={xyz[1]:+.3f} m  [{status}]", outline)
            )
        if not legend:
            legend.append((f"no {color} apple detected", "white"))

        width, height = raw.size
        title_h, line_h = 18, 14
        sheet = Image.new("RGB", (width * 3, title_h + height + line_h * len(legend) + 6), "black")
        sheet_draw = ImageDraw.Draw(sheet)
        for i, (panel, title) in enumerate(
            [(raw, "camera"), (mask_img, f"{color} mask"), (annotated, "detections")]
        ):
            sheet.paste(panel, (i * width, title_h))
            sheet_draw.text((i * width + 4, 3), title, fill="white")
        for k, (text, fill) in enumerate(legend):
            sheet_draw.text((4, title_h + height + 3 + k * line_h), text, fill=fill)
        sheet.save(self._path(f"detect_{color}", ".png"))
