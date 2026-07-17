from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


DEPTH_ALPHA = 0.55
WINDOW_NAME = "Depth PNG inspector"
ZOOM_STEP = 1.25
MAX_ZOOM = 32.0
PAN_STEP_DISPLAY_PIXELS = 50

LEFT_KEYS = (81, 2424832, 65361)
UP_KEYS = (82, 2490368, 65362)
RIGHT_KEYS = (83, 2555904, 65363)
DOWN_KEYS = (84, 2621440, 65364)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect individual depth pixels in an interactive JET view."
    )
    parser.add_argument("depth_png", type=Path, help="Path to a depth PNG file.")
    return parser.parse_args()


def load_depth_image(path: Path) -> np.ndarray:
    depth_image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise SystemExit(f"Could not read depth PNG: {path}")
    if depth_image.ndim != 2:
        raise SystemExit(
            f"Expected a single-channel depth PNG, got shape {depth_image.shape}"
        )
    if depth_image.size == 0:
        raise SystemExit(f"Depth PNG is empty: {path}")
    return depth_image


class DepthImageViewer:
    def __init__(self, depth_image: np.ndarray) -> None:
        self.depth_image = depth_image
        self.height, self.width = depth_image.shape
        self.colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=DEPTH_ALPHA),
            cv2.COLORMAP_JET,
        )

        self.zoom = 1.0
        self.view_x = 0.0
        self.view_y = 0.0
        self.selected_pixel: tuple[int, int] | None = None
        self.dirty = True

    def display_to_image(self, display_x: int, display_y: int) -> tuple[int, int]:
        image_x = int(self.view_x + display_x / self.zoom)
        image_y = int(self.view_y + display_y / self.zoom)
        return (
            int(np.clip(image_x, 0, self.width - 1)),
            int(np.clip(image_y, 0, self.height - 1)),
        )

    def clamp_view(self) -> None:
        max_x = max(0.0, self.width - self.width / self.zoom)
        max_y = max(0.0, self.height - self.height / self.zoom)
        self.view_x = float(np.clip(self.view_x, 0.0, max_x))
        self.view_y = float(np.clip(self.view_y, 0.0, max_y))

    def zoom_at(self, display_x: int, display_y: int, factor: float) -> None:
        anchor_x = self.view_x + display_x / self.zoom
        anchor_y = self.view_y + display_y / self.zoom
        new_zoom = float(np.clip(self.zoom * factor, 1.0, MAX_ZOOM))
        if new_zoom == self.zoom:
            return

        self.zoom = new_zoom
        self.view_x = anchor_x - display_x / self.zoom
        self.view_y = anchor_y - display_y / self.zoom
        self.clamp_view()
        self.dirty = True

    def pan(self, display_dx: int, display_dy: int) -> None:
        self.view_x += display_dx / self.zoom
        self.view_y += display_dy / self.zoom
        self.clamp_view()
        self.dirty = True

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.view_x = 0.0
        self.view_y = 0.0
        self.dirty = True

    def on_mouse(
        self, event: int, x: int, y: int, _flags: int, _param: object
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            image_x, image_y = self.display_to_image(x, y)
            self.selected_pixel = image_x, image_y
            depth = int(self.depth_image[image_y, image_x])
            print(f"Selected x={image_x}, y={image_y}, depth={depth}")
            self.dirty = True

    def render(self) -> np.ndarray:
        image_x = self.view_x + np.arange(self.width, dtype=np.float32) / self.zoom
        image_y = self.view_y + np.arange(self.height, dtype=np.float32) / self.zoom
        map_x, map_y = np.meshgrid(image_x, image_y)
        rendered = cv2.remap(
            self.colormap,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REPLICATE,
        )

        if self.selected_pixel is not None:
            selected_x, selected_y = self.selected_pixel
            display_x = int(round((selected_x - self.view_x) * self.zoom))
            display_y = int(round((selected_y - self.view_y) * self.zoom))
            if 0 <= display_x < self.width and 0 <= display_y < self.height:
                cv2.drawMarker(
                    rendered,
                    (display_x, display_y),
                    (255, 255, 255),
                    cv2.MARKER_CROSS,
                    20,
                    2,
                    cv2.LINE_AA,
                )

        overlay = rendered.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, 68), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, rendered, 0.3, 0, dst=rendered)
        cv2.putText(
            rendered,
            f"Zoom {self.zoom:.2f}x | +/-: zoom | arrows: pan | R: reset | Q: quit",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        selected_text = "Left-click a pixel to inspect its depth"
        if self.selected_pixel is not None:
            selected_x, selected_y = self.selected_pixel
            depth = int(self.depth_image[selected_y, selected_x])
            selected_text = f"Selected: x={selected_x}, y={selected_y}, depth={depth}"
        cv2.putText(
            rendered,
            selected_text,
            (10, 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return rendered

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        try:
            while True:
                if self.dirty:
                    cv2.imshow(WINDOW_NAME, self.render())
                    self.dirty = False

                key = cv2.waitKeyEx(20)
                if key == -1:
                    continue

                ascii_key = key & 0xFF
                if key == 27 or ascii_key == ord("q"):
                    break
                if ascii_key == ord("r"):
                    self.reset_view()
                elif ascii_key in (ord("+"), ord("=")):
                    self.zoom_at(self.width // 2, self.height // 2, ZOOM_STEP)
                elif ascii_key in (ord("-"), ord("_")):
                    self.zoom_at(
                        self.width // 2, self.height // 2, 1.0 / ZOOM_STEP
                    )
                elif key in LEFT_KEYS:
                    self.pan(-PAN_STEP_DISPLAY_PIXELS, 0)
                elif key in RIGHT_KEYS:
                    self.pan(PAN_STEP_DISPLAY_PIXELS, 0)
                elif key in UP_KEYS:
                    self.pan(0, -PAN_STEP_DISPLAY_PIXELS)
                elif key in DOWN_KEYS:
                    self.pan(0, PAN_STEP_DISPLAY_PIXELS)
        finally:
            cv2.destroyWindow(WINDOW_NAME)


def main() -> None:
    args = parse_args()
    depth_image = load_depth_image(args.depth_png)
    print(
        f"Loaded {args.depth_png}: shape={depth_image.shape}, "
        f"min={int(depth_image.min())}, max={int(depth_image.max())}"
    )
    DepthImageViewer(depth_image).run()


if __name__ == "__main__":
    main()
