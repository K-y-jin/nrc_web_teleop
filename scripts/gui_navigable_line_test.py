#!/usr/bin/env python3
"""is_line_navigable 튜닝용 디버그 GUI.

nav color 이미지를 클릭하면 DEPTH_REF 픽셀에서 클릭 픽셀까지 직선상의
pred_depth 프로파일을 plot 한다. 동시에 is_line_navigable 판정과 진단치
(total/expected per step/max|Δ|/threshold)를 표기한다.

사용:
    python3 scripts/gui_navigable_line_test.py [--dir DIR]

DIR 안의 color_*.png를 listbox에 채우고, 같은 디렉토리의 pred_*.png를
depth로 사용한다. pred 파일이 없으면 Depth-Anything으로 추론.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
import tkinter as tk  # noqa: E402
from tkinter import ttk  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nrc_web_teleop_helpers.constants import (  # noqa: E402
    DEPTH_REF,
    NAVIGABLE_LINE_SAMPLES,
    NAVIGABLE_MAX_STEP_RATIO,
)

DEFAULT_DIR = os.path.join(
    REPO_ROOT, "output", "tilt_neg_45deg_nav_pred", "nav"
)


def is_line_navigable_debug(pred_depth, click_row, click_col):
    """is_line_navigable과 동일한 로직 + 진단치 함께 반환."""
    ref_r, ref_c, _ = DEPTH_REF
    h, w = pred_depth.shape[:2]
    rs = np.clip(
        np.linspace(ref_r, click_row, NAVIGABLE_LINE_SAMPLES).astype(int),
        0, h - 1,
    )
    cs = np.clip(
        np.linspace(ref_c, click_col, NAVIGABLE_LINE_SAMPLES).astype(int),
        0, w - 1,
    )
    line = pred_depth[rs, cs].astype(np.float32)
    total = float(line[-1] - line[0])
    steps = np.diff(line)
    expected = total / (NAVIGABLE_LINE_SAMPLES - 1) if total > 0 else 0.0
    max_abs_step = float(np.max(np.abs(steps))) if steps.size else 0.0
    threshold = expected * NAVIGABLE_MAX_STEP_RATIO
    ok = total > 0.0 and max_abs_step <= threshold
    return {
        "ok": ok,
        "line": line,
        "rs": rs,
        "cs": cs,
        "steps": steps,
        "total": total,
        "expected": expected,
        "max_abs_step": max_abs_step,
        "threshold": threshold,
    }


def load_pred(path_color):
    """color_N.png에 대응하는 pred_N.png를 같은 디렉토리에서 로드."""
    d = os.path.dirname(path_color)
    base = os.path.basename(path_color)
    pred_path = os.path.join(d, base.replace("color_", "pred_", 1))
    if not os.path.isfile(pred_path):
        return None
    return cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)


class NavigableLineGui:
    def __init__(self, root, image_dir):
        self.root = root
        root.title("is_line_navigable debug")
        self.image_dir = image_dir
        self.color_image = None
        self.pred_depth = None
        self.selected_name = None

        top = tk.Frame(root)
        top.pack(fill=tk.X, padx=8, pady=6)
        tk.Label(top, text=f"dir: {image_dir}").pack(side=tk.LEFT)

        body = tk.Frame(root)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = tk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(left, text="color_*.png").pack()
        self.listbox = tk.Listbox(left, width=24, height=24, exportselection=False)
        self.listbox.pack(side=tk.LEFT, fill=tk.Y)
        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.listbox.yview)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        self.fig, (self.ax_img, self.ax_prof) = plt.subplots(
            1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1]}
        )
        self.ax_steps = self.ax_prof.twinx()
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        self.status = tk.Label(root, text="select an image, then click on it",
                               anchor="w", font=("TkDefaultFont", 10))
        self.status.pack(fill=tk.X, padx=8, pady=4)

        self._populate_list()

    def _populate_list(self):
        names = sorted(
            n for n in os.listdir(self.image_dir)
            if n.startswith("color_") and n.endswith(".png")
        )
        for n in names:
            self.listbox.insert(tk.END, n)

    def on_select(self, _evt):
        sel = self.listbox.curselection()
        if not sel:
            return
        name = self.listbox.get(sel[0])
        if name == self.selected_name:
            return
        self.selected_name = name
        path = os.path.join(self.image_dir, name)
        color = cv2.imread(path)
        if color is None:
            self.status.config(text=f"failed to read {path}")
            return
        pred = load_pred(path)
        if pred is None:
            self.status.config(text=f"no pred png for {name}")
            return
        if pred.shape[:2] != color.shape[:2]:
            pred = cv2.resize(pred, (color.shape[1], color.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        self.color_image = color
        self.pred_depth = pred
        self._draw_image()
        self.ax_prof.clear()
        self.ax_steps.clear()
        self.ax_prof.set_title("click a pixel on the left image")
        self.canvas.draw_idle()
        self.status.config(
            text=f"{name}  pred dtype={pred.dtype} "
                 f"min={int(pred.min())} max={int(pred.max())} "
                 f"DEPTH_REF={DEPTH_REF}"
        )

    def _draw_image(self):
        self.ax_img.clear()
        self.ax_img.imshow(cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB))
        self.ax_img.plot(DEPTH_REF[1], DEPTH_REF[0], "go", markersize=8,
                         label="DEPTH_REF")
        self.ax_img.legend(loc="upper right")
        self.ax_img.set_title(self.selected_name or "")
        self.ax_img.set_axis_off()

    def on_click(self, event):
        if event.inaxes is not self.ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.pred_depth is None:
            return
        click_col = int(event.xdata)
        click_row = int(event.ydata)
        info = is_line_navigable_debug(self.pred_depth, click_row, click_col)
        verdict_color = "lime" if info["ok"] else "red"
        verdict = "NAVIGABLE" if info["ok"] else "NOT NAVIGABLE"

        # DEPTH_REF → 클릭 픽셀의 단순 기울기 (depth/px, 이미지상 직선거리 기준).
        ref_r, ref_c, ref_z = DEPTH_REF
        pixel_dist = float(np.hypot(click_row - ref_r, click_col - ref_c))
        click_z = float(self.pred_depth[click_row, click_col])
        slope_per_px = ((click_z - ref_z) / pixel_dist) if pixel_dist > 0 else 0.0

        self._draw_image()
        self.ax_img.plot(info["cs"], info["rs"], color=verdict_color, linewidth=2)
        self.ax_img.plot(click_col, click_row, "o", color=verdict_color,
                         markersize=8)

        self.ax_prof.clear()
        self.ax_steps.clear()
        x = np.arange(NAVIGABLE_LINE_SAMPLES)
        self.ax_prof.plot(x, info["line"], "-o", markersize=3, color="C0",
                          label="pred_depth")
        self.ax_prof.plot([0, NAVIGABLE_LINE_SAMPLES - 1],
                          [info["line"][0], info["line"][-1]],
                          "k--", alpha=0.5, label="endpoint linear")
        self.ax_prof.set_xlabel("sample index (0=ref, N-1=click)")
        self.ax_prof.set_ylabel("pred_depth (raw)", color="C0")
        self.ax_prof.grid(True, alpha=0.3)
        self.ax_prof.legend(loc="upper left")

        self.ax_steps.plot(np.arange(len(info["steps"])) + 0.5, info["steps"],
                           color="orange", alpha=0.7, label="Δ per step")
        self.ax_steps.axhline(info["threshold"], color="orange", linestyle=":",
                              alpha=0.8, label=f"±thr ({info['threshold']:.2f})")
        self.ax_steps.axhline(-info["threshold"], color="orange", linestyle=":",
                              alpha=0.8)
        self.ax_steps.axhline(0, color="gray", linewidth=0.5)
        self.ax_steps.set_ylabel("Δdepth per step", color="orange")
        self.ax_steps.legend(loc="upper right")

        self.ax_prof.set_title(
            f"{verdict}  | total={info['total']:.1f}  "
            f"exp/step={info['expected']:.2f}  "
            f"max|Δ|={info['max_abs_step']:.2f}  "
            f"thr={info['threshold']:.2f}\n"
            f"slope (ref→click) = {slope_per_px:.3f} depth/px  "
            f"(Δz={click_z - ref_z:+.1f}, Δpx={pixel_dist:.1f})",
            color=("green" if info["ok"] else "red"),
        )
        self.canvas.draw_idle()
        self.status.config(
            text=f"click=(r={click_row}, c={click_col}) "
                 f"z={click_z:.1f}  "
                 f"slope={slope_per_px:.3f} depth/px  -> {verdict}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=DEFAULT_DIR,
                   help="directory containing color_*.png and pred_*.png")
    args = p.parse_args()
    if not os.path.isdir(args.dir):
        raise SystemExit(f"not a directory: {args.dir}")
    root = tk.Tk()
    NavigableLineGui(root, args.dir)
    root.mainloop()


if __name__ == "__main__":
    main()
