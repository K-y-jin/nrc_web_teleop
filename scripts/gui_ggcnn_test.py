import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

from nrc_web_teleop_helpers.ggcnn.ggcnn_torch import predict
from nrc_web_teleop_helpers.da.utils import get_pred_depth

DEFAULT_INPUT_DIR = "/root/ament_ws/src/nrc_web_teleop/output/samples/gripper"


def get_ggcnn_results(normalized_depth_image):
    """Depth 이미지를 GGCNN에 입력하여 q_out, ang(rad), width(px), depth를 image 크기로 반환한다."""
    depth_nan_mask = None # normalized_depth_image == 0
    q_out, ang_out, width_out, depth_out = predict(
        normalized_depth_image, process_depth=True, crop_size=None, out_size=None,
        depth_nan_mask=depth_nan_mask, crop_y_offset=0,
        filters=(False, False, False),
    )
    image_size = (normalized_depth_image.shape[1], normalized_depth_image.shape[0])
    q_out = (q_out - q_out.min()) / (q_out.max() - q_out.min() + 1e-8)
    print("q_out shape: ", q_out.shape)
    q_out = cv2.resize((255.0 * np.clip(q_out, 0.0, 1.0)).astype(np.uint8), image_size, cv2.INTER_AREA)
    depth_out = cv2.resize((255.0 * np.clip(depth_out, 0.0, 1.0)).astype(np.uint8), image_size, cv2.INTER_AREA)
    ang_out = cv2.resize(ang_out.astype(np.float32), image_size, cv2.INTER_LINEAR)
    width_out = cv2.resize(width_out.astype(np.float32), image_size, cv2.INTER_LINEAR)
    return q_out, depth_out, ang_out, width_out


def bgr_to_tk(img_bgr, max_size=360):
    """OpenCV BGR 이미지를 max_size에 맞게 축소한 후 Tkinter용 PhotoImage와 적용된 scale을 반환한다."""
    h, w = img_bgr.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(img_rgb)), scale


class GgcnnGui:
    def __init__(self, root):
        """GUI 위젯(상단 버튼/디렉토리 라벨, 좌측 파일 리스트, 우측 이미지 패널)을 구성한다."""
        self.root = root
        root.title("GGCNN Test GUI")

        self.image_dir = None
        self.color_image = None
        self.selected_name = None
        self._imgs = {}
        self._input_scale = 1.0
        self.depth_file_image = None
        self.full_pred_depth = None
        self.predicted_depth = None
        self.ref_rcz = None
        self.q_out = None
        self.ang_out = None
        self.width_out = None
        self.click_rc = None
        self.peak_rc = None
        self.crop_tl = None  # crop top-left (row, col)
        self.crop_br = None  # crop bottom-right (row, col)
        self.crop_selecting = False  # True while selecting crop region

        top = tk.Frame(root)
        top.pack(fill=tk.X, padx=8, pady=6)
        tk.Button(top, text="Select Input Dir", command=self.select_dir).pack(side=tk.LEFT)
        self.dir_label = tk.Label(top, text="(no directory)")
        self.dir_label.pack(side=tk.LEFT, padx=8)
        tk.Label(top, text="q_out radius:").pack(side=tk.LEFT, padx=(16, 2))
        self.radius_var = tk.IntVar(value=40)
        tk.Spinbox(top, from_=1, to=500, width=5, textvariable=self.radius_var).pack(side=tk.LEFT)
        self.crop_btn = tk.Button(top, text="Set Crop", command=self.toggle_crop_mode)
        self.crop_btn.pack(side=tk.LEFT, padx=(16, 2))
        tk.Button(top, text="Clear Crop", command=self.clear_crop).pack(side=tk.LEFT, padx=2)

        body = tk.Frame(root)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = tk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(left, text="color_*.png").pack()
        self.listbox = tk.Listbox(left, width=28, height=20, exportselection=False)
        self.listbox.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.listbox.yview)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox.config(yscrollcommand=sb.set)

        right = tk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8)

        self.input_label = self._make_panel(right, "Input color", 0, 0)
        self.depth_file_label = self._make_panel(right, "Depth file", 0, 1)
        self.depth_label = self._make_panel(right, "Depth (predicted)", 1, 0)
        self.overlay_label = self._make_panel(right, "q_out overlay", 1, 1)
        self.input_label.bind("<Button-1>", self.on_pixel_click)
        self.rgb_label = tk.Label(right, text="RGB: -", font=("TkDefaultFont", 10))
        self.rgb_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=4)
        self.ref_label = tk.Label(right, text="ref_rcz: -", font=("TkDefaultFont", 10))
        self.ref_label.grid(row=3, column=0, columnspan=2, sticky="w", padx=4)
        self.peak_label = tk.Label(right, text="peak: -", font=("TkDefaultFont", 10))
        self.peak_label.grid(row=4, column=0, columnspan=2, sticky="w", padx=4)

    def _make_panel(self, parent, title, row, col, columnspan=1):
        """이미지를 표시할 LabelFrame과 내부 Label을 생성하여 grid에 배치한다."""
        frame = tk.LabelFrame(parent, text=title)
        frame.grid(row=row, column=col, columnspan=columnspan, padx=4, pady=4, sticky="nsew")
        lbl = tk.Label(frame, width=50, height=20, bg="#222")
        lbl.pack(padx=4, pady=4)
        return lbl

    def select_dir(self):
        """입력 디렉토리를 선택 다이얼로그로 고른다."""
        path = filedialog.askdirectory(title="Select input directory")
        if path:
            self.load_dir(path)

    def load_dir(self, path):
        """주어진 디렉토리의 color_*.png 파일들을 리스트박스에 채운다."""
        if not os.path.isdir(path):
            return
        self.image_dir = path
        self.dir_label.config(text=path)
        names = sorted(
            n for n in os.listdir(path)
            if n.startswith("color_") and n.endswith(".png")
        )
        self.listbox.delete(0, tk.END)
        for n in names:
            self.listbox.insert(tk.END, n)

    def on_select(self, _evt):
        """리스트박스 인덱스 변경 이벤트: 선택된 color/depth 파일을 모두 로드해 표시한다."""
        sel = self.listbox.curselection()
        if not sel:
            return
        name = self.listbox.get(sel[0])
        if name == self.selected_name:
            return
        self.selected_name = name
        self._load_color_file(name)
        self._load_depth_file(name)
        self.run()

    def _load_color_file(self, name):
        """color_*.png을 읽어 보관하고 예측 결과 패널과 클릭 상태를 초기화한다."""
        img = cv2.imread(os.path.join(self.image_dir, name))
        if img is None:
            return
        self.color_image = img
        self.predicted_depth = None
        self.q_out = None
        self.ang_out = None
        self.width_out = None
        self.click_rc = None
        self.peak_rc = None
        self.rgb_label.config(text="RGB: -")
        self.peak_label.config(text="peak: -")
        self.depth_label.config(image="")
        self.overlay_label.config(image="")

    def _load_depth_file(self, name):
        """동일 번호의 depth_*.png을 읽어 Depth file 패널에 표시하고 ref_rcz를 계산한다."""
        depth_name = name.replace("color_", "depth_", 1)
        depth_img = cv2.imread(os.path.join(self.image_dir, depth_name), cv2.IMREAD_UNCHANGED)
        self.depth_file_image = depth_img
        self.ref_rcz = self._compute_ref_rcz()
        if self.ref_rcz is None:
            self.ref_label.config(text="ref_rcz: -")
        else:
            rr, rc, rz = self.ref_rcz
            self.ref_label.config(text=f"ref_rcz: ({rr}, {rc}, {rz})")
        self._refresh_input_display()
        if depth_img is None:
            self.depth_file_label.config(image="")
            return
        if depth_img.ndim == 2:
            vis = cv2.applyColorMap(
                cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_JET,
            )
        else:
            vis = depth_img.copy()
        if self.ref_rcz is not None:
            r, c, _ = self.ref_rcz
            cv2.drawMarker(vis, (c, r), (0, 255, 0), markerType=cv2.MARKER_CROSS,
                           markerSize=20, thickness=2)
        self._show(self.depth_file_label, vis)

    def toggle_crop_mode(self):
        """크롭 선택 모드를 토글한다. 활성화 시 첫 클릭=좌상단, 두 번째 클릭=우하단."""
        self.crop_selecting = not self.crop_selecting
        if self.crop_selecting:
            self.crop_tl = None
            self.crop_br = None
            self.crop_btn.config(relief=tk.SUNKEN, text="Selecting...")
        else:
            self.crop_btn.config(relief=tk.RAISED, text="Set Crop")

    def clear_crop(self):
        """크롭 영역을 해제하고 전체 이미지로 GGCNN을 재실행한다."""
        self.crop_tl = None
        self.crop_br = None
        self.crop_selecting = False
        self.crop_btn.config(relief=tk.RAISED, text="Set Crop")
        self._refresh_input_display()
        self.run()

    def _pixel_from_event(self, event):
        """클릭 이벤트로부터 원본 이미지의 (row, col)을 반환한다. 범위 밖이면 None."""
        if self.color_image is None or self._input_scale <= 0:
            return None
        tkimg = self._imgs.get(self.input_label)
        if tkimg is None:
            return None
        lbl_w, lbl_h = self.input_label.winfo_width(), self.input_label.winfo_height()
        img_w, img_h = tkimg.width(), tkimg.height()
        off_c = max((lbl_w - img_w) // 2, 0)
        off_r = max((lbl_h - img_h) // 2, 0)
        dc, dr = event.x - off_c, event.y - off_r
        if not (0 <= dc < img_w and 0 <= dr < img_h):
            return None
        col = int(dc / self._input_scale)
        row = int(dr / self._input_scale)
        h, w = self.color_image.shape[:2]
        if not (0 <= col < w and 0 <= row < h):
            return None
        return (row, col)

    def on_pixel_click(self, event):
        """입력 이미지 패널에서 클릭한 픽셀의 원본 (row, col) 및 RGB/Depth 값을 표시한다."""
        rc = self._pixel_from_event(event)
        if rc is None:
            return
        row, col = rc

        if self.crop_selecting:
            if self.crop_tl is None:
                self.crop_tl = (row, col)
                self.crop_btn.config(text="Click BR...")
                self._refresh_input_display()
                return
            else:
                self.crop_br = (row, col)
                # 좌상단/우하단 정렬
                r0 = min(self.crop_tl[0], self.crop_br[0])
                c0 = min(self.crop_tl[1], self.crop_br[1])
                r1 = max(self.crop_tl[0], self.crop_br[0])
                c1 = max(self.crop_tl[1], self.crop_br[1])
                self.crop_tl = (r0, c0)
                self.crop_br = (r1, c1)
                self.crop_selecting = False
                self.crop_btn.config(relief=tk.RAISED, text="Set Crop")
                self._refresh_input_display()
                self.run_crop()
                return

        h, w = self.color_image.shape[:2]
        b, g, red = self.color_image[row, col]
        d_file = self._sample_at(self.depth_file_image, row, col, h, w)
        d_pred = self._sample_at(self.predicted_depth, row, col, h, w)
        self.click_rc = (row, col)
        try:
            radius = int(self.radius_var.get())
        except (tk.TclError, ValueError):
            radius = 40
        self.peak_rc = self._find_q_peak_near(row, col, h, w, radius=radius)
        self._refresh_input_display()
        self.rgb_label.config(
            text=f"(r={row}, c={col})  RGB: ({red}, {g}, {b})  Depth: {d_file}  Depth(pred): {d_pred}"
        )
        self._update_peak_label(h, w)

    def _update_peak_label(self, ref_h, ref_w):
        """peak_rc 위치의 q_out/ang/width 값을 peak_label에 표기한다."""
        if self.peak_rc is None or self.q_out is None:
            self.peak_label.config(text="peak: -")
            return
        pr, pc = self.peak_rc
        qh, qw = self.q_out.shape[:2]
        qr = min(int(pr * qh / ref_h), qh - 1)
        qc = min(int(pc * qw / ref_w), qw - 1)
        q_v = int(self.q_out[qr, qc])
        ang_v = float(self.ang_out[qr, qc]) if self.ang_out is not None else float("nan")
        width_v = float(self.width_out[qr, qc]) if self.width_out is not None else float("nan")
        self.peak_label.config(
            text=f"peak (r={pr}, c={pc})  q_out={q_v}  ang={ang_v:.3f} rad  width={width_v:.2f}"
        )

    def _find_q_peak_near(self, row, col, ref_h, ref_w, radius=40):
        """q_out 좌표계에서 (row, col) 주변 radius 내 최댓값 위치를 color 좌표계로 환산해 반환한다."""
        q = self.q_out
        if q is None:
            return None
        qh, qw = q.shape[:2]
        qr = int(row * qh / ref_h)
        qc = int(col * qw / ref_w)
        r0 = max(qr - radius, 0)
        r1 = min(qr + radius + 1, qh)
        c0 = max(qc - radius, 0)
        c1 = min(qc + radius + 1, qw)
        patch = q[r0:r1, c0:c1]
        if patch.size == 0:
            return None
        local = int(np.argmax(patch))
        pr, pc = divmod(local, patch.shape[1])
        peak_qr, peak_qc = r0 + pr, c0 + pc
        peak_row = int(peak_qr * ref_h / qh)
        peak_col = int(peak_qc * ref_w / qw)
        return (peak_row, peak_col)

    @staticmethod
    def _sample_at(arr, row, col, ref_h, ref_w):
        """ref_h x ref_w 좌표계의 (row, col)에 대응하는 arr 픽셀 값을 문자열로 반환한다."""
        if arr is None:
            return "-"
        ah, aw = arr.shape[:2]
        sr = min(int(row * ah / ref_h), ah - 1)
        sc = min(int(col * aw / ref_w), aw - 1)
        v = arr[sr, sc]
        if hasattr(v, "__len__"):
            return "(" + ",".join(str(int(ch)) for ch in v) + ")"
        return str(int(v))

    def _refresh_input_display(self):
        """입력 color 이미지에 ref(녹색 +), 클릭(노란 +), q_out peak(빨간 ^) 마커를 오버레이한다."""
        if self.color_image is None:
            return
        vis = self.color_image.copy()
        if self.ref_rcz is not None:
            r, c, _ = self.ref_rcz
            cv2.drawMarker(vis, (c, r), (0, 255, 0), markerType=cv2.MARKER_CROSS,
                           markerSize=20, thickness=2)
        if self.click_rc is not None:
            r, c = self.click_rc
            cv2.drawMarker(vis, (c, r), (0, 255, 255), markerType=cv2.MARKER_CROSS,
                           markerSize=20, thickness=2)
        if self.crop_tl is not None:
            if self.crop_br is not None:
                cv2.rectangle(vis,
                              (self.crop_tl[1], self.crop_tl[0]),
                              (self.crop_br[1], self.crop_br[0]),
                              (255, 0, 255), 2)
            else:
                cv2.drawMarker(vis, (self.crop_tl[1], self.crop_tl[0]),
                               (255, 0, 255), markerType=cv2.MARKER_CROSS,
                               markerSize=20, thickness=2)
        if self.peak_rc is not None:
            r, c = self.peak_rc
            cv2.drawMarker(vis, (c, r), (0, 0, 255), markerType=cv2.MARKER_TRIANGLE_UP,
                           markerSize=20, thickness=2)
            if self.ang_out is not None and self.width_out is not None:
                ah, aw = self.ang_out.shape[:2]
                ih, iw = self.color_image.shape[:2]
                ar = min(int(r * ah / ih), ah - 1)
                ac = min(int(c * aw / iw), aw - 1)
                angle = float(self.ang_out[ar, ac])
                width = float(self.width_out[ar, ac])
                half = max(width / 2.0, 1.0)
                dx = half * np.cos(angle)
                dy = half * np.sin(angle)
                p1 = (int(round(c - dx)), int(round(r - dy)))
                p2 = (int(round(c + dx)), int(round(r + dy)))
                cv2.line(vis, p1, p2, (0, 0, 255), 2)
        self._show(self.input_label, vis)

    def _compute_ref_rcz(self):
        """depth_file_image에서 0/saturation을 제외한 픽셀들 중 z가 최대인 픽셀을 골라 [r, c, z]를 반환한다."""
        d = self.depth_file_image
        if d is None or d.ndim != 2:
            return None
        max_v = np.iinfo(d.dtype).max if np.issubdtype(d.dtype, np.integer) else d.max()
        mask = (d > 0) & (d < max_v)
        rs, cs = np.where(mask)
        if rs.size == 0:
            return None
        vals = d[rs, cs].astype(np.int64)
        idx = int(np.argmax(vals))
        # r, c, z = int(rs[idx]), int(cs[idx]), int(vals[idx])
        r, c, z = [200, 130, d[200, 130]]
        print(f"[ref_rcz] count={vals.size} min={vals.min()} max={vals.max()} picked=({r},{c},{z})")
        return [r, c, z]

    def run(self):
        """전체 RGB로 depth를 예측하고 전체 pred depth로 GGCNN을 실행해 결과를 표시한다."""
        if self.color_image is None:
            return
        # 전체 이미지로 depth 예측
        depth_image = get_pred_depth(self.color_image, ref_rcz=self.ref_rcz)
        self.full_pred_depth = depth_image
        # 전체 pred depth로 GGCNN 추론
        self._run_ggcnn(depth_image, self.color_image)

    def run_crop(self):
        """crop 영역의 pred depth로 GGCNN을 재추론하여 결과를 표시한다."""
        if self.full_pred_depth is None:
            return
        r0, c0 = self.crop_tl
        r1, c1 = self.crop_br
        cropped_depth = self.full_pred_depth[r0:r1, c0:c1]
        cropped_color = self.color_image[r0:r1, c0:c1]
        self._run_ggcnn(cropped_depth, cropped_color)

    def _run_ggcnn(self, depth_image, color_image):
        """depth_image를 정규화하여 GGCNN을 실행하고 depth/overlay 패널을 갱신한다."""
        max_v = float(depth_image.max()) if depth_image.size else 0.0
        if max_v > 0:
            normalized_depth_image = (depth_image / max_v).astype(np.float32)
        else:
            normalized_depth_image = depth_image.astype(np.float32)
        q_out, depth_out, ang_out, width_out = get_ggcnn_results(normalized_depth_image)
        self.predicted_depth = depth_out
        self.q_out = q_out
        self.ang_out = ang_out
        self.width_out = width_out
        self.click_rc = None
        self.peak_rc = None

        depth_vis = cv2.applyColorMap(depth_out, cv2.COLORMAP_JET)

        color = color_image
        if color.shape[:2] != q_out.shape[:2]:
            color = cv2.resize(color, (q_out.shape[1], q_out.shape[0]))
        q_color = cv2.applyColorMap(q_out, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(color, 0.5, q_color, 0.5, 0)

        self._show(self.depth_label, depth_vis)
        self._show(self.overlay_label, overlay)

    def _show(self, label, img_bgr):
        """BGR 이미지를 Tk PhotoImage로 변환해 라벨에 표시하고 GC 방지를 위해 참조를 보관한다."""
        tkimg, scale = bgr_to_tk(img_bgr)
        self._imgs[label] = tkimg
        if label is self.input_label:
            self._input_scale = scale
        label.config(image=tkimg, width=tkimg.width(), height=tkimg.height())


def main():
    """Tk 루트를 생성하고 GUI를 띄워 메인 루프를 시작한다."""
    root = tk.Tk()
    gui = GgcnnGui(root)
    if os.path.isdir(DEFAULT_INPUT_DIR):
        gui.load_dir(DEFAULT_INPUT_DIR)
    root.mainloop()


if __name__ == "__main__":
    main()
