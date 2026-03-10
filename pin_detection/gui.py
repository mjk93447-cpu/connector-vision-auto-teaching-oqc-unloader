"""
Pin Detection GUI — Train/Inference, local file upload, YOLO26 training.
All UI in English for global use.
"""
import os
import platform
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .inference import run_inference, split_upper_lower, compute_spacing_mm
from .excel_io import load_excel_format, write_result_excel
from .dataset import extract_cell_id


def _select_dir(parent, title: str) -> str:
    path = filedialog.askdirectory(parent=parent, title=title)
    return path or ""


def _select_file(parent, title: str, types=None) -> str:
    types = types or [("All", "*.*"), ("Images", "*.jpg *.jpeg *.png *.bmp")]
    path = filedialog.askopenfilename(parent=parent, title=title, filetypes=types)
    return path or ""


def _get_cpu_info() -> str:
    """Return CPU model/cores for display."""
    try:
        n = os.cpu_count() or 4
        proc = platform.processor() or "Unknown"
        return f"{proc} ({n} cores)"
    except Exception:
        return f"{os.cpu_count() or 4} cores"


def _estimate_training_time(n_images: int, imgsz: int, epochs: int, workers: int) -> float:
    """
    Conservative estimate in seconds for CPU training.
    Base: ~6 sec per image per epoch at 640px (CPU, no GPU).
    Scale by (imgsz/640)^2. Workers help data loading (~1.2x with 4 workers).
    imgsz from ROI only; no cap.
    """
    if n_images <= 0 or imgsz <= 0:
        return 0.0
    base_per_epoch = n_images * 6.0 * (imgsz / 640) ** 2
    worker_factor = 1.0 + 0.05 * min(workers, 4)
    return base_per_epoch * epochs / worker_factor


def _format_duration(sec: float) -> str:
    if sec < 60:
        return f"~{int(sec)} sec"
    if sec < 3600:
        return f"~{int(sec / 60)} min"
    return f"~{sec / 3600:.1f} hr"


_HELP_CONTENT = """
================================================================================
  CONNECTOR PIN DETECTION — USER GUIDE
================================================================================

OVERVIEW
--------
This tool detects connector pins in top-view images using YOLO26. It has two modes:
1) Training: Learn from masked/unmasked image pairs (one-time setup).
2) Inference: Run detection on new images.

================================================================================
  TRAIN MODE
================================================================================

STEP 1: Prepare your data
  • Unmasked images: Original connector top-view photos (10+ recommended).
  • Masked images: Same photos with pin regions marked in green dots.
  • Pairing: by filename stem OR by cell ID (A2HDxxxx in filename, e.g. 20250101_120000_A2HD001.jpg).

STEP 2: Select folders
  • Unmasked images folder: Browse to the folder containing original images.
  • Masked images folder: Browse to the folder containing masked images.

STEP 3: Output folder
  • Where the trained model will be saved (default: pin_models).

STEP 4: Training parameters
  • Epochs: Number of training passes (100–200 typical). More = better but slower.
  • Image size: ROI box dimensions only. No manual input; crop region = analysis size.
  • Workers: Data loading threads. Set to match your CPU cores for faster training.
  • Suggested: After selecting folders and Edit ROI, click "Apply suggested" to auto-set epochs, val_split.

STEP 5: ROI Editor (optional, for large images)
  • Click "Edit ROI" to see unmasked (YOLO input) and masked (ground truth) side by side.
  • Drag on the left (unmasked) to set ROI. Prev/Next or Left/Right keys to navigate.
  • Save ROI map to roi_map.json. Training uses it when present in the output folder.

STEP 6: Start training
  • Click "Start training". The graph shows Loss, Precision, and Recall during training.
  • When done, the model path appears in the Inference tab.

================================================================================
  INFERENCE MODE
================================================================================

STEP 1: Select input image
  • Browse to a connector top-view image (no mask needed).

STEP 2: Select model
  • Browse to the trained .pt file (e.g. pin_models/pin_run/weights/best.pt).

STEP 3: Excel format ref (optional)
  • Select multi-row Excel (row 1=headers, row 2+=one per cell). If image has A2HDxxxx in filename,
    the result is written to the matching row. Otherwise a new result.xlsx is created.

STEP 4: Run inference
  • Click "Run inference". Outputs:
    - Masked image: <filename>_masked.png (green dots on detected pins)
    - Excel: updated at cell row (if A2HD format) or result.xlsx

================================================================================
  OUTPUT INTERPRETATION
================================================================================

  • OK: 20 upper pins + 20 lower pins detected.
  • NG: Different count (missing or extra).

  • Pin spacing: Approximate left-right spacing in mm (based on 0.5mm pin width).

================================================================================
  TIPS
================================================================================

  • First run: Use 10 image pairs for training. Ensure green dots cover the full pin area.
  • Slow training? Increase Workers; reduce epochs. (imgsz from ROI, batch auto-scales.)
  • OOM? Batch auto-scales by imgsz; Edit ROI to smaller region if needed.
  • Poor detection? Try more epochs (150–200); ROI region should cover all pins.
"""


class PinDetectionGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Connector Pin Detection — YOLO26")
        self.root.geometry("720x620")
        self.root.minsize(500, 500)

        self.unmasked_dir = tk.StringVar()
        self.masked_dir = tk.StringVar()
        self.model_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="pin_models")
        self.epochs_var = tk.IntVar(value=100)
        self._imgsz_from_roi = 0  # Read-only, derived from ROI (0 = not set)
        self.workers_var = tk.IntVar(value=min(os.cpu_count() or 4, 4))  # cap 4 for Windows stability
        self._train_stop = threading.Event()

        self._build_ui()

    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Train tab
        train_f = ttk.Frame(nb, padding=8)
        nb.add(train_f, text="Train")

        row = 0
        ttk.Label(train_f, text="Original connector photos. Pair with masked images by filename.", font=("Segoe UI", 9), foreground="gray").grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
        row += 1
        ttk.Label(train_f, text="Unmasked images folder:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(train_f, textvariable=self.unmasked_dir, width=45).grid(row=row, column=1, padx=4, pady=2)
        ttk.Button(train_f, text="Browse", command=lambda: self._on_unmasked_browse()).grid(row=row, column=2, pady=2)
        row += 1

        ttk.Label(train_f, text="Masked images folder:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(train_f, textvariable=self.masked_dir, width=45).grid(row=row, column=1, padx=4, pady=2)
        ttk.Button(train_f, text="Browse", command=lambda: self._on_masked_browse()).grid(row=row, column=2, pady=2)
        row += 1

        ttk.Label(train_f, text="Output folder:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(train_f, textvariable=self.output_dir, width=45).grid(row=row, column=1, padx=4, pady=2)
        ttk.Button(train_f, text="Browse", command=lambda: self.output_dir.set(_select_dir(self.root, "Output folder") or self.output_dir.get())).grid(row=row, column=2, pady=2)
        row += 1

        ttk.Label(train_f, text="Epochs:").grid(row=row, column=0, sticky=tk.W, pady=2)
        sb_epochs = ttk.Spinbox(train_f, from_=10, to=500, textvariable=self.epochs_var, width=8, command=self._update_eta)
        sb_epochs.grid(row=row, column=1, sticky=tk.W, padx=4, pady=2)
        row += 1

        # ROI box size = analysis size; read-only, no manual input
        ttk.Label(train_f, text="Image size:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.imgsz_label = ttk.Label(train_f, text="—", foreground="gray")
        self.imgsz_label.grid(row=row, column=1, sticky=tk.W, padx=4, pady=2)
        ttk.Label(train_f, text="(from ROI, auto)").grid(row=row, column=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Label(train_f, text="Workers:").grid(row=row, column=0, sticky=tk.W, pady=2)
        sb_workers = ttk.Spinbox(train_f, from_=0, to=min(16, os.cpu_count() or 8), textvariable=self.workers_var, width=8, command=self._update_eta)
        sb_workers.grid(row=row, column=1, sticky=tk.W, padx=4, pady=2)
        ttk.Label(train_f, text="(data loading threads)").grid(row=row, column=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Label(train_f, text="Val split:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.val_split_var = tk.DoubleVar(value=0.2)
        sb_val = ttk.Spinbox(train_f, from_=0.01, to=0.5, increment=0.05, textvariable=self.val_split_var, width=8)
        sb_val.grid(row=row, column=1, sticky=tk.W, padx=4, pady=2)
        ttk.Label(train_f, text="(0.1–0.3 typical)").grid(row=row, column=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Label(train_f, text="Mosaic aug:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.mosaic_var = tk.DoubleVar(value=0.0)
        sb_mosaic = ttk.Spinbox(train_f, from_=0.0, to=1.0, increment=0.1, textvariable=self.mosaic_var, width=8)
        sb_mosaic.grid(row=row, column=1, sticky=tk.W, padx=4, pady=2)
        ttk.Label(train_f, text="(0=fast, 0.5=quality)").grid(row=row, column=2, sticky=tk.W, pady=2)
        row += 1

        # Hardware & ETA
        ttk.Label(train_f, text="CPU:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.cpu_label = ttk.Label(train_f, text=_get_cpu_info())
        self.cpu_label.grid(row=row, column=1, columnspan=2, sticky=tk.W, padx=4, pady=2)
        row += 1

        ttk.Label(train_f, text="Dataset:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.dataset_label = ttk.Label(train_f, text="—")
        self.dataset_label.grid(row=row, column=1, columnspan=2, sticky=tk.W, padx=4, pady=2)
        row += 1

        ttk.Label(train_f, text="Est. time:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.eta_label = ttk.Label(train_f, text="—")
        self.eta_label.grid(row=row, column=1, columnspan=2, sticky=tk.W, padx=4, pady=2)
        row += 1

        ttk.Label(train_f, text="Suggested:").grid(row=row, column=0, sticky=tk.W, pady=2)
        sug_f = ttk.Frame(train_f)
        sug_f.grid(row=row, column=1, columnspan=2, sticky=tk.W, padx=4, pady=2)
        self.suggested_label = ttk.Label(sug_f, text="—", foreground="gray")
        self.suggested_label.pack(side=tk.LEFT)
        self.apply_suggested_btn = ttk.Button(sug_f, text="Apply suggested", command=self._on_apply_suggested, state=tk.DISABLED)
        self.apply_suggested_btn.pack(side=tk.LEFT, padx=8)
        row += 1

        self.train_progress = ttk.Progressbar(train_f, mode="indeterminate")
        self.train_progress.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=8)
        row += 1

        self.train_status = ttk.Label(train_f, text="")
        self.train_status.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=2)
        row += 1

        # Graph frame + Training log
        graph_f = ttk.LabelFrame(train_f, text="Training metrics & log")
        graph_f.grid(row=row, column=0, columnspan=3, sticky=tk.NSEW, pady=8)
        graph_f.columnconfigure(0, weight=1)
        graph_f.rowconfigure(0, weight=1)
        graph_f.rowconfigure(1, weight=0)
        self.graph_frame = ttk.Frame(graph_f)  # Canvas goes here (children cleared on poll)
        self.graph_frame.grid(row=0, column=0, sticky=tk.NSEW)
        # Log text (epoch progress, ETA)
        log_f = ttk.Frame(graph_f)
        log_f.grid(row=1, column=0, sticky=tk.EW, padx=4, pady=2)
        log_f.columnconfigure(0, weight=1)
        self.train_log_text = tk.Text(log_f, height=5, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED)
        log_scroll = ttk.Scrollbar(log_f, command=self.train_log_text.yview)
        self.train_log_text.config(yscrollcommand=log_scroll.set)
        self.train_log_text.grid(row=0, column=0, sticky=tk.NSEW)
        log_scroll.grid(row=0, column=1, sticky=tk.NS)
        row += 1

        btn_f = ttk.Frame(train_f)
        btn_f.grid(row=row, column=1, pady=12)
        self.train_btn = ttk.Button(btn_f, text="Start training", command=self._on_train)
        self.train_btn.pack(side=tk.LEFT, padx=4)
        self.stop_btn = ttk.Button(btn_f, text="Stop training", command=self._on_stop_train, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_f, text="Edit ROI", command=self._on_edit_roi).pack(side=tk.LEFT, padx=4)

        train_f.columnconfigure(1, weight=1)

        # Bind spinbox for ETA update (lightweight — no scan restart)
        for c in [sb_epochs, sb_workers]:
            c.bind("<KeyRelease>", lambda e: self._update_eta_label())

        # Inference tab
        inf_f = ttk.Frame(nb, padding=8)
        nb.add(inf_f, text="Inference")

        ttk.Label(inf_f, text="Run detection on new images. Requires a trained model from the Train tab.", font=("Segoe UI", 9), foreground="gray").grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
        ttk.Label(inf_f, text="Input image:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.inference_image = tk.StringVar()
        ttk.Entry(inf_f, textvariable=self.inference_image, width=50).grid(row=1, column=1, padx=4, pady=2)
        ttk.Button(inf_f, text="Browse", command=lambda: self.inference_image.set(_select_file(self.root, "Select image"))).grid(row=1, column=2, pady=2)

        ttk.Label(inf_f, text="Model (.pt):").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(inf_f, textvariable=self.model_path, width=50).grid(row=2, column=1, padx=4, pady=2)
        ttk.Button(inf_f, text="Browse", command=lambda: self.model_path.set(_select_file(self.root, "Select model", [("PyTorch", "*.pt")]))).grid(row=2, column=2, pady=2)

        ttk.Label(inf_f, text="Excel format ref (optional):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.excel_format = tk.StringVar()
        ttk.Entry(inf_f, textvariable=self.excel_format, width=50).grid(row=3, column=1, padx=4, pady=2)
        ttk.Button(inf_f, text="Browse", command=lambda: self.excel_format.set(_select_file(self.root, "Excel", [("Excel", "*.xlsx *.xls")]))).grid(row=3, column=2, pady=2)

        ttk.Label(inf_f, text="Output folder (optional):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.inference_output_dir = tk.StringVar()
        ttk.Entry(inf_f, textvariable=self.inference_output_dir, width=50).grid(row=4, column=1, padx=4, pady=2)
        ttk.Button(inf_f, text="Browse", command=lambda: self.inference_output_dir.set(_select_dir(self.root, "Output folder") or self.inference_output_dir.get())).grid(row=4, column=2, pady=2)
        ttk.Label(inf_f, text="(empty = same as input image)", font=("Segoe UI", 8), foreground="gray").grid(row=5, column=1, sticky=tk.W, padx=4)

        ttk.Label(inf_f, text="Confidence threshold:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.conf_var = tk.DoubleVar(value=0.25)
        sb_conf = ttk.Spinbox(inf_f, from_=0.01, to=0.5, increment=0.05, textvariable=self.conf_var, width=8)
        sb_conf.grid(row=6, column=1, sticky=tk.W, padx=4, pady=2)
        ttk.Label(inf_f, text="(0.01–0.5, lower=more recall)").grid(row=6, column=2, sticky=tk.W, pady=2)

        self.inference_status = ttk.Label(inf_f, text="")
        self.inference_status.grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=8)

        ttk.Button(inf_f, text="Run inference", command=self._on_inference).grid(row=8, column=1, pady=12)

        # Help tab
        help_f = ttk.Frame(nb, padding=8)
        nb.add(help_f, text="Help")
        help_text = tk.Text(help_f, wrap=tk.WORD, width=70, height=24, font=("Segoe UI", 10), relief=tk.FLAT, padx=8, pady=8)
        scr = ttk.Scrollbar(help_f, command=help_text.yview)
        help_text.config(yscrollcommand=scr.set)
        help_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scr.pack(side=tk.RIGHT, fill=tk.Y)
        help_text.insert(tk.END, _HELP_CONTENT)
        help_text.config(state=tk.DISABLED)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var).pack(side=tk.BOTTOM, pady=4)

    def _on_unmasked_browse(self):
        p = _select_dir(self.root, "Unmasked folder")
        if p:
            self.unmasked_dir.set(p)
            self._update_eta()

    def _on_masked_browse(self):
        p = _select_dir(self.root, "Masked folder")
        if p:
            self.masked_dir.set(p)
            self._update_eta()

    def _update_eta_label(self):
        """Update ETA label only (no scan). Called on epochs/workers spinbox change."""
        u = self.unmasked_dir.get().strip()
        m = self.masked_dir.get().strip()
        if not u or not m:
            return
        try:
            from .dataset import IMG_EXTS
            from .train import _default_workers
            pu, pm = Path(u), Path(m)
            u_files = [f for f in pu.iterdir() if f.suffix.lower() in IMG_EXTS]
            n = len(u_files)
            try:
                workers = int(self.workers_var.get())
            except Exception:
                workers = _default_workers()
            imgsz = getattr(self, "_imgsz_from_roi", 0) or 640
            sec = _estimate_training_time(n, imgsz, self.epochs_var.get(), workers)
            self.eta_label.config(text=_format_duration(sec))
        except Exception:
            pass

    def _update_eta(self):
        """Quick update: n_images, w×h, ETA. No heavy analyze (runs in background)."""
        u = self.unmasked_dir.get().strip()
        m = self.masked_dir.get().strip()
        if not u or not m:
            self.dataset_label.config(text="—")
            self.eta_label.config(text="—")
            self.suggested_label.config(text="—")
            self.apply_suggested_btn.config(state=tk.DISABLED)
            self._suggested = None
            self._scan_cancel = True
            return
        try:
            from .dataset import IMG_EXTS
            from .train import _default_workers
            pu, pm = Path(u), Path(m)
            u_files = [f for f in pu.iterdir() if f.suffix.lower() in IMG_EXTS]
            n = len(u_files)
            w, h = 0, 0
            if u_files:
                try:
                    from PIL import Image
                    im = Image.open(u_files[0])
                    w, h = im.size
                    im.close()
                except Exception:
                    pass
            self.dataset_label.config(text=f"{n} images, {w}×{h} px")

            # ETA only — no analyze (avoids 2min freeze, EXE_ARTIFACT_ISSUES #2)
            try:
                workers = int(self.workers_var.get())
            except Exception:
                workers = _default_workers()
            # imgsz from ROI only (ROADMAP: ROI sole determinant)
            out = self.output_dir.get().strip()
            if out:
                try:
                    from .dataset import load_roi_map_and_imgsz
                    _, imgsz_val = load_roi_map_and_imgsz(Path(out), pu, pm)
                    self._imgsz_from_roi = imgsz_val
                    self.imgsz_label.config(text=str(imgsz_val) if imgsz_val > 0 else "—")
                except Exception:
                    pass
            imgsz = getattr(self, "_imgsz_from_roi", 0) or 640  # 640 only for ETA when unknown
            sec = _estimate_training_time(n, imgsz, self.epochs_var.get(), workers)
            self.eta_label.config(text=_format_duration(sec))

            # Start background scan for "Apply suggested" (non-blocking, EXE_ARTIFACT_ISSUES #2)
            self._scan_cancel = True  # cancel any in-flight scan before starting new one
            self._scan_paths = (u, m)
            self._scan_cancel = False
            self.suggested_label.config(text="Scanning...")
            self.apply_suggested_btn.config(state=tk.DISABLED)

            def _scan():
                try:
                    from .dataset import analyze_dataset_for_training
                    out = self.output_dir.get().strip()
                    out_p = Path(out) if out else None
                    s = analyze_dataset_for_training(pu, pm, max_samples=3, output_dir=out_p)
                    if getattr(self, "_scan_cancel", True):
                        return
                    self.root.after(0, lambda: self._on_scan_done(s, (u, m)))
                except Exception:
                    if not getattr(self, "_scan_cancel", True):
                        self.root.after(0, lambda: self._on_scan_done(None, (u, m)))

            t = threading.Thread(target=_scan, daemon=True)
            t.start()
        except Exception as e:
            self.dataset_label.config(text=str(e)[:40])
            self.eta_label.config(text="—")
            self.suggested_label.config(text="—")
            self.apply_suggested_btn.config(state=tk.DISABLED)
            self._suggested = None

    def _on_scan_done(self, suggested: dict | None, scan_paths: tuple):
        """Called from main thread after background scan."""
        if scan_paths != getattr(self, "_scan_paths", None):
            return  # folders changed, ignore stale result
        if suggested is None:
            self.suggested_label.config(text="—")
            self.apply_suggested_btn.config(state=tk.DISABLED)
            self._suggested = None
            return
        self._suggested = suggested
        epochs = suggested.get("epochs", 3)
        mosaic = suggested.get("mosaic", 0.0)
        note = suggested.get("note", "")
        imgsz_sug = suggested.get("imgsz", 0)
        self._imgsz_from_roi = imgsz_sug
        self.imgsz_label.config(text=str(imgsz_sug) if imgsz_sug > 0 else "—")
        imgsz_txt = str(imgsz_sug) if imgsz_sug > 0 else "—"
        self.suggested_label.config(text=f"imgsz:{imgsz_txt}, epochs:{epochs}, mosaic:{mosaic}" + (f" ({note})" if note else ""))
        self.apply_suggested_btn.config(state=tk.NORMAL)

    def _on_apply_suggested(self):
        if not getattr(self, "_suggested", None):
            return
        s = self._suggested
        self._imgsz_from_roi = s.get("imgsz", 0)
        self.imgsz_label.config(text=str(self._imgsz_from_roi) if self._imgsz_from_roi > 0 else "—")
        self.epochs_var.set(s.get("epochs", 3))
        self.val_split_var.set(s.get("val_split", 0.2))
        self.mosaic_var.set(s.get("mosaic", 0.0))
        self._update_eta()

    def _on_edit_roi(self):
        """Open ROI Editor to manually draw ROI per image (ROADMAP 10.20)."""
        u = self.unmasked_dir.get().strip()
        m = self.masked_dir.get().strip()
        out = self.output_dir.get().strip()
        if not u or not m:
            messagebox.showerror("Error", "Select unmasked and masked folders first.")
            return
        if not out:
            messagebox.showerror("Error", "Select output folder first.")
            return
        try:
            from .dataset import IMG_EXTS
            pu, pm = Path(u), Path(m)
            if not pu.exists() or not pu.is_dir():
                messagebox.showerror("Error", f"Unmasked folder does not exist: {u}")
                return
            if not pm.exists() or not pm.is_dir():
                messagebox.showerror("Error", f"Masked folder does not exist: {m}")
                return
            u_files = [f for f in pu.iterdir() if f.suffix.lower() in IMG_EXTS]
            if not u_files:
                messagebox.showerror("Error", f"No images in {u}. Use .jpg, .png, .bmp.")
                return
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        from .roi_editor import run_roi_editor
        run_roi_editor(u, m, Path(out), parent=self.root)

    def _on_train(self):
        u = self.unmasked_dir.get().strip()
        m = self.masked_dir.get().strip()
        out = self.output_dir.get().strip()
        if not u or not m:
            messagebox.showerror("Error", "Select unmasked and masked folders.")
            return
        if not out:
            messagebox.showerror("Error", "Select output folder first.")
            return
        try:
            pu, pm, po = Path(u), Path(m), Path(out)
            if not pu.exists() or not pu.is_dir():
                messagebox.showerror("Error", f"Unmasked folder does not exist: {u}")
                return
            if not pm.exists() or not pm.is_dir():
                messagebox.showerror("Error", f"Masked folder does not exist: {m}")
                return
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self._train_stop.clear()
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        def run():
            try:
                from .debug_log import clear_log, log_step
                clear_log()
                log_step("0. Train thread started")
                self.root.after(0, lambda: self.train_progress.start(10))
                self.root.after(0, lambda: self.train_status.config(text="Preparing dataset..."))
                self.root.after(0, lambda: self.status_var.set("Training..."))

                from .train import train_pin_model

                out_dir = self.output_dir.get()
                try:
                    val_split = float(self.val_split_var.get())
                    val_split = max(0.01, min(0.5, val_split))
                except Exception:
                    val_split = 0.2

                def _on_dataset_progress(current: int, total: int, path):
                    c, t, p = current, total, path
                    self.root.after(0, lambda c=c, t=t, p=p: self._update_train_log(f"Building dataset ({c}/{t}) — {p.name}", clear=False))

                # Start graph poll: YOLO saves to output_dir/pin_run (abs path) or runs/detect/<name>/pin_run (rel)
                candidates = [
                    Path(out_dir) / "pin_run",
                    Path("runs") / "detect" / Path(out_dir).name / "pin_run",
                ]
                log_step("0a. Starting graph poll (matplotlib)")
                self._start_graph_poll(candidates)
                log_step("0b. Graph poll started")

                try:
                    workers = int(self.workers_var.get())
                except Exception:
                    workers = None
                try:
                    mosaic_val = float(self.mosaic_var.get())
                    mosaic_val = max(0.0, min(1.0, mosaic_val))
                except Exception:
                    mosaic_val = 0.0
                log_step("0c. Calling train_pin_model (imgsz from ROI)")
                model_path = train_pin_model(
                    unmasked_dir=u,
                    masked_dir=m,
                    output_dir=out_dir,
                    epochs=self.epochs_var.get(),
                    workers=workers,
                    val_split=val_split,
                    stop_event=self._train_stop,
                    mosaic=mosaic_val,
                    use_roi=True,
                    on_progress=_on_dataset_progress,
                )

                self.root.after(0, lambda: self._stop_graph_poll())
                self.root.after(0, lambda: self.model_path.set(str(model_path)))
                self.root.after(0, lambda: self.train_status.config(text=f"Done: {model_path}"))
                self.root.after(0, lambda: messagebox.showinfo("Done", f"Model saved:\n{model_path}"))
            except Exception as e:
                self.root.after(0, lambda: self._stop_graph_poll())
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.root.after(0, lambda: self.train_status.config(text=f"Error: {e}"))
            finally:
                self.root.after(0, lambda: self.train_progress.stop())
                self.root.after(0, lambda: self.status_var.set("Ready"))
                self.root.after(0, lambda: self.train_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))

        threading.Thread(target=run, daemon=True).start()

    def _on_stop_train(self):
        self._train_stop.set()
        self.train_status.config(text="Stopping at end of epoch...")

    def _start_graph_poll(self, save_dir: Path | list[Path]):
        """Poll results.csv and update graph + log. save_dir can be Path or list of candidate Paths."""
        self._graph_data = []
        self._graph_start_time = time.time()
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            fig = Figure(figsize=(5, 2.5), dpi=80)
            self._graph_ax = fig.add_subplot(111)
            self._graph_fig = fig
            for w in self.graph_frame.winfo_children():
                w.destroy()
            self._graph_canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            self._graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception:
            self._graph_canvas = None
            self._graph_ax = None
        self._graph_save_dir = save_dir if isinstance(save_dir, list) else [save_dir]
        self._graph_poll_id = None
        self._last_logged_epoch = -1
        self._update_train_log("Preparing dataset...", clear=True)
        self._poll_graph()

    def _update_train_log(self, msg: str, clear: bool = False):
        """Append to training log text (thread-safe via root.after)."""
        def _do():
            t = getattr(self, "train_log_text", None)
            if t:
                t.config(state=tk.NORMAL)
                if clear:
                    t.delete("1.0", tk.END)
                t.insert(tk.END, msg + "\n")
                t.see(tk.END)
                t.config(state=tk.DISABLED)
        self.root.after(0, _do)

    def _poll_graph(self):
        candidates = getattr(self, "_graph_save_dir", None) or []
        csv_path = None
        for d in candidates:
            p = Path(d) / "results.csv"
            if p.exists():
                csv_path = p
                break
        if csv_path:
            try:
                import csv
                with open(csv_path) as f:
                    r = csv.DictReader(f)
                    rows = list(r)
                if rows:
                    self._graph_data = rows
                    if getattr(self, "_graph_canvas", None):
                        self._draw_graph()
                    last = rows[-1]
                    ep = int(last.get("epoch", len(rows)))
                    if ep != getattr(self, "_last_logged_epoch", -1):
                        self._last_logged_epoch = ep
                        box = last.get("train/box_loss") or last.get("train_box_loss") or ""
                        prec = last.get("metrics/precision(B)") or last.get("metrics_precision(B)") or ""
                        rec = last.get("metrics/recall(B)") or last.get("metrics_recall(B)") or ""
                        epochs_total = 3
                        try:
                            epochs_total = int(self.epochs_var.get())
                        except Exception:
                            pass
                        eta = ""
                        try:
                            elapsed = time.time() - getattr(self, "_graph_start_time", time.time())
                            if ep > 0 and elapsed > 0:
                                sec_per_ep = elapsed / ep
                                remaining = (epochs_total - ep) * sec_per_ep
                                eta = f"ETA ~{int(remaining)}s"
                        except Exception:
                            pass
                        log_line = f"Epoch {ep}/{epochs_total}: loss={box[:6] if box else '-'} P={prec[:5] if prec else '-'} R={rec[:5] if rec else '-'} {eta}"
                        self._update_train_log(log_line)
            except Exception:
                pass
        self._graph_poll_id = self.root.after(1000, self._poll_graph)

    def _draw_graph(self):
        if not self._graph_data or not getattr(self, "_graph_ax", None):
            return
        try:
            self._graph_ax.clear()
            epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(self._graph_data)]
            # Support various Ultralytics column names
            col_map = [
                (["train/box_loss", "train_box_loss"], "Loss"),
                (["metrics/precision(B)", "metrics_precision(B)"], "Precision"),
                (["metrics/recall(B)", "metrics_recall(B)"], "Recall"),
            ]
            for cols, label in col_map:
                vals = []
                for r in self._graph_data:
                    v = ""
                    for c in cols:
                        if c in r and r[c] is not None:
                            v = str(r[c]).strip()
                            break
                    vals.append(float(v) if v else 0)
                if any(vals):
                    self._graph_ax.plot(epochs, vals, label=label, alpha=0.8)
            self._graph_ax.legend(loc="upper right", fontsize=8)
            self._graph_ax.set_xlabel("Epoch")
            self._graph_ax.grid(True, alpha=0.3)
            self._graph_canvas.draw()
        except Exception:
            pass

    def _stop_graph_poll(self):
        if getattr(self, "_graph_poll_id", None):
            self.root.after_cancel(self._graph_poll_id)
            self._graph_poll_id = None

    def _on_inference(self):
        img_path = self.inference_image.get().strip()
        model_path = self.model_path.get().strip()
        if not img_path or not model_path:
            messagebox.showerror("Error", "Select image and model path.")
            return

        out_dir = self.inference_output_dir.get().strip()
        from .results_path import get_results_root, get_timestamped_dir
        if out_dir:
            base_dir = get_timestamped_dir(Path(out_dir))
        else:
            base_dir = get_timestamped_dir(get_results_root())
        out_img_path = base_dir / f"{Path(img_path).stem}_masked.png"
        excel_out = base_dir / "result.xlsx"
        try:
            conf = float(self.conf_var.get())
            conf = max(0.01, min(0.5, conf))
        except Exception:
            conf = 0.25

        def run():
            try:
                self.root.after(0, lambda: self.status_var.set("Inference..."))
                self.root.after(0, lambda: self.inference_status.config(text="Processing..."))

                base_dir.mkdir(parents=True, exist_ok=True)
                img, detections, masked = run_inference(
                    model_path=model_path,
                    image_path=img_path,
                    output_image_path=out_img_path,
                    conf_threshold=conf,
                )
                h, w = img.shape[:2]
                upper, lower = split_upper_lower(detections)
                upper_spacings = compute_spacing_mm(upper, w)
                lower_spacings = compute_spacing_mm(lower, w)

                format_ref = None
                ef = self.excel_format.get().strip()
                if ef:
                    format_ref = load_excel_format(ef)

                cell_id = extract_cell_id(Path(img_path))
                update_existing = bool(ef and cell_id and Path(ef).exists())
                excel_path_to_use = Path(ef) if update_existing else excel_out

                write_result_excel(
                    excel_path_to_use,
                    upper_count=len(upper),
                    lower_count=len(lower),
                    upper_spacings=upper_spacings,
                    lower_spacings=lower_spacings,
                    format_ref=format_ref,
                    cell_id=cell_id,
                    update_existing=update_existing,
                )

                ok = len(upper) == 20 and len(lower) == 20
                excel_msg = excel_path_to_use
                msg = f"Upper: {len(upper)}, Lower: {len(lower)} → {'OK' if ok else 'NG'}\nImage: {out_img_path}\nExcel: {excel_msg}"
                self.root.after(0, lambda: self.inference_status.config(text=msg))
                self.root.after(0, lambda: messagebox.showinfo("Done", msg))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.root.after(0, lambda: self.inference_status.config(text=f"Error: {e}"))
            finally:
                self.root.after(0, lambda: self.status_var.set("Ready"))

        threading.Thread(target=run, daemon=True).start()


def main():
    root = tk.Tk()
    root.tk.call("tk", "scaling", 1.2)
    app = PinDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
