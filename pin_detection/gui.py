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
    Rough estimate in seconds. Base: ~2 sec per image per epoch at 640px.
    Scale by (imgsz/640)^2. Workers help data loading (~1.3x with 4 workers).
    """
    if n_images <= 0:
        return 0.0
    base_per_epoch = n_images * 2.0 * (imgsz / 640) ** 2
    worker_factor = 1.0 + 0.1 * min(workers, 8)  # up to ~1.8x speedup
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
  • Unmasked images: Original connector top-view photos (10 images recommended).
  • Masked images: Same photos with pin regions marked in green dots.
  • Each pair must have the same filename (e.g. 01.jpg, 01_masked.jpg).

STEP 2: Select folders
  • Unmasked images folder: Browse to the folder containing original images.
  • Masked images folder: Browse to the folder containing masked images.
  • Excel folder (optional): Reference Excel files for output format.

STEP 3: Output folder
  • Where the trained model will be saved (default: pin_models).

STEP 4: Training parameters
  • Epochs: Number of training passes (100–200 typical). More = better but slower.
  • Image size (imgsz): Input resolution. 640–1280 for small pins. Larger = more accurate but slower.
  • Workers: Data loading threads. Set to match your CPU cores for faster training.

STEP 5: Start training
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
  • If you used a reference Excel during training, select it to match output format.

STEP 4: Run inference
  • Click "Run inference". Outputs:
    - Masked image: <filename>_masked.png (green dots on detected pins)
    - Excel: result.xlsx (pin count, OK/NG, spacing values)

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
  • Slow training? Increase Workers; reduce imgsz or epochs.
  • Poor detection? Try more epochs (150–200) or larger imgsz (1280).
"""


class PinDetectionGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Connector Pin Detection — YOLO26")
        self.root.geometry("720x620")
        self.root.minsize(500, 500)

        self.unmasked_dir = tk.StringVar()
        self.masked_dir = tk.StringVar()
        self.excel_dir = tk.StringVar()
        self.model_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="pin_models")
        self.epochs_var = tk.IntVar(value=100)
        self.imgsz_var = tk.IntVar(value=640)
        self.workers_var = tk.IntVar(value=min(os.cpu_count() or 4, 8))
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

        ttk.Label(train_f, text="Excel folder (optional):").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(train_f, textvariable=self.excel_dir, width=45).grid(row=row, column=1, padx=4, pady=2)
        ttk.Button(train_f, text="Browse", command=lambda: self.excel_dir.set(_select_dir(self.root, "Excel folder"))).grid(row=row, column=2, pady=2)
        row += 1

        ttk.Label(train_f, text="Output folder:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(train_f, textvariable=self.output_dir, width=45).grid(row=row, column=1, padx=4, pady=2)
        ttk.Button(train_f, text="Browse", command=lambda: self.output_dir.set(_select_dir(self.root, "Output folder") or self.output_dir.get())).grid(row=row, column=2, pady=2)
        row += 1

        ttk.Label(train_f, text="Epochs:").grid(row=row, column=0, sticky=tk.W, pady=2)
        sb_epochs = ttk.Spinbox(train_f, from_=10, to=500, textvariable=self.epochs_var, width=8, command=self._update_eta)
        sb_epochs.grid(row=row, column=1, sticky=tk.W, padx=4, pady=2)
        row += 1

        ttk.Label(train_f, text="Image size (imgsz):").grid(row=row, column=0, sticky=tk.W, pady=2)
        sb_imgsz = ttk.Spinbox(train_f, from_=320, to=1280, increment=64, textvariable=self.imgsz_var, width=8, command=self._update_eta)
        sb_imgsz.grid(row=row, column=1, sticky=tk.W, padx=4, pady=2)
        ttk.Label(train_f, text="(640–1280 for small pins)").grid(row=row, column=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Label(train_f, text="Workers:").grid(row=row, column=0, sticky=tk.W, pady=2)
        sb_workers = ttk.Spinbox(train_f, from_=0, to=min(16, os.cpu_count() or 8), textvariable=self.workers_var, width=8, command=self._update_eta)
        sb_workers.grid(row=row, column=1, sticky=tk.W, padx=4, pady=2)
        ttk.Label(train_f, text="(data loading threads)").grid(row=row, column=2, sticky=tk.W, pady=2)
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

        self.train_progress = ttk.Progressbar(train_f, mode="indeterminate")
        self.train_progress.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=8)
        row += 1

        self.train_status = ttk.Label(train_f, text="")
        self.train_status.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=2)
        row += 1

        # Graph frame
        graph_f = ttk.LabelFrame(train_f, text="Training metrics")
        graph_f.grid(row=row, column=0, columnspan=3, sticky=tk.NSEW, pady=8)
        self.graph_frame = graph_f
        row += 1

        self.train_btn = ttk.Button(train_f, text="Start training", command=self._on_train)
        self.train_btn.grid(row=row, column=1, pady=12)

        train_f.columnconfigure(1, weight=1)

        # Bind spinbox for ETA update
        for c in [sb_epochs, sb_imgsz, sb_workers]:
            c.bind("<KeyRelease>", lambda e: self._update_eta())

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

        self.inference_status = ttk.Label(inf_f, text="")
        self.inference_status.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=8)

        ttk.Button(inf_f, text="Run inference", command=self._on_inference).grid(row=5, column=1, pady=12)

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

    def _update_eta(self):
        u = self.unmasked_dir.get().strip()
        m = self.masked_dir.get().strip()
        if not u or not m:
            self.dataset_label.config(text="—")
            self.eta_label.config(text="—")
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
            try:
                workers = int(self.workers_var.get())
            except Exception:
                workers = _default_workers()
            sec = _estimate_training_time(n, self.imgsz_var.get(), self.epochs_var.get(), workers)
            self.eta_label.config(text=_format_duration(sec))
        except Exception as e:
            self.dataset_label.config(text=str(e)[:40])
            self.eta_label.config(text="—")

    def _on_train(self):
        u = self.unmasked_dir.get().strip()
        m = self.masked_dir.get().strip()
        if not u or not m:
            messagebox.showerror("Error", "Select unmasked and masked folders.")
            return

        self._train_stop.clear()
        self.train_btn.config(state=tk.DISABLED)

        def run():
            try:
                self.root.after(0, lambda: self.train_progress.start(10))
                self.root.after(0, lambda: self.train_status.config(text="Preparing dataset..."))
                self.root.after(0, lambda: self.status_var.set("Training..."))

                from .train import train_pin_model
                from .dataset import get_dataset_info, prepare_yolo_dataset_from_dirs

                out_dir = self.output_dir.get()
                dataset_dir = Path(out_dir) / "dataset"
                dataset_dir.mkdir(parents=True, exist_ok=True)
                data_yaml = prepare_yolo_dataset_from_dirs(Path(u), Path(m), dataset_dir)

                # Start graph poll in main thread
                save_dir = Path(out_dir) / "pin_run"
                self._start_graph_poll(save_dir)

                try:
                    workers = int(self.workers_var.get())
                except Exception:
                    workers = None
                model_path = train_pin_model(
                    unmasked_dir=u,
                    masked_dir=m,
                    output_dir=out_dir,
                    epochs=self.epochs_var.get(),
                    imgsz=self.imgsz_var.get(),
                    workers=workers,
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

        threading.Thread(target=run, daemon=True).start()

    def _start_graph_poll(self, save_dir: Path):
        """Poll results.csv and update graph."""
        self._graph_data = []
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
        except ImportError:
            self._graph_canvas = None
            return
        self._graph_save_dir = save_dir
        self._graph_poll_id = None
        self._poll_graph()

    def _poll_graph(self):
        if not getattr(self, "_graph_canvas", None):
            return
        csv_path = self._graph_save_dir / "results.csv"
        if csv_path.exists():
            try:
                import csv
                with open(csv_path) as f:
                    r = csv.DictReader(f)
                    rows = list(r)
                if rows:
                    self._graph_data = rows
                    self._draw_graph()
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

        def run():
            try:
                self.root.after(0, lambda: self.status_var.set("Inference..."))
                self.root.after(0, lambda: self.inference_status.config(text="Processing..."))

                img, detections, masked = run_inference(
                    model_path=model_path,
                    image_path=img_path,
                    output_image_path=Path(img_path).parent / f"{Path(img_path).stem}_masked.png",
                    conf_threshold=0.25,
                )
                h, w = img.shape[:2]
                upper, lower = split_upper_lower(detections)
                upper_spacings = compute_spacing_mm(upper, w)
                lower_spacings = compute_spacing_mm(lower, w)

                format_ref = None
                ef = self.excel_format.get().strip()
                if ef:
                    format_ref = load_excel_format(ef)

                excel_out = Path(img_path).parent / "result.xlsx"
                write_result_excel(
                    excel_out,
                    upper_count=len(upper),
                    lower_count=len(lower),
                    upper_spacings=upper_spacings,
                    lower_spacings=lower_spacings,
                    format_ref=format_ref,
                )

                out_img = Path(img_path).parent / f"{Path(img_path).stem}_masked.png"
                ok = len(upper) == 20 and len(lower) == 20
                msg = f"Upper: {len(upper)}, Lower: {len(lower)} → {'OK' if ok else 'NG'}\nImage: {out_img}\nExcel: {excel_out}"
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
