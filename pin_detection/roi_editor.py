"""
ROI Editor GUI: manually draw ROI rectangles per image.
Shows unmasked (YOLO input) and masked (ground-truth annotation) side by side.
Saves to roi_map.json: { "stem": [x1, y1, x2, y2] } — stem = unmasked filename stem.
ROADMAP 10.20, EXE_TEST_FEEDBACK 10.20.2.
"""
import json
import tkinter as tk
from pathlib import Path
from typing import Callable

from PIL import Image, ImageTk

from .dataset import IMG_EXTS, _find_masked_pair

try:
    from tkinter import ttk
except ImportError:
    import tkinter.ttk as ttk


def load_roi_map(path: Path) -> dict:
    """Load roi_map.json. Returns {} if missing or invalid."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return {k: list(v) for k, v in data.items() if isinstance(v, (list, tuple)) and len(v) == 4}
    except Exception:
        return {}


def save_roi_map(path: Path, roi_map: dict) -> None:
    """Save roi_map to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(roi_map, f, indent=2)


def run_roi_editor(
    unmasked_dir: str | Path,
    masked_dir: str | Path,
    output_dir: str | Path,
    parent: tk.Tk | tk.Toplevel | None = None,
    on_save: Callable[[Path], None] | None = None,
) -> None:
    """
    Open ROI Editor window.
    - unmasked_dir, masked_dir: image folders (paired by filename)
    - output_dir: where to save roi_map.json (output_dir/roi_map.json)
    - on_save: callback(path) when user saves

    Shows unmasked (left) and masked (right) side by side.
    ROI is drawn on unmasked — YOLO input (where to find pins).
    Masked provides ground-truth annotations (how to mask).
    Same ROI coordinates apply to both (identical dimensions).
    """
    unmasked_dir = Path(unmasked_dir)
    masked_dir = Path(masked_dir)
    output_dir = Path(output_dir)
    roi_map_path = output_dir / "roi_map.json"

    u_files = sorted(
        [f for f in unmasked_dir.iterdir() if f.suffix.lower() in IMG_EXTS],
        key=lambda p: p.name,
    )
    if not u_files:
        return  # caller should show error

    roi_map = load_roi_map(roi_map_path)

    root = tk.Toplevel(parent) if parent else tk.Toplevel()
    root.title("ROI Editor — Unmasked (YOLO input) | Masked (ground truth) — drag on left to set ROI")
    root.geometry("1200x700")
    root.minsize(800, 500)

    # Split pane: left = unmasked, right = masked
    paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _make_canvas_frame(label: str) -> tuple[tk.Frame, tk.Canvas]:
        f = ttk.LabelFrame(paned, text=label)
        paned.add(f, weight=1)
        c = tk.Canvas(f, bg="#1a1a1a", highlightthickness=0)
        c.pack(fill=tk.BOTH, expand=True)
        return f, c

    left_f, canvas_u = _make_canvas_frame("Unmasked (YOLO input — drag here)")
    right_f, canvas_m = _make_canvas_frame("Masked (ground truth)")

    # State
    idx = [0]
    photo_u = [None]
    photo_m = [None]
    scale = [1.0]
    img_size = [0, 0]
    rect_u_id = [None]
    rect_m_id = [None]
    drag_start = [None]
    current_roi = [None]

    def _find_masked(u_path: Path) -> Path | None:
        try:
            return _find_masked_pair(u_path, masked_dir)
        except FileNotFoundError:
            return None

    def _display_to_image(x: float, y: float) -> tuple[int, int]:
        s = scale[0]
        return (int(x / s), int(y / s))

    def _draw_roi_on_canvas(canvas: tk.Canvas, rect_id_ref: list, roi: list[int] | None):
        if rect_id_ref[0]:
            canvas.delete(rect_id_ref[0])
            rect_id_ref[0] = None
        if roi and len(roi) == 4:
            x1, y1, x2, y2 = roi
            s = scale[0]
            r = (int(x1 * s), int(y1 * s), int(x2 * s), int(y2 * s))
            rect_id_ref[0] = canvas.create_rectangle(*r, outline="lime", width=2)

    def _load_image():
        u_path = u_files[idx[0]]
        m_path = _find_masked(u_path)
        try:
            img_u = Image.open(u_path).convert("RGB")
        except Exception:
            img_u = Image.new("RGB", (640, 480), (40, 40, 40))
        if m_path:
            try:
                img_m = Image.open(m_path).convert("RGB")
            except Exception:
                img_m = img_u.copy()
        else:
            img_m = img_u.copy()

        img_size[0], img_size[1] = img_u.size
        cw = 500  # half width per pane
        ch = canvas_u.winfo_height() or 400
        if ch < 10:
            ch = 400
        rw = cw / img_u.size[0]
        rh = ch / img_u.size[1]
        scale[0] = min(rw, rh, 1.0)
        new_w = int(img_u.size[0] * scale[0])
        new_h = int(img_u.size[1] * scale[0])

        img_u_resized = img_u.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_m_resized = img_m.resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo_u[0] = ImageTk.PhotoImage(img_u_resized)
        photo_m[0] = ImageTk.PhotoImage(img_m_resized)

        canvas_u.delete("all")
        canvas_u.create_image(0, 0, anchor=tk.NW, image=photo_u[0])
        canvas_u.config(scrollregion=(0, 0, new_w, new_h))
        canvas_m.delete("all")
        canvas_m.create_image(0, 0, anchor=tk.NW, image=photo_m[0])
        canvas_m.config(scrollregion=(0, 0, new_w, new_h))

        stem = u_path.stem
        roi = roi_map.get(stem)
        if roi:
            current_roi[0] = roi
            _draw_roi_on_canvas(canvas_u, rect_u_id, roi)
            _draw_roi_on_canvas(canvas_m, rect_m_id, roi)
        else:
            rect_u_id[0] = None
            rect_m_id[0] = None
            current_roi[0] = None

        pair_info = f" + {m_path.name}" if m_path else " (no masked pair)"
        nav_label.config(
            text=f"Image {idx[0] + 1}/{len(u_files)} — {u_path.name}{pair_info}"
            + (f"  ROI: {roi}" if roi else "  (drag on left to set ROI)")
        )

    def _on_mouse_down(event, canvas: tk.Canvas):
        x, y = canvas.canvasx(event.x), canvas.canvasy(event.y)
        if 0 <= x < img_size[0] * scale[0] and 0 <= y < img_size[1] * scale[0]:
            ix, iy = _display_to_image(x, y)
            drag_start[0] = (ix, iy)
            if rect_u_id[0]:
                canvas_u.delete(rect_u_id[0])
            if rect_m_id[0]:
                canvas_m.delete(rect_m_id[0])
            rect_u_id[0] = canvas_u.create_rectangle(x, y, x, y, outline="lime", width=2)
            rect_m_id[0] = canvas_m.create_rectangle(x, y, x, y, outline="lime", width=2)
            current_roi[0] = [ix, iy, ix, iy]

    def _on_mouse_move(event, canvas: tk.Canvas):
        if drag_start[0] is None:
            return
        try:
            x, y = canvas.canvasx(event.x), canvas.canvasy(event.y)
        except Exception:
            return
        ix, iy = _display_to_image(x, y)
        x1, y1 = drag_start[0][0], drag_start[0][1]
        s = scale[0]
        r = (int(min(x1, ix) * s), int(min(y1, iy) * s), int(max(x1, ix) * s), int(max(y1, iy) * s))
        if rect_u_id[0]:
            canvas_u.coords(rect_u_id[0], *r)
        if rect_m_id[0]:
            canvas_m.coords(rect_m_id[0], *r)
        current_roi[0] = [min(x1, ix), min(y1, iy), max(x1, ix), max(y1, iy)]

    def _on_mouse_up(event):
        if drag_start[0] is None:
            return
        stem = u_files[idx[0]].stem
        if current_roi[0]:
            x1, y1, x2, y2 = current_roi[0]
            if abs(x2 - x1) > 4 and abs(y2 - y1) > 4:
                roi_map[stem] = [x1, y1, x2, y2]
        drag_start[0] = None

    canvas_u.bind("<ButtonPress-1>", lambda e: _on_mouse_down(e, canvas_u))
    canvas_u.bind("<B1-Motion>", lambda e: _on_mouse_move(e, canvas_u))
    canvas_m.bind("<B1-Motion>", lambda e: _on_mouse_move(e, canvas_m))
    root.bind("<ButtonRelease-1>", _on_mouse_up)
    root.bind("<Left>", lambda e: _go(-1))
    root.bind("<Right>", lambda e: _go(1))

    def _go(delta: int):
        idx[0] = (idx[0] + delta) % len(u_files)
        _load_image()

    def _save():
        save_roi_map(roi_map_path, roi_map)
        if on_save:
            on_save(roi_map_path)
        from tkinter import messagebox
        messagebox.showinfo("Saved", f"ROI map saved to:\n{roi_map_path}")

    def _clear_current():
        stem = u_files[idx[0]].stem
        roi_map.pop(stem, None)
        if rect_u_id[0]:
            canvas_u.delete(rect_u_id[0])
            rect_u_id[0] = None
        if rect_m_id[0]:
            canvas_m.delete(rect_m_id[0])
            rect_m_id[0] = None
        current_roi[0] = None
        nav_label.config(
            text=f"Image {idx[0] + 1}/{len(u_files)} — {u_files[idx[0]].name}  (ROI cleared, drag on left to set)"
        )

    nav_f = tk.Frame(root)
    nav_f.pack(fill=tk.X, padx=8, pady=4)
    ttk.Button(nav_f, text="Prev", command=lambda: _go(-1)).pack(side=tk.LEFT, padx=4)
    ttk.Button(nav_f, text="Next", command=lambda: _go(1)).pack(side=tk.LEFT, padx=4)
    nav_label = tk.Label(nav_f, text="", font=("Segoe UI", 10))
    nav_label.pack(side=tk.LEFT, padx=12)
    tk.Label(nav_f, text="(Left/Right keys)", font=("Segoe UI", 8), foreground="gray").pack(side=tk.LEFT)
    ttk.Button(nav_f, text="Clear ROI", command=_clear_current).pack(side=tk.RIGHT, padx=4)
    ttk.Button(nav_f, text="Save ROI map", command=_save).pack(side=tk.RIGHT, padx=4)

    root.after(80, _load_image)
