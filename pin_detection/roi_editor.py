"""
ROI Editor GUI: manually draw ROI rectangles per image.
Saves to roi_map.json: { "stem": [x1, y1, x2, y2] }.
ROADMAP 10.20.
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
    - unmasked_dir, masked_dir: image folders
    - output_dir: where to save roi_map.json (output_dir/roi_map.json)
    - on_save: callback(path) when user saves
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
    root.title("ROI Editor — Draw rectangle per image")
    root.geometry("900x700")
    root.minsize(600, 400)

    # Canvas for image + rectangle
    canvas_frame = tk.Frame(root)
    canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    canvas = tk.Canvas(canvas_frame, bg="#1a1a1a", highlightthickness=0)
    scroll_y = tk.Scrollbar(canvas_frame)
    scroll_x = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)

    canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
    scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
    scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # State
    idx = [0]  # mutable for closure
    photo_ref = [None]
    scale = [1.0]
    img_size = [0, 0]
    rect_id = [None]
    drag_start = [None]
    current_roi = [None]  # [x1,y1,x2,y2] in display coords

    def _find_masked(u_path: Path) -> Path | None:
        try:
            return _find_masked_pair(u_path, masked_dir)
        except FileNotFoundError:
            return None

    def _display_to_image(x: float, y: float) -> tuple[int, int]:
        s = scale[0]
        return (int(x / s), int(y / s))

    def _image_to_display(x: int, y: int) -> tuple[int, int]:
        s = scale[0]
        return (int(x * s), int(y * s))

    def _load_image():
        u_path = u_files[idx[0]]
        m_path = _find_masked(u_path)
        img_path = m_path if m_path else u_path
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (640, 480), (40, 40, 40))
        img_size[0], img_size[1] = img.size
        # Fit to canvas
        cw = canvas.winfo_width() or 800
        ch = canvas.winfo_height() or 500
        if cw < 10:
            cw = 800
        if ch < 10:
            ch = 500
        rw = cw / img.size[0]
        rh = ch / img.size[1]
        scale[0] = min(rw, rh, 1.0)
        new_w = int(img.size[0] * scale[0])
        new_h = int(img.size[1] * scale[0])
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo_ref[0] = ImageTk.PhotoImage(img_resized)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo_ref[0])
        canvas.config(scrollregion=(0, 0, new_w, new_h))
        # Restore or init ROI
        stem = u_path.stem
        roi = roi_map.get(stem)
        if roi:
            x1, y1, x2, y2 = roi
            sx, sy = scale[0], scale[0]
            r1 = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
            rect_id[0] = canvas.create_rectangle(*r1, outline="lime", width=2)
            current_roi[0] = [x1, y1, x2, y2]
        else:
            rect_id[0] = None
            current_roi[0] = None
        # Title
        nav_label.config(text=f"Image {idx[0] + 1} / {len(u_files)} — {u_path.name}" + (f"  ROI: {roi}" if roi else "  (drag to set ROI)"))

    def _on_mouse_down(event):
        x, y = canvas.canvasx(event.x), canvas.canvasy(event.y)
        if 0 <= x < img_size[0] * scale[0] and 0 <= y < img_size[1] * scale[0]:
            ix, iy = _display_to_image(x, y)
            drag_start[0] = (ix, iy)
            if rect_id[0]:
                canvas.delete(rect_id[0])
            rect_id[0] = canvas.create_rectangle(x, y, x, y, outline="lime", width=2)
            current_roi[0] = [ix, iy, ix, iy]

    def _on_mouse_move(event):
        if drag_start[0] is None:
            return
        x, y = canvas.canvasx(event.x), canvas.canvasy(event.y)
        ix, iy = _display_to_image(x, y)
        x1, y1 = drag_start[0][0], drag_start[0][1]
        sx, sy = scale[0], scale[0]
        r1 = (int(x1 * sx), int(y1 * sy), int(ix * sx), int(iy * sy))
        canvas.coords(rect_id[0], *r1)
        current_roi[0] = [
            min(x1, ix), min(y1, iy),
            max(x1, ix), max(y1, iy),
        ]

    def _on_mouse_up(event):
        if drag_start[0] is None:
            return
        stem = u_files[idx[0]].stem
        if current_roi[0]:
            x1, y1, x2, y2 = current_roi[0]
            if abs(x2 - x1) > 4 and abs(y2 - y1) > 4:
                roi_map[stem] = [x1, y1, x2, y2]
        drag_start[0] = None

    canvas.bind("<ButtonPress-1>", _on_mouse_down)
    canvas.bind("<B1-Motion>", _on_mouse_move)
    canvas.bind("<ButtonRelease-1>", _on_mouse_up)
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
        if rect_id[0]:
            canvas.delete(rect_id[0])
            rect_id[0] = None
        current_roi[0] = None
        nav_label.config(text=f"Image {idx[0] + 1} / {len(u_files)} — {u_files[idx[0]].name}  (ROI cleared, drag to set)")

    # Navigation bar
    nav_f = tk.Frame(root)
    nav_f.pack(fill=tk.X, padx=8, pady=4)
    ttk.Button(nav_f, text="Prev", command=lambda: _go(-1)).pack(side=tk.LEFT, padx=4)
    ttk.Button(nav_f, text="Next", command=lambda: _go(1)).pack(side=tk.LEFT, padx=4)
    nav_label = tk.Label(nav_f, text="", font=("Segoe UI", 10))
    nav_label.pack(side=tk.LEFT, padx=12)
    ttk.Button(nav_f, text="Clear ROI", command=_clear_current).pack(side=tk.RIGHT, padx=4)
    ttk.Button(nav_f, text="Save ROI map", command=_save).pack(side=tk.RIGHT, padx=4)

    # Defer first load so canvas has layout dimensions
    root.after(80, _load_image)
