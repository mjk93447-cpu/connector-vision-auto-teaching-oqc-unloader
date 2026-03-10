"""
ROI Editor GUI: ROI rectangles, zoom, brush masking, split ROI (upper/lower).
Shows unmasked (YOLO input) and masked (ground-truth) side by side.
Saves to roi_map.json. ROADMAP 10.20, 10.24.
"""
import json
import tkinter as tk
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageTk

from .dataset import IMG_EXTS, _find_masked_pair

try:
    from tkinter import ttk
except ImportError:
    import tkinter.ttk as ttk

# Zoom limits
ZOOM_MIN, ZOOM_MAX = 0.25, 4.0
ZOOM_STEP = 1.15
BRUSH_RADIUS = 4
SQUARE_SIZE = 12  # Small square per-pin marker (Action #33)
ERASE_RADIUS = 10
# Red for target masking (avoids confusion with original green markers, YOLO learns from this)
TARGET_MARKER_RGB = (255, 0, 0)


def load_roi_map(path: Path) -> dict:
    """Load roi_map.json. Supports [x1,y1,x2,y2] or {upper:[...], lower:[...]}."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        out = {}
        for k, v in data.items():
            if isinstance(v, (list, tuple)) and len(v) == 4:
                out[k] = list(v)
            elif isinstance(v, dict) and "upper" in v and "lower" in v:
                u, l = v.get("upper"), v.get("lower")
                if isinstance(u, (list, tuple)) and len(u) == 4 and isinstance(l, (list, tuple)) and len(l) == 4:
                    out[k] = {"upper": list(u), "lower": list(l)}
        return out
    except Exception:
        return {}


def save_roi_map(path: Path, roi_map: dict) -> None:
    """Save roi_map to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(roi_map, f, indent=2)


def _roi_to_bbox(roi) -> tuple[int, int, int, int] | None:
    """Convert roi (list or dict) to single bbox for crop. Uses union if split."""
    if roi is None:
        return None
    if isinstance(roi, (list, tuple)) and len(roi) == 4:
        return tuple(roi)
    if isinstance(roi, dict) and "upper" in roi and "lower" in roi:
        u = roi["upper"]
        l = roi["lower"]
        x1 = min(u[0], l[0])
        y1 = min(u[1], l[1])
        x2 = max(u[2], l[2])
        y2 = max(u[3], l[3])
        return (x1, y1, x2, y2)
    return None


def run_roi_editor(
    unmasked_dir: str | Path,
    masked_dir: str | Path,
    output_dir: str | Path,
    parent: tk.Tk | tk.Toplevel | None = None,
    on_save: Callable[[Path], None] | None = None,
) -> None:
    """
    Open ROI Editor window.
    - Zoom: MouseWheel to zoom in/out
    - Rectangle: drag on left to set ROI
    - Brush: paint on right (masked) to mark pins (green)
    - Split ROI: Upper ROI / Lower ROI for 20+20 pins with different lighting
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
        return

    roi_map = load_roi_map(roi_map_path)

    root = tk.Toplevel(parent) if parent else tk.Toplevel()
    root.title("ROI Editor — Square: click right to add pin | Erase: click right to remove | Brush: paint | Scroll: zoom")
    root.geometry("1280x750")
    root.minsize(900, 550)

    paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _make_canvas_frame(label: str) -> tuple[tk.Frame, tk.Canvas, tk.Frame]:
        f = ttk.LabelFrame(paned, text=label)
        paned.add(f, weight=1)
        inner = tk.Frame(f)
        inner.pack(fill=tk.BOTH, expand=True)
        c = tk.Canvas(inner, bg="#1a1a1a", highlightthickness=0)
        sb_y = ttk.Scrollbar(inner)
        sb_x = ttk.Scrollbar(inner, orient=tk.HORIZONTAL)
        c.config(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        sb_y.config(command=c.yview)
        sb_x.config(command=c.xview)
        sb_y.pack(side=tk.RIGHT, fill=tk.Y)
        sb_x.pack(side=tk.BOTTOM, fill=tk.X)
        c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        return f, c, inner

    left_f, canvas_u, _ = _make_canvas_frame("Unmasked (YOLO input — drag ROI)")
    right_f, canvas_m, _ = _make_canvas_frame("Masked (red = target pins for YOLO — Square/Brush add, Erase remove)")

    # State
    idx = [0]
    photo_u = [None]
    photo_m = [None]
    scale = [1.0]
    img_size = [0, 0]
    img_u_pil = [None]
    img_m_pil = [None]
    rect_u_id = [None]
    rect_m_id = [None]
    drag_start = [None]
    current_roi = [None]
    mode = ["rect"]
    split_mode = [False]
    current_roi_upper = [None]
    current_roi_lower = [None]
    brush_active = [False]
    last_brush_xy = [None]
    square_squares = []  # List of (x1,y1,x2,y2) for current image's square markers (for redraw)

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

    def _redraw_at_scale():
        """Redraw canvases at current scale (for zoom)."""
        if img_u_pil[0] is None:
            return
        new_w = int(img_size[0] * scale[0])
        new_h = int(img_size[1] * scale[0])
        img_u_r = img_u_pil[0].resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_m_r = img_m_pil[0].resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo_u[0] = ImageTk.PhotoImage(img_u_r)
        photo_m[0] = ImageTk.PhotoImage(img_m_r)
        canvas_u.delete("all")
        canvas_u.create_image(0, 0, anchor=tk.NW, image=photo_u[0])
        canvas_m.delete("all")
        canvas_m.create_image(0, 0, anchor=tk.NW, image=photo_m[0])
        canvas_u.config(scrollregion=(0, 0, new_w, new_h))
        canvas_m.config(scrollregion=(0, 0, new_w, new_h))
        s = scale[0]
        if split_mode[0]:
            if current_roi_upper[0]:
                r = current_roi_upper[0]
                canvas_u.create_rectangle(int(r[0]*s), int(r[1]*s), int(r[2]*s), int(r[3]*s), outline="cyan", width=2)
                canvas_m.create_rectangle(int(r[0]*s), int(r[1]*s), int(r[2]*s), int(r[3]*s), outline="cyan", width=2)
            if current_roi_lower[0]:
                r = current_roi_lower[0]
                canvas_u.create_rectangle(int(r[0]*s), int(r[1]*s), int(r[2]*s), int(r[3]*s), outline="orange", width=2)
                canvas_m.create_rectangle(int(r[0]*s), int(r[1]*s), int(r[2]*s), int(r[3]*s), outline="orange", width=2)
        elif current_roi[0]:
            r = current_roi[0]
            canvas_u.create_rectangle(int(r[0]*s), int(r[1]*s), int(r[2]*s), int(r[3]*s), outline="lime", width=2)
            canvas_m.create_rectangle(int(r[0]*s), int(r[1]*s), int(r[2]*s), int(r[3]*s), outline="lime", width=2)

    def _draw_split_roi():
        rect_u_id[0] = None
        rect_m_id[0] = None
        s = scale[0]
        if current_roi_upper[0]:
            r = current_roi_upper[0]
            canvas_u.create_rectangle(int(r[0]*s), int(r[1]*s), int(r[2]*s), int(r[3]*s), outline="cyan", width=2)
            canvas_m.create_rectangle(int(r[0]*s), int(r[1]*s), int(r[2]*s), int(r[3]*s), outline="cyan", width=2)
        if current_roi_lower[0]:
            r = current_roi_lower[0]
            canvas_u.create_rectangle(int(r[0]*s), int(r[1]*s), int(r[2]*s), int(r[3]*s), outline="orange", width=2)
            canvas_m.create_rectangle(int(r[0]*s), int(r[1]*s), int(r[2]*s), int(r[3]*s), outline="orange", width=2)

    def _on_zoom(delta: int, event):
        old_s = scale[0]
        new_s = old_s * (ZOOM_STEP ** delta)
        new_s = max(ZOOM_MIN, min(ZOOM_MAX, new_s))
        if abs(new_s - old_s) < 0.01:
            return
        scale[0] = new_s
        _redraw_at_scale()

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
        img_u_pil[0] = img_u
        img_m_pil[0] = img_m
        square_squares.clear()

        cw = 550
        ch = max(canvas_u.winfo_height() or 400, 300)
        rw = cw / img_u.size[0]
        rh = ch / img_u.size[1]
        scale[0] = min(rw, rh, 1.0)
        new_w = int(img_u.size[0] * scale[0])
        new_h = int(img_u.size[1] * scale[0])

        img_u_resized = img_u.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_m_resized = img_m.resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo_u[0] = ImageTk.PhotoImage(img_u_resized)
        photo_u[0]._img = img_u_resized
        photo_m[0] = ImageTk.PhotoImage(img_m_resized)
        photo_m[0]._img = img_m_resized

        canvas_u.delete("all")
        canvas_u.create_image(0, 0, anchor=tk.NW, image=photo_u[0])
        canvas_u.config(scrollregion=(0, 0, new_w, new_h))
        canvas_m.delete("all")
        canvas_m.create_image(0, 0, anchor=tk.NW, image=photo_m[0])
        canvas_m.config(scrollregion=(0, 0, new_w, new_h))

        stem = u_path.stem
        roi = roi_map.get(stem)
        if isinstance(roi, dict):
            split_mode[0] = True
            current_roi_upper[0] = roi.get("upper")
            current_roi_lower[0] = roi.get("lower")
            current_roi[0] = None
            _draw_split_roi()
        elif isinstance(roi, (list, tuple)) and len(roi) == 4:
            split_mode[0] = False
            current_roi[0] = list(roi)
            current_roi_upper[0] = None
            current_roi_lower[0] = None
            _draw_roi_on_canvas(canvas_u, rect_u_id, roi)
            _draw_roi_on_canvas(canvas_m, rect_m_id, roi)
        else:
            split_mode[0] = False
            rect_u_id[0] = None
            rect_m_id[0] = None
            current_roi[0] = None
            current_roi_upper[0] = None
            current_roi_lower[0] = None

        pair_info = f" + {m_path.name}" if m_path else " (no masked pair)"
        mode_txt = mode[0].capitalize() if mode[0] in ("brush", "square", "erase") else ("Split ROI" if split_mode[0] else "Rectangle")
        nav_label.config(
            text=f"Image {idx[0] + 1}/{len(u_files)} — {u_path.name}{pair_info}  |  Mode: {mode_txt}"
        )

    def _on_mouse_down(event, canvas: tk.Canvas, is_left: bool):
        try:
            x, y = canvas.canvasx(event.x), canvas.canvasy(event.y)
        except Exception:
            return
        ix, iy = _display_to_image(x, y)
        if not (0 <= ix < img_size[0] and 0 <= iy < img_size[1]):
            return

        if mode[0] == "brush" and not is_left:
            brush_active[0] = True
            last_brush_xy[0] = (ix, iy)
            _apply_brush(ix, iy)
            return

        if mode[0] == "square" and not is_left:
            _apply_square(ix, iy)
            _redraw_masked_canvas()
            _save_masked_image()
            return

        if mode[0] == "erase" and not is_left:
            _apply_erase(ix, iy)
            _redraw_masked_canvas()
            _save_masked_image()
            return

        if mode[0] == "rect" and is_left:
            drag_start[0] = (ix, iy)
            for r in [rect_u_id, rect_m_id]:
                if r[0]:
                    canvas_u.delete(r[0])
                    canvas_m.delete(r[0])
                    r[0] = None
            s = scale[0]
            if split_mode[0]:
                if current_roi_upper[0] is None:
                    rect_u_id[0] = canvas_u.create_rectangle(x, y, x, y, outline="cyan", width=2)
                    rect_m_id[0] = canvas_m.create_rectangle(x, y, x, y, outline="cyan", width=2)
                else:
                    rect_u_id[0] = canvas_u.create_rectangle(x, y, x, y, outline="orange", width=2)
                    rect_m_id[0] = canvas_m.create_rectangle(x, y, x, y, outline="orange", width=2)
            else:
                rect_u_id[0] = canvas_u.create_rectangle(x, y, x, y, outline="lime", width=2)
                rect_m_id[0] = canvas_m.create_rectangle(x, y, x, y, outline="lime", width=2)
            current_roi[0] = [ix, iy, ix, iy]

    def _apply_brush(ix: int, iy: int):
        if img_m_pil[0] is None:
            return
        draw = ImageDraw.Draw(img_m_pil[0])
        rad = BRUSH_RADIUS
        draw.ellipse((ix - rad, iy - rad, ix + rad, iy + rad), fill=TARGET_MARKER_RGB, outline=TARGET_MARKER_RGB)

    def _apply_square(ix: int, iy: int):
        """Add one small red square at click (Action #33)."""
        if img_m_pil[0] is None:
            return
        half = SQUARE_SIZE // 2
        x1 = max(0, ix - half)
        y1 = max(0, iy - half)
        x2 = min(img_size[0], ix + half)
        y2 = min(img_size[1], iy + half)
        if x2 <= x1 or y2 <= y1:
            return
        draw = ImageDraw.Draw(img_m_pil[0])
        draw.rectangle((x1, y1, x2, y2), fill=TARGET_MARKER_RGB, outline=TARGET_MARKER_RGB)
        square_squares.append([x1, y1, x2, y2])

    def _apply_erase(ix: int, iy: int):
        """Remove marker at click by restoring unmasked pixels (Action #33)."""
        if img_m_pil[0] is None or img_u_pil[0] is None:
            return
        rad = ERASE_RADIUS
        x1 = max(0, ix - rad)
        y1 = max(0, iy - rad)
        x2 = min(img_size[0], ix + rad + 1)
        y2 = min(img_size[1], iy + rad + 1)
        region_u = img_u_pil[0].crop((x1, y1, x2, y2))
        img_m_pil[0].paste(region_u, (x1, y1))
        # Remove squares that overlap this region
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        square_squares[:] = [
            b for b in square_squares
            if not (abs((b[0] + b[2]) // 2 - cx) < rad + SQUARE_SIZE and abs((b[1] + b[3]) // 2 - cy) < rad + SQUARE_SIZE)
        ]
        new_w = int(img_size[0] * scale[0])
        new_h = int(img_size[1] * scale[0])
        img_m_resized = img_m_pil[0].resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo_m[0] = ImageTk.PhotoImage(img_m_resized)
        canvas_m.delete("all")
        canvas_m.create_image(0, 0, anchor=tk.NW, image=photo_m[0])
        s = scale[0]
        if split_mode[0]:
            if current_roi_upper[0]:
                box = current_roi_upper[0]
                canvas_m.create_rectangle(int(box[0]*s), int(box[1]*s), int(box[2]*s), int(box[3]*s), outline="cyan", width=2)
            if current_roi_lower[0]:
                box = current_roi_lower[0]
                canvas_m.create_rectangle(int(box[0]*s), int(box[1]*s), int(box[2]*s), int(box[3]*s), outline="orange", width=2)
        elif current_roi[0]:
            box = current_roi[0]
            canvas_m.create_rectangle(int(box[0]*s), int(box[1]*s), int(box[2]*s), int(box[3]*s), outline="lime", width=2)

    def _redraw_masked_canvas():
        """Redraw masked canvas after square/erase (no PIL change for brush - it does inline)."""
        if img_m_pil[0] is None:
            return
        new_w = int(img_size[0] * scale[0])
        new_h = int(img_size[1] * scale[0])
        img_m_resized = img_m_pil[0].resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo_m[0] = ImageTk.PhotoImage(img_m_resized)
        canvas_m.delete("all")
        canvas_m.create_image(0, 0, anchor=tk.NW, image=photo_m[0])
        s = scale[0]
        if split_mode[0]:
            if current_roi_upper[0]:
                box = current_roi_upper[0]
                canvas_m.create_rectangle(int(box[0]*s), int(box[1]*s), int(box[2]*s), int(box[3]*s), outline="cyan", width=2)
            if current_roi_lower[0]:
                box = current_roi_lower[0]
                canvas_m.create_rectangle(int(box[0]*s), int(box[1]*s), int(box[2]*s), int(box[3]*s), outline="orange", width=2)
        elif current_roi[0]:
            box = current_roi[0]
            canvas_m.create_rectangle(int(box[0]*s), int(box[1]*s), int(box[2]*s), int(box[3]*s), outline="lime", width=2)

    def _on_mouse_move(event, canvas: tk.Canvas, is_left: bool):
        try:
            x, y = canvas.canvasx(event.x), canvas.canvasy(event.y)
        except Exception:
            return
        ix, iy = _display_to_image(x, y)

        if brush_active[0] and not is_left:
            if last_brush_xy[0]:
                lx, ly = last_brush_xy[0]
                steps = max(1, int(((ix - lx) ** 2 + (iy - ly) ** 2) ** 0.5 / 2))
                for i in range(1, steps + 1):
                    tx = int(lx + (ix - lx) * i / steps)
                    ty = int(ly + (iy - ly) * i / steps)
                    _apply_brush(tx, ty)
            last_brush_xy[0] = (ix, iy)
            return

        if drag_start[0] is None:
            return
        x1, y1 = drag_start[0][0], drag_start[0][1]
        s = scale[0]
        r = (int(min(x1, ix) * s), int(min(y1, iy) * s), int(max(x1, ix) * s), int(max(y1, iy) * s))
        if rect_u_id[0]:
            canvas_u.coords(rect_u_id[0], *r)
        if rect_m_id[0]:
            canvas_m.coords(rect_m_id[0], *r)
        current_roi[0] = [min(x1, ix), min(y1, iy), max(x1, ix), max(y1, iy)]

    def _on_mouse_up(event):
        if brush_active[0]:
            brush_active[0] = False
            last_brush_xy[0] = None
            _save_masked_image()
            return
        if drag_start[0] is None:
            return
        stem = u_files[idx[0]].stem
        if split_mode[0]:
            if current_roi_upper[0] is None and current_roi[0]:
                x1, y1, x2, y2 = current_roi[0]
                if abs(x2 - x1) > 4 and abs(y2 - y1) > 4:
                    current_roi_upper[0] = [x1, y1, x2, y2]
            elif current_roi_lower[0] is None and current_roi[0]:
                x1, y1, x2, y2 = current_roi[0]
                if abs(x2 - x1) > 4 and abs(y2 - y1) > 4:
                    current_roi_lower[0] = [x1, y1, x2, y2]
                    roi_map[stem] = {"upper": current_roi_upper[0], "lower": current_roi_lower[0]}
        elif current_roi[0]:
            x1, y1, x2, y2 = current_roi[0]
            if abs(x2 - x1) > 4 and abs(y2 - y1) > 4:
                roi_map[stem] = [x1, y1, x2, y2]
        drag_start[0] = None

    def _save_masked_image():
        m_path = _find_masked(u_files[idx[0]])
        if m_path and img_m_pil[0] is not None:
            try:
                img_m_pil[0].save(m_path)
            except Exception:
                pass

    canvas_u.bind("<ButtonPress-1>", lambda e: _on_mouse_down(e, canvas_u, True))
    canvas_u.bind("<B1-Motion>", lambda e: _on_mouse_move(e, canvas_u, True))
    canvas_m.bind("<ButtonPress-1>", lambda e: _on_mouse_down(e, canvas_m, False))
    canvas_m.bind("<B1-Motion>", lambda e: _on_mouse_move(e, canvas_m, False))
    root.bind("<ButtonRelease-1>", _on_mouse_up)
    canvas_u.bind("<MouseWheel>", lambda e: _on_zoom(1 if e.delta > 0 else -1, e))
    canvas_m.bind("<MouseWheel>", lambda e: _on_zoom(1 if e.delta > 0 else -1, e))
    root.bind("<Left>", lambda e: _go(-1))
    root.bind("<Right>", lambda e: _go(1))

    def _go(delta: int):
        _save_masked_image()
        idx[0] = (idx[0] + delta) % len(u_files)
        _load_image()

    def _set_mode(m: str):
        mode[0] = m
        nav_label.config(text=nav_label.cget("text").split("|")[0] + f"  |  Mode: {m}")

    def _toggle_split():
        split_mode[0] = not split_mode[0]
        if not split_mode[0]:
            stem = u_files[idx[0]].stem
            if isinstance(roi_map.get(stem), dict):
                bbox = _roi_to_bbox(roi_map[stem])
                if bbox:
                    roi_map[stem] = list(bbox)
            current_roi_upper[0] = None
            current_roi_lower[0] = None
        _load_image()

    def _save():
        _save_masked_image()
        to_save = {}
        for k, v in roi_map.items():
            if isinstance(v, dict) and "upper" in v and "lower" in v:
                u, l = v.get("upper"), v.get("lower")
                if isinstance(u, (list, tuple)) and len(u) == 4 and isinstance(l, (list, tuple)) and len(l) == 4:
                    to_save[k] = v
            elif isinstance(v, (list, tuple)) and len(v) == 4:
                to_save[k] = list(v)
        save_roi_map(roi_map_path, to_save)
        if on_save:
            on_save(roi_map_path)
        from tkinter import messagebox
        messagebox.showinfo("Saved", f"ROI map saved to:\n{roi_map_path}")

    def _clear_current():
        stem = u_files[idx[0]].stem
        roi_map.pop(stem, None)
        rect_u_id[0] = None
        rect_m_id[0] = None
        current_roi[0] = None
        current_roi_upper[0] = None
        current_roi_lower[0] = None
        _load_image()

    nav_f = tk.Frame(root)
    nav_f.pack(fill=tk.X, padx=8, pady=4)
    ttk.Button(nav_f, text="Prev", command=lambda: _go(-1)).pack(side=tk.LEFT, padx=4)
    ttk.Button(nav_f, text="Next", command=lambda: _go(1)).pack(side=tk.LEFT, padx=4)
    nav_label = tk.Label(nav_f, text="", font=("Segoe UI", 10))
    nav_label.pack(side=tk.LEFT, padx=12)
    tk.Label(nav_f, text="(Left/Right keys | Scroll: zoom)", font=("Segoe UI", 8), fg="gray").pack(side=tk.LEFT, padx=4)
    ttk.Button(nav_f, text="Rectangle", command=lambda: _set_mode("rect")).pack(side=tk.LEFT, padx=2)
    ttk.Button(nav_f, text="Square", command=lambda: _set_mode("square")).pack(side=tk.LEFT, padx=2)
    ttk.Button(nav_f, text="Brush", command=lambda: _set_mode("brush")).pack(side=tk.LEFT, padx=2)
    ttk.Button(nav_f, text="Erase", command=lambda: _set_mode("erase")).pack(side=tk.LEFT, padx=2)
    ttk.Button(nav_f, text="Split ROI", command=_toggle_split).pack(side=tk.LEFT, padx=2)
    ttk.Button(nav_f, text="Clear ROI", command=_clear_current).pack(side=tk.RIGHT, padx=4)
    ttk.Button(nav_f, text="Save ROI map", command=_save).pack(side=tk.RIGHT, padx=4)

    root.after(80, _load_image)
