"""
Pin detection GUI — 학습/추론 모드, 로컬 파일 업로드, YOLO26 학습.
"""
import threading
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


class PinDetectionGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Connector Pin Detection — YOLO26")
        self.root.geometry("600x500")
        self.root.minsize(400, 400)

        self.unmasked_dir = tk.StringVar()
        self.masked_dir = tk.StringVar()
        self.excel_dir = tk.StringVar()
        self.model_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="pin_models")
        self.epochs_var = tk.IntVar(value=100)
        self.imgsz_var = tk.IntVar(value=640)

        self._build_ui()

    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 학습 탭
        train_f = ttk.Frame(nb, padding=8)
        nb.add(train_f, text="학습 (Train)")

        ttk.Label(train_f, text="마스킹 전 이미지 폴더:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(train_f, textvariable=self.unmasked_dir, width=50).grid(row=0, column=1, padx=4, pady=2)
        ttk.Button(train_f, text="찾아보기", command=lambda: self.unmasked_dir.set(_select_dir(self.root, "마스킹 전 폴더"))).grid(row=0, column=2, pady=2)

        ttk.Label(train_f, text="마스킹 후 이미지 폴더:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(train_f, textvariable=self.masked_dir, width=50).grid(row=1, column=1, padx=4, pady=2)
        ttk.Button(train_f, text="찾아보기", command=lambda: self.masked_dir.set(_select_dir(self.root, "마스킹 후 폴더"))).grid(row=1, column=2, pady=2)

        ttk.Label(train_f, text="엑셀 폴더 (선택):").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(train_f, textvariable=self.excel_dir, width=50).grid(row=2, column=1, padx=4, pady=2)
        ttk.Button(train_f, text="찾아보기", command=lambda: self.excel_dir.set(_select_dir(self.root, "엑셀 폴더"))).grid(row=2, column=2, pady=2)

        ttk.Label(train_f, text="출력 폴더:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(train_f, textvariable=self.output_dir, width=50).grid(row=3, column=1, padx=4, pady=2)
        ttk.Button(train_f, text="찾아보기", command=lambda: self.output_dir.set(_select_dir(self.root, "출력 폴더") or self.output_dir.get())).grid(row=3, column=2, pady=2)

        ttk.Label(train_f, text="Epochs:").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(train_f, from_=10, to=500, textvariable=self.epochs_var, width=8).grid(row=4, column=1, sticky=tk.W, padx=4, pady=2)

        ttk.Label(train_f, text="이미지 크기 (imgsz):").grid(row=5, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(train_f, from_=320, to=1280, increment=64, textvariable=self.imgsz_var, width=8).grid(row=5, column=1, sticky=tk.W, padx=4, pady=2)
        ttk.Label(train_f, text="(소형 핀: 640~1280 권장)").grid(row=5, column=2, sticky=tk.W, pady=2)

        self.train_progress = ttk.Progressbar(train_f, mode="indeterminate")
        self.train_progress.grid(row=6, column=0, columnspan=3, sticky=tk.EW, pady=8)
        self.train_status = ttk.Label(train_f, text="")
        self.train_status.grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=2)

        ttk.Button(train_f, text="학습 시작", command=self._on_train).grid(row=8, column=1, pady=12)

        # 추론 탭
        inf_f = ttk.Frame(nb, padding=8)
        nb.add(inf_f, text="추론 (Inference)")

        ttk.Label(inf_f, text="추론할 이미지:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.inference_image = tk.StringVar()
        ttk.Entry(inf_f, textvariable=self.inference_image, width=50).grid(row=0, column=1, padx=4, pady=2)
        ttk.Button(inf_f, text="찾아보기", command=lambda: self.inference_image.set(_select_file(self.root, "이미지 선택"))).grid(row=0, column=2, pady=2)

        ttk.Label(inf_f, text="학습된 모델 (.pt):").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(inf_f, textvariable=self.model_path, width=50).grid(row=1, column=1, padx=4, pady=2)
        ttk.Button(inf_f, text="찾아보기", command=lambda: self.model_path.set(_select_file(self.root, "모델 선택", [("PyTorch", "*.pt")]))).grid(row=1, column=2, pady=2)

        ttk.Label(inf_f, text="엑셀 형식 참조 (선택):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.excel_format = tk.StringVar()
        ttk.Entry(inf_f, textvariable=self.excel_format, width=50).grid(row=2, column=1, padx=4, pady=2)
        ttk.Button(inf_f, text="찾아보기", command=lambda: self.excel_format.set(_select_file(self.root, "엑셀", [("Excel", "*.xlsx *.xls")]))).grid(row=2, column=2, pady=2)

        self.inference_status = ttk.Label(inf_f, text="")
        self.inference_status.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=8)

        ttk.Button(inf_f, text="추론 실행", command=self._on_inference).grid(row=4, column=1, pady=12)

        # 상태바
        self.status_var = tk.StringVar(value="준비")
        ttk.Label(self.root, textvariable=self.status_var).pack(side=tk.BOTTOM, pady=4)

    def _on_train(self):
        u = self.unmasked_dir.get().strip()
        m = self.masked_dir.get().strip()
        if not u or not m:
            messagebox.showerror("오류", "마스킹 전/후 폴더를 선택하세요.")
            return

        def run():
            try:
                self.root.after(0, lambda: self.train_progress.start(10))
                self.root.after(0, lambda: self.train_status.config(text="데이터셋 준비 중..."))
                self.root.after(0, lambda: self.status_var.set("학습 중..."))

                from .train import train_pin_model
                model_path = train_pin_model(
                    unmasked_dir=u,
                    masked_dir=m,
                    output_dir=self.output_dir.get(),
                    epochs=self.epochs_var.get(),
                    imgsz=self.imgsz_var.get(),
                )

                self.root.after(0, lambda: self.model_path.set(str(model_path)))
                self.root.after(0, lambda: self.train_status.config(text=f"완료: {model_path}"))
                self.root.after(0, lambda: messagebox.showinfo("완료", f"모델 저장 완료:\n{model_path}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("오류", str(e)))
                self.root.after(0, lambda: self.train_status.config(text=f"오류: {e}"))
            finally:
                self.root.after(0, lambda: self.train_progress.stop())
                self.root.after(0, lambda: self.status_var.set("준비"))

        threading.Thread(target=run, daemon=True).start()

    def _on_inference(self):
        img_path = self.inference_image.get().strip()
        model_path = self.model_path.get().strip()
        if not img_path or not model_path:
            messagebox.showerror("오류", "이미지와 모델 경로를 선택하세요.")
            return

        def run():
            try:
                self.root.after(0, lambda: self.status_var.set("추론 중..."))
                self.root.after(0, lambda: self.inference_status.config(text="처리 중..."))

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
                msg = f"위핀: {len(upper)}, 아래핀: {len(lower)} → {'OK' if len(upper) == 20 and len(lower) == 20 else 'NG'}\n이미지: {out_img}\n엑셀: {excel_out}"
                self.root.after(0, lambda: self.inference_status.config(text=msg))
                self.root.after(0, lambda: messagebox.showinfo("완료", msg))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("오류", str(e)))
                self.root.after(0, lambda: self.inference_status.config(text=f"오류: {e}"))
            finally:
                self.root.after(0, lambda: self.status_var.set("준비"))

        threading.Thread(target=run, daemon=True).start()


def main():
    root = tk.Tk()
    root.tk.call("tk", "scaling", 1.2)
    app = PinDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
