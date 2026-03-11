# Real Factory Data Test Manual

This guide explains how to install the EXE on a factory line PC and test **pin detection** using real captured images. Written for field technicians.

---

## 1. EXE Installation and Run

### 1.1 Download

1. Go to GitHub → **Actions** → **Build Pin Detection EXE** → select the latest successful run
2. Under **Artifacts**, download `pin_detection_gui_windows`
3. Extract the archive and run `pin_detection_gui.exe`

### 1.2 Folder Setup

```
factory_data/
├── unmasked/          ← Original images (connector top view)
│   ├── 01.jpg
│   ├── 02.jpg
│   └── ...
└── masked/            ← Copy for masking (initially same as unmasked)
    ├── 01.jpg
    ├── 02.jpg
    └── ...
```

- **Recommended**: Use at least 10 image pairs
- **File names**: unmasked and masked must have the same names
- **Formats**: .jpg, .jpeg, .png, .bmp supported

---

## 2. ROI Setup (Important)

### 2.1 Why ROI Matters

| Item | Description |
|------|-------------|
| **Image size** | Real factory: ~5000×4000 px |
| **Pin location** | All 40 pins in top-left quadrant, within ~1000×200 px strip |
| **Camera variance** | ±10% horizontal shift possible between images |
| **ROI benefit** | Crop only pin area → faster training, less memory, better detection |

### 2.2 ROI Setup Steps

1. Run EXE → open **Train** tab
2. **Unmasked images folder**: select `unmasked` folder
3. **Masked images folder**: select `masked` folder
4. **Output folder**: select path (e.g. `pin_models`)
5. Click **Edit ROI**

### 2.3 ROI Setup Tips

| Item | Recommended | Avoid |
|------|-------------|-------|
| **Area** | Rectangle that includes all 40 pins | Whole image or area outside pins |
| **Margin** | Small margin around pins (top, bottom, left, right) | Too tight or too wide |
| **Size** | ~1000×200 px (for pin strip) | 2000×2000 or larger (unnecessary area) |
| **Per image** | Set ROI for each image and save | Setting ROI for one image only |

### 2.4 ROI Modes

| Mode | Action | Description |
|------|--------|-------------|
| **Rectangle** | Drag on left panel | Set ROI (top-left → bottom-right) |
| **Split ROI** | Split ROI button | Separate upper 20 pins / lower 20 pins (different lighting/angle) |
| **Clear ROI** | Clear ROI button | Reset ROI for current image |

**Left panel (Unmasked)**:
- Drag to draw ROI rectangle
- ROI is used as crop region during training
- Use **Prev/Next** or **Left/Right** keys to move between images
- **Set ROI for each image and save**

### 2.5 Handling Camera Variance

- Real factory images may have pin position shift of ±10% between shots
- **Set ROI separately for each image**
- Click **Save ROI map** → saves per-image ROI to `output_folder/roi_map.json`

---

## 3. Masking Tips

### 3.1 Masking Modes

| Mode | Button | Action | Use |
|------|--------|--------|-----|
| **Pin (drag)** | Pin (drag) | Drag on right panel to draw pin rectangle (min 50×50 px) | **Recommended** — YOLO tight bbox |
| **Brush** | Brush | Drag to fill with circle | Large area at once |
| **Erase** | Erase | Click to remove marker at that position | Fix wrong masking |

### 3.2 Masking Quality (Better YOLO Training)

| Item | Recommended | Avoid |
|------|-------------|-------|
| **Tight bbox** | Draw rectangle tightly around each pin (touching pin edge) | Much larger rectangle than pin |
| **Min size** | At least 50×50 px | Smaller than 50×50 (may be ignored by YOLO) |
| **Pin count** | 20 upper + 20 lower = 40 pins | Missing or duplicate pins |
| **Zoom** | Zoom 200% or more with mouse wheel for precise drawing | Drawing roughly at low zoom |

### 3.3 Masking Steps

1. Open ROI Editor with **Edit ROI**
2. In **Rectangle** mode, drag ROI on left panel
3. Switch to **Pin (drag)** mode
4. On **right panel (Masked)**, drag rectangle for each pin
5. Use **Erase** mode to remove wrong markers
6. Click **Save ROI map** → saves `roi_map.json`
7. Masked images are saved to `masked/` folder

### 3.4 Split ROI (Upper/Lower)

If upper 20 pins and lower 20 pins have different lighting or angle, use **Split ROI**:

1. Click **Split ROI**
2. Drag **upper area** on left panel → cyan
3. Drag **lower area** on left panel → orange
4. In **Pin (drag)** mode, draw rectangles for pins in each area
5. Click **Save ROI map**

---

## 4. Test Flow

### 4.1 Overall Flow

```
1. Download and extract EXE
2. Prepare unmasked / masked folders
3. Edit ROI → set ROI for each image → Save ROI map
4. Pin (drag) mode → mask 40 pins → Save ROI map
5. Start training (Epochs: 100~200 recommended)
6. Inference tab → select best.pt → run inference test
```

### 4.2 Training Parameters

| Item | Recommended |
|------|-------------|
| **Epochs** | 100~200 (50~100 if many images) |
| **imgsz** | Apply suggested or default (EXE cap: 1920px) |
| **Data count** | At least 10 pairs |

### 4.3 Inference Test

- After training, open **Inference** tab
- Model: `pin_models/pin_run/weights/best.pt`
- Select test image → **Run** → check pin detection result

---

## 5. Checklist

| Step | Done |
|------|------|
| Unmasked folder has original images | ☐ |
| Masked folder has copy of unmasked | ☐ |
| Edit ROI → set ROI **for each image** | ☐ |
| Pin (drag) mode → mask 40 pins (min 50×50 px) | ☐ |
| Save ROI map | ☐ |
| Masked images saved | ☐ |
| Start training | ☐ |
| Inference with best.pt | ☐ |

---

## 6. Reference

- **roi_map.json**: format `{ "01": [x1,y1,x2,y2], ... }`
- **Image size**: Use ROI if image is 2000px or larger
- **EXE imgsz cap**: 1920px (prevents out-of-memory)
- **Red markers**: YOLO training target (RED priority)
