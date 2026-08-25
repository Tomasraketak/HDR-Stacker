# 🌞 Astro HDR Stacker — Solar Eclipse & Exposure Fusion Studio

A professional desktop application in Python (PyQt6 + OpenCV) for stacking exposure brackets into a single High Dynamic Range (HDR) image. Designed specifically for demanding astrophotography scenes such as **Total Solar Eclipses**, the solar corona, prominences, as well as classic landscape HDR photography.

---

## 🚀 Key Features

- **🎯 Lightning-fast ROI Crop (300x300 px / Fast Edit Mode)**:
  - Toggle between the full image and a **crop around the Sun** (300x300, 450x450, 600x600 px) with a single click.
  - In ROI mode, stacking and all adjustments happen in real-time (**response under 30 ms**).
  - The **`☀️ Najít`** (Find) button or clicking on the image immediately centers the crop on the eclipse.
  - During export, the full scene is automatically processed and saved in **100% resolution at 16-bit depth**.
- **Automatic EV and Exposure Time Detection**:
  - Automatically extracts shutter speeds, ISO, and aperture from **EXIF** metadata.
  - Intelligent scene brightness analysis: If EXIF is missing, the app analyzes the histogram and sorts the images from the shortest exposure (-EV) to the longest (+EV).
  - Option to define custom EV steps (e.g., **9 images with a 1.0 EV step**).
- **Alignment and Black Moon Disk Detection**:
  - 🌒 **Moon disk detection in the corona**: Finds the circular black moon disk surrounded by coronal light using a geometric circularity filter.
  - 🛠 **Interactive manual alignment frame by frame**:
    - **Edge Difference**, **50% Blend**, and **Flicker** modes for precise alignment.
    - Easily nudge each frame by 0.2px / 1px / 5px using keyboard arrows or on-screen buttons.
    - Ability to exclude specific frames directly from the alignment window.
- **Stacking Algorithms (HDR & Fusion Engines)**:
  - **Mertens Exposure Fusion** (*Recommended for Solar Eclipses*): Seamless Laplacian pyramid fusion with built-in noise suppression.
  - **Debevec 32-bit HDR** with Camera Response Function (CRF) calibration and tonemapping (Reinhard, Drago, Mantiuk). Highly optimized CRF estimation prevents OOM crashes on large 24MP brackets.
  - **Robertson 32-bit HDR**.
- **Noise Reduction & Solar Corona Filter**:
  - 🧹 **Noise Reduction (Grain filter)**: Adaptive bilateral sensor noise smoothing without blurring fine coronal structures.
  - 🌟 **Eclipse Coronal Detail Filter**: Enhances fine magnetic field lines in the inner and outer corona while strictly protecting the dark sky background.
- **Export**:
  - **16-bit TIFF** (ideal for further post-processing in PixInsight, Photoshop, or Lightroom).
  - **PNG**, **100% JPG**, and **32-bit Radiance HDR**.

---

## 🛠 Installation and Usage

### 1. Requirements
- Python 3.10+ (e.g., Python 3.13)
- Required libraries:

```bash
pip install -r requirements.txt
```

### 2. Running the application
```bash
python main.py
```
or
```bash
py main.py
```

---

## 📖 How to use the application

1. **Load images**:
   - Drag & Drop files (e.g., 9 JPG eclipse brackets) directly into the app window, or click **`+ Přidat fotky`** (Add photos).
2. **Lightning-fast editing (Fast ROI)**:
   - Click **`🎯 ROI`** in the top bar and **`☀️ Najít`** (Find) (or just click on the Sun in the image).
   - Stacking and all sliders will respond **instantly without any waiting**.
3. **Full scene view / Export**:
   - Toggle the ROI button off to view the full scene.
   - Click **`💾 Export`** to save the final 16-bit TIFF or JPG.
