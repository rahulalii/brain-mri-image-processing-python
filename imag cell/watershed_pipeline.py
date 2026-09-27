"""
=============================================================================
  Full Watershed Segmentation Pipeline
  =====================================
  Stages:
    1. Image Acquisition
    2. Pre-Processing
    3. Morphological Cleaning
    4. Watershed Segmentation – Marker Generation
    5. Watershed Segmentation – Object Separation
=============================================================================
"""

import os
import sys
import urllib.request
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # use interactive backend; change to "Agg" if headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Optional: try importing heavy libraries and guide the user if absent ────
try:
    import cv2
except ImportError:
    sys.exit("[ERROR] OpenCV not found. Install with:  pip install opencv-python")

try:
    from scipy import ndimage as ndi
except ImportError:
    sys.exit("[ERROR] SciPy not found. Install with:  pip install scipy")

try:
    from skimage import (
        color, filters, morphology, measure,
        segmentation, feature, exposure
    )
    from skimage.color import label2rgb
except ImportError:
    sys.exit("[ERROR] scikit-image not found. Install with:  pip install scikit-image")

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  (edit here to customise the run)
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # -------------------------------------------------------------------
    # Image source
    #   • Set IMAGE_PATH to a local file path (jpg / png / tif …)
    #     to skip the dialog and load that file directly.
    #   • Leave as None → a pop-up will ask at run-time:
    #       Yes → opens a file-browser so you can pick YOUR image
    #       No  → downloads/uses the built-in demo image
    # -------------------------------------------------------------------
    "IMAGE_PATH": None,          # e.g. r"C:\images\cells.png"

    # -------------------------------------------------------------------
    # Gaussian blur kernel size (must be odd)
    # -------------------------------------------------------------------
    "BLUR_KERNEL": 7,

    # -------------------------------------------------------------------
    # Threshold method: "otsu" | "adaptive" | "manual"
    # -------------------------------------------------------------------
    "THRESHOLD_METHOD": "otsu",
    "MANUAL_THRESH": 128,        # used only when method == "manual"

    # -------------------------------------------------------------------
    # Morphological structuring element radius (disk)
    # -------------------------------------------------------------------
    "MORPH_RADIUS": 5,

    # -------------------------------------------------------------------
    # Minimum object size (pixels) to keep after cleaning
    # -------------------------------------------------------------------
    "MIN_OBJECT_SIZE": 300,

    # -------------------------------------------------------------------
    # Distance-transform peak threshold (fraction of max distance)
    # -------------------------------------------------------------------
    "PEAK_THRESHOLD": 0.4,

    # -------------------------------------------------------------------
    # Display intermediate stages?
    # -------------------------------------------------------------------
    "SHOW_PLOTS": True,

    # -------------------------------------------------------------------
    # Save results to disk?
    # -------------------------------------------------------------------
    "SAVE_RESULTS": True,
    "OUTPUT_DIR": r"c:\Users\rahul\Desktop\New folder\output",
}


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1 – IMAGE ACQUISITION
# ─────────────────────────────────────────────────────────────────────────────
def _open_file_dialog() -> str:
    """
    Open a native Windows file-chooser dialog and return the chosen path.
    Returns an empty string if the user cancels.
    """
    root = tk.Tk()
    root.withdraw()                        # hide the empty root window
    root.attributes("-topmost", True)      # bring dialog to the front
    path = filedialog.askopenfilename(
        title="Select an Image for Watershed Segmentation",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
            ("JPEG",        "*.jpg *.jpeg"),
            ("PNG",         "*.png"),
            ("TIFF",        "*.tif *.tiff"),
            ("BMP",         "*.bmp"),
            ("All files",   "*.*"),
        ],
    )
    root.destroy()
    return path


def stage1_acquire(cfg: dict) -> np.ndarray:
    """
    Load a colour image as a NumPy array (BGR → RGB).

    Priority order:
      1. CONFIG["IMAGE_PATH"]  – if set and file exists, use it directly.
      2. Interactive file-chooser dialog – user picks their own image.
      3. Download a demo image from Wikipedia.
      4. Generate a synthetic image (offline fallback).
    """
    print("\n" + "="*60)
    print("  STAGE 1 — Image Acquisition")
    print("="*60)

    path = cfg["IMAGE_PATH"]

    # ── 1. Use path from CONFIG if valid ───────────────────────────────────
    if path and os.path.isfile(path):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            sys.exit(f"[ERROR] Could not read image from: {path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        print(f"  ✓ Loaded from CONFIG path: {path}")
        print(f"  ✓ Image shape : {img.shape}  |  dtype : {img.dtype}")
        return img

    # ── 2. Ask user: browse for own image or use demo? ─────────────────────
    print("  ℹ  No IMAGE_PATH set – opening image chooser …")
    try:
        # Show a small yes/no dialog first
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        use_own = messagebox.askyesno(
            "Image Source",
            "Do you want to choose YOUR OWN image?\n\n"
            "  Yes → Opens a file browser\n"
            "  No  → Uses a built-in demo image",
            icon="question",
        )
        root.destroy()
    except Exception:
        use_own = False   # no display available (headless)

    if use_own:
        chosen = _open_file_dialog()
        if chosen and os.path.isfile(chosen):
            img_bgr = cv2.imread(chosen)
            if img_bgr is None:
                print(f"  ✗ Could not read chosen file: {chosen}")
                print("  ↩ Falling back to demo image …")
            else:
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                # Save the chosen path back into config for future reference
                cfg["IMAGE_PATH"] = chosen
                print(f"  ✓ Loaded your image : {chosen}")
                print(f"  ✓ Image shape : {img.shape}  |  dtype : {img.dtype}")
                return img
        else:
            print("  ✗ No file selected – falling back to demo image …")

    # ── 3. Download demo image ─────────────────────────────────────────────
    demo_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "a/a7/Camponotus_flavomarginatus_ant.jpg/"
        "800px-Camponotus_flavomarginatus_ant.jpg"
    )
    tmp_path = os.path.join(
        cfg["OUTPUT_DIR"] if cfg["SAVE_RESULTS"] else os.getcwd(),
        "_demo_image.jpg"
    )
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    print("  ↓ Downloading demo image …")
    try:
        urllib.request.urlretrieve(demo_url, tmp_path)
        img_bgr = cv2.imread(tmp_path)
        if img_bgr is None:
            raise ValueError("Downloaded file could not be decoded.")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        print(f"  ✓ Demo image saved to: {tmp_path}")
    except Exception as exc:
        # ── 4. Synthetic fallback ──────────────────────────────────────────
        print(f"  ✗ Download failed ({exc}). Generating synthetic image …")
        img = _synthetic_image()

    print(f"  ✓ Image shape : {img.shape}  |  dtype : {img.dtype}")
    return img


def _synthetic_image(size: int = 512) -> np.ndarray:
    """Create a synthetic RGB image with overlapping bright blobs on dark bg."""
    rng = np.random.default_rng(42)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    centres = [
        (130, 130), (260, 200), (380, 130),
        (180, 370), (320, 360), (420, 300),
        (90, 290),  (460, 430), (230, 460),
    ]
    for (cy, cx) in centres:
        r = rng.integers(40, 75)
        colour = rng.integers(180, 255, size=3).tolist()
        cv2.circle(canvas, (cx, cy), r, colour, -1)
    # add slight overlap between a pair of blobs
    cv2.circle(canvas, (300, 200), 60, (220, 200, 180), -1)
    canvas = cv2.GaussianBlur(canvas, (9, 9), 0)
    noise  = rng.integers(0, 20, canvas.shape, dtype=np.uint8)
    canvas = cv2.add(canvas, noise)
    print("  ✓ Synthetic image generated.")
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2 – PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def stage2_preprocess(img: np.ndarray, cfg: dict) -> dict:
    """
    Convert to greyscale, normalise, enhance contrast, blur,
    and threshold to produce a binary mask.

    Returns a dict of intermediate arrays for display / debugging.
    """
    print("\n" + "="*60)
    print("  STAGE 2 — Pre-Processing")
    print("="*60)

    # 2-a  Greyscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("  ✓ Greyscale conversion")

    # 2-b  Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    print("  ✓ CLAHE contrast enhancement")

    # 2-c  Gaussian blur / noise reduction
    k = cfg["BLUR_KERNEL"]
    if k % 2 == 0:
        k += 1               # kernel must be odd
    blurred = cv2.GaussianBlur(enhanced, (k, k), 0)
    print(f"  ✓ Gaussian blur  (kernel={k}×{k})")

    # 2-d  Thresholding
    method = cfg["THRESHOLD_METHOD"]
    if method == "otsu":
        _, binary = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print(f"  ✓ Otsu thresholding")
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        print("  ✓ Adaptive (Gaussian) thresholding")
    else:
        t = cfg["MANUAL_THRESH"]
        _, binary = cv2.threshold(blurred, t, 255, cv2.THRESH_BINARY)
        print(f"  ✓ Manual thresholding  (T={t})")

    # Ensure foreground is WHITE (bright objects on dark bg)
    if binary.mean() > 127:
        binary = cv2.bitwise_not(binary)
        print("  ✓ Inverted binary (foreground → white)")

    return {
        "gray":     gray,
        "enhanced": enhanced,
        "blurred":  blurred,
        "binary":   binary,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 3 – MORPHOLOGICAL CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def stage3_morph_clean(binary: np.ndarray, cfg: dict) -> dict:
    """
    Remove noise and fill holes in the binary mask using morphological ops.
    """
    print("\n" + "="*60)
    print("  STAGE 3 — Morphological Cleaning")
    print("="*60)

    r    = cfg["MORPH_RADIUS"]
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))

    # 3-a  Opening → removes small bright noise
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  disk, iterations=2)
    print(f"  ✓ Opening  (r={r}, iters=2)  — small noise removed")

    # 3-b  Closing → fills small dark holes inside objects
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, disk, iterations=2)
    print(f"  ✓ Closing  (r={r}, iters=2)  — holes filled")

    # 3-c  Remove small connected components (scikit-image)
    bool_mask  = closed.astype(bool)
    cleaned_sk = morphology.remove_small_objects(
        bool_mask, min_size=cfg["MIN_OBJECT_SIZE"]
    )
    # Fill remaining holes
    cleaned_sk = ndi.binary_fill_holes(cleaned_sk)
    cleaned    = (cleaned_sk * 255).astype(np.uint8)
    print(f"  ✓ Small objects removed  (min_size={cfg['MIN_OBJECT_SIZE']} px)")
    print(f"  ✓ Remaining holes filled")

    return {
        "opened":  opened,
        "closed":  closed,
        "cleaned": cleaned,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 4 – WATERSHED MARKER GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def stage4_markers(cleaned: np.ndarray, cfg: dict) -> dict:
    """
    1. Distance transform on the cleaned binary mask.
    2. Detect local maxima → sure-foreground markers.
    3. Dilate cleaned mask → sure-background markers.
    4. Unknown region = background − foreground.
    """
    print("\n" + "="*60)
    print("  STAGE 4 — Watershed: Marker Generation")
    print("="*60)

    bool_mask = cleaned.astype(bool)

    # 4-a  Euclidean distance transform
    dist = ndi.distance_transform_edt(bool_mask)
    print("  ✓ Distance transform computed")

    # 4-b  Local maxima (sure-foreground seeds)
    peak_thresh = cfg["PEAK_THRESHOLD"] * dist.max()
    local_maxi  = feature.peak_local_max(
        dist,
        min_distance  = int(cfg["MORPH_RADIUS"] * 1.5),
        threshold_abs = peak_thresh,
        exclude_border= False,
        labels        = bool_mask,
    )
    # Build a boolean image of peaks
    markers_bool = np.zeros_like(bool_mask)
    markers_bool[tuple(local_maxi.T)] = True
    markers_bool = morphology.dilation(markers_bool, morphology.disk(3))
    print(f"  ✓ Local maxima detected: {len(local_maxi)} peak(s)")

    # 4-c  Label the seeds
    labeled_seeds, n_seeds = ndi.label(markers_bool)
    print(f"  ✓ Labeled {n_seeds} foreground marker(s)")

    # 4-d  Sure-background (dilated mask)
    sure_bg_disk = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2*cfg["MORPH_RADIUS"]+1, 2*cfg["MORPH_RADIUS"]+1)
    )
    sure_bg = cv2.dilate(cleaned, sure_bg_disk, iterations=3)
    print("  ✓ Sure-background region computed")

    # 4-e  Unknown region
    sure_fg  = (labeled_seeds > 0).astype(np.uint8) * 255
    unknown  = cv2.subtract(sure_bg, sure_fg)
    print("  ✓ Unknown region derived")

    return {
        "dist":          dist,
        "local_maxi":    local_maxi,
        "labeled_seeds": labeled_seeds,
        "sure_fg":       sure_fg,
        "sure_bg":       sure_bg,
        "unknown":       unknown,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 5 – WATERSHED OBJECT SEPARATION
# ─────────────────────────────────────────────────────────────────────────────
def stage5_watershed(img: np.ndarray, cleaned: np.ndarray,
                     marker_data: dict) -> dict:
    """
    Run OpenCV watershed, extract object boundaries, label each region,
    and produce annotated output images.
    """
    print("\n" + "="*60)
    print("  STAGE 5 — Watershed: Object Separation")
    print("="*60)

    labeled_seeds = marker_data["labeled_seeds"]

    # 5-a  Add background label (0 → unknown, will be filled by watershed)
    markers_ws = labeled_seeds.copy().astype(np.int32)
    # Mark unknown pixels as 0 (watershed will decide them)
    markers_ws[marker_data["unknown"] == 255] = 0

    # 5-b  Run OpenCV watershed (needs 8-bit 3-channel input)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.watershed(img_bgr, markers_ws)
    print("  ✓ Watershed algorithm completed")

    # 5-c  Extract boundaries (label == -1 after watershed)
    boundary_mask = (markers_ws == -1)
    print(f"  ✓ Boundary pixels detected")

    # 5-d  Build coloured label overlay
    n_objects = markers_ws.max()
    print(f"  ✓ Objects detected: {n_objects}")

    label_rgb = label2rgb(
        markers_ws,
        image      = img,
        bg_label   = -1,
        alpha      = 0.4,
        kind       = "overlay",
    )

    # 5-e  Draw boundaries on the original image
    img_boundaries = img.copy()
    img_boundaries[boundary_mask] = [255, 0, 0]   # red boundaries

    # 5-f  Bounding boxes & properties
    props = measure.regionprops(markers_ws)
    print(f"  ✓ Region properties computed for {len(props)} region(s)")

    # Draw bounding boxes
    annotated = img.copy()
    annotated[boundary_mask] = [255, 0, 0]
    for p in props:
        if p.label <= 0:
            continue
        r0, c0, r1, c1 = p.bbox
        cv2.rectangle(
            annotated,
            (c0, r0), (c1, r1),
            (0, 255, 0), 2
        )
        cv2.putText(
            annotated,
            str(p.label),
            (c0 + 4, r0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 0), 1, cv2.LINE_AA
        )

    return {
        "markers_ws":    markers_ws,
        "boundary_mask": boundary_mask,
        "label_rgb":     label_rgb,
        "img_boundaries":img_boundaries,
        "annotated":     annotated,
        "n_objects":     n_objects,
        "props":         props,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def visualise_all(img, pre, morph, marker_data, ws_data, cfg):
    """Display all pipeline stages in a grid of subplots."""

    stages = [
        # (title,  image/array,  cmap)
        ("1 – Original Image",        img,                          None),
        ("2a – Greyscale",            pre["gray"],                  "gray"),
        ("2b – CLAHE Enhanced",       pre["enhanced"],              "gray"),
        ("2c – Gaussian Blurred",     pre["blurred"],               "gray"),
        ("2d – Binary Threshold",     pre["binary"],                "gray"),
        ("3a – Morphological Open",   morph["opened"],              "gray"),
        ("3b – Morphological Close",  morph["closed"],              "gray"),
        ("3c – Cleaned Mask",         morph["cleaned"],             "gray"),
        ("4a – Distance Transform",   marker_data["dist"],          "jet"),
        ("4b – Sure Foreground",      marker_data["sure_fg"],       "gray"),
        ("4c – Sure Background",      marker_data["sure_bg"],       "gray"),
        ("4d – Unknown Region",       marker_data["unknown"],       "gray"),
        ("5a – Watershed Labels",     ws_data["label_rgb"],         None),
        ("5b – Object Boundaries",    ws_data["img_boundaries"],    None),
        ("5c – Annotated Result",     ws_data["annotated"],         None),
    ]

    cols = 5
    rows = int(np.ceil(len(stages) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten()

    for ax, (title, arr, cmap) in zip(axes, stages):
        if cmap:
            ax.imshow(arr, cmap=cmap)
        else:
            ax.imshow(arr)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    # hide unused axes
    for ax in axes[len(stages):]:
        ax.axis("off")

    fig.suptitle(
        "Watershed Segmentation Pipeline",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    if cfg["SAVE_RESULTS"]:
        out = os.path.join(cfg["OUTPUT_DIR"], "pipeline_overview.png")
        os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  ✓ Pipeline overview saved → {out}")

    if cfg["SHOW_PLOTS"]:
        plt.show()
    else:
        plt.close()


def visualise_final(img, ws_data, cfg):
    """Large side-by-side comparison: original vs annotated result."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.imshow(img)
    ax1.set_title("Original Image", fontsize=12, fontweight="bold")
    ax1.axis("off")

    ax2.imshow(ws_data["annotated"])
    ax2.set_title(
        f"Segmentation Result  ({ws_data['n_objects']} objects)",
        fontsize=12, fontweight="bold"
    )
    ax2.axis("off")

    # Legend patch
    patches = [
        mpatches.Patch(color=(1, 0, 0), label="Object boundaries"),
        mpatches.Patch(color=(0, 1, 0), label="Bounding boxes"),
        mpatches.Patch(color=(1, 1, 0), label="Object labels"),
    ]
    ax2.legend(handles=patches, loc="lower right", fontsize=8)

    plt.tight_layout()

    if cfg["SAVE_RESULTS"]:
        out = os.path.join(cfg["OUTPUT_DIR"], "final_result.png")
        os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  ✓ Final result saved    → {out}")

    if cfg["SHOW_PLOTS"]:
        plt.show()
    else:
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_report(ws_data):
    props      = ws_data["props"]
    n_objects  = ws_data["n_objects"]

    print("\n" + "="*60)
    print("  SEGMENTATION REPORT")
    print("="*60)
    print(f"  Total objects detected : {n_objects}")
    print()
    print(f"  {'ID':>4}  {'Area (px)':>10}  {'Centroid (r,c)':>18}  "
          f"{'Perimeter':>10}  {'Equiv. Diam.':>12}")
    print("  " + "-"*60)

    for p in props:
        if p.label <= 0:
            continue
        cy, cx = p.centroid
        print(
            f"  {p.label:>4}  {p.area:>10}  "
            f"({cy:7.1f}, {cx:7.1f})  "
            f"{p.perimeter:>10.1f}  "
            f"{p.equivalent_diameter_area:>12.1f}"
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE INDIVIDUAL STAGE IMAGES
# ─────────────────────────────────────────────────────────────────────────────
def save_stages(img, pre, morph, marker_data, ws_data, cfg):
    if not cfg["SAVE_RESULTS"]:
        return
    out_dir = cfg["OUTPUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    def _save(name, arr, cmap=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(arr, cmap=cmap)
        ax.axis("off")
        path = os.path.join(out_dir, name)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    _save("01_original.png",        img)
    _save("02a_gray.png",           pre["gray"],      "gray")
    _save("02b_enhanced.png",       pre["enhanced"],  "gray")
    _save("02c_blurred.png",        pre["blurred"],   "gray")
    _save("02d_binary.png",         pre["binary"],    "gray")
    _save("03a_opened.png",         morph["opened"],  "gray")
    _save("03b_closed.png",         morph["closed"],  "gray")
    _save("03c_cleaned.png",        morph["cleaned"], "gray")
    _save("04a_distance.png",       marker_data["dist"], "jet")
    _save("04b_sure_fg.png",        marker_data["sure_fg"],  "gray")
    _save("04c_sure_bg.png",        marker_data["sure_bg"],  "gray")
    _save("04d_unknown.png",        marker_data["unknown"],  "gray")
    _save("05a_label_rgb.png",      ws_data["label_rgb"])
    _save("05b_boundaries.png",     ws_data["img_boundaries"])
    _save("05c_annotated.png",      ws_data["annotated"])

    print(f"  ✓ All stage images saved to: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cfg = CONFIG

    # ── Stage 1: Acquisition ───────────────────────────────────────────────
    img = stage1_acquire(cfg)

    # ── Stage 2: Pre-Processing ────────────────────────────────────────────
    pre = stage2_preprocess(img, cfg)

    # ── Stage 3: Morphological Cleaning ───────────────────────────────────
    morph = stage3_morph_clean(pre["binary"], cfg)

    # ── Stage 4: Marker Generation ─────────────────────────────────────────
    marker_data = stage4_markers(morph["cleaned"], cfg)

    # ── Stage 5: Object Separation ─────────────────────────────────────────
    ws_data = stage5_watershed(img, morph["cleaned"], marker_data)

    # ── Report ─────────────────────────────────────────────────────────────
    print_report(ws_data)

    # ── Save individual stages ─────────────────────────────────────────────
    save_stages(img, pre, morph, marker_data, ws_data, cfg)

    # ── Visualise ──────────────────────────────────────────────────────────
    visualise_all(img, pre, morph, marker_data, ws_data, cfg)
    visualise_final(img, ws_data, cfg)

    print("\n  Pipeline complete.\n")


if __name__ == "__main__":
    main()
