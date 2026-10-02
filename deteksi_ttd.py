import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob, os, csv


ROI_REL = (0.14, 0.70, 0.40, 0.83)

NORM_W         = 2000   
GLOBAL_T       = 127    
MIN_COMP_AREA  = 40     
RATIO_THRESH   = 0.01   
LARGEST_THRESH = 150    
CONTRAST_GUARD = 0.80  
OUT_DIR        = "hasil"


os.makedirs(OUT_DIR, exist_ok=True)


def load_upright(path):
    """Baca citra, putar agar tegak/landscape, normalisasi ukuran."""
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if h > w:                               
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    s = NORM_W / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def roi_box(img):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = ROI_REL
    return int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)


def make_absent(img):
    """Simulasi 'tanpa tanda tangan': isi ROI dengan warna kertas (median)."""
    out = img.copy()
    x1, y1, x2, y2 = roi_box(out)
    roi = out[y1:y2, x1:x2]
    paper = np.median(roi.reshape(-1, 3), axis=0).astype(np.uint8)
    out[y1:y2, x1:x2] = paper
    return out


def analyze(img):
    """Pipeline: crop -> grayscale -> threshold -> morfologi -> fitur -> keputusan."""
    x1, y1, x2, y2 = roi_box(img)
    roi_bgr = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th_global = cv2.threshold(gray_blur, GLOBAL_T, 255, cv2.THRESH_BINARY_INV)
    otsu_val, th_otsu = cv2.threshold(gray_blur, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th_adapt = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 12)
    k = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(th_otsu, cv2.MORPH_CLOSE, k, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, k, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(morph, connectivity=8)
    clean = np.zeros_like(morph)
    areas = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMP_AREA:
            clean[labels == i] = 255
            areas.append(stats[i, cv2.CC_STAT_AREA])
    fg = int(cv2.countNonZero(clean))
    ratio = fg / clean.size
    largest = max(areas) if areas else 0
    n_comp = len(areas)
    bg_median = float(np.median(gray_blur))
    contrast_ok = otsu_val < CONTRAST_GUARD * bg_median
    present = contrast_ok and ratio >= RATIO_THRESH and largest >= LARGEST_THRESH
    status = "SIGNATURE PRESENT" if present else "SIGNATURE ABSENT"
    return dict(status=status, fg=fg, ratio=ratio, largest=largest, n_comp=n_comp,
                otsu=float(otsu_val), contrast_ok=contrast_ok,
                roi_bgr=roi_bgr, gray=gray_blur, th_global=th_global,
                th_otsu=th_otsu, th_adapt=th_adapt, clean=clean,
                box=(x1, y1, x2, y2))


def visualize(name, img, r, save_path):
    disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    x1, y1, x2, y2 = r["box"]
    cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 6)

    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    ax = ax.ravel()
    panels = [
        (disp, None, "1. Lokasi ROI"),
        (cv2.cvtColor(r["roi_bgr"], cv2.COLOR_BGR2RGB), None, "2. Crop ROI"),
        (r["gray"], "gray", "3. Grayscale"),
        (r["th_global"], "gray", f"4. Global T={GLOBAL_T}\n{cv2.countNonZero(r['th_global'])} px"),
        (r["th_otsu"], "gray", f"5. Otsu T={r['otsu']:.0f}\n{cv2.countNonZero(r['th_otsu'])} px"),
        (r["th_adapt"], "gray", f"6. Adaptive\n{cv2.countNonZero(r['th_adapt'])} px"),
        (r["clean"], "gray", f"7. Morfologi + filter\n{r['fg']} px"),
    ]
    for a, (im, cmap, t) in zip(ax, panels):
        a.imshow(im, cmap=cmap); a.set_title(t, fontsize=10); a.axis("off")
    ax[7].axis("off")
    ax[7].text(0.0, 0.5,
               f"{name}\n\nForeground : {r['fg']} px\nRasio      : {r['ratio']*100:.2f}%\n"
               f"Komp. terbesar: {r['largest']} px\nJumlah komp.: {r['n_comp']}\n"
               f"Kontras OK : {r['contrast_ok']}\n\nHASIL: {r['status']}",
               fontsize=11, family="monospace", va="center")
    plt.tight_layout()
    plt.savefig(save_path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    matplotlib.use("Agg")   
    files = sorted(glob.glob("*.jpg"))
    if not files:
        raise SystemExit("Tidak ada file .jpg di folder ini.")
    print(f"Total gambar ditemukan: {len(files)}\n")

    rows, correct, total = [], 0, 0
    for f in files:
        img = load_upright(f)
        if img is None:
            print(f"Gagal membaca {f}"); continue
        base = os.path.splitext(f)[0]
        for tag, im, truth in [("ASLI", img, "SIGNATURE PRESENT"),
                               ("SIM_ABSENT", make_absent(img), "SIGNATURE ABSENT")]:
            r = analyze(im)
            ok = r["status"] == truth
            correct += ok; total += 1
            rows.append([base, tag, r["fg"], f"{r['ratio']*100:.2f}", r["largest"],
                         r["n_comp"], f"{r['otsu']:.0f}", r["status"], truth, "OK" if ok else "SALAH"])
            print(f"{base[:32]:32s} {tag:10s} px={r['fg']:6d} rasio={r['ratio']*100:5.2f}% "
                  f"-> {r['status']:18s} [{'OK' if ok else 'SALAH'}]")
            visualize(f"{base} ({tag})", im, r, os.path.join(OUT_DIR, f"{base}_{tag}.png"))

    with open(os.path.join(OUT_DIR, "ringkasan.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["citra", "kondisi", "fg_px", "rasio_%", "komp_terbesar", "jml_komp",
                    "otsu_T", "prediksi", "sebenarnya", "cek"])
        w.writerows(rows)
    print(f"\nAkurasi: {correct}/{total} = {correct/total*100:.1f}%")