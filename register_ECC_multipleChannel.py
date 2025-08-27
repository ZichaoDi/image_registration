
import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_module import load_image, rotate_image, snr, find_angle
from register_multiscale import preprocess
from skimage.registration import optical_flow_tvl1, optical_flow_ilk, phase_cross_correlation
from skimage.transform import warp, AffineTransform
from skimage.metrics import normalized_mutual_information, structural_similarity
plt.ion()

def normalize(img):
    return (img - np.mean(img)) / (np.std(img) + 1e-6)

# ===================== Paths & channels =====================
##================multimodal dataset
img_path_tgt = "../../2017_alloy_data/ROI-3/2018011908-5kx-PreSI-2kx2k-20kcps-12hrs-HAADF.png"
img_path_ref = "../../2017_alloy_data/bnp_fly0009.h5"
channels = [2, 4, 8,  9, 10, 16, 20 ] #np.arange(4, 22, 1) # [8, 10, 14, 20, 21] #[6,7, 8, 9, 20] # [2, 3, 5, 6, 8, 22] #np.arange(6,9) #
# Multiscale dataset (your current selection)
# img_path_tgt = "../../bnp_fly0004.h5"
# img_path_ref = "../../bnp_fly0002.h5"
# channels = [2, 3, 5, 6, 8, 22]
ref_key = "MAPS/XRF_fits"
tgt_key = "MAPS/XRF_fits"

# ===================== Config =====================
angle_range = np.linspace(30, 60, 15)  # angles to try (deg)
scale_factors = np.linspace(0.1, 1, 50)
rows = 3  # reference, aligned, overlay
cols = len(channels)

# ===================== Storage for best per-channel =====================
best_by_channel = []  # list of dicts, one per channel
rmse_all = []
mi_all = []
ssim_all = []
snr_all = []

# ===================== Figure & gridspec =====================
fig = plt.figure(figsize=(4 * cols, 10))
gs = fig.add_gridspec(rows + 2, cols, height_ratios=[1, 1, 1, 0.5, 0.6], hspace=0.4)

for ci, channel in enumerate(channels):
    print(f"\n=== Processing channel {channel} ===")

    # Load & preprocess
    img_ref0, channel_name = load_image(img_path_ref, ref_key, channel)
    img_ref0 = preprocess(img_ref0)
    img_ref0 = img_ref0[:400,:]
    ##==========================================
    img_tgt_full, _ = load_image(img_path_tgt, tgt_key, channel)
    img_tgt_full = cv2.flip(img_tgt_full,0)
    img_tgt_full = preprocess(img_tgt_full)
    img_tgt_full = img_tgt_full[100:,:]

    # Keep SNR of original ref (or rotated version later — choice is minor)
    snr_all.append(snr(img_ref0))

    # Sweep angles and keep the BEST (lowest RMSE)
    best_rec = {
        "rmse": np.inf,
        "mi": None,
        "ssim": None,
        "angle": None,
        "scale": None,
        "ref_img_rot": None,
        "ref_target": None,
        "template": None,
        "aligned": None,
        "overlay": None,
        "overlay_of": None,
        "rect": None,
        "channel_name": channel_name,
    }

    for angle in angle_range:
        # Rotate reference (following your original convention)
        img_ref = rotate_image(img_ref0, angle)
        img_tgt = img_tgt_full

        # Multi-scale template match to get coarse location/scale
        best_score = -np.inf
        best_scale = None
        best_top_left = None
        best_resized = None

        for scale in scale_factors:
            resized = cv2.resize(img_tgt, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            if resized.shape[0] > img_ref.shape[0] or resized.shape[1] > img_ref.shape[1]:
                continue

            result = cv2.matchTemplate(img_ref, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_scale = scale
                best_top_left = max_loc
                best_resized = resized

        if best_resized is None:
            # Couldn’t fit this scale; skip this angle safely
            continue

        # Crop matched region from rotated reference
        h, w = best_resized.shape
        top_left = best_top_left
        bottom_right = (top_left[0] + w, top_left[1] + h)
        ref_crop = img_ref[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # ECC refinement
        template = ref_crop.astype(np.float32) / 255.0
        target = best_resized.astype(np.float32) / 255.0

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
        try:
            cc, warp_matrix_ecc = cv2.findTransformECC(
                template, target, warp_matrix,
                motionType=cv2.MOTION_EUCLIDEAN,
                criteria=criteria
            )
            aligned = cv2.warpAffine(
                target, warp_matrix_ecc, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
        except cv2.error:
            # ECC sometimes fails; skip this angle
            continue

        # Metrics
        overlap = (aligned > 0)
        if not np.any(overlap):
            continue
        diff = (aligned - template) * overlap
        rmse = np.sqrt(np.mean(diff[overlap] ** 2))
        mi = normalized_mutual_information(aligned * overlap, template * overlap)
        ssim = structural_similarity(
            aligned * overlap, template * overlap, data_range=1,
            full=False, win_size=7, gaussian_weights=True, sigma=1.5
        )

        # Keep the best (lowest RMSE)
        if rmse < best_rec["rmse"]:
            blended = cv2.addWeighted(template, 0.5, aligned, 0.5, 0)
            best_rec.update({
                "rmse": rmse,
                "mi": mi,
                "ssim": ssim,
                "angle": angle,
                "scale": best_scale,
                "ref_img_rot": img_ref,
                "ref_target": target,
                "template": template,
                "aligned": aligned,
                "overlay": blended,
                "rect": (top_left, w, h),
            })

    ###=================== try optical flow
    image1 = best_rec["ref_target"]
    image0 = best_rec["template"]

    v, u = optical_flow_tvl1(image0, image1)

    nr, nc = image0.shape

    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

    image1_warp = warp(image1, np.array([row_coords + v, col_coords + u]), mode='edge')
    blended_of = cv2.addWeighted(image0, 0.5, image1_warp, 0.5, 0)
    # best_rec["overlay"]=blended_of

    fig1, (ax0,ax1)=plt.subplots(1,2,figsize=(10,5))

    ax0.imshow(image1_warp)
    ax0.set_title("registered sequence")
    ax0.set_axis_off()

    ax1.imshow(image0)
    ax1.set_title("reference")
    ax1.set_axis_off()
    fig1.tight_layout()
    ###===================================================
    # Save best for this channel
    best_by_channel.append(best_rec)
    rmse_all.append(best_rec["rmse"])
    mi_all.append(best_rec["mi"])
    ssim_all.append(best_rec["ssim"])

    # ---------- Plot per-channel best results ----------
    # Row 0: rotated reference + matched rectangle
    ax1 = fig.add_subplot(gs[0, ci])
    ax1.imshow(best_rec["ref_img_rot"], cmap='gray')
    tl, w, h = best_rec["rect"]
    ax1.add_patch(plt.Rectangle(tl, w, h, edgecolor='lime', facecolor='none', linewidth=2))
    ax1.set_title(f"{best_rec['channel_name']}\nθ*={best_rec['angle']:.2f}°, scale*={best_rec['scale']:.2f}")
    ax1.axis("off")

    # Row 1: aligned target (best)
    ax2 = fig.add_subplot(gs[1, ci])
    ax2.imshow(best_rec["aligned"], cmap='gray')
    ax2.set_title("Aligned Target (best)")
    ax2.axis("off")

    # Row 2: overlay
    ax3 = fig.add_subplot(gs[2, ci])
    ax3.imshow(best_rec["overlay"], cmap='gray')
    ax3.set_title("Overlay")
    ax3.axis("off")

# ---------- Bottom rows: SNR + normalized errors per channel ----------
channels_idx = np.arange(1, cols + 1)

# SNR
ax_snr = fig.add_subplot(gs[3, :])
ax_snr.plot(channels_idx, snr_all, marker='o', linewidth=2)
ax_snr.set_xticks(channels_idx)
ax_snr.set_ylabel("SNR")
ax_snr.set_title("Per-channel SNR (reference)")

# Normalize errors across channels
A = np.column_stack([rmse_all, mi_all, ssim_all])  # shape (n_channels, 3)
col_min = A.min(axis=0)
col_max = A.max(axis=0)
denom = np.where(col_max > col_min, col_max - col_min, 1)
A_norm = (A - col_min) / denom

ax_err = fig.add_subplot(gs[4, :])
ax_err.plot(channels_idx, A_norm[:, 0], marker='o', linewidth=2, label="rmse (min better)")
ax_err.plot(channels_idx, A_norm[:, 1], marker='o', linewidth=2, label="mi")
ax_err.plot(channels_idx, A_norm[:, 2], marker='o', linewidth=2, label="ssim")
ax_err.set_xticks(channels_idx)
ax_err.set_xlabel("Channel index (as ordered)")
ax_err.set_ylabel("Normalized metric")
ax_err.legend(loc="best")
ax_err.set_title("Normalized metrics (best angle per channel)")

plt.tight_layout()
plt.show()
