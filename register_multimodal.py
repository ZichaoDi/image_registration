import os
import h5py
import cv2
import numpy as np
from reg_kmeans import register_images
import matplotlib.pyplot as plt
plt.ion()

from hotspot_filter import remove_hotspots_auto, detect_hotspots
from skimage.metrics import normalized_mutual_information


def load_image(path, dataset_key=None, channel=None, as_gray=True):
    """
    Load either:
      • A single channel from an HDF5 dataset (shape C×H×W), or
      • A 2D image file (PNG/JPG/TIFF/etc).

    Returns:
        img (np.ndarray): 2D image array.
        element (str|None): channel name (HDF5 only) or None.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in ('.h5', '.hdf5'):
        if dataset_key is None or channel is None:
            raise ValueError("HDF5 inputs require dataset_key AND channel")
        with h5py.File(path, 'r') as f:
            arr = f[dataset_key][...]              # (C,H,W) or (C,H,W,3)
            names = f["MAPS/channel_names"][...]   # list of bytes
        channel_names = [n.decode() for n in names]
        img = arr[channel]
        element = channel_names[channel]

        if as_gray and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    else:
        # image file
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image at {path!r}")
        element = None

        if as_gray and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, element
def preprocess_with_canny(img,
                          low_threshold=0,
                          high_threshold=150,
                          gaussian_blur_ksize=(5,5),
                          gaussian_blur_sigma=1.0):
    """
    Preprocess image by:
      1) Normalizing to uint8
      2) Gaussian‐blurring to reduce noise
      3) Running Canny edge detector
      4) (Optionally) Dilating edges to thicken

    Returns:
        edge_img (np.ndarray): binary edge map (0 or 255)
    """
    # 1) normalize to [0,255] uint8
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)

    # 2) smooth
    img_blur = cv2.GaussianBlur(img, gaussian_blur_ksize, gaussian_blur_sigma)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    mag = np.hypot(gx, gy)
    med = np.median(mag)
    low_threshold  = 1.66 * med
    high_threshold = 3.33 * med

    # 3) Canny edges
    edges = cv2.Canny(img_blur, low_threshold, high_threshold, L2gradient=True)

    # 4) (optional) thicken edges by dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges



def preprocess(img):
    """Normalize → remove hotspots → equalize → blur."""
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)
    mask = detect_hotspots(img, sigma_factor=3)
    img = remove_hotspots_auto(img, mask, method="inpaint")
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def draw_matches(img1, kp1, img2, kp2, matches, element=None, top_n=20):
    """Draw top_n matches between img1 & img2."""
    img_m = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches[:top_n], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(12, 6))
    plt.imshow(img_m, cmap='gray')
    title = f"{element}: Top {top_n} Matches" if element else f"Top {top_n} Matches"
    plt.title(title)
    plt.axis('off')
    plt.show()


def center_crop_2d(A: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    H, W = A.shape[:2]
    if crop_h > H or crop_w > W:
        raise ValueError("crop size must be ≤ array size")
    start_i = (H - crop_h) // 2
    start_j = (W - crop_w) // 2
    return A[start_i:start_i + crop_h,
             start_j:start_j + crop_w]
def register_cropped_region(
    img_path_ref, ref_key,
    img_path_tgt, tgt_key,
    channel,
    draw_n_matches=20
):
    """
    Align a small target patch (tgt) back into the large reference (ref).
    Returns: aligned, ref, tgt, transform M, error_metrics
    """
    # ── Load & preprocess ──────────────────────────────────────────────────────
    ref_img, ref_element = load_image(img_path_ref, ref_key, channel)
    ref = preprocess(ref_img)
    # ref = ref[150:350, 200:500]

    tgt_img, tgt_element = load_image(img_path_tgt, tgt_key, channel)
    tgt_img = cv2.flip(tgt_img,0)
    tgt = preprocess(tgt_img)

    aligned, M, inliers = register_images(
            ref, tgt,
            use_kmeans=True,
            n_clusters=50,
            warp_mode="affine"
            )
    # visualize
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    for ax,im,title in zip(axes,[ref,tgt,aligned],
                           ["Reference","Target","Aligned"]):
        ax.imshow(im, cmap='gray')
        ax.set_title(title); ax.axis('off')
    plt.show()

    # metrics
    mask_ov = aligned>0
    diff    = (aligned.astype(float)-ref.astype(float))*mask_ov
    rmse    = np.sqrt(np.mean(diff[mask_ov]**2))
    mi      = normalized_mutual_information(aligned*mask_ov,
                                            ref*mask_ov)
    err = np.array([rmse, mi])
    print(f"RMSE={rmse:.2f}, MI={mi:.3f}, inliers={inliers.sum()}/{len(inliers)}")

    return aligned, ref, tgt, ref_img, tgt_img, M, err


# — example usage —
if __name__ == "__main__":
    h5_a = "../2017_alloy_data/bnp_fly0009.h5"
    png_b = "../2017_alloy_data/ROI-3/2018011908-5kx-PreSI-2kx2k-20kcps-12hrs-HAADF.png"
    # h5_a = "../bnp_fly0002.h5"
    # h5_b = "../bnp_fly0003.h5"
    channels = [20] #[6,7, 8, 9, 20] # [2, 3, 5, 6, 8, 22] #np.arange(6,9) #
    # Pre-allocate an array of shape (n_channels, error_dim)
    error_all = np.zeros((len(channels), 2), dtype=float)
    # layout: 2 rows × ceil(n/2) columns (adjust as you like)
    n = len(channels)
    cols = len(channels)  # for 6 channels, 2×3 grid; tweak for your n
    rows = 2 #int(np.ceil(n/cols))+1
    fig = plt.figure(figsize=(4*n, 8))
    gs  = fig.add_gridspec(2, n, height_ratios=[1,1.2], hspace=0.3)


    for i, ch in enumerate(channels):
        row = i // cols
        col = i %  cols
        ax  = fig.add_subplot(gs[row, col])
        # 1) register
        try:
            # aligned, ref_img, tgt_img, ref_img0, tgt_img0, M, error = register_cropped_region(
            #   h5_a, "MAPS/XRF_fits",
            #   h5_b, "MAPS/XRF_fits",
            #   channel=ch
            # )
            aligned, ref_img, tgt_img, ref_img0, tgt_img0, M, error = register_cropped_region(
              h5_a, "MAPS/XRF_fits",
              png_b, None,
              channel=ch
            )
            error_all[i] = error
        except RuntimeError as e:
            print(f"Channel {ch}: {e} — skipping")
            continue    # ← valid here inside the loop

        # 2) print metrics
        print(f"Channel {ch} error metrics: {error}")

        # 3) get the human‐readable channel name (if HDF5)
        _, channel_name = load_image(h5_a, "MAPS/XRF_fits", ch)

        ## 4) overlay & show
        # overlay_images(ref_img, aligned, element=channel_name)
        # build the blended overlay
        blended = cv2.addWeighted(
            ref_img.astype(np.float32), 0.5,
            aligned.astype(np.float32), 0.5,
            0
        )
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # plot into subplot
        # ax = fig.add_subplot(gs[0, i])
        ax.imshow(blended, cmap="gray")
        ax.axis("off")
        ax.set_title(channel_name)


    A = error_all
    # 1) Compute per‐column minima and maxima
    col_min = A.min(axis=0)   # shape (6,)
    col_max = A.max(axis=0)   # shape (6,)

    # 2) Avoid divide‐by‐zero: if max==min, set denom to 1
    denom = np.where(col_max > col_min, col_max - col_min, 1)

    # 3) Normalize
    A_norm = (A - col_min) / denom
    ax_err = fig.add_subplot(gs[1, :])  # row = n_rows-1, all columns

# plot only the last two columns of errors
    ax_err.plot(np.arange(1,n+1),error_all[:, :], marker='o', linewidth=2)
    ax_err.set_xticks(np.arange(1,n+1))
    ax_err.set_xlabel("Channel index")
    ax_err.set_ylabel("Normalized error")
    ax_err.legend(["md","#matches","rmse","mi"], loc="best")
    ax_err.set_title("Normalized errors (last two metrics)")

    plt.tight_layout()
    plt.show()
