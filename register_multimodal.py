import cv2
import numpy as np
import os
import h5py
import cv2
import matplotlib.pyplot as plt
plt.ion()

from hotspot_filter import remove_hotspots_auto, detect_hotspots
from skimage.metrics import normalized_mutual_information
from image_module import snr, load_image, rotate_image, center_crop_2d,find_angle

def preprocess(img):
    """
    1) Normalize → remove hotspots → clip outliers → rescale → equalize → blur.
    """
    # 1) bring to 0–255 uint8
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)

    # 2) remove bright/dark hotspots
    mask = detect_hotspots(img, sigma_factor=3)
    img = remove_hotspots_auto(img, mask, method="blur")
    # import pdb; pdb.set_trace()


    # 3) robust clipping to [mean–2σ, mean+2σ]
    arr = img.astype(np.float32)
    # m, s = np.nanmean(arr), np.nanstd(arr)
    # arr = np.clip(arr, m - 2*s, m + 2*s)

    # 4) rescale back to full 0–255 range
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    img = (arr * 255).astype(np.uint8)

    # 5) histogram equalization
    img = cv2.equalizeHist(img)

    # 6) final smoothing
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



def assign_intensity_centroid_orientation(img: np.ndarray, keypoints, patch_radius=16):
    """
    For each cv2.KeyPoint in keypoints, compute its angle
    as the arctan2 of the intensity‐centroid of a local patch.
    """
    h, w = img.shape
    for kp in keypoints:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        r = int(round(kp.size/2)) if kp.size>0 else patch_radius

        # extract patch, clamped
        x0, x1 = max(x-r,0), min(x+r+1, w)
        y0, y1 = max(y-r,0), min(y+r+1, h)
        patch = img[y0:y1, x0:x1].astype(np.float32)

        # coords relative to center
        ys, xs = np.mgrid[y0:y1, x0:x1]
        u = xs - x
        v = ys - y

        # moments
        m00 = patch.sum() + 1e-6
        m10 = (u * patch).sum()
        m01 = (v * patch).sum()

        # centroid vector and angle
        cx, cy = m10/m00, m01/m00
        angle = (np.degrees(np.arctan2(cy, cx)) + 360) % 360
        kp.angle = angle

    return keypoints


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
    # ref = ref[:400,:]
    # if channel in [17, 18, 19]:
    #     ref = 255-ref
    # ref = cv2.resize(ref, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)

    # ref = rotate_image(ref, angle=45)

    tgt_img, tgt_element = load_image(img_path_tgt, tgt_key, channel)
    tgt_img = cv2.flip(tgt_img,0)
    tgt = preprocess(tgt_img[:,:]) #100
    # import pdb; pdb.set_trace()

    # angle, rot_matrix, aligned = find_angle(tgt, ref[1:1948,1:2048])

    # M = np.load("my_array.npy")
    # H_ref, W_ref = ref.shape
    # tgt = cv2.warpAffine(tgt, M, (W_ref, H_ref))
    ### ── Detect ORB keypoints only ───────────────────────────────────
    ## use a 64×64 patch around each keypoint
    orb = cv2.ORB_create(
      nfeatures=1000,
      patchSize=128,         # ← increase this to cover a larger neighborhood
      edgeThreshold=64,  # must be >= patchSize/2 + 1
      scaleFactor=1.2,
      nlevels=16
    )

    kp_ref = orb.detect(ref, None)
    kp_tgt = orb.detect(tgt, None)
    # # ── Assign orientation by intensity‐centroid ──────────────────
    kp_ref = assign_intensity_centroid_orientation(ref, kp_ref)
    kp_tgt = assign_intensity_centroid_orientation(tgt, kp_tgt)
    kp_ref, des_ref = orb.compute(ref, kp_ref)
    kp_tgt, des_tgt = orb.compute(tgt, kp_tgt)

    if des_ref is None or des_tgt is None:
        raise RuntimeError("No descriptors found in one of the images")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ####### -------use SIFT -------------------------
    # sift = cv2.SIFT_create()
    # kp_ref = sift.detect(ref, None)
    # kp_tgt = sift.detect(tgt, None)
    # # ── Assign orientation by intensity‐centroid ──────────────────
    # kp_ref = assign_intensity_centroid_orientation(ref, kp_ref)
    # kp_tgt = assign_intensity_centroid_orientation(tgt, kp_tgt)
    # kp_ref, des_ref = sift.compute(ref, kp_ref)
    # kp_tgt, des_tgt = sift.compute(tgt, kp_tgt)
    # if des_ref is None or des_tgt is None:
    #   raise RuntimeError("No descriptors found in one of the images")

    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    ############################################################
    # after you’ve computed des_ref, des_tgt and kp_ref, kp_tgt…

    # 1) forward and backward matches
    matches_ab = bf.match(des_ref, des_tgt)   # A→B
    matches_ba = bf.match(des_tgt, des_ref)   # B→A

    # 2) build reverse map from B-queryIdx → A-trainIdx
    ba_map = {m.queryIdx: m.trainIdx for m in matches_ba}

    # 3) mutual consistency test
    good = []
    for m in matches_ab:
        # m.queryIdx indexes into des_ref/kp_ref
        # m.trainIdx indexes into des_tgt/kp_tgt
        if ba_map.get(m.trainIdx, -1) == m.queryIdx:
            good.append(m)

    print(f"{len(good)} mutual matches found")

    # 4) guard against empty set
    if len(good) < 3:
        raise RuntimeError("Too few mutual matches; aborting registration")

    # 5) sort by distance and then build your point arrays
    matches = sorted(good, key=lambda m: m.distance)

    pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts_tgt = np.float32([kp_tgt[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # draw_matches(ref, kp_ref, tgt, kp_tgt,
    #            matches, element=ref_element,
    #            top_n=draw_n_matches)



    # ── Estimate transform ────────────────────────────────────────────────────
    M_affine, mask = cv2.estimateAffinePartial2D(
        pts_tgt, pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,   # pixels: lower → stricter
        maxIters=2000,
        confidence=0.995
    )
    if M_affine is not None:
        M = M_affine.astype(np.float32)
        warp_fn = cv2.warpAffine
    else:
        M_homo, mask = cv2.findHomography(pts_tgt, pts_ref,
                                          cv2.RANSAC, 3.0)
        if M_homo is None:
            raise RuntimeError("Transform estimation failed")
        M = M_homo.astype(np.float32)
        warp_fn = cv2.warpPerspective

    # ── Warp target into reference frame ───────────────────────────────────────
    # M = np.load("my_array.npy")
    H_ref, W_ref = ref.shape
    aligned = warp_fn(tgt, M, (W_ref, H_ref))
    # plt.figure()
    # plt.imshow(aligned)
    # plt.show()


    # ── Compute error metrics over overlap ────────────────────────────────────
    overlap = (aligned > 0)
    diff = (aligned.astype(float) - ref.astype(float)) * overlap
    rmse = np.sqrt(np.mean(diff[overlap]**2))
    mi   = normalized_mutual_information(aligned*overlap, ref*overlap)
    md   = np.mean([m.distance for m in matches])
    inl  = int(mask.sum()) if mask is not None else 0
    err = np.array([md, len(matches), rmse, mi])

    # # Show absolute‐difference map
    # plt.figure(figsize=(6,6))
    # plt.imshow(diff, cmap="gray")
    # plt.colorbar(label="|aligned – ref|")
    # plt.title("Absolute Difference (darker = better)")
    # plt.axis("off")
    # plt.show()

    return aligned, ref, tgt, ref_img, tgt_img, M, err

# — example usage —
if __name__ == "__main__":
    h5_a = "../../2017_alloy_data/bnp_fly0009.h5"
    png_b = "../../2017_alloy_data/ROI-3/2018011908-5kx-PreSI-2kx2k-20kcps-12hrs-HAADF.png"
    # h5_a = "../../bnp_fly0002.h5"
    # h5_b = "../../bnp_fly0003.h5"
    channels = [0, 2, 4, 8,  9, 10, 16, 20 ] #np.arange(4, 22, 1) # [8, 10, 14, 20, 21] #[6,7, 8, 9, 20] # [2, 3, 5, 6, 8, 22] #np.arange(6,9) #
    # Pre-allocate an array of shape (n_channels, error_dim)
    error_all = np.zeros((len(channels), 4), dtype=float)
    snr_ratio = np.zeros((len(channels), 1), dtype=float)
    # layout: 2 rows × ceil(n/2) columns (adjust as you like)
    n = len(channels)
    cols = len(channels)  # for 6 channels, 2×3 grid; tweak for your n
    rows = 3 #int(np.ceil(n/cols))+1
    fig = plt.figure(figsize=(4*n, 8))
    gs  = fig.add_gridspec(rows, n, height_ratios=[1,1.2, 1.2], hspace=0.3)


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
            print(np.max(ref_img0))

            error_all[i] = error
            snr_ratio[i] = snr(ref_img0)
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
    ax_snr = fig.add_subplot(gs[1, :])  # row = n_rows-1, all columns
    ax_snr.plot(np.arange(1,n+1),snr_ratio[:], marker='o', linewidth=2)
    ax_snr.set_ylabel("SNR")
    ax_err = fig.add_subplot(gs[2, :])  # row = n_rows-1, all columns

# plot only the last two columns of errors
    ax_err.plot(np.arange(1,n+1),A_norm[:, :], marker='o', linewidth=2)
    ax_err.set_xticks(np.arange(1,n+1))
    ax_err.set_xlabel("Channel index")
    ax_err.set_ylabel("Normalized error")
    ax_err.legend(["md","#matches","rmse","mi"], loc="best")
    ax_err.set_title("Normalized errors (last two metrics)")

    plt.tight_layout()
    plt.show()
