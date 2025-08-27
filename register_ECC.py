import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_module import load_image, rotate_image, snr, find_angle
from register_multiscale import preprocess
from skimage.metrics import normalized_mutual_information, structural_similarity
plt.ion()

def normalize(img):
    return (img - np.mean(img)) / (np.std(img) + 1e-6)

# Set paths
##================multimodal dataset
# img_path_tgt = "../../2017_alloy_data/ROI-3/2018011908-5kx-PreSI-2kx2k-20kcps-12hrs-HAADF.png"
# img_path_ref = "../../2017_alloy_data/bnp_fly0009.h5"
# channels = [8]#, 2, 4, 8,  9, 10, 16, 20 ] #np.arange(4, 22, 1) # [8, 10, 14, 20, 21] #[6,7, 8, 9, 20] # [2, 3, 5, 6, 8, 22] #np.arange(6,9) #
##================multiscale dataset
img_path_tgt = "../../bnp_fly0003.h5"
img_path_ref = "../../bnp_fly0002.h5"
channels =  [2, 3, 5, 6, 8, 22] #np.arange(6,9) #
##==========================================
ref_key = "MAPS/XRF_fits"
tgt_key = "MAPS/XRF_fits"

# error_all = np.zeros((len(channels), 3))  # RMSE, MI
# snr_all = np.zeros(len(channels))

n = len(channels)
rows = 3  # reference, aligned, overlay
cols = len(channels)

angle_range = np.linspace(0,20,15)
error_all = np.zeros((len(angle_range), 3))  # RMSE, MI
snr_all = np.zeros(len(angle_range))
n = len(angle_range)
cols = len(angle_range)
fig = plt.figure(figsize=(4 * cols, 9))
gs = fig.add_gridspec(rows + 2, cols, height_ratios=[1, 1, 1, 0.5, 0.5], hspace=0.4)

for t, angle in enumerate(angle_range):

    for i, channel in enumerate(channels):
        row = t // cols
        col = t %  cols
        print(f"\nProcessing channel {channel}...")

        img_ref0, channel_name = load_image(img_path_ref, ref_key, channel)

        ##=================more specific preprocess
        # img_ref0 = preprocess(img_ref0[:400, :])
        img_ref0 = preprocess(img_ref0)
        # if channel in [16, 17, 18]:
        #     img_ref0 = 255 - img_ref0

        img_tgt_full, _ = load_image(img_path_tgt, tgt_key, channel)
        # img_tgt_full = cv2.flip(img_tgt_full, 0)
        img_tgt_full = preprocess(img_tgt_full[:, :])
        plt.figure(), plt.imshow(img_ref0), plt.show()
        plt.figure(), plt.imshow(img_tgt_full), plt.show()
        # scale=0.125
        # resized = cv2.resize(img_tgt_full, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        # h,w=np.shape(resized)
        # angle, _, _ = find_angle(resized, img_ref0[:h,:w])
        # plt.figure(), plt.imshow(img_ref0[:256,:256]), plt.show()
        # plt.figure(), plt.imshow(resized), plt.show()
        # angle = 45
        img_ref0 = rotate_image(img_ref0, angle)
        ##==================simple preprocess (for multiscale only)
        # img_ref0 = preprocess(img_ref0)
        # img_tgt_full = preprocess(img_tgt_full)
        ##===============================================
        # img_ref = normalize(img_ref0.astype(np.float32))
        # img_tgt = normalize(img_tgt_full.astype(np.float32))
        img_ref = img_ref0
        img_tgt = img_tgt_full



        # Define a range of downscale factors to try
        scale_factors = np.linspace(0.1,2.5, 50)  # From 20% to 100% of original size

        best_score = -np.inf
        best_scale = None
        best_top_left = None
        best_resized = None

        # Template match at multiple scales
        for scale in scale_factors:
            resized = cv2.resize(img_tgt, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            if resized.shape[0] > img_ref.shape[0] or resized.shape[1] > img_ref.shape[1]:
                continue  # Skip if resized target is larger than reference

            result = cv2.matchTemplate(img_ref, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_scale = scale
                best_top_left = max_loc
                best_resized = resized.copy()

        # Crop matched region from reference
        h, w = best_resized.shape
        top_left = best_top_left
        bottom_right = (top_left[0] + w, top_left[1] + h)
        ref_crop = img_ref[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # ECC refinement
        template = ref_crop.astype(np.float32) / 255.0
        target = best_resized.astype(np.float32) / 255.0

        # angle1, _, _ = find_angle(target, template)

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
        (cc, warp_matrix_ecc) = cv2.findTransformECC(template, target, warp_matrix,
                                                     motionType=cv2.MOTION_EUCLIDEAN,
                                                     criteria=criteria)

        # Warp the target to match the reference crop
        aligned = cv2.warpAffine(target, warp_matrix_ecc, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        blended = cv2.addWeighted(template, 0.5, aligned, 0.5, 0)

        # Plot results
        ax1 = fig.add_subplot(gs[0, t])
        ax1.imshow(img_ref, cmap='gray')
        ax1.set_title(f"{channel_name}\nRotation {angle:.2f}\nscale{best_scale:.2f}")
        ax1.add_patch(plt.Rectangle(top_left, w, h, edgecolor='lime', facecolor='none', linewidth=2))
        # for spine in ax.spines.values():
        #     spine.set_visible(False)

        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[1, t])
        ax2.imshow(aligned, cmap='gray')
        ax2.set_title("Aligned Target")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[2, t])
        ax3.imshow(blended, cmap='gray')
        ax3.set_title("Overlay")
        ax3.axis("off")

        # Compute metrics
        overlap = (aligned > 0)
        diff = (aligned - template) * overlap
        rmse = np.sqrt(np.mean(diff[overlap] ** 2))
        mi = normalized_mutual_information(aligned * overlap, template * overlap)
        ssim = structural_similarity(aligned*overlap, template*overlap, data_range=1, full=False, win_size=7, gaussian_weights=True, sigma=1.5)
    error_all[t] = [rmse, mi, ssim]
    snr_all[t] = snr(img_ref0)

A = error_all
# 1) Compute per‐column minima and maxima
col_min = A.min(axis=0)   # shape (6,)
col_max = A.max(axis=0)   # shape (6,)

# 2) Avoid divide‐by‐zero: if max==min, set denom to 1
denom = np.where(col_max > col_min, col_max - col_min, 1)

# 3) Normalize
A_norm = (A - col_min) / denom

ax_snr = fig.add_subplot(gs[3, :])  # row = n_rows-1, all columns
ax_snr.plot(np.arange(1,n+1),snr_all[:], marker='o', linewidth=2)
ax_snr.set_ylabel("SNR")
ax_err = fig.add_subplot(gs[4, :])  # row = n_rows-1, all columns

# plot only the last two columns of errors
ax_err.plot(np.arange(1,n+1),A_norm[:, :], marker='o', linewidth=2)
ax_err.set_xticks(np.arange(1,n+1))
ax_err.set_xlabel("Channel index")
ax_err.set_ylabel("Normalized error")
ax_err.legend(["rmse","mi","ssim"], loc="best")
ax_err.set_title("Normalized errors (last two metrics)")

plt.tight_layout()
plt.show()
