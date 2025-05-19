import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()  # Enable interactive mode

import importlib
import hotspot_filter
from hotspot_filter import remove_hotspots_auto, remove_hotspots_manual, detect_hotspots
from skimage.metrics import normalized_mutual_information, mean_squared_error, structural_similarity

importlib.reload(hotspot_filter)


def register_images_from_two_h5(h5_path_1, ref_key, h5_path_2, tgt_key, channel,c_ratio):
    """
    Registers a target image (low-resolution/cropped) to a reference image (high-resolution)
    when stored in two separate HDF5 files.

    Args:
        h5_path_1 (str): Path to the HDF5 file containing the reference image.
        ref_key (str): Dataset key for the reference (high-resolution) image.
        h5_path_2 (str): Path to the HDF5 file containing the target image.
        tgt_key (str): Dataset key for the target (low-resolution) image.

    Returns:
        np.ndarray: The registered image.
    """
    # Load reference image from HDF5 file 1
    with h5py.File(h5_path_1, "r") as h5f1:
        ref_img = np.array(h5f1[ref_key])
        ref_img = ref_img[channel,:,:]

    # Load target image from HDF5 file 2
    with h5py.File(h5_path_2, "r") as h5f2:
        tgt_img = np.array(h5f2[tgt_key])
        tgt_img = tgt_img[channel,:,:]

    with h5py.File(h5_file_1, "r") as h5f1:
        channel_names = np.array(h5f1["MAPS/channel_names"])
        channel_names = np.array([name.decode("utf-8") for name in channel_names])
    # Ensure images are grayscale
    if len(ref_img.shape) == 3:
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    if len(tgt_img.shape) == 3:
        tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_RGB2GRAY)

    # Normalize data to uint8 for OpenCV
    if ref_img.dtype != np.uint8:
        ref_img = cv2.normalize(ref_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mask_ref = detect_hotspots(ref_img, sigma_factor=3)
    ref_img = remove_hotspots_auto(ref_img, mask_ref, method="inpaint")

    if tgt_img.dtype != np.uint8:
        tgt_img = cv2.normalize(tgt_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mask_tgt = detect_hotspots(tgt_img, sigma_factor=3)
    tgt_img = remove_hotspots_auto(tgt_img, mask_tgt, method="inpaint")

    tgt_img = cv2.equalizeHist(tgt_img)
    tgt_img = cv2.GaussianBlur(tgt_img, (3, 3), 0)

    ref_img = cv2.equalizeHist(ref_img)
    ref_img = cv2.GaussianBlur(ref_img, (3, 3), 0)
    ref_img = ref_img[c_ratio:, c_ratio:]


    # (optionally blur/equalize these crops again to remove edge artifacts)
    tgt_img = cv2.equalizeHist(tgt_img)
    tgt_img = cv2.GaussianBlur(tgt_img, (3,3), 0)


    # Detect ORB features and compute descriptors
    # orb = cv2.ORB_create(10000)
    # keypoints1, descriptors1 = orb.detectAndCompute(ref_img, None)
    # keypoints2, descriptors2 = orb.detectAndCompute(tgt_img, None)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(ref_img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(tgt_img, None)

    # Use BFMatcher with Hamming distance for ORB
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    if descriptors1 is None or descriptors2 is None:
        print("Error: One or both images have no keypoints detected!")
        return None

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort by best matches

    # Compute Mean Distance of Matches
    distances = [match.distance for match in matches]
    mean_distance = np.mean(distances) if distances else float('inf')


    # Draw top 50 matches
    match_img = cv2.drawMatches(ref_img, keypoints1, tgt_img, keypoints2, matches[:15], None, flags=2)
    plt.figure(figsize=(10, 5))
    plt.imshow(match_img, cmap='gray')
    plt.title("Feature Matching")
    plt.show()

    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # good_matches = [m for m in matches if m.distance < 250]  # Adjust threshold as needed


    # # # Ensure enough matches are found
    # if len(good_matches) > 10:
    #     src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #     dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # # Estimate homography using RANSAC
    # H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)
    H, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts,  method=cv2.RANSAC)

    # print(H)

    inliers = np.count_nonzero(mask)
    homography_error = np.mean(mask) if mask is not None else float('inf')

    # Warp the target image to match the reference
    h, w = ref_img.shape
    # aligned_img = cv2.warpPerspective(tgt_img, H, (w, h))
    aligned_img = cv2.warpAffine(tgt_img, H, (w, h))
    if np.all(aligned_img == 0):
        print("Error: Output image is completely black!")

    # ###  OPTICAL FLOW REFINEMENT (Farneback)
    flow = cv2.calcOpticalFlowFarneback(ref_img, aligned_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Create remap grid
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    flow_map = np.dstack((grid_x, grid_y)).astype(np.float32) + flow
    flow_map = flow_map.astype(np.float32)

    # Warp img2 using dense optical flow
    aligned_img = cv2.remap(aligned_img, flow_map, None, interpolation=cv2.INTER_LINEAR)


    # Compute distance of the whole image
    # registered_resized = cv2.resize(aligned_img, (ref_img.shape[1], ref_img.shape[0]))
    rmse = np.sqrt(np.mean((aligned_img-ref_img) ** 2))
    mi = normalized_mutual_information(aligned_img, ref_img)

    print("mean distance:", mean_distance, "| inliners:", inliers, "| homography error:",homography_error, "| number of matches:",len(matches), "|RMSE:", rmse,"|mutual information:", mi)
    error = np.array([mean_distance, inliers, homography_error, len(matches), rmse, mi])
    print(error)
    # import pdb; pdb.set_trace()

    # Show the registered image
    # fig, axes = plt.subplots(1,4, figsize=(10,5))
    # axes = axes.ravel()
    # axes[0].imshow(ref_img)
    # axes[0].set_title(f"{channel_names[channel]}: Reference Image")
    # axes[0].axis("off")
    # axes[1].imshow(aligned_img)
    # axes[1].set_title("Registered Image")
    # axes[1].axis("off")
    # axes[2].imshow(np.abs(aligned_img-ref_img))
    # axes[2].set_title("difference")
    # axes[2].axis("off")
    # axes[3].imshow(tgt_img)
    # axes[3].set_title("Targetedd Image")
    # axes[3].axis("off")
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(f"registration_{channel_names[channel]}.png", dpi=300)  # Save with high resolution (300 DPI)
    fig, ax = plt.subplots()
    im = ax.imshow(ref_img)
    ax.set_title(f"{channel_names[channel]}: Reference Image")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # tweak size/spacing if needed
    cbar.set_label("Intensity")                               # optional label
    fig.savefig(f"reference_{channel_names[channel]}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    blended = overlay_images(ref_img, aligned_img, alpha=0.5)
    fig, ax = plt.subplots()
    im = ax.imshow(blended,cmap='gray')
    ax.set_title("overlay")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # tweak size/spacing if needed

    plt.savefig(f"overlay_{channel_names[channel]}.png", dpi=300)  # Save with high resolution (300 DPI)
    plt.show()
    # plt.close(fig)

    diff = np.abs(ref_img.astype(float) - aligned_img.astype(float))
    fig, ax = plt.subplots()
    im = ax.imshow(diff,cmap='gray')
    ax.set_title("difference")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # tweak size/spacing if needed

    plt.savefig(f"difference_{channel_names[channel]}.png", dpi=300)  # Save with high resolution (300 DPI)
    plt.show()
    # plt.close(fig)



    fig, ax = plt.subplots()
    im = ax.imshow(aligned_img)
    ax.set_title("Registered Image")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # tweak size/spacing if needed
    cbar.set_label("Intensity")                               # optional label
    plt.savefig(f"registrered_{channel_names[channel]}.png", dpi=300)  # Save with high resolution (300 DPI)
    plt.close(fig)


    fig, ax = plt.subplots()
    im = ax.imshow(tgt_img)
    ax.set_title("Targeted Image")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # tweak size/spacing if needed
    cbar.set_label("Intensity")                               # optional label
    plt.savefig(f"targeted_{channel_names[channel]}.png", dpi=300)  # Save with high resolution (300 DPI)
    plt.close(fig)

    return aligned_img, ref_img, tgt_img, H, error
def overlay_images(reference_img, registered_img, alpha=0.5):
    """
    Creates a blended overlay of the reference and registered images.

    Args:
        reference_img (np.ndarray): Reference image (grayscale).
        registered_img (np.ndarray): Registered image (grayscale).
        alpha (float): Blending factor (0 = only reference, 1 = only registered).

    Returns:
        None (displays visualization)
    """
    # Resize registered image to match reference size
    registered_resized = cv2.resize(registered_img, (reference_img.shape[1], reference_img.shape[0]))

    # Blend images (convert to float for proper blending)
    blended = cv2.addWeighted(reference_img.astype(np.float32), alpha,
                              registered_resized.astype(np.float32), 1 - alpha, 0)

    # Convert back to uint8 for visualization
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended

# Example Usage
h5_file_1 = "../bnp_fly0002.h5"   # Path to HDF5 file containing the reference image
reference_dataset = "MAPS/XRF_fits"  # Dataset key for the reference image


h5_file_2 = "../bnp_fly0003.h5"  # Path to HDF5 file containing the target image
target_dataset = "MAPS/XRF_fits"  # Dataset key for the target image

# registered_image, ref_img, tgt_img, H, err = register_images_from_two_h5(h5_file_1, reference_dataset, h5_file_2, target_dataset,6,3)

#
# c = [2, 10, 20, 30, 50]
# H1 = np.zeros((len(c), 2, 3))  # Preallocate a 3D array
# # H2 = np.zeros((10, 3, 3))
#
# for i in range(0,5):
#     registered_image, ref_img, tgt_img, H1[i], error[i]= register_images_from_two_h5(h5_file_1, reference_dataset, h5_file_2, target_dataset,6,c[i])


# for i in range(0,5):
#    registered_image, H2[i] = register_images_from_two_h5(h5_file_1, reference_dataset, h5_file_2, target_dataset,6,c[i])
#
ch = [2, 3, 5, 6, 8, 16, 22]
error = np.zeros((len(ch),) + (6,))  # Preallocate storage

Hc = np.zeros((len(ch), 2, 3))
for i in range(0,len(ch)):
    try:
        registered_image, ref_img, tgt_img, Hc[i], error[i]= register_images_from_two_h5(h5_file_1, reference_dataset, h5_file_2,
                target_dataset,ch[i],0)
    except Exception as e:
        continue
#
