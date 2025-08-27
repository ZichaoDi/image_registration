import os
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()



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

        # C, H, W = arr.shape
        # fig, axes = plt.subplots(1, C, figsize=(4*C, 4))
        # if C == 1:
        #     axes = [axes]  # make it iterable even when there's only one channel

        # for i, ax in enumerate(axes):
        #     im = arr[i]
        #     ax.imshow(im, cmap='gray')
        #     ax.set_title(channel_names[i])
        #     ax.axis('off')

        # plt.tight_layout()
        # plt.show()

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

def rotate_image(
    img: np.ndarray,
    angle: float,
    scale: float = 1.0,
    border_mode: int = cv2.BORDER_REFLECT
) -> np.ndarray:
    """
    Rotate an image by `angle` degrees around its center.

    Parameters
    ----------
    img : np.ndarray
        Input image (H×W or H×W×C).
    angle : float
        Rotation angle in degrees (positive = counter-clockwise).
    scale : float, optional
        Isotropic scale factor (default 1.0).
    border_mode : int, optional
        Pixel extrapolation method (see cv2.BORDER_*). Default: cv2.BORDER_REFLECT.

    Returns
    -------
    rotated : np.ndarray
        The rotated image, same size as input.
    """
    (h, w) = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    # Compute rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Step 2: Compute the new bounding box size after rotation
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Step 3: Adjust the rotation matrix to center the image
    # M[0, 2] += (new_w / 2) - center[0]
    # M[1, 2] += (new_h / 2) - center[1]

    # Step 4: Apply the rotation
    rotated = cv2.warpAffine(img, M, (new_w, new_h)) ##adding new boarder
    # rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,borderMode=border_mode)
    return rotated

def snr(img):
    mean_signal = np.mean(img)
    std_noise = np.std(img)
    snr_ratio = 20 * np.log10(mean_signal / std_noise)
    return snr_ratio

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


def center_crop_2d(A: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    H, W = A.shape[:2]
    if crop_h > H or crop_w > W:
        raise ValueError("crop size must be ≤ array size")
    start_i = (H - crop_h) // 2
    start_j = (W - crop_w) // 2
    return A[start_i:start_i + crop_h,
             start_j:start_j + crop_w]
def overlay_images(ref, aligned, alpha=0.5, element=None):
    """
    Display ref and aligned on the same axes in true grayscale,
    with the aligned image semi-transparent on top.
    """
    # ensure uint8 0–255
    ref8    = np.clip(ref,    0, 255).astype(np.uint8)
    aligned8 = np.clip(aligned, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(6,6))
    # show reference first
    im_ref = ax.imshow(ref8,    cmap='gray', vmin=0, vmax=255)
    # overlay aligned with alpha
    ax.imshow(aligned8, cmap='gray', vmin=0, vmax=255, alpha=alpha)

    ax.axis('off')
    title = f"{element}: Overlay" if element else "Overlay"
    ax.set_title(title)

    # one colorbar for the reference intensity scale
    cbar = fig.colorbar(im_ref, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pixel intensity (0–255)")

    plt.tight_layout()
    plt.show()


def overlay_images1(ref, aligned, alpha=0.5, element=None):
    """Blend to see where the small patch landed."""
    blended = cv2.addWeighted(ref.astype(np.float32), alpha,
                              aligned.astype(np.float32), 1-alpha, 0)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    plt.figure(figsize=(6,6))
    plt.imshow(blended, cmap="gray")
    title = f"{element}: Overlay" if element else "Overlay"
    plt.title(title)
    plt.axis("off")
    plt.show()


def find_angle(img1, img2):
    """Use log-polar transform centered at image center."""
    center = (img1.shape[1] // 2, img1.shape[0] // 2)
    M = img1.shape[0] / np.log(img1.shape[0])  # Magnitude scale
    # Convert to float32
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)

    log_polar1 = cv2.logPolar(img1_f, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    log_polar2 = cv2.logPolar(img2_f, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    # import pdb; pdb.set_trace()

    # Phase correlation gives rotation angle offset
    (shift, _) = cv2.phaseCorrelate(log_polar1, log_polar2)

    # Compensate for wrap-around (angle is in degrees)
    angle = shift[1]/img1.shape[1]*360  # Reverse rotation direction
    print(f"Estimated rotation angle: {angle:.2f} degrees")

    # Now apply rotation to img2 to align it with img1
    (h, w) = img2.shape
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(img2, rot_matrix, (w, h))

    # # Display results
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 3, 1); plt.imshow(img1, cmap='gray'); plt.title("Reference")
    # plt.subplot(1, 3, 2); plt.imshow(img2, cmap='gray'); plt.title("Rotated Input")
    # plt.subplot(1, 3, 3); plt.imshow(aligned, cmap='gray'); plt.title("Aligned (Rotation Only)")
    # plt.tight_layout(); plt.show()

    return angle, rot_matrix, aligned
