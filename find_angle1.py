
import cv2
import numpy as np
from image_module import rotate_image

def find_angle1(img1, img2):
    # make sure gray and float32
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim==3 else img1
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim==3 else img2

    H, W = img1.shape[:2]
    center = (W/2, H/2)
    R = np.hypot(W/2, H/2)
    M = W / np.log(R)  # scale factor
    M = 1

    # log–polar remap
    flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP
    lp1 = cv2.logPolar(img1_f, center, M, flags)
    lp2 = cv2.logPolar(img2_f, center, M, flags)

    # phase correlation
    (shift_rho, shift_theta), _ = cv2.phaseCorrelate(lp1, lp2)

    # convert to degrees
    print(shift_theta)
    print(H)
    angle =  (shift_rho / H) * 360.0
    # flip sign if needed for your convention
    angle = -angle

    print(f"Estimated rotation: {angle:.2f}°")

    # rotate img2 back
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(img2, rot, (W, H))

    return angle, aligned


# mg_rot=rotate_image(img_ref0,45)
# angle = find_angle1(img_rot[:526,:463],img_ref0)
