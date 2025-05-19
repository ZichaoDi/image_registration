import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.data import stereo_motorcycle, vortex
from skimage.transform import warp, AffineTransform
from skimage.registration import optical_flow_tvl1, optical_flow_ilk, phase_cross_correlation
from scipy.ndimage import fourier_shift
from skimage.io import imread
from skimage.metrics import normalized_mutual_information, mean_squared_error, structural_similarity
import sys

# --- Load the sequence
# image0, image1, disp = stereo_motorcycle()

# --- Convert the images to gray level: color is not supported.
# image0 = imread('Si/Coated_Sample6_Frame1.png')
image0 = imread('Si/Uncoated_Sample5_Frame1.png')
image0 = image0[:1800,:1800, :3]
# image1 = imread('Si/Coated_Sample6_Frame900.png')
image1 = imread('Si/Uncoated_Sample5_Frame300.png')
image1 = image1[:1800,:1800, :3]
image0 = rgb2gray(image0)
image1 = rgb2gray(image1)

# breakpoint()  # Execution pauses here
# --- Compute the optical flow
v, u = optical_flow_tvl1(image0, image1)
v1, u1 = optical_flow_ilk(image0, image1)


nr, nc = image0.shape

row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

image1_warp = warp(image1, np.array([row_coords + v, col_coords + u]), mode='edge')
image2_warp = warp(image1, np.array([row_coords + v1, col_coords + u1]), mode='edge')

### cross-correlation
shift = phase_cross_correlation(image0,image1)
# --- Use the estimated optical flow for registration
translation = AffineTransform(translation=(-shift[0][1], -shift[0][0]))  # -shift[1] is for x-axis, -shift[0] for y-axis
# Apply the translation using warp
image2 = warp(image1, translation)

#==== cross-correlation first and then optical flow
v_cc, u_cc = optical_flow_tvl1(image0,image2)


image_cc_of = warp(image2, np.array([row_coords + v_cc, col_coords + u_cc]), mode='edge')
##=====================================


# build an RGB image with the unregistered sequence
seq_im = np.zeros((nr, nc, 3))
seq_im[..., 0] = image1
seq_im[..., 1] = image0
seq_im[..., 2] = image0

# build an RGB image with the registered sequence
reg_im = np.zeros((nr, nc, 3))
reg_im[..., 0] = image1_warp
reg_im[..., 1] = image0
reg_im[..., 2] = image0

# build an RGB image with the registered sequence
reg_im1 = np.zeros((nr, nc, 3))
reg_im1[..., 0] = image2_warp
reg_im1[..., 1] = image0
reg_im1[..., 2] = image0
# build an RGB image with the registered sequence
target_im = np.zeros((nr, nc, 3))
target_im[..., 0] = image2
target_im[..., 1] = image0
target_im[..., 2] = image0
###
target_im_cc_of = np.zeros((nr, nc, 3))
target_im_cc_of[..., 0] = image_cc_of
target_im_cc_of[..., 1] = image0
target_im_cc_of[..., 2] = image0
##check registration quality based on different metrics
mi = np.zeros(5)
mi[0] = normalized_mutual_information(image0, image1)
mi[1] = normalized_mutual_information(image0, image1_warp)
mi[2] = normalized_mutual_information(image0, image2)
mi[3] = normalized_mutual_information(image0, image2_warp)
mi[4] = normalized_mutual_information(image0, image_cc_of)

rmse = np.zeros(5)
rmse[0] = mean_squared_error(image0, image1)
rmse[1] = mean_squared_error(image0, image1_warp)
rmse[2] = mean_squared_error(image0, image2)
rmse[3] = mean_squared_error(image0, image2_warp)
rmse[4] = mean_squared_error(image0, image_cc_of)
# --- Show the result

fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1,5, figsize=(10, 5))

ax0.imshow(seq_im)
ax0.set_title("Unregistered sequence")
ax0.set_axis_off()

ax1.imshow(reg_im)
ax1.set_title("OF_tv Registered sequence")
ax1.set_axis_off()

ax2.imshow(target_im)
ax2.set_title("CC Registered")
ax2.set_axis_off()

ax3.imshow(reg_im)
ax3.set_title("OF_ilk Registered sequence")
ax3.set_axis_off()

ax4.imshow(target_im_cc_of)
ax4.set_title("CC_OF_tv Registered sequence")
ax4.set_axis_off()

fig.tight_layout()

x = np.linspace(1, 5, 5)
y1 = mi       # Data for the first y-axis
y2 = rmse   # Data for the second y-axis

fig, ax1 = plt.subplots()

# Plot on the first axis
ax1.plot(x, y1, 'b-', label="mutual information")
ax1.set_xlabel('X-axis')
ax1.set_ylabel('MI', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
ax2.plot(x, y2, 'r--', label="RMSE")
ax2.set_ylabel('RMSE', color='r')
ax2.tick_params(axis='y', labelcolor='r')
plt.xticks(ticks=[1, 2, 3, 4, 5], labels=["raw","OF_tv","CC", "OF_ilk", "CC_OF_tv"])

plt.show(block=False)

# sys.exit(0)
# --- Compute the optical flow
# v, u = optical_flow_ilk(image0, image1, radius=15)

# --- Compute flow magnitude
norm = np.sqrt(u_cc**2 + v_cc**2)

# --- Display
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

# --- Sequence image sample

ax0.imshow(image0, cmap='gray')
ax0.set_title("Sequence image sample")
ax0.set_axis_off()

# --- Quiver plot arguments

nvec = 20  # Number of vectors to be displayed along each image dimension
nl, nc = image0.shape
step = max(nl // nvec, nc // nvec)

y, x = np.mgrid[:nl:step, :nc:step]
u_ = u[::step, ::step]
v_ = v[::step, ::step]

ax1.imshow(norm)
ax1.quiver(x, y, u_, v_, color='r', units='dots', angles='xy', scale_units='xy', lw=3)
ax1.set_title("Optical flow after CC magnitude and vector field")
ax1.set_axis_off()
fig.tight_layout()

plt.show()
