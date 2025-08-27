
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from image_module import load_image, rotate_image, snr, find_angle
from register_multiscale import preprocess
import cv2

def register_multimodal(fixed, moving,
                        initial_transform=None,
                        num_pyramid_levels=4,
                        sampling_percentage=0.05):
    """
    Register `moving` to `fixed` using Mattes mutual information.

    Parameters
    ----------
    fixed : np.ndarray
        Fixed image (reference).
    moving : np.ndarray
        Moving image (to be warped).
    initial_transform : sitk.Transform, optional
        Initial guess (e.g. translation).
    num_pyramid_levels : int
        Number of shrink levels for pyramid.
    sampling_percentage : float
        Fraction of voxels to sample for MI estimation.

    Returns
    -------
    sitk.Transform, np.ndarray
        Final transform and warped moving image.
    """
    # Convert to SITK images
    sitk_fixed  = sitk.GetImageFromArray(fixed.astype(np.float32))
    sitk_moving = sitk.GetImageFromArray(moving.astype(np.float32))

    # Initialize registration
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingPercentage(0.2)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetShrinkFactorsPerLevel(shrinkFactors = [2**i for i in range(num_pyramid_levels)][::-1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[i for i in range(num_pyramid_levels)][::-1])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Choose optimizer
    reg.SetOptimizerAsGradientDescent(learningRate=1.0,
                                      numberOfIterations=200,
                                      convergenceMinimumValue=1e-6,
                                      convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift()

    # Choose transform (start with translation or provided guess)
    if initial_transform is None:
        initial_transform = sitk.TranslationTransform(sitk_fixed.GetDimension())
    reg.SetInitialTransform(initial_transform, inPlace=False)

    # Multi-resolution & interpolation
    reg.SetInterpolator(sitk.sitkLinear)

    # Run registration
    final_transform = reg.Execute(sitk_fixed, sitk_moving)
    print("Optimizer stop condition:", reg.GetOptimizerStopConditionDescription())
    print("Final metric value:     ", reg.GetMetricValue())

    # Warp moving image
    warped = sitk.Resample(sitk_moving, sitk_fixed, final_transform,
                           sitk.sitkLinear, 0.0, sitk_moving.GetPixelID())
    warped_np = sitk.GetArrayFromImage(warped)
    return final_transform, warped_np

# Example usage:
if __name__ == "__main__":
    # Load your two NumPy arrays here:
	img_path_tgt = "../../2017_alloy_data/ROI-3/2018011908-5kx-PreSI-2kx2k-20kcps-12hrs-Mg.png"
	img_path_ref = "../../2017_alloy_data/bnp_fly0009.h5"
	##================multiscale dataset
	# img_path_tgt = "../../bnp_fly0003.h5"
	# img_path_ref = "../../bnp_fly0002.h5"
	##==========================================
	ref_key = "MAPS/XRF_fits"
	tgt_key = "MAPS/XRF_fits"
	channels = np.append(np.arange(4, 16, 1),[20, 21]) # [8, 10, 14, 20, 21] #[6,7, 8, 9, 20] # [2, 3, 5, 6, 8, 22] #np.arange(6,9) #
	channel = 20#, 10, 14, 20, 21]
	img_ref0, channel_name = load_image(img_path_ref, ref_key, channel)

	##=================more specific preprocess
	fixed = preprocess(img_ref0[:400, :])
	# if channel in [16, 17, 18]:
	#     img_ref0 = 255 - img_ref0

	img_tgt_full, _ = load_image(img_path_tgt, tgt_key, channel)
	img_tgt_full = cv2.flip(img_tgt_full, 0)
	moving = preprocess(img_tgt_full[100:, :])


	tfm, aligned = register_multimodal(fixed, moving, num_pyramid_levels=3, sampling_percentage=0.1)

    # Display result
	fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
	ax1.imshow(fixed,  cmap="gray");  ax1.set_title("Fixed")
	ax2.imshow(aligned,cmap="gray");  ax2.set_title("Aligned moving")
	plt.show()
