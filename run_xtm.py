# In the same directory as xtm_module.py
import numpy as np
from xtm_module import phantom3d, generate_noise, intersection_set

# 1. Generate a 3D phantom (e.g. for n=50)
vol = phantom3d(n=50)
print("Phantom volume shape:", vol.shape)  # ➔ (50, 50, 50)

# 2. Simulate your XRT/XRF projection with XTM_Tensor
#    (assuming you also imported XTM_Tensor from the module)
from xtm_module import XTM_Tensor
angles = np.linspace(0, 2*np.pi, 30)
projections, Lmat = XTM_Tensor(50,30,1e-2)
print("Projection data shape:", projections.shape)


