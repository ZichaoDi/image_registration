import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from skimage.metrics import normalized_mutual_information

def register_images(ref_img, tgt_img,
                    detector='SIFT',
                    matcher='BF',
                    warp_mode='affine',
                    use_kmeans=False,
                    n_clusters=100,
                    ransac_thresh=3.0,
                    ransac_iters=2000,
                    ransac_confidence=0.995):
    """
    Align tgt_img to ref_img.  If use_kmeans=True, we first cluster SIFT
    descriptors from the REF image into n_clusters visual words, then
    match centroids of those words between ref & tgt.
    """

    # 1) detect & describe SIFT
    if detector != 'SIFT':
        raise ValueError("k-means code only supports SIFT detector")
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref_img, None)
    kp2, des2 = sift.detectAndCompute(tgt_img, None)
    if des1 is None or des2 is None:
        raise RuntimeError("No descriptors found")

    if use_kmeans:
        # 2) cluster ref descriptors into visual words
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(des1)
        labels1 = kmeans.labels_
        # 3) assign tgt descriptors to nearest word
        labels2 = kmeans.predict(des2)

        # 4) compute centroids of keypoints for each word
        def compute_centroids(kp_list, labels):
            centroids = {}
            for w in range(n_clusters):
                idx = np.where(labels == w)[0]
                if len(idx) >= 1:
                    pts = np.array([kp_list[i].pt for i in idx])
                    centroids[w] = pts.mean(axis=0)
            return centroids

        cen1 = compute_centroids(kp1, labels1)
        cen2 = compute_centroids(kp2, labels2)

        # 5) keep only words present in both images
        common = sorted(set(cen1.keys()).intersection(cen2.keys()))
        if len(common) < 4:
            raise RuntimeError("Too few common visual words for registration")

        src = np.float32([ cen1[w] for w in common ]).reshape(-1,1,2)
        dst = np.float32([ cen2[w] for w in common ]).reshape(-1,1,2)
        matches = bf.match(src, dst)


    else:
        # standard descriptor matching path
        if matcher == 'BF':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = sorted(bf.match(des1, des2), key=lambda m:m.distance)
        else:
            raise ValueError("only BF matcher supported in non-kmeans mode")

        if len(matches) < 4:
            raise RuntimeError("Too few matches for registration")

        src = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    # 6) RANSAC transform estimation
    if warp_mode == 'affine':
        M, mask = cv2.estimateAffinePartial2D(
            dst, src,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=ransac_iters,
            confidence=ransac_confidence
        )
        warp_fn = lambda im: cv2.warpAffine(im, M,
                                            (ref_img.shape[1], ref_img.shape[0]))
    elif warp_mode == 'homography':
        M, mask = cv2.findHomography(
            dst, src,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=ransac_iters,
            confidence=ransac_confidence
        )
        warp_fn = lambda im: cv2.warpPerspective(im, M,
                                                 (ref_img.shape[1],
                                                  ref_img.shape[0]))
    else:
        raise ValueError(warp_mode)

    if M is None:
        raise RuntimeError("RANSAC failed")

    aligned = warp_fn(tgt_img)
    return aligned, M.astype(np.float32), mask.ravel().astype(bool)


# ─── example usage ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ref = cv2.imread("ref.png", cv2.IMREAD_GRAYSCALE)
    tgt = cv2.imread("tgt.png", cv2.IMREAD_GRAYSCALE)

    aligned, M, inliers = register_images(
        ref, tgt,
        use_kmeans=True,
        n_clusters=50,
        warp_mode='affine'
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
    print(f"RMSE={rmse:.2f}, MI={mi:.3f}, inliers={inliers.sum()}/{len(inliers)}")

