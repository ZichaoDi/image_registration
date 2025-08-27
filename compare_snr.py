# channels = np.arange(start_ind,10,1)
from skimage.measure import shannon_entropy
n = len(channels)
snr_ratio = np.zeros(len(channels))
entropy = np.zeros(len(channels))
fig = plt.figure()
gs = fig.add_gridspec(3,len(channels),height_ratios=[1,1, 1.2],hspace=0.3)
for i, ch in enumerate(channels):
    row = (i) // n
    col = (i) % n
    ref_img, ele = load_image(img_path_ref, "MAPS/XRF_fits",ch)
    ref_img = preprocess(ref_img)
    snr_ratio[i]=snr(ref_img)
    entropy[i] = shannon_entropy(ref_img)
    ax = fig.add_subplot(gs[row,col])
    ax.imshow(ref_img, cmap="gray")
    ax.axis("off")
    ax.set_title(ele)
    ax_hist = fig.add_subplot(gs[row+1,col])
    ax_hist.hist(ref_img.ravel(), bins=50,color='steelblue',edgecolor='black')
    ax_hist.set_xlabel("value")
    ax_hist.set_ylabel("frequency")

ax_snr = fig.add_subplot(gs[2,:])
ax_snr.plot(np.arange(1,n+1),snr_ratio,marker='o',linewidth=2)
color1 = 'tab:blue'
ax_snr.set_xlabel("channel index")
ax_snr.set_ylabel("SNR",color=color1)

ax2 = ax_snr.twinx()
color2='tab:red'
ax2.plot(np.arange(1,n+1),entropy,marker='*',linewidth=2, color=color2)
ax2.set_ylabel('Entropy',color=color2)


plt.tight_layout()
plt.show()



