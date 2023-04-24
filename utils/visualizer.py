import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

def map_visualizer(opt, img_depth, img_binary, predict, file_name='data/sample/output.jpg'):
    row, col = 1, 3
    plt.figure(figsize=(12, 3))
    # --- Depth iamge
    depth = np.flipud(img_depth)
    plt.subplot(row, col, 1)
    plt.title("Scene Image")
    plt.imshow(depth, vmin=0, vmax=255, interpolation='none')
    plt.axis("off")
    # --- Binary image
    mask = np.flipud(img_binary)
    plt.subplot(row, col, 2)
    plt.title("Target Image")
    plt.imshow(mask, vmin=0, vmax=255, interpolation='none')
    plt.axis("off")
    # --- Heatmap
    heatmap = np.flipud(predict[0, :, :, 0]*255)
    heatmap = gaussian_filter(heatmap, sigma=5.0)
    plt.subplot(row, col, 3)
    plt.title("Predicted Result")
    plt.imshow(heatmap, cmap="jet", vmin=0, vmax=255, interpolation='none') # "viridis"
    plt.colorbar()
    plt.axis("off")

    plt.savefig(file_name)
    plt.clf()
    plt.close()

def map_visualizer_sample(img_depth, img_binary, predict):
    row, col = 1, 3
    plt.figure(figsize=(12, 3))
    # --- Depth iamge
    depth = np.flipud(img_depth)
    plt.subplot(row, col, 1)
    plt.title("Scene Image")
    plt.imshow(depth, vmin=0, vmax=255, interpolation='none')
    plt.axis("off")
    # --- Binary image
    mask = np.flipud(img_binary)
    plt.subplot(row, col, 2)
    plt.title("Target Image")
    plt.imshow(mask, vmin=0, vmax=255, interpolation='none')
    plt.axis("off")
    # --- Heatmap
    heatmap = np.flipud(predict[0, :, :, 0]*255)
    heatmap = gaussian_filter(heatmap, sigma=5.0)
    plt.subplot(row, col, 3)
    plt.title("Predicted Result")
    plt.imshow(heatmap, cmap="jet", vmin=0, vmax=255, interpolation='none') # "viridis"
    # plt.colorbar()
    plt.axis("off")

    name = './data/sample/output.jpg'
    plt.savefig(name)
    plt.clf()
    plt.close()