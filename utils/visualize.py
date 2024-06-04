import matplotlib.pyplot as plt
import numpy as np

def visualize(original, processed):
    """
    comparing image (used for development purpose)
    """

    plt.figure(figsize=(10, 6))

    plt.subplot(221)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image') # plot original
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(processed, cmap='gray')
    plt.title('Processed Image') # plot processed
    plt.axis('off')

    plt.tight_layout()
    plt.show()