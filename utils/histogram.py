import numpy as np
from skimage.io import imread
from grayscale import to_gray
from visualize import visualize

def histogram_equalization(image):
    """
    Histogram equalization to reduce the image contrast
    """

    # check if image is greyscale
    if len(image.shape) != 2:
        raise ValueError("Image must be a grayscaled first")

    rows, cols = image.shape

    # calculate the histogram
    histogram, bins = np.histogram(image.flatten(), bins=256, range=(0, 256)) # pixel count for each of the 256 possible values in the grayscale image

    # calculate the cumulative distribution function
    cdf = histogram.cumsum()
    cdf = cdf / (rows * cols)  # normalize

    # create the transformation function
    transformation_function = np.round(cdf * 255).astype(np.uint8)

    # equalize the image
    equalized_image = transformation_function[image.flatten().astype(np.uint8)]
    equalized_image = equalized_image.reshape(rows, cols)

    # reduce contrast by stretching the central values of the histogram to compress the overall range
    stretch_factor = 0.8  # adjust to control
    mid_point = int(rows * cols / 2)
    equalized_image[:mid_point] = (equalized_image[:mid_point] * stretch_factor).astype(np.uint8)
    equalized_image[mid_point:] = (equalized_image[mid_point:] * (2 - stretch_factor)).astype(np.uint8)

    return equalized_image

def compute_histogram(image):
    histogram = np.zeros(256, dtype=np.uint8)
    rows, cols = image.shape
    gray_image = np.uint8(image * 255)
    for i in range(rows):
        for j in range(cols):
            histogram[gray_image[i][j]] = histogram[gray_image[i][j]] + 1
    n = rows * cols
    for i in range(rows):
        histogram[i]/=n
    
    return histogram       

if __name__ == "__main__":
    image = imread('C:/Users/manzi/VSCoding/cataract_classification/image_test/image_cataract_2.png')
    image = to_gray(image)
    compute_histogram(image)

    equalized_image = histogram_equalization(image)
    visualize(image, equalized_image)