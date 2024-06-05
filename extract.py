import numpy as np
from skimage.io import imread
from skimage.transform import resize

from utils.grayscale import to_gray
from utils.visualize import visualize
from utils.histogram import histogram_equalization

from glcm.experiment_classes import CoOccurance, ExtractFeatures

co = CoOccurance()
extractor = ExtractFeatures()

def extract(image_path):
    image = imread(image_path)
    gray_image = to_gray(image)
    visualize(image, gray_image)
    rows, cols = gray_image.shape
    resized_gray_image = resize(gray_image, (rows / 6, cols / 6), anti_aliasing=True)
    visualize(gray_image, resized_gray_image)
    histographed = histogram_equalization(resized_gray_image)
    visualize(resized_gray_image, histographed)
    histographed = np.uint8(histographed * 255)

    glcm_0 = co.deg_0(histographed)
    glcm_45 = co.deg_45(histographed)
    glcm_90 = co.deg_90(histographed)
    glcm_135 = co.deg_135(histographed)

    feature_0 = extractor.extract_all(glcm_0)
    feature_45 = extractor.extract_all(glcm_45)
    feature_90 =  extractor.extract_all(glcm_90)
    feature_135 = extractor.extract_all(glcm_135)

    features = {
        'contrast': [feature_0[0], feature_45[0], feature_90[0], feature_135[0]],
        'homogeneity': [feature_0[1], feature_45[1], feature_90[1], feature_135[1]],
        'energy': [feature_0[2], feature_45[2], feature_90[2], feature_135[2]],
        'correlation': [feature_0[3], feature_45[3], feature_90[3], feature_135[3]]
    }

    return features