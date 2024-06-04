import numpy as np
from skimage.io import imread
from skimage.transform import resize

from utils.grayscale import to_gray
from utils.histogram import histogram_equalization

from glcm.experiment_classes import CoOccurance, ExtractFeatures

co = CoOccurance()
extractor = ExtractFeatures()

def extract(image_path):
    image = imread(image_path)
    gray_image = to_gray(image)
    gray_image = resize(gray_image, (256, 256), anti_aliasing=True)
    gray_image = histogram_equalization(gray_image)
    gray_image = np.uint8(gray_image * 255)

    glcm_0 = co.deg_0(gray_image)
    glcm_45 = co.deg_45(gray_image)
    glcm_90 = co.deg_90(gray_image)
    glcm_135 = co.deg_135(gray_image)

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