import numpy as np

def to_gray(image):
    ''' convert rgb to grayscale '''
    if image is None or not image.size:
        raise ValueError("Invalid image")
    
    # check if image is already gray
    if len(image.shape) == 2:
        return image
    
    # check if image has alpha channel
    if image.shape[-1] == 4:
        # remove it if there is
        image = image[..., :3]

    # convert rgb to grayscale
    gray_img = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    
    return gray_img