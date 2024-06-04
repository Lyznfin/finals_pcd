import numpy as np
import math

class ExtractFeatures:
    '''formula from: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycoprops'''

    def __extract_contrast(self, glcm):
        rows, cols = glcm.shape
        contrast = 0

        for row in range(rows):
            for col in range(cols):
                ij_square = (row-col) ** 2
                contrast += glcm[row][col] * ij_square

        return contrast
            
    def __extract_homogeneity(self, glcm):
        rows, cols = glcm.shape
        homogeneity = 0

        for row in range(rows):
            for col in range(cols):
                ij_square = (row-col) ** 2
                homogeneity += glcm[row][col] / (1+(ij_square))

        return homogeneity

    def __extract_energy(self, glcm):
        rows, cols = glcm.shape
        asm = 0
        energy = 0

        for row in range(rows):
            for col in range(cols):
                asm += glcm[row][col] ** 2

        energy = math.sqrt(asm)
        return energy
    
    def __extract_correlation(self, glcm):
        rows, cols = glcm.shape
        ux = 0
        uy = 0 
        sigx = 0
        sigy = 0
        correlation = 0

        for row in range(rows):
            for col in range(cols):
                ux += row * glcm[row][col]
                uy += col * glcm[row][col]

        for row in range(rows):
            for col in range(cols):
                sigx += ((row-ux) ** 2) * glcm[row][col]
                sigy += ((row-uy) ** 2) * glcm[row][col]

        for row in range(rows):
            for col in range(cols):
                correlation += ((row-ux)*(col-uy)*glcm[row][col]) / math.sqrt(ux*uy)

        return correlation
    
    def extract_all(self, glcm):
        return [self.__extract_contrast(glcm), self.__extract_homogeneity(glcm), self.__extract_energy(glcm), self.__extract_correlation(glcm)]

class CoOccurance:
    '''
    calculate the glcm of each angle.
    theory from: https://yunusmuhammad007.medium.com/feature-extraction-gray-level-co-occurrence-matrix-glcm-10c45b6d46a1
    '''
    def _normalize_matrix(self, glcm):
        '''
        glcm norm = glcm value / sum of glcm
        '''
        simmetry = glcm + np.transpose(glcm) # the sim of coocurence matrix is its value + its transpose
        normal_matrix = simmetry/np.sum(simmetry, keepdims=True) # norm of each pixel is equal to its value divided by the sum of all pixel
        return normal_matrix

    def deg_0(self, gray_image):
        rows, cols = gray_image.shape
        glcm = np.zeros((256, 256), dtype=np.uint32)
        for row in range(rows):
            for col in range(cols - 1):
                if row + 1 < cols:
                    current_pixel = gray_image[row][col]
                    neighbor_pixel = gray_image[row][col + 1] # pixel | right
                    glcm[current_pixel][neighbor_pixel] += 1
        return self._normalize_matrix(glcm)
    
    def deg_45(self, gray_image):
        rows, cols = gray_image.shape
        glcm = np.zeros((256, 256), dtype=np.uint32)
        for row in range(rows):
            for col in range(cols - 1):
                if row - 1 >= 0 and col + 1 < cols:
                    current_pixel = gray_image[row][col]
                    neighbor_pixel = gray_image[row - 1][col + 1] # pixel | top right
                    glcm[current_pixel][neighbor_pixel] += 1
        return self._normalize_matrix(glcm)
    
    def deg_90(self, gray_image):
        rows, cols = gray_image.shape
        glcm = np.zeros((256, 256), dtype=np.uint32)
        for row in range(rows):
            for col in range(cols - 1):
                if row - 1 >= 0:
                    current_pixel = gray_image[row][col]
                    neighbor_pixel = gray_image[row - 1][col] # pixel | top
                    glcm[current_pixel][neighbor_pixel] += 1
        return self._normalize_matrix(glcm)
    
    def deg_135(self, gray_image):
        rows, cols = gray_image.shape
        glcm = np.zeros((256, 256), dtype=np.uint32)
        for row in range(rows):
            for col in range(cols - 1):
                current_pixel = gray_image[row][col]
                neighbor_pixel = gray_image[row - 1][col - 1] # pixel | top left
                glcm[current_pixel][neighbor_pixel] += 1
        return self._normalize_matrix(glcm)