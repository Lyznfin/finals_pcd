import numpy as np
import math

class ExtractFeatures:
    ''' formula from: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycoprops '''

    def extract_all(self, glcm):
        rows, cols = glcm.shape
        contrast = 0
        homogeneity = 0

        asm = 0
        energy = 0

        ux = 0
        uy = 0 
        sigx = 0
        sigy = 0
        correlation = 0

        for row in range(rows):
            for col in range(cols):
                ij_square = (row-col) ** 2
                contrast += glcm[row][col] * ij_square
                homogeneity += glcm[row][col] / (1+(ij_square))

                asm += glcm[row][col] ** 2

                ux += row * glcm[row][col]
                uy += col * glcm[row][col]

        energy = math.sqrt(asm)

        for row in range(rows):
            for col in range(cols):
                sigx += ((row-ux) ** 2) * glcm[row][col]
                sigy += ((row-uy) ** 2) * glcm[row][col]

        for row in range(rows):
            for col in range(cols):
                correlation += ((row-ux)*(col-uy)*glcm[row][col]) / math.sqrt(ux*uy)

        return [contrast, homogeneity, energy, correlation]

class CoOccurance:
    def _normalize_matrix(self, glcm):
        ''' symmetry = glcm + transpose matrix | glcm pixel normalization = glcm pixel / sum of glcm '''
        glcm = glcm.astype(np.float64)
        symmetry = glcm + np.transpose(glcm)
        return symmetry / symmetry.sum()

    def __calculate_glcm(self, gray_image, dx, dy):
        ''' theory: https://yunusmuhammad007.medium.com/feature-extraction-gray-level-co-occurrence-matrix-glcm-10c45b6d46a1 '''
        glcm = np.zeros((256, 256), dtype=np.uint32)
        rows, cols = gray_image.shape

        for row in range(rows):
            for col in range(cols):
                new_row = row + dy
                new_col = col + dx
                if 0 <= new_row < rows and 0 <= new_col < cols:
                    current_pixel = gray_image[row, col]
                    neighbor_pixel = gray_image[new_row, new_col]
                    glcm[current_pixel, neighbor_pixel] += 1

        return self._normalize_matrix(glcm)
    
    def deg_0(self, gray_image):
        return self.__calculate_glcm(gray_image, 1, 0) # pixel | right

    def deg_45(self, gray_image):
        return self.__calculate_glcm(gray_image, 1, -1) # pixel | top right

    def deg_90(self, gray_image):
        return self.__calculate_glcm(gray_image, 0, -1) # pixel | top

    def deg_135(self, gray_image):
        return self.__calculate_glcm(gray_image, -1, -1) # pixel | top left