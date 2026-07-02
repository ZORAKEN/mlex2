import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


#
# NO MORE MODULES ALLOWED
#

def gaussFilter(img_in, ksize, sigma):
    #Remove noise before edge detection for smoothening
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """

    kernel = np.zeros((ksize, ksize), dtype=np.float64)

    center = ksize // 2

    for y in range(ksize):
        for x in range(ksize):
            dx = x - center
            dy = y - center

            kernel[y, x] = (
                1.0 / (2 * np.pi * sigma**2)#2d gaussian formula
                * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            )#center has the greatest value, values decrease as we move away from the center

    # Normalize kernel so that image brightness remains unchanged
    kernel /= np.sum(kernel)

    # Filter image
    filtered = convolve(img_in, kernel).astype(int)

    return kernel, filtered
    


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])#intensity changes in x direction,detects vertical edges 

    sobel_y = np.array([
    [ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1]
])#horizontal edges
    
# compensate for scipy's kernel flip
 #  sobel_x = np.flip(sobel_x)
# sobel_y = np.flip(sobel_y)

    gx = convolve(img_in, sobel_x).astype(int)
    gy = convolve(img_in, sobel_y).astype(int)# two derivative images

    return gx, gy

def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    
    g = np.sqrt(gx**2 + gy**2).astype(int)#gradient magnitude and type,measures edge strength
    theta = np.arctan2(gy, gx) # angle in radians tan^-1(gy/gx), gives edge direction, range (-pi, pi)

    return g, theta
   

def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    """
    Input angle is usually in radians.
    Some tests pass huge values, so we always:
      1. convert to degrees
      2. map to [0,180)
      3. quantize
    """

    angle = np.degrees(angle)
    angle = angle % 180

    if angle < 22.5:
        return 0
    elif angle < 67.5:
        return 45
    elif angle < 112.5:
        return 90
    elif angle < 157.5:
        return 135
    else:
        return 0

def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    #checks in all directions and keeps the maximum pixel value
    rows, cols = g.shape

    max_sup = np.zeros_like(g)

    # skip image border
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):

            angle = convertAngle(theta[y, x])

            current = g[y, x]

            # theta = 0°
            # compare left and right
            if angle == 0:

                if (current >= g[y, x - 1] and
                        current >= g[y, x + 1]):
                    max_sup[y, x] = current

            # theta = 45°
            # compare up-right and down-left
            elif angle == 45:

                if (current >= g[y - 1, x + 1] and
                        current >= g[y + 1, x - 1]):
                    max_sup[y, x] = current

            # theta = 90°
            # compare up and down
            elif angle == 90:

                if (current >= g[y - 1, x] and
                        current >= g[y + 1, x]):
                    max_sup[y, x] = current

            # theta = 135°
            # compare upper-left and lower-right
            elif angle == 135:

                if (current >= g[y - 1, x - 1] and
                        current >= g[y + 1, x + 1]):
                    max_sup[y, x] = current

    return max_sup



def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
    rows, cols = max_sup.shape

    thresh = np.zeros_like(max_sup, dtype=int)

    thresh[max_sup <= t_low] = 0
    thresh[(max_sup > t_low) & (max_sup <= t_high)] = 1
    thresh[max_sup > t_high] = 2#nonedge,weak edge,strong edge

    result = np.zeros_like(max_sup, dtype=int)

    for y in range(rows):
        for x in range(cols):

            if thresh[y, x] == 2:

                result[y, x] = 255

                for dy in (-1, 0, 1):#for evry strong eddge, check its 8 neighbors, if they are weak edges, make them strong edges and check their neighbors as well, repeat until no more weak edges are connected to strong edges
                    for dx in (-1, 0, 1):

                        ny = y + dy
                        nx = x + dx

                        if (0 <= ny < rows and
                            0 <= nx < cols and
                            thresh[ny, nx] >= 1):

                            result[ny, nx] = 255

    return result
#Is a weak edge connected to a strong edge?

#If YES:
#keep it.

#If NO:
#remove it.

#This is the heart of hysteresis.
def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result

# to display the image
img = plt.imread("data/contrast.jpg")

if img.ndim == 3:
   img = img.mean(axis=2)

result = canny(img)

plt.figure()
plt.imshow(result, cmap='gray')
plt.title("Canny Result")
plt.axis('off')
plt.show()