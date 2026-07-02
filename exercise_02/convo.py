from PIL import Image
import numpy as np


def make_kernel(ksize, sigma):
    """
    Create a normalized Gaussian kernel."unsharp masking" actually makes an image sharper by enhancing edges.
    Original Image
      │
      ▼
Gaussian Blur
      │
      ▼
Compute Difference (Mask)
      │
      ▼
Add Difference Back
      │
      ▼
Sharpened Image

Mask=Original − Blurred
Sharpened=Original+Mask
or
Sharpened=2⋅Original − Blurred

    """
    kernel = np.zeros((ksize, ksize), dtype=np.float64)

    center = ksize // 2

    for y in range(ksize):
        for x in range(ksize):
            dx = x - center
            dy = y - center#the center contributes the most

            kernel[y, x] = (
                1.0 / (2 * np.pi * sigma ** 2)
                * np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))#gausian formula,center has the greatest value, values decrease as we move away from the center
            )

    # Normalize so that the kernel sums to 1,brightness of the image remains unchanged
    kernel /= np.sum(kernel)

    return kernel


def slow_convolve(arr, k):
    """
    Convolution with zero-padding.
    arr=image, k=kernel
    Works for grayscale and RGB images.
    """
    k = np.flip(k)#convolution requires flipping the kernel before applying it.aussian kernels ae symmetric, so flipping doesn't change the kernel, but it's important to do it for the convolution operation to be correct.
    kh, kw = k.shape#(5,5)
    pad_h = kh // 2#2
    pad_w = kw // 2#2

    # Grayscale image,only two dimensions
    if arr.ndim == 2:
        h, w = arr.shape

        padded = np.pad (
            arr,
            ((pad_h, pad_h), (pad_w, pad_w)),
            mode='constant',
            constant_values=0
        )#Adds zeros around the image.

        result = np.zeros((h, w), dtype=np.float64)

        for i in range(h):
            for j in range(w):
                value = 0.0#initialize the sum

                for u in range(kh):
                    for v in range(kw):
                        value += k[u, v] * padded[i + u, j + v]#∑kernel×neighborhood

                result[i, j] = value

        return result

    # RGB image
    elif arr.ndim == 3:
        h, w, c = arr.shape

        padded = np.pad(
            arr,
            ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode='constant',
            constant_values=0
        )

        result = np.zeros((h, w, c), dtype=np.float64)

        for channel in range(c):
            for i in range(h):
                for j in range(w):
                    value = 0.0

                    for u in range(kh):
                        for v in range(kw):
                            value += k[u, v] * padded[i + u, j + v, channel]

                    result[i, j, channel] = value

        return result

    else:
        raise ValueError("Unsupported image shape")


if __name__ == '__main__':

    # Gaussian parameters (good starting point)
    k = make_kernel(5, 1.0)

    # Choose one image
    im = np.array(Image.open('data/input1.jpg'))

    # Blur image
    blurred = slow_convolve(im.astype(np.float64), k)
#Edges become softer. after multiplied with the kernel, the pixel values are averaged with their neighbors, resulting in a smoother image with reduced detail and softer edges.
    # Unsharp masking
    mask = im.astype(np.float64) - blurred#subtracting blurred removes low frequencies smooth areas leaving high frequencies edges and details, creating a mask that highlights the edges and details in the image
    sharpened = im.astype(np.float64) + mask
    """Original:200

Blurred:150

Mask: 50 (200-150 stong edge)

Sharpened:

250

Edge becomes stronger."""

    # Clip to valid image range(0 ... 255)
    sharpened = np.clip(sharpened, 0, 255)

    # Convert back to uint8 (123.45 becomes 123, 255.67 becomes 255)
    sharpened = sharpened.astype(np.uint8)

    # Save result
    Image.fromarray(sharpened).save('sharpened.png')

    print("Saved sharpened image as sharpened.png")