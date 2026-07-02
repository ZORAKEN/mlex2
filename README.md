# mlex2
# Image Processing Algorithms in Python

## Overview

This repository contains implementations of fundamental image processing algorithms written from scratch using Python and NumPy dome as apart of my coursework "Introduction to machine Learning". The project demonstrates core computer vision concepts including Gaussian filtering, convolution, edge detection, image sharpening, and binary morphological operations.

The implementations avoid relying on high-level image processing libraries for the main algorithms, making the code suitable for learning and educational purposes.

---

## Features

### 1. Gaussian Filtering

* Generates a Gaussian kernel using a specified kernel size and sigma.
* Applies Gaussian smoothing to reduce image noise.
* Uses convolution for filtering.

### 2. Manual Convolution

* Custom implementation of 2D convolution.
* Supports:

  * Grayscale images
  * RGB images
* Uses zero-padding.

### 3. Image Sharpening (Unsharp Masking)

* Blurs the image using Gaussian convolution.
* Computes the detail mask:

  ```
  Mask = Original − Blurred
  ```
* Produces a sharpened image:

  ```
  Sharpened = Original + Mask
  ```

### 4. Canny Edge Detection

The Canny implementation includes:

* Gaussian smoothing
* Sobel edge detection
* Gradient magnitude computation
* Gradient direction estimation
* Non-maximum suppression
* Double thresholding
* Hysteresis edge tracking

### 5. Binary Morphological Operations

Implements classic binary morphology:

* Erosion
* Dilation
* Opening
* Closing

These operations use configurable structuring elements and demonstrate image refinement techniques such as noise removal and hole filling.

---

## Project Structure

```
.
├── CannyEdgeDetector.py      # Complete Canny edge detector
├── morphological.py          # Binary morphology operations
├── convo.py                  # Gaussian convolution & image sharpening
├── data/
│   ├── contrast.jpg
│   ├── input1.jpg
│   ├── erosion_image_raw.png
│   └── dilation_image_raw.png
└── README.md
```

---

## Requirements

Install the required packages:

```bash
pip install numpy matplotlib scipy pillow
```

---

## Running the Programs

### Canny Edge Detection

```bash
python CannyEdgeDetector.py
```

Displays:

* Sobel X response
* Sobel Y response
* Gradient magnitude
* Gradient direction
* Non-maximum suppression result
* Final Canny edge image

---

### Morphological Operations

```bash
python morphological.py
```

Performs:

* Binary erosion
* Binary dilation
* Opening
* Closing

Outputs processed binary images and displays the results.

---

### Image Sharpening

```bash
python convo.py
```

Produces:

```
sharpened.png
```

using Gaussian blur followed by unsharp masking.

---

## Algorithms Implemented

* Gaussian Kernel Generation
* 2D Convolution
* Sobel Operator
* Gradient Magnitude & Direction
* Non-Maximum Suppression
* Hysteresis Thresholding
* Binary Erosion
* Binary Dilation
* Opening
* Closing
* Unsharp Masking

---

## Learning Objectives

This project demonstrates:

* Digital image filtering
* Edge detection techniques
* Gradient-based feature extraction
* Binary image processing
* Mathematical morphology
* Spatial domain image enhancement

---

## Technologies Used

* Python 3
* NumPy
* SciPy
* Matplotlib
* Pillow (PIL)
