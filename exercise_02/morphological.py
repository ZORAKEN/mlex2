
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def extract_region(padded_image: np.ndarray, center_row: int, center_col: int, window_size: int) -> np.ndarray:
    # The function receives a padded image (pad_image) and the current pixel of our padded image.
    # ToDo: Return the surrounding area around that center pixel with the given size (window_size).
    # ToDo: Use slicing.
    radius = window_size // 2

    return padded_image[
        center_row - radius:center_row + radius + 1,
        center_col - radius:center_col + radius + 1
    ]


def pad_image(image: np.ndarray, padding_size: int) -> np.ndarray:
    # Pad the image with zeros.
    return np.pad(image, pad_width=padding_size, mode='constant', constant_values=0)


def erode_binary(image: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    # Apply erosion on the given image using the structuring element.
    se_size = structuring_element.shape[0]
    assert se_size == structuring_element.shape[1], "SE must be quadratic."
    assert se_size % 2 == 1, "SE size must be uneven."

    # ToDo: Create the padded image and an empty output image that can be filled later.
    radius = se_size // 2
    padded = pad_image(image, radius)
    output = np.zeros_like(image)

    # ToDo: Iterate over the provided image and perform erosion around each pixel.
    # ToDo: Hint: Use the extract_region function to get the area around each pixel.
    # ToDo: Hint: Don't forget that the extract region function receives the padded image and the corresponding centers.
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            region = extract_region(
                padded,
                r + radius,
                c + radius,
                se_size
            )

            if np.all(region[structuring_element == 1] == 1):
                output[r, c] = 1
                #Erosion RuleALL pixels must be white otherwise:pixel becomes black

    return output


def dilate_binary(image: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    # Apply dilation on the given image using the structuring element.
    se_size = structuring_element.shape[0]
    assert se_size == structuring_element.shape[1], "SE must be quadratic."
    assert se_size % 2 == 1, "SE size must be uneven."

    # ToDo: Create the padded image and an empty output image that can be filled later.
    radius = se_size // 2
    padded = pad_image(image, radius)
    output = np.zeros_like(image)

    # ToDo: Iterate over the provided image and perform dilation around each pixel.
    # ToDo: Hint: Use the extract_region function to get the area around each pixel.
    # ToDo: Hint: Don't forget that the extract region function receives the padded image and the corresponding centers.
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            region = extract_region(
                padded,
                r + radius,
                c + radius,
                se_size
            )

            if np.any(region[structuring_element == 1] == 1):
                output[r, c] = 1#AT LEAST ONE pixel = 1 otherwise:pixel becomes black

    return output


def open_binary(input_image: np.ndarray, structuring_element: np.ndarray, iterations: int = 1) -> np.ndarray:
    # ToDo: Perform opening (erosion followed by dilation).#remove smmall noise
    result = input_image.copy()

    for _ in range(iterations):
        result = erode_binary(result, structuring_element)
        result = dilate_binary(result, structuring_element)

    return result


def close_binary(input_image: np.ndarray, structuring_element: np.ndarray, iterations: int = 1) -> np.ndarray:
    # ToDo: Perform closing (dilation followed by erosion).
    result = input_image.copy()

    for _ in range(iterations):
        result = dilate_binary(result, structuring_element)
        result = erode_binary(result, structuring_element)

    return result#filling the cracks and small holes


def load_binary(filepath: str) -> np.ndarray:
    # Load the image and binarize it again with a simple threshold.
    img = Image.open(filepath).convert('L')
    arr = np.array(img, dtype=np.uint8)  # type: ignore
    binary_arr = (arr > 128).astype(np.uint8)
    return binary_arr


def save_binary(image_array: np.ndarray, filepath: str):
    # Save the binary image.
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img.save(filepath)


def show_image(image_array: np.ndarray, title: str = ""):
    plt.imshow(image_array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Paths.
    raw_erosion_image_path = 'data/erosion_image_raw.png'
    raw_dilation_image_path = 'data/dilation_image_raw.png'
    erosion_out_path = 'data/erosion_output.png'
    dilation_out_path = 'data/dilation_output.png'

    # Load images.
    erosion_input = load_binary(raw_erosion_image_path)
    dilation_input = load_binary(raw_dilation_image_path)

    # Structuring element.
    SE = np.ones((5, 5), dtype=np.uint8)

    # Erosion.
    # ToDo: Perform erosion multiple times until the circles separate from each other.
    eroded = erosion_input.copy()
    erosion_iterations = 0

    while True:
        previous = eroded.copy()
        eroded = erode_binary(eroded, SE)
        erosion_iterations += 1

        if np.array_equal(previous, eroded):
            break

    print(f"Erosion iterations: {erosion_iterations}")

    save_binary(eroded, erosion_out_path)
    show_image(eroded, "Erosion Output")

    # Dilation.
    # ToDo: Perform dilation multiple times until the hole closes.
    dilated = dilation_input.copy()
    dilation_iterations = 0

    while True:
        previous = dilated.copy()
        dilated = dilate_binary(dilated, SE)
        dilation_iterations += 1

        if np.array_equal(previous, dilated):
            break #No more changes

    print(f"Dilation iterations: {dilation_iterations}")

    save_binary(dilated, dilation_out_path)
    show_image(dilated, "Dilation Output")