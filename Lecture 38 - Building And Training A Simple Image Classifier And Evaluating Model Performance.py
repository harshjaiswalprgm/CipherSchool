import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt

# Function to read and display an image
def read_and_display_image(image_path):
    # Read an image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not open or find the image.")
        return
    
    # Display the image
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return image

# Function to perform various image processing taskss
def process_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to 100x100
    resized_image = cv2.resize(image, (100, 100))
    
    # Draw a rectangle
    cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), 2)
    
    # Draw a circle
    cv2.circle(image, (100, 100), 50, (0, 255, 0), 2)
    
    # Display the image with shapes
    cv2.imshow('Shapes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return gray_image

# Function to apply filters and edge detection
def apply_filters(image, gray_image):
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 100, 200)
    
    # Display the edges
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to extract and display HOG features
def extract_and_display_hog(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not open or find the image.")
        return
    
    # Resize the image
    image = cv2.resize(image, (128, 64))
    
    # Extract HOG features
    features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    
    # Display the HOG image
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Features')
    plt.show()

# Main code execution
image_path = 'path_to_image.jpg'
image = read_and_display_image(image_path)
if image is not None:
    gray_image = process_image(image)
    apply_filters(image, gray_image)
    extract_and_display_hog(image_path)


### Explanation of Changes:

# 1. **Error Handling**: Added checks to ensure the image is correctly loaded.
# 2. **Consistent Variable Names**: Ensured the same variable names are used consistently across different functions.
# 3. **Modular Functions**: Organized the code into modular functions for better readability and reusability.
# 4. **Improved Functionality**: Ensured that each function focuses on a specific task, making the code easier to debug and maintain.

# This code should be placed in a script file and run. Ensure that the path to the image is correctly specified. The functions will perform the following tasks:
# 1. Read and display the original image.
# 2. Convert the image to grayscale, resize it, and draw shapes on it.
# 3. Apply Gaussian blur and Canny edge detection, displaying the results.
# 4. Extract and display HOG features from the grayscale image.