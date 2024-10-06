import cv2
import requests
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt

# Function to download and convert an image from a URL
def read_image_from_url(url):
    try:
        # Fetch image from URL
        response = requests.get(url)
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        
        # Decode the image from the array into an OpenCV image
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            print("Error: Could not open or find the image from URL.")
            return None
        
        # Display the image
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return image
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Function to perform various image processing tasks
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
def extract_and_display_hog(image):
    # Resize the image
    image = cv2.resize(image, (128, 64))
    
    # Extract HOG features
    features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    
    # Display the HOG image
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Features')
    plt.show()

# Main code execution
url = 'https://upload.wikimedia.org/wikipedia/commons/4/41/Sunflower_from_Silesia2.jpg'  # Replace this with a valid image URL
image = read_image_from_url(url)

if image is not None:
    gray_image = process_image(image)
    apply_filters(image, gray_image)
    extract_and_display_hog(gray_image)
