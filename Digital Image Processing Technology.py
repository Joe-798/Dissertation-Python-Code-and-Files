#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
import cv2
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from itertools import combinations
from multiprocessing import Pool


# In[2]:


# Adjust the image path and resize the image
def adjust_image_path(image_path, target_size=(128, 128)):
    image = Image.open(image_path)
    image = image.resize(target_size)  # Resize
    image_array = np.array(image) / 255.0  # Normalisation
    return image_array

# Adjust the image array and resize the image
def adjust_image_array(image_array, target_size=(128, 128)):
    image = Image.fromarray(image_array)
    image = image.resize(target_size)  # Resize
    image_array = np.array(image) / 255.0  # Normalisation
    return image_array

# Preprocess the image
def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise (Gaussian Blur)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sharpen the image
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)

    return sharpened



# In[2]:


def perform_canny_edge_detection(image_paths):
    canny_edge_detection = []

    # Canny edge detection for each image
    for image_path in image_paths:
        # Read the image
        image = cv2.imread(image_path)
        
        # Check if the image was successfully loaded
        if image is None:
            print(f"Error: Unable to load image {image_path}")
            continue
        
        # Canny edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Find contours, only external contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a copy for drawing contours
        result = image.copy()
        
        # Draw contours
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        # Convert the result to RGB format and add to the list
        canny_edge_detection.append(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    return canny_edge_detection


# In[3]:


def canny_edge_detection_with_various_apertures(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny edge detection with different aperture sizes
    edges_aperture_3 = cv2.Canny(gray, 100, 200, apertureSize=3)
    edges_aperture_5 = cv2.Canny(gray, 100, 200, apertureSize=5)
    edges_aperture_7 = cv2.Canny(gray, 100, 200, apertureSize=7)

    # Display results
    plt.figure(figsize=(12, 8))

    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    # Canny with apertureSize = 3
    plt.subplot(2, 2, 2)
    plt.imshow(edges_aperture_3, cmap='gray')
    plt.title('Canny: apertureSize = 3')

    # Canny with apertureSize = 5
    plt.subplot(2, 2, 3)
    plt.imshow(edges_aperture_5, cmap='gray')
    plt.title('Canny: apertureSize = 5')

    # Canny with apertureSize = 7
    plt.subplot(2, 2, 4)
    plt.imshow(edges_aperture_7, cmap='gray')
    plt.title('Canny: apertureSize = 7')

    plt.tight_layout()
    plt.show()


# In[6]:


def detect_close_parallel_lines(image_path, angle_threshold=5, distance_threshold=10):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)

    # Canny edge detection
    edges = cv2.Canny(sharpened, 50, 150, apertureSize=3)
    
    # Hough transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    
    if lines is None:
        print(f"No lines detected in {image_path}")
        return image
    
    # Extract line parameters (rho, theta)
    lines = lines[:, 0, :]
    
    close_parallel_lines = []
    
    # Iterate over each pair of lines using two for loops
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            rho1, theta1 = lines[i]
            rho2, theta2 = lines[j]
            
            # Calculate the angle difference between lines
            angle_diff = abs(theta1 - theta2) * (180 / np.pi)
            
            # Calculate the distance between lines
            distance = abs(rho1 - rho2)
            
            # If both angle difference and distance are within the given threshold,
            # consider the lines as parallel and close
            if angle_diff < angle_threshold and distance < distance_threshold:
                close_parallel_lines.append((lines[i], lines[j]))
    
    # Draw the detected pairs of lines
    for line1, line2 in close_parallel_lines:
        for rho, theta in (line1, line2):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return image

def detect_close_parallel_lines_images(image_paths, angle_threshold=5, distance_threshold=10):
    # Use multiprocessing to process images in parallel
    with Pool() as pool:
        results = pool.starmap(detect_close_parallel_lines, [(path, angle_threshold, distance_threshold) for path in image_paths])
    return results


# In[15]:


def detect_double_solid_lines(image_paths):
    
    def is_double_solid_line(line1, line2, distance_threshold=20):

        x11, y11, x12, y12 = line1[0]
        x21, y21, x22, y22 = line2[0]
        if abs(y11 - y21) < distance_threshold and abs(y12 - y22) < distance_threshold:
            return True
        return False

    results = []

    for image_path in image_paths:
        # 1. Load the image
        image = cv2.imread(image_path)

        # 2. Preprocess the image
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #Sharpen the image
        kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
        sharpened = cv2.filter2D(blurred, -1, kernel)


        # 3. Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # 4. Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=5, maxLineGap=5)

        # 5. Filter double solid lines
        double_solid_lines = []
        if lines is not None:
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    if is_double_solid_line(lines[i], lines[j]):
                        double_solid_lines.append((lines[i], lines[j]))

        # Draw detected double solid lines on the image
        if double_solid_lines:
            for line_pair in double_solid_lines:
                line1, line2 = line_pair
                x1, y1, x2, y2 = line1[0]
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                x1, y1, x2, y2 = line2[0]
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Append results for this image
        results.append({
            'image_path': image_path,
            'double_solid_lines': double_solid_lines,
            'processed_image': image
        })

    return results


# In[4]:


# Main folder path
main_folder_path = 'C:\\Users\\18229\\Desktop\\Sunderland'

# Find images in all subfolders
image_paths = []
for root, dirs, files in os.walk(main_folder_path):
    for file in files:
        if file.endswith('jpg'):
            image_paths.append(os.path.join(root, file))

# Check if 15 images are found
if len(image_paths) != 15:
    raise ValueError(f"Expected 15 images, but found {len(image_paths)} images.")

print(image_paths)


# In[8]:


# Load and preprocess images
images = np.array([adjust_image_path(path) for path in image_paths])
images_flattened = np.array([img.flatten() for img in images])

# Perform t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=5, random_state=42)  # Set perplexity to 5
images_tsne = tsne.fit_transform(images_flattened)

# Visualize the results
fig, ax = plt.subplots(figsize=(12, 8))

# Add each image thumbnail to the scatter plot of t-SNE results
for xy, img_path in zip(images_tsne, image_paths):
    img = Image.open(img_path)
    img.thumbnail((80, 80), Image.Resampling.LANCZOS)  # Resize thumbnail to a larger size
    imagebox = OffsetImage(img)
    ab = AnnotationBbox(imagebox, xy, frameon=False)
    ax.add_artist(ab)

# Set display range to avoid generating an overly large image
ax.set_xlim(images_tsne[:, 0].min() - 15, images_tsne[:, 0].max() + 15)
ax.set_ylim(images_tsne[:, 1].min() - 15, images_tsne[:, 1].max() + 15)

ax.set_title('t-SNE Visualisation of Images')
ax.set_xlabel('t-SNE Feature 1')
ax.set_ylabel('t-SNE Feature 2')

plt.show()


# In[12]:


processed_images = [preprocess_image(path) for path in image_paths]
images = np.array([adjust_image_array(img) for img in processed_images])
images_flattened = np.array([img.flatten() for img in images])

# Perform dimensionality reduction using t-SNE
tsne = TSNE(n_components=2, perplexity=5, random_state=42)  # Set the perplexity to 5
images_tsne = tsne.fit_transform(images_flattened)

# Visualize the results
fig, ax = plt.subplots(figsize=(12, 8))

# Add thumbnails of each image to the scatter plot of the t-SNE results
for xy, img_array in zip(images_tsne, processed_images):
    img = Image.fromarray(img_array)
    img.thumbnail((80, 80), Image.Resampling.LANCZOS)  # Resize the thumbnail to a larger size
    imagebox = OffsetImage(img, cmap='gray')
    ab = AnnotationBbox(imagebox, xy, frameon=False)
    ax.add_artist(ab)

# Set the display range to ensure no oversized image is generated
ax.set_xlim(images_tsne[:, 0].min() - 15, images_tsne[:, 0].max() + 15)
ax.set_ylim(images_tsne[:, 1].min() - 15, images_tsne[:, 1].max() + 15)

ax.set_title('t-SNE Visualisation of Processed Images')
ax.set_xlabel('t-SNE Feature 1')
ax.set_ylabel('t-SNE Feature 2')

plt.show()


# In[11]:


Image.fromarray(canny_edge_detection_images[0]).show()


# In[5]:


results_canny_edge_detection = perform_canny_edge_detection(image_paths)
Image.fromarray(results_canny_edge_detection[0]).show()


# In[7]:


result_detect_close_parallel_lines = detect_close_parallel_lines(image_paths[0])
Image.fromarray(result_detect_close_parallel_lines).show()


# In[ ]:


result_double_solid_lines = detect_double_solid_lines(image_paths[0])
Image.fromarray(result_double_solid_lines).show()


# In[5]:


canny_edge_detection_with_various_apertures(image_paths[0])


# In[ ]:





# In[ ]:




