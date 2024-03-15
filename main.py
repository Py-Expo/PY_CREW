
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import glob2
import os, fnmatch
from pathlib import Path
# import mtcnn
from mtcnn.mtcnn import MTCNN

def extract_multiple_videos(intput_filenames, image_path_infile):
    """Extract video files into sequence of images."""
    i = 1  # Counter of first video
# Iterate file names:
    cap = cv2.VideoCapture(intput_filenames)
    if (cap.isOpened()== False):
        print("Error opening file")
# Keep iterating break
    while True:
        ret, frame = cap.read()  # Read frame from first video
            
        if ret:
            cv2.imwrite(os.path.join(image_path_infile , str(i) + '.jpg'), frame)  # Write frame to JPEG file (1.jpg, 2.jpg, ...)
# you can uncomment this line if you want to view them.
#           cv2.imshow('frame', frame)  # Display frame for testing
            i += 1 # Advance file counter
        else:
            # Break the interal loop when res status is False.
            break
    cv2.waitKey(50) #Wait 50msec
    cap.release()

# extract_multiple_videos(r'C:\Projects\DPFAKE\df\test.mp4', r'C:\Projects\DPFAKE\df\tst2')
# extract_multiple_videos(r'C:\Projects\DPFAKE\df\test.mp4', r'C:\Projects\DPFAKE\df\tst1')

from skimage import measure
def mse(image_path1, image_path2):
    
    # Convert images to grayscale
    imageA_gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageB_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    
    # Calculate MSE
    err = np.sum((imageA_gray.astype("float") - imageB_gray.astype("float")) ** 2)
    err /= float(imageA_gray.shape[0] * imageA_gray.shape[1])
    # return the MSE
    return err


def compare_images(imageA, imageB, title):
    # Compute the mean squared error
    m = mse(imageA, imageB)
    print("Mean Squared Error (MSE):",m)
    # Define a threshold for considering images as fake or not
    threshold = 1000  # You can adjust this threshold as needed
    
    # Determine if the images are fake or not based on the MSE
    if m > threshold:
        fake_label = "Fake"
    else:
        fake_label = "Not Fake"
    
    # Setup the figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Show the first image
    axes[0].imshow(imageA, cmap=plt.cm.gray)
    axes[0].set_title("Image A")
    axes[0].axis("off")
    
    # Show the second image
    axes[1].imshow(imageB, cmap=plt.cm.gray)
    axes[1].set_title("Image B")
    axes[1].axis("off")
    
    # Show the images and the fake label
    plt.suptitle(f"{title} - {fake_label}")
    plt.show()

# Test the function
image_path1 = r'C:\Projects\DPFAKE\df\tst1\1.jpg'
image_path2 = r'C:\Projects\DPFAKE\df\tst2\1.jpg'
imageA = cv2.imread(image_path1)
imageB = cv2.imread(image_path2)
compare_images(imageA, imageB, "Comparison")

