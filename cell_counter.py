"""
Created on Tue Jan  9 09:50:17 2024
@author: mcanela
"""

import nd2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import cv2
from kneed import KneeLocator
from scipy.signal import find_peaks
from skimage.measure import regionprops
from math import pi
from scipy.ndimage import label
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import skimage
from skimage.feature import peak_local_max


# Load the .nd2 file
directory = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-10 - TRAP2/Microscope TRAP2/Females/female_2/female_2_cg_001.nd2'
image = nd2.imread(directory)

def split_layers(image):
    layers = {}
    for n in range(image.shape[0]):
        my_layer = image[n]
        layers['layer_' + str(n)] = my_layer
    return layers

def draw_ROI(layer_data):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(layer_data, cmap=None)
    print('Draw a ROI on this layer and double click to finish.')
    print('To use the next layer to draw the ROI, just close this window.')

    polygon_coords = []  # List to store selected points
    def onselect(verts):
        polygon_coords.append(verts)
        print('The ROI has been correctly saved.')
    polygon_selector = PolygonSelector(ax, onselect) 
    plt.show(block=True)
    
    if len(polygon_coords) == 0:
        return None
    else:
        return polygon_coords[0]

def background_threshold(blurred_normalized):
        hist, bins = np.histogram(blurred_normalized[blurred_normalized != 0], bins=64, range=(1, 256)) 
    
        # Identify peak values closest to 255 and 255
        peaks, _ = find_peaks(hist)
        closest_peak_index = np.argmax(hist[peaks])
        
        # Create a subset of histogram and bins values between the identified peak and 255
        subset_hist = hist[peaks[closest_peak_index]:]
        subset_bins = bins[peaks[closest_peak_index]:-1]
        
        # Find the elbow using KneeLocator on the subset
        knee = KneeLocator(subset_bins, subset_hist, curve='convex', direction='decreasing')
        elbow_value = knee.elbow
        
        # Visualize the histogram and the identified peaks
        # plt.figure()
        # plt.plot(bins[:-1], hist)
        # plt.plot(bins[peaks], hist[peaks], 'ro')
        # plt.axvline(x=closest_peak_value, color='g', linestyle='--', label=f'Closest Peak to 255: {closest_peak_value:.2f}')
        # plt.axvline(x=elbow_value, color='b', linestyle='--', label=f'Elbow: {elbow_value:.2f}')
        # plt.title('Histogram with Identified Peaks and Elbow')
        # plt.xlabel('Pixel Value')
        # plt.ylabel('Frequency')
        # plt.legend()
        # plt.show()
        
        # Create a threshold mask
        # threshold = elbow_value
        # threshold_mask = layer_roi < threshold
        # layer_threshold = np.where(blurred_normalized < threshold, 255, blurred_normalized)
        # plt.imshow(layer_threshold, cmap='grey');
        
        return elbow_value
    

def watershed(binary_image):
        
    img = binary_image.astype(np.uint8)
    img = (img == 0).astype(np.uint8) * 255
    
    dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    plt.imshow(dist_transform);
    
    local_max_location_coords = peak_local_max(dist_transform, min_distance=1)
    local_max_location = np.zeros_like(dist_transform, dtype=bool)
    local_max_location[local_max_location_coords] = True
    
    local_max_boolean_coords = peak_local_max(dist_transform, min_distance=1)
    local_max_boolean = np.zeros_like(dist_transform, dtype=bool)
    local_max_boolean[local_max_boolean_coords] = True
    
    markers, _ = ndi.label(local_max_boolean)
    
    segmented = skimage.segmentation.watershed(255-dist_transform, markers, mask=img)

    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    
    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[1].imshow(-dist_transform, cmap=plt.cm.gray)
    ax[1].set_title('Distance transform')
    ax[2].imshow(segmented, cmap=plt.cm.nipy_spectral)
    ax[2].set_title('Separated objects')
    
    for a in ax:
        a.set_axis_off()
    
    fig.tight_layout()
    plt.show()
    
    
    
    
    
    
    
    
    
    return





def compiler(image):
    layers = split_layers(image)
    
    for layer_name, layer_data in layers.items():
        roi_coords = draw_ROI(layer_data)
        if roi_coords is not None:
            roi = np.array([roi_coords], dtype=np.int32)
            break
    
    # Apply the defined ROI
    cfos = layers['layer_1']  
    mask = np.zeros_like(cfos)
    cv2.fillPoly(mask, roi, 255)
    layer_roi = np.where(mask == 255, cfos, 0)
    # plt.imshow(layer_roi, cmap='grey');
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(layer_roi, (5, 5), 0)
    # plt.imshow(blurred, cmap='grey');
    
    # Normalize the cfos layer
    def normalize_array(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        normalized_arr = (arr - min_val) / (max_val - min_val) * 255
        return normalized_arr
    blurred_normalized = normalize_array(blurred)
    # plt.imshow(blurred_normalized, cmap='grey');
  
    # Apply a threshold for the background
    elbow_value = background_threshold(blurred_normalized)
    binary_image = blurred_normalized < elbow_value
    # plt.imshow(binary_image, cmap='grey');
    
    # Label connected clusters
    labeled_array, num_clusters = label(~binary_image)  # Invert the array because we want to label False values











    # Find properties of each labeled region
    regions = regionprops(labeled_array)
    
    # Parameters for filtering
    threshold_circularity = 0.5  # Circularity threshold
    
    # Calculate the minimum area in pixels using the conversion factor
    min_area_threshold_micron = 10  # Minimum area in µm^2
    min_area_threshold_pixels = min_area_threshold_micron * (1.55 ** 2)
    
    # Filter elliptical/circular clusters based on circularity and minimum area
    filtered_clusters = []
    
    for region in regions:
        # Calculate circularity: 4 * pi * area / (perimeter^2)
        circularity = 4 * pi * region.area / (region.perimeter ** 2)
        
        # Check circularity and minimum area
        if circularity >= threshold_circularity and region.area >= min_area_threshold_pixels:
            # Add the cluster to the list of filtered clusters
            filtered_clusters.append(region)
    
    # Extract sizes of filtered clusters
    sizes = [region.area for region in filtered_clusters]
    
    plt.hist(sizes, bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Cluster Sizes')
    plt.xlabel('Cluster Size (pixels)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    
    
    
    
    
    
    # Count the size of each cluster
    sizes = [np.sum(labeled_array == i) for i in range(1, num_clusters + 1)]
    


    
    
    
    
    
    




        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        