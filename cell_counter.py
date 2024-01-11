"""
Created on Tue Jan  9 09:50:17 2024
@author: mcanela
"""

import nd2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import cv2
from kneed import KneeLocator
from scipy.signal import find_peaks
from skimage.measure import regionprops
from math import pi
from scipy.ndimage import label
from scipy import ndimage as ndi
import os
from PIL import Image

directory = 'C:/Users/mcanela/Desktop/image_test/'
ratio = 1.55 # px/µm



def split_layers(image):
    layers = {}
    for n in range(image.shape[0]):
        my_layer = image[n]
        layers['layer_' + str(n)] = my_layer
    return layers

def draw_ROI(layer_data, tag):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(layer_data, cmap='grey')
    ax.set_title(tag)
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
        
        return elbow_value

def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val) * 255
    return normalized_arr.astype(int)

def image_to_binary(image, tag):
    layers = split_layers(image)
    
    for layer_name, layer_data in layers.items():
        roi_coords = draw_ROI(layer_data, tag)
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
    blurred_normalized = normalize_array(blurred)
    # plt.imshow(blurred_normalized, cmap='grey');
  
    # Apply a threshold for the background
    elbow_value = background_threshold(blurred_normalized)
    binary_image = blurred_normalized < elbow_value
    # plt.imshow(binary_image, cmap='grey');
    
    # Plot identified cells
    # identified = cfos[:]
    # identified[binary_image] = 0
    # plt.imshow(identified, cmap='grey');
    
    return [binary_image, roi, elbow_value, layer_roi]

def watershed(binary_image):
    # Label connected clusters
    labeled_array, num_clusters = label(~binary_image)  # Invert the array because we want to label False values

    # Find properties of each labeled region
    regions = regionprops(labeled_array)
    
    # Parameters for filtering
    threshold_circularity = 0.75  # Circularity threshold
    
    # Calculate the minimum area in pixels using the conversion factor
    min_area_threshold_micron = 10  # Minimum area in µm^2
    min_area_threshold_pixels = min_area_threshold_micron * (ratio ** 2)
        
    # Filter elliptical/circular clusters based on circularity
    circular_clusters = []
    artifact_clusters = []
    
    for region in regions:
        # Calculate circularity: 4 * pi * area / (perimeter^2)
        if region.perimeter != 0:
            circularity = 4 * pi * region.area / (region.perimeter ** 2)
        else:
            circularity = 0
        
        # Check circularity and minimum area
        if circularity >= threshold_circularity:
            if region.area >= min_area_threshold_pixels:
                circular_clusters.append(region)
        elif circularity < threshold_circularity:
            artifact_clusters.append(region)
    
    # # Create a new image displaying the artifacts
    # artifacts_binary = np.zeros(binary_image.shape, dtype=bool)
    # for region in artifact_clusters:
    #     coords = region.coords  # Coordinates of the current region
    #     artifacts_binary[coords[:, 0], coords[:, 1]] = True
    # # plt.imshow(artifacts_binary);

    # Create a collection of images of artifacts
    my_artifacts = []
    for region in artifact_clusters:
        # Select only those artifacts that are big
        if region.area >= min_area_threshold_pixels:        
            artifacts_binary = np.zeros(binary_image.shape, dtype=bool)
            coords = region.coords  # Coordinates of the current region
            artifacts_binary[coords[:, 0], coords[:, 1]] = True
            # Find rows and columns that are all False
            # rows_to_keep = np.any(artifacts_binary, axis=1)
            # cols_to_keep = np.any(artifacts_binary, axis=0)
            # Crop the array based on the identified rows and columns
            # cropped_arr = artifacts_binary[rows_to_keep][:, cols_to_keep]
            # Save the array of each individual artifact and its area
            artifact_info = [artifacts_binary, region.area]
            my_artifacts.append(artifact_info)
    # plt.imshow(my_artifacts[25][0]);

    # Analyze each artifact individually
    separated_artifacts = []
    for artifact in my_artifacts:
        # Calculate the distance to the edge
        distance_artifact = ndi.distance_transform_edt(artifact[0]) #Select the array
        distance_normalized_artifact = normalize_array(distance_artifact)
        # plt.imshow(distance_normalized);
        # Select only the center/s of the artifact
        threshold_artifact = 0.8 * 255
        binary_artifact = distance_normalized_artifact < threshold_artifact
        # plt.imshow(binary_artifact);
        # Count the number and sizes of the centers
        labeled_array_artifact, num_clusters_artifact = label(~binary_artifact)  # Invert the array because we want to label False values
        regions_artifact = regionprops(labeled_array_artifact)
        # Calculate the poderated mean of the area of the artifact by the area of its centers
        # Consider only if passes the min_area
        # Then append the coordinates of each
        total = sum(region.area for region in regions_artifact)
        for region in regions_artifact:
            factor = region.area/total
            separated_area = artifact[1] * factor
            if separated_area >= min_area_threshold_pixels:
                separated_artifacts.append(region)
    
    output_coords = []
    for circular_cluster in circular_clusters:
        output_coords.append(circular_cluster.coords)
    for separated_artifact in separated_artifacts:
        output_coords.append(separated_artifact.coords)
        
    return output_coords

def calculate_roi_area(roi):
    # Ensure the input array has the correct shape
    if roi.shape[0] != 1 or roi.shape[2] != 2:
        raise ValueError("Input array should have shape (1, n, 2)")

    # Extract the coordinates from the array
    x_coords = roi[0, :, 0]
    y_coords = roi[0, :, 1]

    # Apply the Shoelace formula to calculate the area
    area = 0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, 1)) - np.dot(y_coords, np.roll(x_coords, 1)))

    return area
    
# =============================================================================
# Run the whole script
# =============================================================================

def compiler(directory, ratio):
    dict_of_binary = {}
    
    for filename in os.listdir(directory):
        if filename.endswith(".nd2"):
            file_path = os.path.join(directory, filename)
            image = nd2.imread(file_path)
            binary_and_roi_and_elbow_and_layerroi = image_to_binary(image, filename[:-4])
            dict_of_binary[filename] = binary_and_roi_and_elbow_and_layerroi
    
    output_df = pd.DataFrame(columns=['file_name',
                                      'background_threshold',
                                      'num_cells',
                                      'roi_area',
                                      'cells/mm^2'
                                      ])   
    
    for key, value in dict_of_binary.items():
        output_coords = watershed(value[0])
        roi_area = calculate_roi_area(value[1])
        roi_area = roi_area * (ratio ** 2)
        cells_mm_2 = 10**6*len(output_coords)/roi_area
        output_df = output_df.append({'file_name': key[:-4],
                                      'background_threshold': value[2],
                                      'num_cells': len(output_coords),
                                      'roi_area': roi_area,
                                      'cells/mm^2': cells_mm_2},
                                      ignore_index=True)
        
        # Create an image with the results
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
        # Plot the first array in the first panel
        im1 = axes[0].imshow(value[3], cmap='grey')
        axes[0].set_title('Original figure')
        
        # Plot the second array in the second panel
        artifical_binary = np.zeros(value[3].shape, dtype=bool)
        for coords in output_coords:
            artifical_binary[coords[:, 0], coords[:, 1]] = True
            
        axes[1].imshow(artifical_binary, cmap='grey')
        axes[1].set_title('Identified cells')
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        
        # Save the image as a JPEG file
        name = key[:-4] + '.jpg'
        file_path = os.path.join(directory, name)
        plt.savefig(file_path)
        plt.close()
        
    output_df_path = os.path.join(directory, 'results.csv')
    output_df.to_csv(output_df_path, index=False)

compiler(directory, ratio)

    

    
    
    
    
    
    




        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
             
        
        
        
