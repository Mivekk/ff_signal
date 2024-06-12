import os
import json
import numpy as np
from scipy.signal import savgol_filter

def read_json_files(directory_path):
    json_contents = []
    # Iterate over all files in the directory
    for file_name in os.listdir(directory_path):
        # Check if the file is a JSON file
        if file_name.endswith('.json'):
            file_path = os.path.join(directory_path, file_name)
            # Open and read the content of the JSON file
            with open(file_path, 'r') as file:
                content = json.load(file)
                json_contents.append(content)
    
    return np.array(json_contents)

def process_sensor_data(json_contents, window_length=51, poly_order=3):
    interior_data = []
    exterior_data = []
    peaks_data = []
    # Extract interior, exterior, and peaks data from JSON contents
    for file_content in json_contents:
        for entry in file_content:
            interior_data.append(entry["interior"])
            exterior_data.append(entry["exterior"])
            peaks_data.append(entry["peaks"]["serie"])
    
    # Smooth the interior and exterior data using Savitzky-Golay filter
    smoothed_interior = savgol_filter(interior_data, window_length=window_length, polyorder=poly_order)
    smoothed_exterior = savgol_filter(exterior_data, window_length=window_length, polyorder=poly_order)

    # Combine the smoothed interior and exterior data
    combined_data = np.stack([smoothed_interior, smoothed_exterior], axis=-1)
    
    return combined_data, np.expand_dims(np.array(peaks_data), axis=-1)

def create_sliding_windows(data, targets, index, window_size, stride):
    windows = []
    target_windows = []
    # Create sliding windows from the data and targets
    for i in range((data.shape[1] - window_size) // stride + 1):
        windows.append(data[index][i * stride : i * stride + window_size])
        target_windows.append(targets[index][i * stride : i * stride + window_size])
        
    return np.array(windows), np.array(target_windows)

def load_and_create_sliding_windows(directory_path):  
    window_size = 250  
    target_window_size = 25
    stride = 50
    
    # Read JSON files from the specified directory
    json_contents = read_json_files(directory_path)

    # Process sensor data to extract and smooth interior, exterior, and peaks data
    sensor_data, peaks_data = process_sensor_data(json_contents)
    
    all_windows, all_target_windows = [], []
    # Create sliding windows for each data sample
    for idx in range(sensor_data.shape[0]):
        windows, target_windows = create_sliding_windows(sensor_data, peaks_data, idx,
                window_size=window_size, stride=stride)
        
        all_windows.append(windows)
        all_target_windows.append(target_windows)
        
    all_windows, all_target_windows = np.array(all_windows), np.array(all_target_windows)
        
    # Reshape the arrays to the required format
    all_windows = all_windows.reshape(-1, all_windows.shape[2] * all_windows.shape[3])
    all_target_windows = all_target_windows.reshape((-1, all_target_windows.shape[2]))
    
    # Create target labels based on the presence of a peak in the target window
    target_labels = [
        [1] in all_target_windows[i][window_size // 2 - target_window_size : window_size // 2 + target_window_size]
        for i in range(all_target_windows.shape[0])
    ]
    
    return all_windows, np.array(target_labels).reshape(-1, 1)