import matplotlib.pyplot as plt
import numpy as np
import torch
from model import PeakPredictorNN

from load_data import process_sensor_data, read_json_files

def predict(model, data):
    model.eval()
    with torch.no_grad():
        input = torch.tensor(data=data, dtype=torch.float32).transpose(0, 1)
        outputs = model(input)
        predicted = (outputs > 0.5).float()
        
    return predicted

def main():
    # Path to the directory containing JSON files
    directory_path = "./data/"
    # Load and create sliding windows
    data_windows, target_labels = process_sensor_data(read_json_files(directory_path))
    
    window_size = 250
    stride = 50
    
    input_size = window_size * 2
    output_size = 1
    model = PeakPredictorNN(input_size, output_size)
    model.load_state_dict(torch.load("./model.pth"))
    
    idx = 3
    
    plt.plot(data_windows[idx])
    
    for i in range((data_windows[idx].shape[0] - window_size) // stride + 1):
        data = data_windows[idx][i * stride : i * stride + window_size].reshape(-1, 1)
        prediction = predict(model, data)
        
        if prediction:
            plt.axvline(i * stride + window_size // 2, color='r', linestyle='--', linewidth=2)
            
    for i in range(len(target_labels[idx])):
        if target_labels[idx][i]:
            plt.axvline(i, color='g', linestyle='--', linewidth=2)
    
    # plt.legend()
    plt.show()

if __name__ == "__main__":
    main()