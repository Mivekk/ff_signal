import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
from load_data import load_and_create_sliding_windows
from model import PeakPredictorNN

# Set the random seed for NumPy
np.random.seed(0)

# Set the random seed for PyTorch
torch.manual_seed(0)

def train_model(model, criterion, optimizer, train_loader, num_epochs):
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        tp, tn, fp, fn = 0, 0, 0, 0
        total = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            
            total += len(predicted)
            for i in range(len(predicted)):
                tp += 1.0 if predicted[i] == labels[i] and predicted[i] else 0
                tn += 1.0 if predicted[i] == labels[i] and not predicted[i] else 0
                fp += 1.0 if predicted[i] != labels[i] and predicted[i] else 0
                fn += 1.0 if predicted[i] != labels[i] and not predicted[i] else 0
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        print(f"Accuracy: {100 * (tp + tn) / total:.2f}%")
        print(f"Precision: {100 * precision:.2f}%")
        print(f"Recall: {100 * recall:.2f}%")
        print(f"F1 Score: {100 * 2 * precision * recall / (precision + recall):.2f}%")

def create_data_loaders(data_windows, target_labels):
    data_windows = torch.tensor(data_windows, dtype=torch.float32)
    target_labels = torch.tensor(target_labels, dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(data_windows, target_labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

def main():
    # Path to the directory containing JSON files
    directory_path = "./data/"
    # Load and create sliding windows
    data_windows, target_labels = load_and_create_sliding_windows(directory_path)

    train_loader, test_loader = create_data_loaders(data_windows, target_labels)
    
    input_size = data_windows.shape[1]
    output_size = 1
    model = PeakPredictorNN(input_size, output_size)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, criterion, optimizer, train_loader, num_epochs=5)
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to model.pth")
    
    model.load_state_dict(torch.load("./model.pth"))
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()