import torch
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import GCN
from utils import *

# Function to train the model
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for data in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Convert logits to predictions
        preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    average_loss = total_loss / len(train_loader)

    return average_loss, accuracy

# Function to evaluate the model on the validation set
def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation"):
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()

            # Convert logits to predictions
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    average_loss = total_loss / len(val_loader)

    return average_loss, accuracy

def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()

            # Convert logits to predictions
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

            # Visualize the graph for one batch (you can adjust this based on your needs)
            # visualize_graph(data)

    accuracy = accuracy_score(all_labels, all_preds)
    average_loss = total_loss / len(test_loader)

    return average_loss, accuracy

def visualize_graph(data):
    # Convert PyTorch Geometric Data object to NetworkX graph
    graph = nx.to_networkx_graph(data)
    # Visualize the graph
    plt.figure(figsize=(8, 8))
    nx.draw(graph, with_labels=True)
    plt.show()

def main():
    test_size = 0.2
    validation_size = 0.2
    feature_dimensions=32
    order=2
    threshold=0.5
    batch_size = 4 
    label=[]
    subject_fc_matrices=np.load('source_data/fc/fc_matrices.npy')
    print(f"Number of Subjects {len(subject_fc_matrices)}")
    for i in range(len(subject_fc_matrices)):
        if i <25:
            label.append(0)
        if i >= 25 and i<48:
            label.append(1)
        if i >=48:
            label.append(2)

    all_labels = np.array(label)
    num_classes=len(set(label))
    # Find unique labels and their counts
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    # Create a dictionary to store the counts for each unique label
    label_counts = dict(zip(unique_labels, counts))
    # Print the results
    for label, count in label_counts.items():
        print(f"Label {label}: {count} subjects")

    # First, split into training and test sets
    subject_fc_train, subject_fc_test, labels_train, labels_test = train_test_split(
        subject_fc_matrices, all_labels, test_size=test_size, random_state=42)
    # Then, split the training set into training and validation sets
    subject_fc_train, subject_fc_val, labels_train, labels_val = train_test_split(
        subject_fc_train, labels_train, test_size=validation_size / (1 - test_size), random_state=42)

    # Example: Create PyTorch Geometric Data objects for each set
    data_train = [create_data_object(sub_conn_matrix,labels, feature_dimensions, order,threshold) for sub_conn_matrix, labels in zip(subject_fc_train, labels_train)]
    data_val = [create_data_object(sub_conn_matrix,labels, feature_dimensions, order,threshold) for sub_conn_matrix, labels in zip(subject_fc_val, labels_val)]
    data_test = [create_data_object(sub_conn_matrix,labels, feature_dimensions, order,threshold) for sub_conn_matrix, labels in zip(subject_fc_test, labels_test)]
    
    data_loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    data_loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)

    # Instantiate your model, optimizer, and criterion
    model = GCN(hidden_channels=64, number_of_features=feature_dimensions, number_of_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Lists to store losses for plotting
    train_losses = []
    val_losses = []

    # Training loop
    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train(model, data_loader_train, optimizer, criterion)
        val_loss, val_accuracy = evaluate(model, data_loader_val, criterion)

        print(f"Epoch {epoch}/{num_epochs}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        # Append losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Plotting the losses
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
        
    test_loss, test_accuracy = test(model, data_loader_test, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        
if __name__ == '__main__':
    main()