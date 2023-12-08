import torch
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
from models import *
from utils import *
from tqdm import tqdm

# Function to train the model
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for data in train_loader:# Iterate in batches over the training dataset.
        data = data.to(device)  # Move data to GPU
        out = model(data.x, data.edge_index, data.batch) # Perform a single forward pass.
        loss = criterion(out, data.y) # Compute the loss.
        optimizer.zero_grad()   # Clear the previous gradients.
        loss.backward() # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        total_loss += loss.item()

        # Convert logits to predictions
        preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    average_loss = total_loss / max(1, len(train_loader))  # Avoid division by zero

    return average_loss, accuracy

# Function to evaluate the model on the validation set
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)  # Move data to GPU
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()

            # Convert logits to predictions
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    average_loss = total_loss / max(1, len(val_loader))  # Avoid division by zero

    return average_loss, accuracy

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)  # Move data to GPU
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()

            # Convert logits to predictions
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    average_loss = total_loss / max(1, len(test_loader))  # Avoid division by zero

    return average_loss, accuracy


def visualize_graph(data):
    # Convert PyTorch Geometric Data object to NetworkX graph
    graph = nx.to_networkx_graph(data)
    # Visualize the graph
    plt.figure(figsize=(8, 8))
    nx.draw(graph, with_labels=True)
    plt.show()

def count(l):
    l=np.array(l)
    print(f'label 0: = {np.sum(l==0)}\n')
    print(f'label 1: = {np.sum(l==1)}\n')
    print(f'label 2: = {np.sum(l==2)}\n')
    

def main():
    val_size = 0.6
    feature_dimensions=43
    # feature_dimensions=11
    # feature_dimensions=32
    batch_size = 4
    # Set the number of folds
    num_folds = 5
    subject_data,all_labels,num_classes=load_subject_data()

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Instantiate your model, optimizer, and criterion outside the loop
    # model = GCN(hidden_channels=16, number_of_features=feature_dimensions, number_of_classes=num_classes)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = torch.nn.CrossEntropyLoss().to(device)

    # Lists to store losses for plotting
    train_losses = []
    val_losses = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(subject_data, all_labels)):
        model = GCN(hidden_channels=64, number_of_features=feature_dimensions, number_of_classes=num_classes)
        model, device, criterion, optimizer, scheduler = prepare_model(model)
        subject_train, subject_test = [subject_data[i] for i in train_idx], [subject_data[i] for i in test_idx]
        labels_train, labels_test = [all_labels[i] for i in train_idx], [all_labels[i] for i in test_idx]

        # Split the training set into training and validation sets
        subject_test, subject_val, labels_test, labels_val = train_test_split(
            subject_test, labels_test, test_size=val_size, random_state=42)
        
        # print('Train:= \n')
        # count(labels_train)
        # print('Test:= \n')
        # count(labels_test)
        # print('Val:= \n')
        # count(labels_val)       

        # Create PyTorch Geometric Data objects for each set
        data_train = [create_data_object(sub_conn_matrix, sub_ts, labels) for (sub_conn_matrix, sub_ts), labels in tqdm(zip(subject_train, labels_train), total=len(subject_train), desc="Processing training data")]
        data_val = [create_data_object(sub_conn_matrix, sub_ts, labels) for (sub_conn_matrix, sub_ts), labels in tqdm(zip(subject_val, labels_val), total=len(subject_val), desc="Processing validation data")]
        data_test = [create_data_object(sub_conn_matrix, sub_ts, labels) for (sub_conn_matrix, sub_ts), labels in tqdm(zip(subject_test, labels_test), total=len(subject_test), desc="Processing test data")]
        
        # Create DataLoader for each set
        data_loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True,pin_memory=True)
        data_loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False,pin_memory=True)
        data_loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False,pin_memory=True)

        print(f"fold number: {fold}. len(Train)={len(data_train)}, len(Val)={len(data_val)}, len(Test)={len(data_test)}" )

        # Training loop
        num_epochs = 100
        for epoch in range(1, num_epochs + 1):
            train_loss, train_accuracy = train(model, data_loader_train, optimizer, criterion,device)
            val_loss, val_accuracy = evaluate(model, data_loader_val, criterion,device)
            # Access the current learning rate from the optimizer
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
                  f"Learning Rate: {current_lr:.6f}")
            
            # Append losses for plotting
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
        
        test_loss, test_accuracy = test(model, data_loader_test, criterion,device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Plotting the losses
    plt.plot(range(1, (num_epochs*num_folds)+1), train_losses, label='Train Loss')
    plt.plot(range(1, (num_epochs*num_folds)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
        
if __name__ == '__main__':
    main()
