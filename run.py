import torch
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
        optimizer.zero_grad()   # Clear the previous gradients.
        data = data.to(device)  # Move data to GPU
        out = model(data.x, data.edge_index, data.batch) # for GCN forward pass
        # out = model(data.x, data.edge_index,data.edge_attr, data.batch) # for GCNe forward pass.
        loss = criterion(out, data.y) # Compute the loss.
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

    compute_metrics(all_labels,all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    print(all_preds)
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
    val_size = 0.2
    test_size = 0.5
    num_aug=10
    feature_dimensions=116+11
    # feature_dimensions=43
    # feature_dimensions=11
    # feature_dimensions=32
    batch_size = 32
    subject_data,all_labels,num_classes=load_subject_data(num_aug)
    # Extract the first 17 subjects
    subject_train = subject_data[:(20 * 10) + 1]
    train_label_0 = all_labels[:(20 * 10) + 1]

    # Extract one augmented data from the next 8 subjects for each block of 23 subjects
    subject_test = []
    labels_test = []
    for i in range(200, 251):
        subject_test.append(subject_data[i])
        labels_test.append(all_labels[i])

    subject_train.extend(subject_data[251:(44 * 10)+1])
    train_label_1 = all_labels[251:(44 * 10)+1]

    for i in range(440, 481):
        subject_test.append(subject_data[i])
        labels_test.append(all_labels[i])

    subject_train.extend(subject_data[481:(68 * 10)+1])
    train_label_2 = all_labels[481:(68 * 10)+1]

    for i in range(681, 710):
        subject_test.append(subject_data[i])
        labels_test.append(all_labels[i])

    labels_train = np.concatenate((train_label_0, train_label_1, train_label_2), axis=0)

    subject_train,labels_train=shuffle(subject_train,labels_train,random_state=42)

    subject_train, subject_val, labels_train, labels_val = train_test_split(
        subject_train, labels_train, test_size=val_size,shuffle=True, random_state=42)
    # Split the training set into training and validation sets
    # subject_train, subject_val, labels_train, labels_val = train_test_split(
    #     subject_data, all_labels, test_size=val_size,shuffle=True, random_state=42)
    # subject_test, subject_val, labels_test, labels_val = train_test_split(
    #     subject_val, labels_val, test_size=test_size,shuffle=True, random_state=42)

    # model = GCN(hidden_channels=80, number_of_features=feature_dimensions, number_of_classes=num_classes)
    model = GCN(hidden_channels=84, number_of_features=feature_dimensions, number_of_classes=num_classes)
    model, device, criterion, optimizer, scheduler = prepare_model(model)
    # model, device, criterion, optimizer = prepare_model(model)
    # Create PyTorch Geometric Data objects for each set
    data_train = [create_data_object(sub_conn_matrix, sub_ts, labels) for (sub_conn_matrix, sub_ts), labels in tqdm(zip(subject_train, labels_train), total=len(subject_train), desc="Processing training data")]
    data_val = [create_data_object(sub_conn_matrix, sub_ts, labels) for (sub_conn_matrix, sub_ts), labels in tqdm(zip(subject_val, labels_val), total=len(subject_val), desc="Processing validation data")]
    data_test = [create_data_object(sub_conn_matrix, sub_ts, labels) for (sub_conn_matrix, sub_ts), labels in tqdm(zip(subject_test, labels_test), total=len(subject_test), desc="Processing test data")]
    
    # Create DataLoader for each set
    data_loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True,pin_memory=True)
    data_loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False,pin_memory=True)
    data_loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False,pin_memory=True)

    print(f"len(Train)={len(data_train)}, len(Val)={len(data_val)}, len(Test)={len(data_test)}" )
    count(labels_test)

    # Lists to store losses for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train(model, data_loader_train, optimizer, criterion,device)
        val_loss, val_accuracy = evaluate(model, data_loader_val, criterion,device)
        torch.cuda.empty_cache()
        # Access the current learning rate from the optimizer
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
                f"Learning Rate: {current_lr:.6f}")
        
        # Append losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        scheduler.step(val_loss)
    
    test_loss, test_accuracy = test(model, data_loader_test, criterion,device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    plot_metrics(num_epochs,train_accuracies,val_accuracies,train_losses,val_losses)

        
if __name__ == '__main__':
    main()
