from karateclub import NetMF
import torch
from torch_geometric.data import Data
import numpy as np
import networkx as nx
from tsfresh import select_features, extract_features,feature_extraction
from tsfresh.utilities.dataframe_functions import impute
from sklearn import metrics,preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.functional import normalize


def extract_tsfresh_features(x):

    functions_to_test = {
        "asof": feature_extraction.feature_calculators.absolute_sum_of_changes,
        # "ae": feature_extraction.feature_calculators.approximate_entropy,
        "bc": feature_extraction.feature_calculators.benford_correlation,
        "c3": feature_extraction.feature_calculators.c3,
        "cid_ce": feature_extraction.feature_calculators.cid_ce,
        "cam": feature_extraction.feature_calculators.count_above_mean,
        "cbm": feature_extraction.feature_calculators.count_below_mean,
        "lsam": feature_extraction.feature_calculators.longest_strike_above_mean,
        "var": feature_extraction.feature_calculators.variance,
        "std": feature_extraction.feature_calculators.standard_deviation,
        "skw": feature_extraction.feature_calculators.skewness,
        # "sentr": feature_extraction.feature_calculators.sample_entropy,
        "qua": feature_extraction.feature_calculators.quantile,
    }

    computed_feature_list = list()

    for key, function in functions_to_test.items():
        # start = time.time()
        for i in range(1):
            if key == "ae":
                computed_feature_list.append(np.float32(function(x, 10, 2)))
            elif key == "c3":
                computed_feature_list.append(np.float32(function(x, 7)))
            elif key == "cid_ce":
                computed_feature_list.append(np.float32(function(x, True)))
            elif key == "qua":
                computed_feature_list.append(np.float32(function(x, 0.25)))
            else:
                computed_feature_list.append(np.float32(function(x)))
        # print(computed_feature_list)
        # end = time.time()
        # duration = end-start
        # print(key, duration)
    return computed_feature_list

# NetMF embedding logic using karate club module
def netmf_embedding(graph):
    dimensions=32 
    order=2
    model = NetMF(dimensions=dimensions, order=order,seed=21)
    model.fit(graph)
    return model.get_embedding()

def temporal_feature_extraction(subject_time_series):
    temporal_features=[]
    for node in range(len(subject_time_series[:,0])):
        feature=extract_tsfresh_features(subject_time_series[:,node])
        temporal_features.append(feature)
    return np.array(temporal_features)


def create_data_object(sub_conn_matrix, sub_ROI_ts, label):
    threshold = 0.5
    adjacency_matrix = np.where(sub_conn_matrix < threshold, 0, sub_conn_matrix)
    # adjacency_matrix = (sub_conn_matrix > threshold).astype(float)
    
    temporal_embeddings = preprocessing.normalize(temporal_feature_extraction(sub_ROI_ts))
    
    # Extract node features from the embeddings
    node_features = torch.tensor(np.concatenate((sub_conn_matrix, temporal_embeddings), axis=1), dtype=torch.float32)

    # Normalize edge attributes
    edge_attr = normalize(torch.tensor(sub_conn_matrix, dtype=torch.float32), dim=0)
    
    # Convert the list of numpy arrays to a single numpy array
    edge_index_np = np.array(adjacency_matrix.nonzero())

    # Create a tensor from the single numpy array
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    
    # Create PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(label, dtype=torch.long))
    
    return data

def load_subject_data(num_aug):
    label=[]
    subject_fc_matrices=np.load('source_data/fc/augmented_fc_matrices_10.npy')
    subject_time_series=np.load('source_data/time_series/augmented_time_series_10.npy')
    combined_data = [(fc_matrix, time_series) for fc_matrix, time_series in zip(subject_fc_matrices, subject_time_series)]
    print(f"Number of Subjects {len(subject_fc_matrices)}")
    for i in range(len(subject_fc_matrices)):
        if i <25*num_aug:
            label.append(0)
        # if i>= 25:
        #     label.append(1)
        if i >= 25*num_aug and i<48*num_aug:
            label.append(1)
        if i >=48*num_aug:
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
    return combined_data, all_labels,num_classes

def plot_metrics(num_epochs,train_accuracies,val_accuracies,train_losses,val_losses):
    # Plotting the losses and accuracies
    plt.figure(figsize=(10, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, (num_epochs) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, (num_epochs) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(1, (num_epochs) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, (num_epochs) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def compute_metrics(all_labels, all_preds, save_path='metrics/model'):
    """
    A function which computes various classification metrics and saves the results in .xlsx files.
    :param all_labels: Ground truth labels
    :param all_preds: Predicted labels
    :param save_path: The path to save the results
    :return: None
    """
    # compute the metrics
    mcc = metrics.matthews_corrcoef(y_true=all_labels, y_pred=all_preds)
    f1macro = metrics.f1_score(y_true=all_labels, y_pred=all_preds, average="macro")
    f1micro = metrics.f1_score(y_true=all_labels, y_pred=all_preds, average="micro")
    f1weighted = metrics.f1_score(y_true=all_labels, y_pred=all_preds, average="weighted")
    accuracy_score = metrics.accuracy_score(y_true=all_labels, y_pred=all_preds)
    balanced_accuracy_score = metrics.balanced_accuracy_score(y_true=all_labels, y_pred=all_preds)
    precision = metrics.precision_score(y_true=all_labels, y_pred=all_preds, average='macro')
    recall_score = metrics.recall_score(y_true=all_labels, y_pred=all_preds, average='macro')

    print(metrics.classification_report(y_true=all_labels, y_pred=all_preds))
    report = metrics.classification_report(y_true=all_labels, y_pred=all_preds, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_excel(f"{save_path}_classif_report_1.xlsx")

    # label_dictionary = classes
    cm = metrics.confusion_matrix(y_true=all_labels, y_pred=all_preds)
    cm_as_df = pd.DataFrame(cm, columns=sorted(set(all_labels)), index=sorted(set(all_labels)))
    cm_as_df.to_excel(f"{save_path}_confusion_matrix_1.xlsx")

    _metrics = {
        "MCC": mcc,
        "F1macro": f1macro,
        "F1micro": f1micro,
        "F1weighted": f1weighted,
        "Accuracy": accuracy_score,
        "Balanced_accuracy_score": balanced_accuracy_score,
        "precision": precision,
        "recall_score": recall_score
    }

    dfmetrics = pd.DataFrame.from_dict(_metrics, orient='index', columns=['Value'])
    dfmetrics.to_excel(f"{save_path}_metric_results_1.xlsx")
    print(dfmetrics)


