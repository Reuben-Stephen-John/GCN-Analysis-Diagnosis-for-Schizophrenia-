from karateclub import NetMF
import torch
from torch_geometric.data import Data
import numpy as np
import networkx as nx
from tsfresh import select_features, extract_features,feature_extraction
from tsfresh.utilities.dataframe_functions import impute
from sklearn import metrics
import pandas as pd


def extract_tsfresh_features(x):
    """
    For MRI data sets. Function used for node features extraction from node time series. The features to be computed
    from tsfresh package are defined in functions_to_test dictionary
    :param x: numpy array containing a time series
    :return: a list of values of computed features
    """

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
    for node in range(len(subject_time_series[0,:])):
        feature=extract_tsfresh_features(subject_time_series[:,node])
        temporal_features.append(feature)
    return np.array(temporal_features)

def create_data_object(sub_conn_matrix, sub_ROI_ts, label):
    threshold = -2
    adjacency_matrix = np.where(sub_conn_matrix < threshold, 0, sub_conn_matrix)
    # adjacency_matrix = (sub_conn_matrix > threshold).astype(float)
    graph = nx.from_numpy_matrix(adjacency_matrix)
    
    netmf_embeddings = netmf_embedding(graph)  # Replace this with your actual embedding logic
    temporal_embeddings = temporal_feature_extraction(sub_ROI_ts)
    
    # Extract node features from the embeddings
    node_features = torch.tensor(np.concatenate((netmf_embeddings, temporal_embeddings), axis=1), dtype=torch.float32)
    # node_features = torch.tensor(temporal_embeddings, dtype=torch.float32)
    # Extract edge attributes from the connection matrix
    edge_attr = torch.tensor(sub_conn_matrix, dtype=torch.float32)
    
    edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)
    
    # Create PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(label, dtype=torch.long))
    
    return data

def load_subject_data():
    label=[]
    subject_fc_matrices=np.load('source_data/fc/fc_matrices_1.npy')
    subject_time_series=np.load('source_data/time_series/time_series.npy')
    combined_data = [(fc_matrix, time_series) for fc_matrix, time_series in zip(subject_fc_matrices, subject_time_series)]
    print(f"Number of Subjects {len(subject_fc_matrices)}")
    for i in range(len(subject_fc_matrices)):
        if i <25:
            label.append(0)
        if i>= 25:
            label.append(1)
        # if i >= 25 and i<48:
        #     label.append(1)
        # if i >=48:
        #     label.append(2)

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

def compute_metrics(df_res, test_run_name, dataset_name):
    """
    A function which computes various classification metrics from a precomputed pandas data frame (in .xlsx format)
    :param df_res: pandas data frame in .xlsx format with columns: "trues" - ground truth labels, "preds" - predicted
    labels
    :param test_run_name: the name of the test run which the results data frame is representing
    :return: None, but saves .xlsx with metric results on disk
    """
    save_path = f"./{directories['training visualizations']}/{dataset_name}/{test_run_name}"
    try:
        os.makedirs(f"./{directories['training visualizations']}/{dataset_name}")
    except FileExistsError:
        pass
    df_res.to_excel(f"{save_path}_trues_preds.xlsx")

    # code borrowed from https://gist.github.com/nickynicolson/202fe765c99af49acb20ea9f77b6255e
    def cm2df(cm, labels):
        df = pd.DataFrame()
        # rows
        for i, row_label in enumerate(labels):
            rowdata = {}
            # columns
            for j, col_label in enumerate(labels):
                rowdata[col_label] = cm[i, j]
            df = df.append(pd.DataFrame.from_dict({row_label: rowdata}, orient='index'))
        return df[labels]

    # define classes and indexes of true values for each class. For each model the true index values are the
    # same since the test set was the same.
    classes = set(df_res["trues"])
    cls_index = dict()
    for cls in classes:
        cls_index[cls] = df_res[df_res["trues"] == cls].index.to_list()

    # compute the metrics
    allmetrics = dict()
    model_metrics = dict()
    mcc = metrics.matthews_corrcoef(y_true=df_res["trues"], y_pred=df_res["preds"])
    f1macro = metrics.f1_score(y_true=df_res["trues"], y_pred=df_res["preds"], average="macro")
    f1micro = metrics.f1_score(y_true=df_res["trues"], y_pred=df_res["preds"], average="micro")
    f1weighted = metrics.f1_score(y_true=df_res["trues"], y_pred=df_res["preds"], average="weighted")
    accuracy_score = metrics.accuracy_score(y_true=df_res["trues"], y_pred=df_res["preds"])
    balanced_accuracy_score = metrics.balanced_accuracy_score(y_true=df_res["trues"], y_pred=df_res["preds"])
    precision = metrics.precision_score(y_true=df_res["trues"], y_pred=df_res["preds"], average='macro')
    recall_score = metrics.recall_score(y_true=df_res["trues"], y_pred=df_res["preds"], average='macro')

    print(metrics.classification_report(y_true=df_res["trues"], y_pred=df_res["preds"]))
    report = metrics.classification_report(y_true=df_res["trues"], y_pred=df_res["preds"], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_excel(f"{save_path}_classif_report.xlsx")

    # label_dictionary = classes
    cm = metrics.confusion_matrix(y_true=df_res["trues"], y_pred=df_res["preds"])
    cm_as_df = cm2df(cm, classes)
    cm_as_df.to_excel(f"{save_path}_confusion_matrix.xlsx")

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
    for metric in _metrics.keys():
        model_metrics[metric] = _metrics[metric]

    allmetrics[0] = model_metrics

    dfmetrics = pd.DataFrame.from_dict(allmetrics)
    dfmetrics.to_excel(f"{save_path}_metric_results.xlsx")
    print(dfmetrics)
