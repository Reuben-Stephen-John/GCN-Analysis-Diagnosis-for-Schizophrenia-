from nilearn import datasets, maskers, plotting,image,connectome
import matplotlib.pyplot as plt
import numpy as np
from bids.layout import BIDSLayout
import networkx as nx
from sklearn.metrics import pairwise_distances

#base directory
def create_fc_matrices():
    fmriprep_dir='../Full-Dataset'
    layout=BIDSLayout(fmriprep_dir)
    atlas_ho = datasets.fetch_atlas_aal('SPM12')
    # Location of HarvardOxford parcellation atlas
    atlas_file = atlas_ho.maps

    print(f'Number of labels in the atlas: {len(atlas_ho.labels)}\n')

    fmri_files=layout.get(datatype='func',extension='.nii.gz',return_type="file")

    print(f'Number of Subjects in the Dataset:{len(fmri_files)}\n')

    masker = maskers.NiftiLabelsMasker(labels_img=atlas_file, standardize=True, verbose=1,
                            memory="nilearn_cache", memory_level=2)

    subject_corr=[]

    for file in fmri_files:
        fmri_data = image.load_img(fmri_files[0])
        time_series = masker.fit_transform(fmri_data)
        corr_matrix=connectome.ConnectivityMeasure(kind='correlation', standardize='zscore_sample').fit_transform([time_series])[0]
        np.fill_diagonal(corr_matrix, 0)
        subject_corr.append(corr_matrix)
    
    return subject_corr


def create_knn_graph(fc_matrices, k):
    # Calculate group FC matrix by averaging individual FC matrices
    group_fc_matrix = np.mean(fc_matrices, axis=0)

    # Calculate pairwise correlation coefficients
    correlation_matrix = np.corrcoef(group_fc_matrix, rowvar=False)

    # Create k-NN graph
    G = nx.Graph()
    num_nodes = correlation_matrix.shape[0]

    for i in range(num_nodes):
        # Find the indices of the top k edges in terms of connectivity strength
        top_k_indices = np.argsort(correlation_matrix[i, :])[-k:]

        # Add edges to the graph
        for j in top_k_indices:
            if i != j:  # Avoid self-loops
                G.add_edge(i, j, weight=correlation_matrix[i, j])

    return G
