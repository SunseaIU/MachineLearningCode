from sklearn import metrics
import numpy as np
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]

'''
Parameters:
labels_true:int array-like of shape (n_samples,). Ground truth class labels to be used as a reference.
labels_pred:int array-like of shape (n_samples,). Cluster labels to evaluate.

Returns:
RI:float
Similarity score between 0.0 and 1.0, 1.0 stands for perfect match.
'''
RI=metrics.rand_score(labels_true, labels_pred)

'''
Parameters:
labels_true:int array-like of shape (n_samples,). Ground truth class labels to be used as a reference.
labels_pred:int array-like of shape (n_samples,). Cluster labels to evaluate.

Returns:
ARI:float
Similarity score between -0.5 and 1.0, Random labelings have an ARI close to 0.0, 1.0 stands for perfect match.
'''
ARI=metrics.adjusted_rand_score(labels_true, labels_pred)

'''
Parameters:
labels_true:int array-like of shape (n_samples,). Ground truth class labels to be used as a reference.
labels_pred:int array-like of shape (n_samples,). Cluster labels to evaluate.

Returns:
NMI:float
Score between 0.0 and 1.0 in normalized nats (based on the natural logarithm), 1.0 stands for perfectly complete labeling.
'''
NMI=metrics.normalized_mutual_info_score(labels_true, labels_pred)

'''
Parameters:
labels_true:int array-like of shape (n_samples,). Ground truth class labels to be used as a reference.
labels_pred:int array-like of shape (n_samples,). Cluster labels to evaluate.

Returns:
Purity:float
Similarity score between 0.0 and 1.0, 1.0 indicates that each sample is grouped into a separate cluster.
'''
def purity(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]
Purity=purity(labels_true, labels_pred)

print("RI is %.2f，ARI is %.2f，NMI is %.2f，Purity is %.2f" %(RI,ARI,NMI,Purity))