import numpy as np
import pandas as pd
from utils import load_data, read_lookup_table
import scipy.io as sio
import scipy.sparse as sp
import random
import tensorflow as tf


def adj_to_bias(adj, sizes, nhood = 1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0

    return -1e9 * (1.0-mt)


def load_data1(multiplication_factor):
    exp_file = 'Data/mDC/TF+1000 Nonspecific/mDC1000-ExpressionData.csv'
    lookup_file = 'Data/mDC/TF+1000 Nonspecific/Transformed_Train_set_lookup_table.csv'

    data_input = pd.read_csv(exp_file, index_col=0)
    print("Initial feature values (before normalization and lookup):")
    print(data_input.iloc[:5, :5])

    geneName = data_input.index
    loader = load_data(data_input)
    normalized_data = loader.exp_data()

    print("Normalized feature values:")
    print(normalized_data[:5, :5])

    lookup_dict = read_lookup_table(lookup_file)
    positive_values = [value for value in lookup_dict.values() if value > 0]

    if positive_values:
        percentiles = {
            '90': np.percentile(positive_values, 90),
            '80': np.percentile(positive_values, 80),
            '70': np.percentile(positive_values, 70),
            '60': np.percentile(positive_values, 60),
            '50': np.percentile(positive_values, 50),
            '40': np.percentile(positive_values, 40),
            '30': np.percentile(positive_values, 30),
            '20': np.percentile(positive_values, 20),
            '10': np.percentile(positive_values, 10),
        }
    else:
        percentiles = {key: 0 for key in ['90', '80', '70', '60', '50', '40', '30', '20', '10']}

    # Construct feature array using the percentiles and multiplication factor
    feature = np.array([
        row + multiplication_factor * percentiles['90'] if lookup_dict.get(geneName[i], 0) > percentiles['90'] else (
            row + multiplication_factor * percentiles['80'] if lookup_dict.get(geneName[i], 0) > percentiles['80'] else (
                row + multiplication_factor * percentiles['70'] if lookup_dict.get(geneName[i], 0) > percentiles['70'] else (
                    row + multiplication_factor * percentiles['60'] if lookup_dict.get(geneName[i], 0) > percentiles['60'] else (
                        row + multiplication_factor * percentiles['50'] if lookup_dict.get(geneName[i], 0) > percentiles['50'] else (
                            row - multiplication_factor * percentiles['50'] if lookup_dict.get(geneName[i], 0) > percentiles['40'] else (
                                row - multiplication_factor * percentiles['60'] if lookup_dict.get(geneName[i], 0) > percentiles['30'] else (
                                    row - multiplication_factor * percentiles['70'] if lookup_dict.get(geneName[i], 0) > percentiles['20'] else (
                                        row - multiplication_factor * percentiles['80'] if lookup_dict.get(geneName[i], 0) > percentiles['10'] else (
                                            row - multiplication_factor * percentiles['90'])))))))))
    for i, row in enumerate(normalized_data)
    ])

    print("Feature values after applying lookup table:")
    print(feature[:5, :5])

    geneNum = feature.shape[0]

    train_file = 'Data/Train_validation_test/mDC 1000 Nonspecific/Train_set.csv'
    test_file = 'Data/Train_validation_test/mDC 1000 Nonspecific/Test_set.csv'
    val_file = 'Data/Train_validation_test/mDC 1000 Nonspecific/Validation_set.csv'
    train_data = pd.read_csv(train_file, index_col=0).values
    validation_data = pd.read_csv(val_file, index_col=0).values
    test_data = pd.read_csv(test_file, index_col=0).values

    train_data = train_data[np.lexsort(-train_data.T)]
    train_index = np.sum(train_data[:, 2])

    validation_data = validation_data[np.lexsort(-validation_data.T)]
    validation_index = np.sum(validation_data[:, 2])

    test_data = test_data[np.lexsort(-test_data.T)]
    test_index = np.sum(test_data[:, 2])

    logits_train = sp.csr_matrix((train_data[0:train_index, 2], (train_data[0:train_index, 0], train_data[0:train_index, 1])),
                                 shape=(geneNum, geneNum)).toarray()
    neg_logits_train = sp.csr_matrix((np.ones(train_data[train_index:, 2].shape), (train_data[train_index:, 0], train_data[train_index:, 1])),
                                     shape=(geneNum, geneNum)).toarray()
    interaction = logits_train
    interaction = interaction + np.eye(interaction.shape[0])
    interaction = sp.csr_matrix(interaction)
    logits_train = logits_train.reshape([-1, 1])
    neg_logits_train = neg_logits_train.reshape([-1, 1])

    logits_test = sp.csr_matrix((test_data[0:test_index, 2], (test_data[0:test_index, 0], test_data[0:test_index, 1])),
                                shape=(geneNum, geneNum)).toarray()
    neg_logits_test = sp.csr_matrix((np.ones(test_data[test_index:, 2].shape), (test_data[test_index:, 0], test_data[test_index:, 1])),
                                    shape=(geneNum, geneNum)).toarray()
    logits_test = logits_test.reshape([-1, 1])
    neg_logits_test = neg_logits_test.reshape([-1, 1])
    logits_validation = sp.csr_matrix((validation_data[0:validation_index, 2], (validation_data[0:validation_index, 0], validation_data[0:validation_index, 1])),
                                      shape=(geneNum, geneNum)).toarray()
    neg_logits_validation = sp.csr_matrix(
        (np.ones(validation_data[validation_index:, 2].shape), (validation_data[validation_index:, 0], validation_data[validation_index:, 1])),
        shape=(geneNum, geneNum)).toarray()
    logits_validation = logits_validation.reshape([-1, 1])
    neg_logits_validation = neg_logits_validation.reshape([-1, 1])

    train_mask = np.array(logits_train[:, 0], dtype=bool).reshape([-1, 1])
    test_mask = np.array(logits_test[:, 0], dtype=bool).reshape([-1, 1])
    validation_mask = np.array(logits_validation[:, 0], dtype=bool).reshape([-1, 1])

    return geneName, feature, logits_train, logits_test, logits_validation, train_mask, test_mask, validation_mask, interaction, neg_logits_train, neg_logits_test, neg_logits_validation, train_data, validation_data, test_data


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    adj_normalized = degree_mat_inv_sqrt.dot(adj_)
    adj_normalized = adj_normalized.dot(degree_mat_inv_sqrt).tocoo()

    return sparse_to_tuple(adj_normalized)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
