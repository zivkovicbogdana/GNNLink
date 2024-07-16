import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from scipy.stats import pearsonr
from Transorfomer import TransorfomerModel  # Assuming this is your model class
from inits import load_data1, preprocess_graph  # Assuming these are your utility functions
import time

# Constants
epochs = 100
batch_size = 1
T = 1
cv_num = 1
seed = 123

def Main(cell_type, gene_number, network_type):
    # Define the file paths
    data_path = f'Data/{cell_type}/TF+{gene_number} {network_type}/{cell_type}{gene_number}-ExpressionData.csv'
    lookup_file = f'Data/{cell_type}/TF+{gene_number} {network_type}/Transformed_Train_set_lookup_table.csv'
    tf_file = f'Data/{cell_type}/TF+{gene_number} {network_type}/TF.csv'

    # Check if files exist
    if not (os.path.exists(data_path) and os.path.exists(lookup_file) and os.path.exists(tf_file)):
        print(f"Files for {cell_type} with {gene_number} genes and {network_type} network type are missing. Skipping...")
        return None, None, None

    geneName, feature, logits_train, logits_test, logits_validation, train_mask, test_mask, validation_mask, interaction, neg_logits_train, neg_logits_test, neg_logits_validation, train_data, validation_data, test_data = load_data1(cell_type, gene_number, network_type)
    biases = preprocess_graph(interaction)
    model = TransorfomerModel(feature, do_train=False)

    with tf.compat.v1.Session() as sess:
        init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        sess.run(init_op)
        train_loss_avg = 0
        train_acc_avg = 0

        for epoch in range(epochs):
            t = time.time()
            tr_step = 0
            tr_size = 1
            if tr_step * batch_size < tr_size:
                _, loss_value_tr, acc_tr = sess.run([model.train_op, model.loss, model.accuracy],
                                                    feed_dict={
                                                        model.encoded_gene: feature,
                                                        model.bias_in: biases,
                                                        model.lbl_in: logits_train,
                                                        model.msk_in: train_mask,
                                                        model.neg_msk: neg_logits_train
                                                    })
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            score, _, _ = sess.run([model.logits, model.loss, model.accuracy],
                                   feed_dict={
                                       model.encoded_gene: feature,
                                       model.bias_in: biases,
                                       model.lbl_in: logits_validation,
                                       model.msk_in: validation_mask,
                                       model.neg_msk: neg_logits_validation
                                   })
            score = score.reshape((feature.shape[0], feature.shape[0]))
            auc_val, aupr_val, rec, prec = evaluate(validation_data, score)
            print("Epoch: %04d | Training: loss = %.5f, acc = %.5f, auc = %.5f, aupr = %.5f, time = %.5f" % (
                epoch, train_loss_avg, train_acc_avg, auc_val, aupr_val, time.time() - t))

        print("Finish training.")

        ts_size = 1
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        print("Start to test")
        while ts_step * batch_size < ts_size:
            out_come, loss_value_ts, acc_ts = sess.run([model.logits, model.loss, model.accuracy],
                                                       feed_dict={
                                                           model.encoded_gene: feature,
                                                           model.bias_in: biases,
                                                           model.lbl_in: logits_test,
                                                           model.msk_in: test_mask,
                                                           model.neg_msk: neg_logits_test})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1
        print('Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step)

        out_come = out_come.reshape((feature.shape[0], feature.shape[0]))

        return geneName, out_come, test_data

def calculate_errors_for_tfs(temp_pre, rel_test_label):
    errors = {}
    for tf_index in np.unique(rel_test_label[:, 0]):
        tf_pre = temp_pre[temp_pre[:, 0] == tf_index]
        if tf_pre.size > 0:
            true_labels = rel_test_label[rel_test_label[:, 0] == tf_index][:, 2]
            predicted_probs = tf_pre[:, 2]
            tf_errors = (true_labels - predicted_probs) ** 2
            errors[tf_index] = np.mean(tf_errors)

            # Print labels and predictions for the current TF
            # print(f"TF Index: {tf_index}")
            # print(f"True Labels: {len(true_labels)}")
            # print(f"True Labels: {true_labels}")
            # print(f"Predicted Probabilities: {len(predicted_probs)}")
            # print(f"Predicted Probabilities: {predicted_probs}")
            # print("\n")
    return errors

def evaluate(rel_test_label, pre_test_label):
    temp_pre = []
    for i in range(rel_test_label.shape[0]):
        l = []
        m = rel_test_label[i, 0]
        n = rel_test_label[i, 1]
        l.append(m)
        l.append(n)
        l.append(pre_test_label[m, n])
        temp_pre.append(l)
    temp_pre = np.asarray(temp_pre)
    prec, rec, thr = precision_recall_curve(rel_test_label[:, 2], temp_pre[:, 2])
    aupr_val = auc(rec, prec)
    fpr, tpr, thr = roc_curve(rel_test_label[:, 2], temp_pre[:, 2])
    auc_val = auc(fpr, tpr)

    return auc_val, aupr_val, rec, prec

def run_all_combinations():
    cell_types = ['hESC', 'hHEP', 'mESC', 'mDC']
    gene_numbers = [500, 1000]
    network_types = ['STRING', 'Cellspecific', 'Nonspecific', 'LOFGOF']

    results = []

    for cell_type in cell_types:
        for gene_number in gene_numbers:
            for network_type in network_types:
                try:
                    lookup_file = f'Data/{cell_type}/TF+{gene_number} {network_type}/Transformed_Train_set_lookup_table.csv'
                    lookup_df = pd.read_csv(lookup_file)
                    tf_file = f'Data/{cell_type}/TF+{gene_number} {network_type}/TF.csv'
                    tf_df = pd.read_csv(tf_file)
                    tf_name_to_index = dict(zip(tf_df['TF'], tf_df['index']))
                except FileNotFoundError:
                    print(f"Files for {cell_type} with {gene_number} genes and {network_type} network not found. Skipping.")
                    continue

                aupr_vec = []
                auroc_vec = []
                geneName, pre_test_label, rel_test_label = Main(cell_type, gene_number, network_type)

                if geneName is None:
                    continue  # Skip to the next combination if files are missing

                temp_pre = []
                for i in range(rel_test_label.shape[0]):
                    m = rel_test_label[i, 0]
                    n = rel_test_label[i, 1]
                    l = [m, n, pre_test_label[m, n]]
                    temp_pre.append(l)
                temp_pre = np.asarray(temp_pre)
                errors = calculate_errors_for_tfs(temp_pre, rel_test_label)

                errors_df = pd.DataFrame(list(errors.items()), columns=['TF_index', 'error'])

                errors_df['TF'] = errors_df['TF_index'].map({v: k for k, v in tf_name_to_index.items()})

                merged_df = pd.merge(errors_df, lookup_df, on='TF', how='left')

                filtered_df = merged_df.dropna(subset=['Normalized_Count'])

                output_directory = os.path.dirname(lookup_file)

                plt.figure(figsize=(10, 6))
                plt.scatter(filtered_df['Normalized_Count'], filtered_df['error'], alpha=0.5)
                plt.xlabel('Connectivity (Normalized Count)')
                plt.ylabel('Mean Squared Error')
                plt.title(f'Scatter Plot of Connectivity vs Error for {cell_type} {gene_number} {network_type}')
                plt.grid(True)
                scatter_plot_path = os.path.join(output_directory,
                                                 f'scatter_plot_{cell_type}_{gene_number}_{network_type}.png')
                plt.savefig(scatter_plot_path)
                plt.close()

                correlation = filtered_df[['Normalized_Count', 'error']].corr().iloc[0, 1]
                print(f"Correlation between Connectivity and Error for {cell_type} {gene_number} {network_type}: {correlation}")

                corr_coef, p_value = pearsonr(filtered_df['Normalized_Count'], filtered_df['error'])
                print(f"Pearson correlation coefficient: {corr_coef} for {cell_type} {gene_number} {network_type}")
                print(f"P-value: {p_value} for {cell_type} {gene_number} {network_type}")

                prec, rec, _ = precision_recall_curve(rel_test_label[:, 2], temp_pre[:, 2])
                aupr_val = auc(rec, prec)
                aupr_vec.append(aupr_val)
                print(f"AUPR for {cell_type} {gene_number} {network_type}: {aupr_val}")
                plt.figure()
                plt.plot(rec, prec, label=f'AUPR = {aupr_val:.2f}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend(loc='lower left')
                pr_curve_path = os.path.join(output_directory,
                                             f'pr_curve_{cell_type}_{gene_number}_{network_type}.png')
                plt.savefig(pr_curve_path)
                plt.close()

                fpr, tpr, _ = roc_curve(rel_test_label[:, 2], temp_pre[:, 2])
                auc_val = auc(fpr, tpr)
                auroc_vec.append(auc_val)
                print(f"AUROC for {cell_type} {gene_number} {network_type}: {auc_val}")
                plt.figure()
                plt.plot(fpr, tpr, label=f'AUROC = {auc_val:.2f}')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic Curve')
                plt.legend(loc='lower right')
                roc_curve_path = os.path.join(output_directory,
                                              f'roc_curve_{cell_type}_{gene_number}_{network_type}.png')
                plt.savefig(roc_curve_path)
                plt.close()

                # Store the results for this combination
                results.append({
                    'Cell Type': cell_type,
                    'Gene Number': gene_number,
                    'Network Type': network_type,
                    'AUPR': aupr_val,
                    'AUROC': auc_val
                })

    # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    results_file = 'results_summary.csv'
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")


run_all_combinations()
