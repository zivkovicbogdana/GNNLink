import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import time
from inits import load_data1, preprocess_graph
from Transorfomer import TransorfomerModel

epochs = 100
batch_size = 1
window_size = 10
average_last_n = 10


def Main(multiplication_factors, seed=123):
    best_factor = None
    last_epoch_aucs = {}

    for multiplication_factor in multiplication_factors:
        tf.compat.v1.reset_default_graph()
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

        geneName, feature, logits_train, logits_test, logits_validation, train_mask, test_mask, validation_mask, interaction, neg_logits_train, neg_logits_test, neg_logits_validation, train_data, validation_data, test_data = load_data1(
            multiplication_factor)
        biases = preprocess_graph(interaction)
        model = TransorfomerModel(feature, do_train=False)

        with tf.compat.v1.Session() as sess:
            init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
            sess.run(init_op)

            auc_window = []
            auc_log = []
            mf_log = []

            for epoch in range(epochs):
                t = time.time()
                train_loss_avg = 0
                train_acc_avg = 0

                ######## train #########
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

                ######## validate #########
                score, _, _ = sess.run([model.logits, model.loss, model.accuracy],
                                       feed_dict={
                                           model.encoded_gene: feature,
                                           model.bias_in: biases,
                                           model.lbl_in: logits_validation,
                                           model.msk_in: validation_mask,
                                           model.neg_msk: neg_logits_validation
                                       })
                score = score.reshape((feature.shape[0], feature.shape[0]))
                auc_val, aupr_val = evaluate(validation_data, score)

                auc_window.append(auc_val)
                if len(auc_window) > window_size:
                    auc_window.pop(0)

                # Calculate the average of the last n AUC values in the window
                if len(auc_window) >= average_last_n:
                    avg_auc_val = np.mean(auc_window[-average_last_n:])
                else:
                    avg_auc_val = np.mean(auc_window)

                auc_log.append(avg_auc_val)
                mf_log.append(multiplication_factor)

                print(
                    f"Epoch {epoch}: avg_auc_val = {avg_auc_val}, multiplication_factor = {multiplication_factor}")
                print(
                    "Epoch: %04d | Training: loss = %.5f, acc = %.5f, auc = %.5f, aupr = %.5f, time = %.5f" % (
                        epoch, train_loss_avg, train_acc_avg, auc_val, aupr_val, time.time() - t))

            print("Finish training for factor:", multiplication_factor)

            # Store the last validation AUC for this factor
            last_epoch_aucs[multiplication_factor] = auc_log[-1]

            # Log the changes in AUC and multiplication factor for later analysis
            log_results(auc_log, mf_log, multiplication_factor)

    # Determine the best multiplication factor based on the last validation AUC
    best_factor = max(last_epoch_aucs, key=last_epoch_aucs.get)
    print(f"Best multiplication factor based on last validation AUC: {best_factor}")
    return best_factor, last_epoch_aucs


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

    return auc_val, aupr_val


def log_results(auc_log, mf_log, factor):
    df = pd.DataFrame({
        'epoch': list(range(len(auc_log))),
        'avg_auc_val': auc_log,
        'multiplication_factor': mf_log
    })
    df.to_csv(f'training_log_factor_{factor}.csv', index=False)
    print(f"Training log saved to training_log_factor_{factor}.csv")


seed = 123
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

T = 1
cv_num = 1

multiplication_factors = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]  # List of factors to test
for t in range(T):
    aupr_vec = []
    auroc_ver = []
    best_factor, last_epoch_aucs = Main(multiplication_factors, seed)

    # Print out the last validation AUC for each multiplication factor
    for factor, performance in last_epoch_aucs.items():
        print(f"Multiplication factor: {factor}, Last validation AUC: {performance}")

    # Optionally: Plot the results
    plt.figure()
    factors = list(last_epoch_aucs.keys())
    performances = list(last_epoch_aucs.values())
    plt.plot(factors, performances, marker='o')
    plt.xlabel('Multiplication Factor')
    plt.ylabel('Last Validation AUC')
    plt.title('Performance of Different Multiplication Factors')
    plt.show()

    # Testing with the best multiplication factor
    tf.compat.v1.reset_default_graph()
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    geneName, feature, logits_train, logits_test, logits_validation, train_mask, test_mask, validation_mask, interaction, neg_logits_train, neg_logits_test, neg_logits_validation, train_data, validation_data, test_data = load_data1(
        best_factor)
    biases = preprocess_graph(interaction)
    model = TransorfomerModel(feature, do_train=False)

    with tf.compat.v1.Session() as sess:
        init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        sess.run(init_op)

        ts_size = 1
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        print("Start to test with the best multiplication factor")
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

        # Ensure consistent length between true labels and predicted scores
        true_labels = test_data[:, 2]
        predicted_scores = out_come[test_data[:, 0].astype(int), test_data[:, 1].astype(int)]

        # Evaluate the final test performance
        auc_val, aupr_val = evaluate(test_data, out_come)
        print("Final Test AUC: %.6f, AUPR: %.6f" % (auc_val, aupr_val))

        # Optionally: Plot the final ROC and PR curves
        fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

        prec, rec, _ = precision_recall_curve(true_labels, predicted_scores)
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.show()
