import pandas as pd
import numpy as np
from pandas.core.indexing import convert_from_missing_indexer_tuple
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay


def get_binary_metircs(y_true, y_predict):
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true, y_predict, average="binary", zero_division=0
    )
    return precision, recall, fscore


def get_confusion_matrix(y_true, y_predict):
    cm = confusion_matrix(y_true, y_predict)
    tn, fp, fn, tp = cm.ravel()
    return cm, tn, fp, fn, tp


def get_actual_buy_time(df, target_window, target_threshold):
    arr_predict = df["predict"].values
    arr_buy = df["buy"].values
    actual_buy = np.zeros(df.shape[0])

    idx_predict = np.where(arr_predict == 1)[0]
    jump_idx = -1
    for i in idx_predict:
        if i > jump_idx:
            actual_buy[i] = 1
            for j in range(i + 1, i + 1 + target_window):
                jump_idx = j
                if arr_buy[j] / arr_buy[i] >= 1 + target_threshold:
                    break
    return actual_buy.astype(int)


def estimate_earnings(df, target_threshold):
    # df's columns: ["buy", "sell_last", "sell_max", "earn_last", "earn_max", "target", "predict", "actual_buy"]

    idx_target = (df["target"] == 1).values
    idx_predict = (df["predict"] == 1).values
    idx_actual = (df["actual_buy"] == 1).values

    idx_tp = idx_target & idx_predict
    idx_fp = ~idx_target & idx_predict
    idx_tp_a = idx_target & idx_actual
    idx_fp_a = ~idx_target & idx_actual

    god_avg_earnings = df[idx_target]["earn_max"].mean()
    # model_best_avg_earnings = (df[idx_tp]["earn_max"].sum() + df[idx_fp]["earn_last"].sum()) / sum(idx_predict)
    model_real_avg_earnings = (sum(idx_tp) * target_threshold + df[idx_fp]["earn_last"].sum()) / sum(idx_predict)
    model_actual_buy_avg_earnings = (sum(idx_tp_a) * target_threshold + df[idx_fp_a]["earn_last"].sum()) / sum(
        idx_actual
    )

    idx_neg_earnings = idx_fp_a & (df["earn_last"] < 0).values
    neg_earnings = df[idx_neg_earnings]["earn_last"].values
    neg_earnings_time = df[idx_neg_earnings].index.tolist()

    return (
        god_avg_earnings,
        # model_best_avg_earnings,
        model_real_avg_earnings,
        model_actual_buy_avg_earnings,
        neg_earnings,
        neg_earnings_time,
    )


def show_evaluations(df, target_threshold):
    # df's columns: ["buy", "sell_last", "sell_max", "earn_last", "earn_max", "target", "predict"]

    (
        god_avg_earnings,
        # model_best_avg_earnings,
        model_real_avg_earnings,
        model_actual_buy_avg_earnings,
        neg_earnings,
        neg_earnings_time,
    ) = estimate_earnings(df, target_threshold)
    precision, recall, fscore = get_binary_metircs(df["target"], df["predict"])
    cm, _, fp, _, tp = get_confusion_matrix(df["target"], df["predict"])
    disp_cm = ConfusionMatrixDisplay(cm)
    print("- Target Earning Rate per Transaction: {:.3f}".format(target_threshold))
    print("- God's Average Earning Ratio: {:.5f}".format(god_avg_earnings))
    # print("- Our Model's Best Average Earning Ratio: {:.5f}".format(model_best_avg_earnings))
    print("- Our Model's Average Earning Ratio: {:.5f}".format(model_real_avg_earnings))
    print(
        "- Our Model's Actual Buy Average Earning Ratio: {:.5f}, {}".format(
            model_actual_buy_avg_earnings, df["actual_buy"].sum()
        )
    )
    print("- Negative Transactions: {}, \n  {}".format(len(neg_earnings), neg_earnings))
    print("- Precions {:.3f}, Recall: {:.3f}, F1Score: {:.3f}".format(precision, recall, fscore))
    print("- Confustion Matrix")
    disp_cm.plot(colorbar=False)
