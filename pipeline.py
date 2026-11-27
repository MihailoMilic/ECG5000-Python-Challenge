import numpy as np
import pandas as pd

def data_prep():

    data_train = np.loadtxt("ECG5000/ECG5000_TRAIN.txt")
    data_test = np.loadtxt("ECG5000/ECG5000_TEST.txt")

    df_train = pd.DataFrame(data_train, columns=["class"] + [f"value_{i}" for i in range(1, data_train.shape[1])])
    df_test = pd.DataFrame(data_test, columns=["class"] + [f"value_{i}" for i in range(1, data_test.shape[1])])

    df_train["binary_label"] = (df_train["class"] != 1).astype(int)

    df_normal = df_train[df_train["binary_label"] == 0]
    df_abnormal = df_train[df_train["binary_label"] == 1]

    X_normal = df_normal.iloc[:, 1:-1].to_numpy(np.float16)
    X_abnormal = df_abnormal.iloc[:, 1:-1].to_numpy(np.float16)
    r,c = X_normal.shape
    X_train = X_normal[:r // 10/7, :]
    X_val = np.stack(X_normal[r//10/7:,:],X_abnormal)


