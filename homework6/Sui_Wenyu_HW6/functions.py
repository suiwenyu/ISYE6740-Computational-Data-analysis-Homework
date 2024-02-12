# import packages
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import functions

def read_spam_data():
    # Read data
    spam_data_df = pd.read_csv("data\spambase\spambase.data", delimiter=",", header=None)

    # Define column headers
    spam_col_header = [
        "word_freq_make",
        "word_freq_address",
        "word_freq_all",
        "word_freq_3d",
        "word_freq_our",
        "word_freq_over",
        "word_freq_remove",
        "word_freq_internet",
        "word_freq_order",
        "word_freq_mail",
        "word_freq_receive",
        "word_freq_will",
        "word_freq_people",
        "word_freq_report",
        "word_freq_addresses",
        "word_freq_free",
        "word_freq_business",
        "word_freq_email",
        "word_freq_you",
        "word_freq_credit",
        "word_freq_your",
        "word_freq_font",
        "word_freq_000",
        "word_freq_money",
        "word_freq_hp",
        "word_freq_hpl",
        "word_freq_george",
        "word_freq_650",
        "word_freq_lab",
        "word_freq_labs",
        "word_freq_telnet",
        "word_freq_857",
        "word_freq_data",
        "word_freq_415",
        "word_freq_85",
        "word_freq_technology",
        "word_freq_1999",
        "word_freq_parts",
        "word_freq_pm",
        "word_freq_direct",
        "word_freq_cs",
        "word_freq_meeting",
        "word_freq_original",
        "word_freq_project",
        "word_freq_re",
        "word_freq_edu",
        "word_freq_table",
        "word_freq_conference",
        "char_freq_;",
        "char_freq_(",
        "char_freq_[",
        "char_freq_!",
        "char_freq_$",
        "char_freq_#",
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
        "Class"]

    spam_data_df.columns = spam_col_header
    return(spam_data_df)




def calculate_test_error(clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    error = (cm[0,1] + cm[1,0]) / y_test.shape[0]

    return error


def calculate_MSE(X_train, X_test, beta_star, sigma_sq):
    lam_range = np.arange(0.1, 10, 0.1)
    bias_sq_list = []
    var_list = []
    MSE_list = []

    for lam in lam_range:
        M = np.dot(np.linalg.inv(np.dot(X_train.T, X_train) + lam * np.identity(X_train.shape[1])), X_train.T)

        # Calculate Bias and Variance
        bias = X_test.T @ M @ X_train @ beta_star - X_test.T @ beta_star
        bias_sq = np.square(bias)
        bias_sq_list += [float(bias_sq)]

        # calculate variance
        var = sigma_sq * X_test.T @ M @ M.T @ X_test
        var_list += [float(var)]

        MSE_list += [float(bias_sq) + float(var)]

    plt.figure(figsize=(6, 4))
    plt.plot(lam_range, bias_sq_list, label="bias squared")
    plt.plot(lam_range, var_list, label="variance")
    plt.plot(lam_range, MSE_list, label="MSE")
    plt.title("Bias Squared, Variance and MSE")
    plt.legend()
    plt.show()