import sys
import os
import random
import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
import catboost as cat
import shap
# mamba install -c anaconda pandas numpy scikit-learn -y
# mamba install -c conda-forge imbalanced-learn matplotlib xgboost -y
# conda install -c conda-forge lightgbm
# conda install -c conda-forge catboost
# conda install -c conda-forge shap

'''
a single line in the manifest file:
1st place - trial_index: single integer, from 1 to 10, $(seq 1 10)
2nd place - model_choice: cat mlp cat_pca mlp_pca
3rd place - file_choice: fn_all fn_baby fn_mom
output - one CSV file for shap values per fold
'''
# parsing inputs from the shell script
trial_index = int(sys.argv[1])
model_choice = sys.argv[2]
file_choice = sys.argv[3]
output_name = str(trial_index) + '_' + str(model_choice) + '_' + str(file_choice)

####################################################################################################
# select file
if file_choice == 'fn_all':
    df = pd.read_csv('family_normalized_reads_all.csv')
    layer_num = 3
elif file_choice == 'fn_baby':
    df = pd.read_csv('family_normalized_reads_baby.csv')
    layer_num = 3
elif file_choice == 'fn_mom':
    df = pd.read_csv('family_normalized_reads_mom.csv')
    layer_num = 1
else:
    print('Input file name not found.')

####################################################################################################
# select data partition
# Case-Neg from T1&2, Case-Pos from T3, samples are from the same person
df.patientID = df.patientID.astype('category').cat.codes
metadata_list = ['sampleID', 'EverCovid', 'CovidStatus', 'CovidLabel', 'Timepoint']  # keep patientID for later use
df0 = df.loc[df['Timepoint'].isin([1, 2, 3])]
df1 = df0.loc[df0['CovidLabel'] == 1]  # case-negatives (CN)
df2 = df0.loc[df0['CovidLabel'] == 2]  # case-positives (CP)
# reformat y values from [1, 2] to [0, 1], to work with XGB and just in case
df1['CovidLabel'] = 0
df2['CovidLabel'] = 1
df_CN_CP = pd.concat([df1, df2], ignore_index=True)  # reset the index after concat
df_CN_CP_otu = df_CN_CP.drop(columns=metadata_list)
X = df_CN_CP_otu.to_numpy()
y = df_CN_CP['CovidLabel'].to_numpy()
groups = df_CN_CP.patientID.to_list()  # for StratifiedGroupKFold cv use
groups = [int(x) for x in groups]  # groups must be integer array

####################################################################################################
# log transform the OTU data, the last 3 columns are Age, HIVstatus, patientID
addons = np.min(X[:, :-3][np.nonzero(X[:, :-3])]) / 2
X[:, :-3] += addons
X[:, :-3] = np.log2(X[:, :-3])

####################################################################################################
# set up training process
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
n = int(trial_index)
set_seed(n)
score_df = []  # text output of the AUC scores
cv = StratifiedGroupKFold(n_splits=4, random_state=n, shuffle=True)  # get 75:25 split for each fold
for fold, (train_index, test_index) in enumerate(cv.split(X, y, groups)):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    if file_choice == 'fn_baby':
        X_train, y_train = ADASYN(random_state=n, n_neighbors=2).fit_resample(X_train, y_train)  # baby samples are too less, need smaller n_neighbors
    else:  # mom and all dataset are large enough
        X_train, y_train = ADASYN(random_state=n).fit_resample(X_train, y_train)  # patientID also increased
    if model_choice == 'cat':
        model = cat.CatBoostClassifier(
            random_seed=n,
            verbose=False,
            depth=3,
            iterations=3,
            learning_rate=0.0001,
        )
        model.fit(X_train[:, :-1], y_train)
        y_pred = model.predict_proba(X_test[:, :-1])
    elif model_choice == 'mlp':
        model = MLPClassifier(
            random_state=n,
            activation='identity',
            alpha=0.0001,
            hidden_layer_sizes=layer_num,
            learning_rate='constant',
            max_iter=30,
            solver='lbfgs',
        )
        X_train[:, -3] = X_train[:, -3] / X_train[:, -3].max(axis=0)  # scale Age column only
        model.fit(X_train[:, :-1], y_train)
        X_test[:, -3] = X_test[:, -3] / X_test[:, -3].max(axis=0)
        y_pred = model.predict_proba(X_test[:, :-1])
    else:
        pass
    explainer = shap.KernelExplainer(model.predict_proba, X_train[:, :-1])  # shap need the larger part of the data
    sv = explainer.shap_values(X_train[:, :-1])
    # shap.summary_plot(sv[1], X_train[:, :-1], max_display=7)  # show how covid positive label is decided
    aggregated_sv = np.abs(sv).mean(1)
    sv_df = pd.DataFrame(aggregated_sv.T)
    sv_top = sv_df.loc[sv_df.sum(1).sort_values(ascending=True).index[-20:]]  # save top 20 important features of each fold
    sv_top.to_csv(str(output_name) + '_' + str(fold) + '_shap_values.csv')