import sys
import pandas as pd
import numpy as np
import sklearn
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, RocCurveDisplay, auc
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


'''
a single line in the manifest file:
1st place - loop index: from 1 to 50, single integer
2nd place - model choice: RF, SVM, XGB
3rd place - data partition: all, NN_CP, CN_CP
4th place - sampling: none, smote, adasyn
output - one figure per loop, has unique name as loopIndex_model_dataPartition_sampling.png
'''
# parsing inputs from the shell script
loop_index = sys.argv[1]
model_choice = sys.argv[2]
data_choice = sys.argv[3]
sampling_choice = sys.argv[4]
output_fig_name = str(loop_index) + '_' + str(model_choice) + '_' + str(data_choice) + '_' + str(sampling_choice)


# select data partitions
df = pd.read_csv('genus_relative_abundance_otu_drop6infants.csv')
df.patientID = df.patientID.astype('category').cat.codes
metadata_list = ['sampleID', 'EverCovid', 'CovidStatus', 'patientID', 'CovidLabel', 'PersonCode', 'Timepoint']
if data_choice == 'all':
    df_otu = df.drop(columns=metadata_list)
    X = df_otu.to_numpy()
    y = df['CovidStatus'].to_numpy()  # y is 0 or 1
    groups = df.patientID.to_list()
elif data_choice == 'CN_CP':
    # Case-Neg from T1&2, Case-Pos from T3
    df0 = df.loc[df['Timepoint'].isin([1, 2, 3])]  # all three time points T1,2,3
    df1 = df0.loc[df0['CovidLabel']==1]  # case-negatives (CN) only exist in T1&2
    df2 = df0.loc[df0['CovidLabel']==2]  # case-positives (CP) only exist in T3
    # reformat y values from [1, 2] to [0, 1], to work with XGB
    df1['CovidLabel'] = 0
    df2['CovidLabel'] = 1
    df_CN_CP = pd.concat([df1, df2])
    df_CN_CP_otu = df_CN_CP.drop(columns=metadata_list)
    X = df_CN_CP_otu.to_numpy()
    y = df_CN_CP['CovidLabel'].to_numpy()
    groups = df_CN_CP.patientID.to_list()
else:
    # Control from T1&2, Case-Pos from T3
    df0 = df.loc[df['Timepoint'].isin([1, 2])]  # select rows in T1, T2
    df1 = df0.loc[df0['CovidLabel'] == 0]  # controls (NN) exist in T1&2, not using the ones in T3
    df0 = df.loc[df['Timepoint'].isin([1, 2, 3])]  # all three time points T1,2,3
    df2 = df0.loc[df0['CovidLabel'] == 2]  # case-positives (CP) only exist in T3
    # reformat y values from [0, 2] to [0, 1], to work with XGB
    df2['CovidLabel'] = 1
    df_NN_CP = pd.concat([df1, df2])
    df_NN_CP_otu = df_NN_CP.drop(columns=metadata_list)
    X = df_NN_CP_otu.to_numpy()
    y = df_NN_CP['CovidLabel'].to_numpy()
    groups = df_NN_CP.patientID.to_list()


# set up nested loop
####################################################################################################
# begin of outer loop, for each outer loop, do once split of the 3-fold cv
n = int(loop_index)
outer_cv = StratifiedGroupKFold(n_splits=4, random_state=n, shuffle=True)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(12, 12))
# begin outer cv loop: access each fold of the above one split
for fold, (train_index, test_index) in enumerate(outer_cv.split(X, y, groups)):
    # set up train set
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    if sampling_choice == 'none':
        pass
    elif sampling_choice == 'smote':  # increase label 1 samples from 10% to 50% in one train set
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)  # not sure if the groups is still correct after this, for inner_cv
    else:  # adasyn, also increase label 1 samples from 10% to 50% in one train set
        X_train, y_train = ADASYN().fit_resample(X_train, y_train)
    rf_parameters = {
        'n_estimators': [8, 10, 15, 20, 22, 24, 26],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [3, 5, 10],  # can't use None for it causes overfit
        'min_samples_split': [2, 3, 5, 8],
        'min_samples_leaf': [1, 3, 5, 8],
        'criterion': ['gini', 'entropy', 'log_loss']}
    xgb_parameters = {
        "min_child_weight": [1, 5, 10],  # default 1
        "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.5, 1, 2],  # default 0
        "learning_rate": [0.01, 0.1, 0.3],  # default 0.3
        "max_depth": [5, 20, 50, 100],  # default 3
        "n_estimators": [50, 100, 150, 200],  # default 100
        "subsample": [0.3, 0.6, 0.8, 1.0]}  # default 1
    svm_parameters = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 'scale', 'auto'],
        'kernel': ['poly', 'rbf', 'sigmoid'],
        'degree': [1, 2, 3, 4, 5]}
    inner_scores = []
    inner_best_models = []
    ##################################################################################################
    # begin inner loop, run once 3-fold cv for each m
    for m in range(1, 11):
        inner_cv = StratifiedGroupKFold(n_splits=3, random_state=m, shuffle=True)
        if model_choice == 'RF':
            rf_grid = GridSearchCV(
                RandomForestClassifier(random_state=m),
                rf_parameters,
                cv=inner_cv,
                n_jobs=-1,
                error_score=0,
                scoring='roc_auc')
            rf_grid.fit(X_train, y_train)
            y_pred = rf_grid.predict(X_test)
            with open(output_fig_name + '_log.txt', 'a') as f:
                print('Best parameters found for random state ' + str(n) + '_' + str(m), rf_grid.best_params_, file=f)
                print('\n', file=f)
                print('Scores using the above parameters: ', classification_report(y_test, y_pred), file=f)
                print('\n', file=f)
                print('\n', file=f)
            inner_best_models.append(rf_grid.best_estimator_)
            inner_scores.append(rf_grid.best_score_)  # max (mean score across 5 folds) for each loop
        elif model_choice == 'SVM':
            svm_grid = GridSearchCV(
                SVC(),
                svm_parameters,
                cv=inner_cv,
                n_jobs=-1,
                error_score=0,
                scoring='roc_auc')
            svm_grid.fit(X_train, y_train)
            y_pred = svm_grid.predict(X_test)
            with open(output_fig_name + '_log.txt', 'a') as f:
                print('Best parameters found for random state ' + str(n) + '_' + str(m), svm_grid.best_params_, file=f)
                print('\n', file=f)
                print('Scores using the above parameters: ', classification_report(y_test, y_pred), file=f)
                print('\n', file=f)
                print('\n', file=f)
            inner_best_models.append(svm_grid.best_estimator_)
            inner_scores.append(svm_grid.best_score_)  # max (mean score across 5 folds) for each loop
        else:  # model_choice == 'XGB'
            xgb_grid = GridSearchCV(
                XGBClassifier(objective='binary:logistic', seed=m),  # or try binary:logistic with predict_proba
                xgb_parameters,
                cv=inner_cv,
                n_jobs=-1,
                error_score=0,
                scoring='roc_auc')
            xgb_grid.fit(X_train, y_train)
            y_pred = xgb_grid.predict_proba(X_test)
            with open(output_fig_name + '_log.txt', 'a') as f:
                print('Best parameters found for random state ' + str(n) + '_' + str(m), xgb_grid.best_params_, file=f)
                print('\n', file=f)
                print('Scores using the above parameters: ', classification_report(y_test, y_pred), file=f)
                print('\n', file=f)
                print('\n', file=f)
            inner_best_models.append(xgb_grid.best_estimator_)
            inner_scores.append(xgb_grid.best_score_)  # max (mean score across 5 folds) for each loop
    # end of the inner loop
    ###################################################################################################
    max_inner_score_index = np.argmax(inner_scores)  # return the first max item
    max_inner_model = inner_best_models[max_inner_score_index]  # best model after 50 inner_loop, of the current outer_cv fold
    max_inner_model.fit(X_train, y_train)
    viz = RocCurveDisplay.from_estimator(
        max_inner_model,
        X_test,
        y_test,
        name=f"ROC fold {fold}",
        alpha=0.3,
        lw=1,
        ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
# end of the outer cv loop
############################################################################################################
ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.")
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title=f"Mean ROC curve with variability\n(Positive label)")
ax.axis("square")
ax.legend(loc="lower right")
plt.savefig(str(output_fig_name) + '.png')
