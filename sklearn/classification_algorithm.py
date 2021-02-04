from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.impute import SimpleImputer 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn import datasets


from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, classification_report, confusion_matrix


# Classification Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier


# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
from pycm import *
import joblib

def rocvis(true, prob, label) :
    AUC = np.mean(true == np.round(prob.ravel()).astype(int))
    if type(true[0]) == str :
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        true = le.fit_transform(true)
    else :
        pass
    fpr, tpr, thresholds = roc_curve(true, prob)
    plt.plot(fpr, tpr, marker = '.', label = "AUC : {:.2f} , {}".format(AUC,label))

print("데이터 처리 중")
data = pd.read_csv('https://raw.githubusercontent.com/eunjong147/tech/main/sklearn/dataset_ver_sun.csv')
data = np.round(data, decimals=5)
feature_list = list(data)[:-1]
data_input = data[feature_list].to_numpy()
# print(data_input)
data_target = data['C'].to_numpy()
# print(data_target)
x_train, x_test, y_train, y_test = train_test_split(data_input, data_target)

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
print("알고리즘 모델 형성 중")
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(random_state=42))])

pipe_rf = Pipeline([('scl', StandardScaler()),
                    ('clf', RandomForestClassifier(random_state=42))])

pipe_svm = Pipeline([('scl', StandardScaler()),
                    ('clf', SVC(random_state=42))])

pipe_lda = Pipeline([('scl', StandardScaler()),
                    ('clf', LinearDiscriminantAnalysis())])

pipe_qda = Pipeline([('scl', StandardScaler()),
                    ('clf', QuadraticDiscriminantAnalysis())])

pipe_gbm = Pipeline([('scl', StandardScaler()),
                    ('clf', GradientBoostingClassifier(random_state=42))])

pipe_knn = Pipeline([('scl', StandardScaler()),
                    ('clf', KNeighborsClassifier())])

pipe_ridge = Pipeline([('scl', StandardScaler()),
                    ('clf', RidgeClassifier())])

pipe_nb = Pipeline([('scl', StandardScaler()),
                    ('clf', GaussianNB())])

param_range = [1, 2, 3 , 4, 5, 6, 7, 8, 9, 10]
depth_range = [7, 8, 9]
min_samples_split_range = [0.5, 0.7, 0.9]
param_range = [0.5, 0.1]
param_range_f1 = np.logspace(0, -5, 5)

grid_params_lr = [{'clf__penalty' : ['none', 'l2'],
                   'clf__C' : [1.0, 0.5, 0.1],
                   'clf__solver' : ['newton-cg', 'sag', 'saga', 'lbfgs'],
                   'clf__max_iter' : [10000, 20000] }]

grid_params_rf = [{'clf__criterion' : ['gini', 'entropy'],
                   'clf__min_samples_leaf' : [0.5, 0.1], 
                   'clf__max_depth' : [5, 6, 7, 8, 9],
                   'clf__min_samples_split' : [0.5, 0.7, 0.9]}]

grid_params_svm = [{'clf__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
                    'clf__C' : [1.0, 0.5, 0.1],
                    'clf__max_iter' : [10000, 20000]}]

grid_params_lda = [{'clf__tol' : [0.5, 0.1],
                    'clf__solver' : ['svd', 'lsqr', 'eigen']}]

grid_params_qda = [{'clf__tol' : np.logspace(0, -5, 5)}]

grid_params_gbm = [{'clf__tol' : np.logspace(0, -5, 5),
                    'clf__max_depth' : [7, 8, 9],
                    'clf__min_samples_leaf' : [0.5, 0.1],
                    'clf__loss' : ['deviance', 'exponential']}]

grid_params_knn = [{'clf__n_neighbors' : [2, 3, 4], 
                    'clf__weights' : ['uniform', 'distance'],
                    'clf__algorithm' : ['ball_tree', 'kd_tree', 'brute']}]

grid_params_ridge = [{'clf__solver' : ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                      'clf__tol' : np.logspace(0, -5, 5),
                      'clf__alpha' : np.logspace(0, -5, 5)}]

grid_params_nb = [{'clf__var_smoothing' : [1e-9, 1e-8]}]


pipe = [
    pipe_lr, pipe_rf, pipe_svm,
    pipe_qda, pipe_knn, pipe_ridge, 
    pipe_lda, pipe_nb
]

pipe1 = [
    pipe_lr, pipe_svm,
    pipe_knn, pipe_nb
    
]
pipe2 = [
    pipe_rf, pipe_qda, pipe_ridge, pipe_lda
]

params = [
    grid_params_lr, grid_params_rf, grid_params_svm,
    grid_params_qda, grid_params_knn, grid_params_ridge, 
    grid_params_lda, grid_params_nb
]

params1 = [
    grid_params_lr, grid_params_svm,
    grid_params_knn, grid_params_nb    
]

params2 = [
    grid_params_rf, grid_params_qda, grid_params_ridge, grid_params_lda
]


grid_dict = {0: 'Logistic Regression',
             1: 'Random Forest', 
             2: 'Support Vector Machine',
             3: 'Quadratic Discriminant Anlysis',
             4: 'KNNClassifier',
             5: 'RidgeClassifier',
             6: 'Linear Discriminant Analysis',
             7: 'Naive Bayes'
            }

grid_dict1 = {0: 'Logistic Regression',
             1: 'Support Vector Machine',
             2: 'KNNclassifier',
             3: 'Naive Bayes'}

grid_dict2 = {
    0: 'Random Forest',
    1: 'Quadratic Discriminant Anlysis',
    2: 'RidgeCClassifier',
    3: 'Linear Discriminant Anlysis'
}

model_prob = {}
model_result = {}
model_best_params = {}
model_confusion = {}
# plt.style.use('ggplot')
# fig, ax = plt.subplots(figsize = (20,10))
# plt.plot([0,1], [0,1], linestyle='--')
print("알고리즘 모델 적용 중")
for idx, (param, model) in enumerate(zip(params, pipe)) :
    print(f'알고리즘 index : {idx}')
    search = GridSearchCV(model, param, cv=cv) # verbose = ? 
    search.fit(x_train, y_train)
    y_pred = search.predict(x_test)
    model_result[grid_dict.get(idx)] = f1_score(y_test, y_pred, average='micro')
    model_best_params[grid_dict.get(idx)] = search.best_params_
    

print(model_result)
print(model_best_params)
"""
for idx, (param, model) in enumerate(zip(params, pipe)) : 
    print(f'index : {idx}')
    search = GridSearchCV(model, param, cv=cv, n_jobs=jobs, verbose=-1)
    search.fit(x_train, y_train)
    y_pred = search.predict(x_test)
    try :
        y_prob = search.predict_proba(x_test)
    except Exception as e :
        pass
    # rocvis(true=y_test, prob=y_prob[:,1], label = grid_dict.get(idx))
    # model_result[grid_dict.get(idx)] = roc_auc_score(y_test, y_pred, multi_class='ovo')
    model_result[grid_dict.get(idx)] = f1_score(y_test, y_pred, average='micro')
    model_prob[grid_dict.get(idx)] = y_prob
    model_best_params[grid_dict.get(idx)] = search.best_params_
    model_confusion[grid_dict.get(idx)] = confusion_matrix(y_test, y_pred)
'''
plt.legend(fontsize = 20, loc='center', shadow=True)
plt.title("Models Roc Curve", fontsize=25)
plt.savefig("./Model_result.png")
'''
output = pd.DataFrame([model_result.keys(), model_result.values()], index=['algo', 'r2']).T
output.sort_values(["r2"], ascending=False, inplace=True)
fig, ax = plt.subplots(figsize=(20, 10))
sns.set(font_scale = 2)
sns.barplot(y='algo', x='r2', data=output)
plt.show()
"""
# cp = Compare(model_confusion)




