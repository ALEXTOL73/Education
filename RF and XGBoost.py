from main import *
import pandas as pd
import numpy as np
from sklearn import ensemble, model_selection, metrics
import xgboost as xgb
from Preprocess import features_creation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

path_ch0,path_ch1,path_ch2,path_ch3 = paths
pca = PCA(n_components=120)

df_fc_ch0,columns,_,_ =  features_creation(path_ch0, sr, duration, mono, mfccs_num, hop_length)
df_fc_ch1,_,_,_ =  features_creation(path_ch1, sr, duration, mono, mfccs_num, hop_length)
df_fc_ch2,_,_,_ =  features_creation(path_ch2, sr, duration, mono, mfccs_num, hop_length)
df_fc_ch3,_,_,_ =  features_creation(path_ch3, sr, duration, mono, mfccs_num, hop_length)

# Convert features into a Panda dataframe for fully connected NN
ch0_df = pd.DataFrame(columns = columns)
ch1_df = pd.DataFrame(columns = columns)
ch2_df = pd.DataFrame(columns = columns)
ch3_df = pd.DataFrame(columns = columns)

for i in range(np.shape(df_fc_ch0)[0]):
    ch0_df.loc[i] = df_fc_ch0[i]
    ch1_df.loc[i] = df_fc_ch1[i]
    ch2_df.loc[i] = df_fc_ch2[i]
    ch3_df.loc[i] = df_fc_ch3[i]

#separate data into features and class labels
features_fc_ch0    = ch0_df.iloc[:,0:-1]
classes_fc_ch0     = ch0_df.iloc[:,-1]
classes_fc_ch0_str = classes_fc_ch0.astype(str)

features_fc_ch1    = ch1_df.iloc[:,0:-1]
classes_fc_ch1     = ch1_df.iloc[:,-1]
classes_fc_ch1_str = classes_fc_ch1.astype(str)

features_fc_ch2    = ch2_df.iloc[:,0:-1]
classes_fc_ch2     = ch2_df.iloc[:,-1]
classes_fc_ch2_str = classes_fc_ch2.astype(str)

features_fc_ch3    = ch3_df.iloc[:,0:-1]
classes_fc_ch3     = ch3_df.iloc[:,-1]
classes_fc_ch3_str = classes_fc_ch3.astype(str)

scaler = StandardScaler()

features_ch0_scaled = scaler.fit_transform(features_fc_ch0)
features_ch1_scaled = scaler.fit_transform(features_fc_ch1)
features_ch2_scaled = scaler.fit_transform(features_fc_ch2)
features_ch3_scaled = scaler.fit_transform(features_fc_ch3)

x_pca_ch0 = pca.fit_transform(features_ch0_scaled)
y_pca_ch0 = ch0_df['class_label']
x_pca_ch1 = pca.fit_transform(features_ch1_scaled)
y_pca_ch1 = ch1_df['class_label']
x_pca_ch2 = pca.fit_transform(features_ch2_scaled)
y_pca_ch2 = ch2_df['class_label']
x_pca_ch3 = pca.fit_transform(features_ch3_scaled)
y_pca_ch3 = ch3_df['class_label']

model_under_test_RF = ensemble.RandomForestClassifier()
model_under_test_RF.get_params().keys()


parameters_grid = {
    'n_estimators' : [600],
    'min_samples_split' : [1,2,3,4],
    'min_samples_leaf' : [1,2,3,4],
    'n_jobs' : [-1]
}

cv = model_selection.StratifiedShuffleSplit(n_splits=10, test_size = 0.1, random_state = 0)
RFC = ensemble.RandomForestClassifier()

grid_cv_ch0 = model_selection.GridSearchCV(RFC, parameters_grid, scoring = 'accuracy', cv = cv, verbose = 1)
grid_cv_ch0.fit(x_pca_ch0, y_pca_ch0)
print('RandomForest for ch0 is calculated')

grid_cv_ch1 = model_selection.GridSearchCV(RFC, parameters_grid, scoring = 'accuracy', cv = cv, verbose = 1)
grid_cv_ch1.fit(x_pca_ch1, y_pca_ch1)
print('RandomForest for ch1 is calculated')

grid_cv_ch2 = model_selection.GridSearchCV(RFC, parameters_grid, scoring = 'accuracy', cv = cv, verbose = 1)
grid_cv_ch2.fit(x_pca_ch2, y_pca_ch2)
print('RandomForest for ch2 is calculated')

grid_cv_ch3 = model_selection.GridSearchCV(RFC, parameters_grid, scoring = 'accuracy', cv = cv, verbose = 1)
grid_cv_ch3.fit(x_pca_ch3, y_pca_ch3)
print('RandomForest for ch3 is calculated')

print('Random Forest for channel 0')
print(grid_cv_ch0.best_score_)
print(grid_cv_ch0.best_params_)
print('Random Forest for channel 1')
print(grid_cv_ch1.best_score_)
print(grid_cv_ch1.best_params_)
print('Random Forest for channel 2')
print(grid_cv_ch2.best_score_)
print(grid_cv_ch2.best_params_)
print('Random Forest for channel 3')
print(grid_cv_ch3.best_score_)
print(grid_cv_ch3.best_params_)
#grid_cv.best_estimator_

model_under_test_xgb = xgb.XGBClassifier()
model_under_test_xgb.get_params().keys()

parameters_grid_xgb = {
    'n_estimators' : [5,15,20,30,90], # количество базовых алгоритмов (деревьев)
    'learning_rate' : [0.1,0.5,0.6],  # скорость обучения
    'reg_alpha' : [0,0.05,0.1,0.5],   # регуляризация
    'n_jobs' : [-1]
}


cv = model_selection.StratifiedShuffleSplit(n_splits=10, test_size = 0.1, random_state = 0)
XGB = xgb.XGBClassifier()

grid_cv_xgb_ch0 = model_selection.GridSearchCV(XGB, parameters_grid_xgb, scoring = 'accuracy', cv = cv, verbose = 1)
grid_cv_xgb_ch0.fit(x_pca_ch0, y_pca_ch0)
print('xgboost for ch0 is calculated')

grid_cv_xgb_ch1 = model_selection.GridSearchCV(XGB, parameters_grid_xgb, scoring = 'accuracy', cv = cv, verbose = 1)
grid_cv_xgb_ch1.fit(x_pca_ch1, y_pca_ch1)
print('xgboost for ch1 is calculated')

grid_cv_xgb_ch2 = model_selection.GridSearchCV(XGB, parameters_grid_xgb, scoring = 'accuracy', cv = cv, verbose = 1)
grid_cv_xgb_ch2.fit(x_pca_ch2, y_pca_ch2)
print('xgboost for ch2 is calculated')

grid_cv_xgb_ch3 = model_selection.GridSearchCV(XGB, parameters_grid_xgb, scoring = 'accuracy', cv = cv, verbose = 1)
grid_cv_xgb_ch3.fit(x_pca_ch3, y_pca_ch3)
print('xgboost for ch3 is calculated')

print('XGBoost for channel 0')
print(grid_cv_xgb_ch0.best_score_)
print(grid_cv_xgb_ch0.best_params_)
print('XGBoost for channel 1')
print(grid_cv_xgb_ch1.best_score_)
print(grid_cv_xgb_ch1.best_params_)
print('XGBoost for channel 2')
print(grid_cv_xgb_ch2.best_score_)
print(grid_cv_xgb_ch2.best_params_)
print('XGBoost Forest for channel 3')
print(grid_cv_xgb_ch3.best_score_)
print(grid_cv_xgb_ch3.best_params_)

exit(0)