from main import *
import pandas as pd
import numpy as np
import time
from sklearn import ensemble, model_selection, metrics
import xgboost as xgb
from Preprocess import features_creation
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

start = time.perf_counter()
pca = PCA(n_components=120)
# Подготовка данных: Convert features into a Panda dataframe, Separate data into features and class labels
for i in range(num_folders):
    exec(f"path_ch{i} = paths[i]")
    exec(f"df_fc_ch{i},columns_fc,_,_ = features_creation(path_ch{i}, sr, duration, mono, mfccs_num, hop_length)")
    exec(f"ch{i}_fc_df = pd.DataFrame(columns=columns_fc)")
    for j in range(np.shape(df_fc_ch0)[0]): # нет ошибки определение переменной в цикле выше
        exec(f"ch{i}_fc_df.loc[j] = df_fc_ch{i}[j]")
    exec(f"features_fc_ch{i} = ch{i}_fc_df.iloc[:, 0:-1]")
    exec(f"classes_fc_ch{i}  = ch{i}_fc_df.iloc[:,-1]")


scaler = StandardScaler()
le = LabelEncoder()
for i in range(num_folders):
    exec(f"features_fc_ch{i}_scaled = scaler.fit_transform(features_fc_ch{i})")
    exec(f"x_pca_ch{i} = pca.fit_transform(features_fc_ch{i}_scaled)")
    exec(f"y_pca_ch{i} = ch{i}_fc_df['class_label']")
    exec(f"y_pca_cat_ch{i} = le.fit_transform(ch{i}_fc_df['class_label'])")

model_under_test_xgb = xgb.XGBClassifier(objective='multi:softprob')
print(model_under_test_xgb.get_params().keys())

parameters_grid_xgb = {
    'n_estimators': [500],  # количество базовых алгоритмов (деревьев)
    'learning_rate': [0.5, 0.7],  # скорость обучения
    'reg_alpha': [0.0, 0.1],  # регуляризация
    'max_depth': [3, 5, 7],  # глубина дерева
    'subsample': [0.5, 0.7, 0.9],  # подвыборка
    'colsample_bytree': [0.5, 0.7, 0.9]  # доля признаков
}

cv = model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
XGB = xgb.XGBClassifier()

for i in range(num_folders):
    exec(
     f"grid_cv_xgb_ch{i} = model_selection.RandomizedSearchCV(XGB,parameters_grid_xgb,\
                                                  scoring='accuracy',cv=cv,verbose=1)")
    exec(f"grid_cv_xgb_ch{i}.fit(x_pca_ch{i}, y_pca_cat_ch{i})")
    exec(f"print('xgboost for ch{i} is calculated')")
    exec(f"print('XGBoost for channel {i}')")
    exec(f"Score{i} = grid_cv_xgb_ch{i}.best_score_")
    exec(f"print(round(Score{i},3))")
    exec(f"print(grid_cv_xgb_ch{i}.best_params_)")

finish = time.perf_counter()
time = round((finish - start), 3)
print("\nВремя обработки = ", time)
exit()