from main import *
import pandas as pd
import numpy as np
import time
from sklearn import ensemble, model_selection, metrics
from Preprocess import features_creation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

start = time.perf_counter()
pca = PCA(n_components=100)
# Подготовка данных: Convert features into a Panda dataframe, Separate data into features and class labels
for i in range(num_folders):
    exec(f"path_ch{i} = paths[i]")
    exec(f"df_fc_ch{i},columns_fc,_,_ = features_creation(path_ch{i}, sr, duration, mono, mfccs_num, hop_length)")
    exec(f"ch{i}_fc_df = pd.DataFrame(columns=columns_fc)")
    for j in range(np.shape(df_fc_ch0)[0]):  # нет ошибки определение переменной в цикле выше
        exec(f"ch{i}_fc_df.loc[j] = df_fc_ch{i}[j]")
    exec(f"features_fc_ch{i} = ch{i}_fc_df.iloc[:, 0:-1]")
    exec(f"classes_fc_ch{i}     = ch{i}_fc_df.iloc[:,-1]")
    exec(f"classes_fc_ch{i}_str = classes_fc_ch{i}.astype(str)")


scaler = StandardScaler()
for i in range(num_folders):
    exec(f"features_fc_ch{i}_scaled = scaler.fit_transform(features_fc_ch{i})")
    exec(f"x_pca_ch{i} = pca.fit_transform(features_fc_ch{i}_scaled)")
    exec(f"y_pca_ch{i} = ch{i}_fc_df['class_label']")


model_under_test_RF = ensemble.RandomForestClassifier()
print(model_under_test_RF.get_params().keys())


parameters_grid = {
    'n_estimators': [500],  # кол-во деревьев
    'max_features': ['sqrt', 'log2'],  # признаки
    'min_samples_split': [2, 4],  # мин кол-во наблюдений
    'min_samples_leaf': [1, 2],  # мин число узлов
}

cv = model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
RFC = ensemble.RandomForestClassifier()


for i in range(num_folders):
    exec(f"grid_cv_ch{i} = model_selection.GridSearchCV(RFC,parameters_grid,scoring='accuracy',cv=cv,verbose=1)")
    exec(f"grid_cv_ch{i}.fit(x_pca_ch{i}, y_pca_ch{i})")
    exec(f"print('RandomForest for ch{i} is calculated')")
    exec(f"print('RandomForest for channel {i}')")
    exec(f"Score{i} = grid_cv_ch{i}.best_score_")
    exec(f"print(round(Score{i},3))")
    exec(f"print(grid_cv_ch{i}.best_params_)")

finish = time.perf_counter()
time = round((finish - start), 3)
print("\nВремя обработки = ", time)

exit()