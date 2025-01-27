from main import *
import pandas as pd
import numpy as np
import time
from sklearn import model_selection, metrics
from sklearn.svm import SVC
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
    exec(f"df_ch{i},columns,_,_ = features_creation(path_ch{i}, sr, duration, mono, mfccs_num, hop_length)")
    exec(f"ch{i}_df = pd.DataFrame(columns=columns)")
    for j in range(np.shape(df_ch0)[0]):  # нет ошибки определение переменной в цикле выше
        exec(f"ch{i}_df.loc[j] = df_ch{i}[j]")
    exec(f"features_ch{i} = ch{i}_df.iloc[:, 0:-1]")
    exec(f"classes_ch{i}     = ch{i}_df.iloc[:,-1]")
    exec(f"classes_ch{i}_str = classes_ch{i}.astype(str)")


scaler = StandardScaler()
for i in range(num_folders):
    exec(f"features_ch{i}_scaled = scaler.fit_transform(features_ch{i})")
    exec(f"x_pca_ch{i} = pca.fit_transform(features_ch{i}_scaled)")
    exec(f"y_pca_ch{i} = classes_ch{i}_str")

model_under_test_svm = SVC()
print(model_under_test_svm.get_params().keys())

parameters_grid = {
    'C': [0.1, 1, 10, 100, 1000],  # штраф
    'gamma': [0.1, 0.01, 0.001, 0.0001],  # кривизна границы
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']  # ядро
}

cv = model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
model_svm = SVC()

for i in range(num_folders):
    exec(f"grid_cv_svm_ch{i} = model_selection.GridSearchCV(SVC(),parameters_grid,scoring='accuracy',cv=cv,verbose=1)")
    exec(f"grid_cv_svm_ch{i}.fit(x_pca_ch{i}, y_pca_ch{i})")
    exec(f"print('SVM for channel {i}')")
    exec(f"Score{i} = grid_cv_svm_ch{i}.best_score_")
    exec(f"print(round(Score{i},3))")
    exec(f"print(grid_cv_svm_ch{i}.best_params_)")

finish = time.perf_counter()
time = round((finish - start), 3)
print("\nВремя обработки = ", time)

exit()


