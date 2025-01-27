from main import *
import pandas as pd
import numpy as np
import time
from sklearn import model_selection, metrics
from sklearn.decomposition import PCA
from Preprocess import features_creation
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
import seaborn as sns
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

tsne = TSNE(n_components=2, random_state=42, perplexity=12)
tsne_transformed_fc_ch0 = tsne.fit_transform(features_fc_ch0_scaled)  # нет ошибки определение переменной в цикле выше
plt.figure(figsize=(14, 12))
sns.scatterplot(
    x=tsne_transformed_fc_ch0[:, 0],
    y=tsne_transformed_fc_ch0[:, 1],
    hue=classes_fc_ch0_str,  # нет ошибки определение переменной в цикле выше
    palette='CMRmap',
    legend='full'
)
plt.savefig('d:\\TSNE_ch0.png')

model_under_test_KMeans = KNeighborsClassifier()
print(model_under_test_KMeans.get_params().keys())
k = len(np.unique(ch0_fc_df['class_label']))  # нет ошибки определение переменной в цикле выше

parameters_grid_KMeans = {
    'n_neighbors': [k],
    'weights': ['uniform', 'distance'],  # взвешивание весов
    'algorithm': ['ball_tree', 'kd_tree', 'brute'],  # алгоритм
    'leaf_size': [1, 5, 10],  # размер листа
    'p': list(np.arange(1, 3)),  # мощность
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']  # метрики для вычисления расстояний
}

cv = model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
KM = KNeighborsClassifier()

for i in range(num_folders):
    exec(
       f"grid_cv_KM_ch{i} = model_selection.GridSearchCV(KM,parameters_grid_KMeans,scoring='accuracy',cv=cv,verbose=1)")
    exec(f"grid_cv_KM_ch{i}.fit(x_pca_ch{i}, y_pca_ch{i})")
    exec(f"print('KMeans for ch{i} is calculated')")
    exec(f"print('KMeans for channel {i}')")
    exec(f"Score{i} = grid_cv_KM_ch{i}.best_score_")
    exec(f"print(round(Score{i},3))")
    exec(f"print(grid_cv_KM_ch{i}.best_params_)")

finish = time.perf_counter()
time = round((finish - start), 3)
print("\nВремя обработки = ", time)
exit()