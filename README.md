Дипломная работа Брицина Алексея Александровича DSU-53

Тема: Распознавание побочных акустических сигналов от нажатия клавиш клавиатуры.

Описание файлов:

  main - основная программа, позволяющая загружать алгоритмы. Алгоритмы загружаются по названию файлов: KNN-KM.py, RandomForest-RF.py,Support Vector Machine-SVM.py,XGBoost-XGB.py,Fully Connected Neural Network-FCNN.py,Convolutional Neural Network-CNN.py,Recurrent Neural Network-RNN.py
  
  Preprocess.py - подпрограмма для расчета признаков, описывающих аудиосигналы
  
  History_Plots - подпрограмма вывода графиков метрик обучения алгоритмов(только для нейронных сетей)-accuracy(например,FCNN_acc.png), loss(например, CNN_loss.png)
  
  Predict_Model - подпрограмма для вывода результатов теста обученных нейросетей для определения  5 лучших результатов(попадание символов клавиатуры в глубину колонки от 1 до 5)
  
  Ensamble_FCNN и Ensamble_CNN - ансамбли FCNN и CNN (запускаются отдельно от основной программы)
  
  model_FCNN.png,model_CNN.png,model_FCNN.png-изображения архитектур нейросетей
  
  TSNE_ch0.png - Кластеры аудио сигналов, полученные с помощью алгоритма t- SNE для канала 1(ch0)
  
  requirements.txt - установленные библиотеки 
  
  РЕЗУЛЬТАТЫ_CPU.txt, РЕЗУЛЬТАТЫ_GPU.txt -текстовые файлы с результатами работы всех алгоритмов с указанием оптимальных параметров и схемы архитектур на CPU, GPU
  
  Дипломная работа.docx - описание работы
  
  Дипломная работа.pptx - презентация работы
