!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:22:20 2023

@author: Equipo DSA
"""

# Importe el conjunto de datos de diabetes y divídalo en entrenamiento y prueba usando scikit-learn
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as nppip
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report,confusion_matrix, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import accuracy_score

import mlflow
import mlflow.sklearn
import mlflow.sklearn
import mlflow.catboost
import mlflow.xgboost
import mlflow.lightgbm

# Importe el conjunto de datos de diabetes y divídalo en entrenamiento y prueba usando scikit-learn
url ='https://github.com/jsperdomobe/proyecto/blob/main/Datos/train.csv?raw=true'
df = pd.read_csv(url, index_col=0)
y = df.Depression
X = df.drop(['id','Name','Depression'], axis=1)

# Divide data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_train = X_train[my_cols].copy()
X_test = X_test[my_cols].copy()

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('scaler', StandardScaler()),
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

#confusion matrix template
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

    #Importe MLFlow para registrar los experimentos, el regresor de bosques aleatorios y la métrica de error cuadrático medio

# defina el servidor para llevar el registro de modelos y artefactos
#mlflow.set_tracking_uri('http://localhost:5000')
# registre el experimento
experiment = mlflow.set_experiment("modelo2")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento. MLflow los crea un experimento para este cuaderno por defecto y guarda las características del experimento y las métricas definidas. 
# Para ver el resultado de las corridas haga click en Experimentos en el menú izquierdo. 
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # defina los parámetros del modelo
    LR = LogisticRegression()
    Tree = DecisionTreeClassifier()
    RF = RandomForestClassifier()
    SVM = SVC()
    KNN = KNeighborsClassifier()
    GB = GradientBoostingClassifier()
    AB = AdaBoostClassifier()
    LGBM = LGBMClassifier()

    model = VotingClassifier(
        estimators=[("Logistic Regression", LR), ("Decision Tree", Tree), ("Random Forest", RF),
                   ("Support Vector Machine",SVM), ("k-nearest neighbor",KNN), ("Gradient Boosting",GB),
                   ("AdaBoost",AB), ("LGBM", LGBM)],
        voting='hard',
    )

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
    # defina los parámetros del modelo
    n_modelos = 8
    my_pipeline.fit(X_train, y_train)
    prediction = my_pipeline.predict(X_test)

    sk_report = classification_report(digits=3,y_true=prediction,y_pred=y_test)
    accuracy = accuracy_score(y_test, prediction)
    print(sk_report)
 
    # Registre los parámetros
    mlflow.log_param("modelos", n_modelos)
       
    # Registre el modelo
    mlflow.sklearn.log_model(model, "model voting classifier")
  
    # Cree y registre la métrica de interés
    mlflow.log_metric("score", sk_report)
    print(sk_report)
    mlflow.log_metric("accuracy", accuracy)
    print(accuracy)

import joblib

# Guardar el pipeline entrenado en un archivo
joblib.dump(my_pipeline, 'modelo_voting_classifier.pkl')