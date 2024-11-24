import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# Cargar el conjunto de datos
url = 'https://github.com/jsperdomobe/proyecto/blob/main/Datos/train.csv?raw=true'
df = pd.read_csv(url, index_col=0)

# Preparar las características y las etiquetas
y = df['Depression']
X = df.drop(['id', 'Name', 'Depression'], axis=1)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crear los transformadores para los datos numéricos y categóricos
numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Crear un preprocesador que combine los transformadores
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Crear los modelos base
LR = LogisticRegression()
Tree = DecisionTreeClassifier()
RF = RandomForestClassifier()
SVM = SVC()

# Crear el VotingClassifier con los modelos base
model = VotingClassifier(estimators=[
    ('Logistic Regression', LR),
    ('Decision Tree', Tree),
    ('Random Forest', RF),
    ('Support Vector Machine', SVM)
], voting='hard')

# Crear el pipeline que incluye el preprocesador y el modelo
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Evaluar el modelo
print("Accuracy on test set:", pipeline.score(X_test, y_test))

# Guardar el modelo entrenado en un archivo
joblib.dump(pipeline, 'modelo_voting_classifier.pkl')
print("Modelo guardado como modelo_voting_classifier.pkl")
