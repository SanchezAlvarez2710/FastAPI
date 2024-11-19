
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importamos los csv (datasets)
df1 = pd.read_csv('./Act3/test.csv')

#Mostramos el dataset
print(df1.head())

#Cambiamos nombre de Var_1 a category
df1.rename(columns={'Var_1': 'Category'}, inplace=True)

# Verificar clases únicas en la columna 'Var_1'
clases_unicas = df1['Category'].unique()
print("Clases en variable objetivo:", clases_unicas)

print(df1.describe())

#Tecnicas dataCleaning
# 1. Reemplazar valores faltantes según el tipo de dato
for column in df1.columns:
    if df1[column].dtype in [np.float64, np.int64]:  # Cuantitativos
        df1[column] = df1[column].fillna(0)
    else:  # Cualitativos
        df1[column] = df1[column].fillna('NaN')

#2. Eliminar duplicados basados únicamente en la columna 'ID'
df1 = df1.drop_duplicates(subset='ID')

#3. Aplicamos Label Encoding a caracteristicas categoricas
from sklearn.preprocessing import LabelEncoder
LabelEncoder = LabelEncoder()
df1['Gender'] = LabelEncoder.fit_transform(df1['Gender']) #0 = Female 1 = Male
df1['Spending_Score'] = LabelEncoder.fit_transform(df1['Spending_Score'])  #0 = Average 1 = High 2 = Low
df1['Ever_Married'] = LabelEncoder.fit_transform(df1['Ever_Married'])  #0 = No 1 = Yes
df1['Graduated'] = LabelEncoder.fit_transform(df1['Graduated'])   #0 = No 1 = Yes
df1['Profession'] = LabelEncoder.fit_transform(df1['Profession'])
df1['Category'] = LabelEncoder.fit_transform(df1['Category'])

# Calcular la matriz de correlación
correlation_matrix = df1.corr()

# Mostrar la matriz de correlación
print(correlation_matrix)

# Visualizar la matriz de correlación usando un mapa de calor (heatmap)
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matriz de Correlación")
plt.show()

print(df1.head())

from sklearn import preprocessing

ds = df1.values
X = ds[:, :-1].astype(float) #variables independientes
Y = ds[:, -1] #variable dependiente

print(f"Variables independientes: {X}")
print(f"Variable dependiente : {Y}")

#Preparación de los datos para el entrenamiento
from  sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val  = train_test_split(X, Y, test_size=0.2, random_state=5)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.15, random_state=5, stratify=Y_train)

print(np.unique(Y, return_counts=True))
print(np.unique(Y_train, return_counts=True))
print(np.unique(Y_test, return_counts=True))
print(np.unique(Y_val, return_counts=True))

#Aplicamos KNN con los hiperparametros en Orange
#Parametros
#Vecinos = 15
#Distribución = Manhattan
#Peso = Uniform

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn = KNeighborsClassifier(n_neighbors= 15, metric='manhattan', weights='uniform')
knn.fit(X_train, Y_train)

train_acc = knn.score(X_train, Y_train)
test_acc = knn.score(X_test, Y_test)
print(f"La precisión del entrenamiento es: {train_acc}")
print(f"La precisión del  test es: {test_acc}")

#Generamos la matríz de confusión
from sklearn.metrics import confusion_matrix
Y_Pred = knn.predict(X_test)
cm  = confusion_matrix(Y_test, Y_Pred)
print(f"Matríz de confusión:\n {cm}")

#Reporte de clasificación
Y_train_pred = knn.predict(X)
print(f"#Reporte de clasificación:\n {classification_report(Y, Y_train_pred)}")