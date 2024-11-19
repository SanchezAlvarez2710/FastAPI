# DataSet Seleccionado
**Nombre:** Customer Segmentation  
**URL:** [Customer Segmentation en Kaggle](https://www.kaggle.com/datasets/abisheksudarshan/customer-segmentation?select=test.csv)

---

# 1. Orange

## Resultados
- **Modelo KNN**  
  - Precisión: 0.638  
  - Exactitud (*Accuracy*): 0.698  
  - Sensibilidad (*Recall*): 0.698  

- **Redes Neuronales**  
  - Precisión: 0.524  
  - Exactitud (*Accuracy*): 0.631  
  - Sensibilidad (*Recall*): 0.631  

- **Máquinas de Soporte Vectorial (SVM)**  
  - Precisión: 0.579  
  - Exactitud (*Accuracy*): 0.509  
  - Sensibilidad (*Recall*): 0.579  

- **Árboles de Decisión**  
  - Precisión: 0.535  
  - Exactitud (*Accuracy*): 0.532  
  - Sensibilidad (*Recall*): 0.532  

## Conclusiones para el ejercicio en Orange
En el ejemplo realizado con Orange se implementaron 4 modelos. A partir de los resultados obtenidos, se concluye que el mejor modelo para este dataset fue el **Modelo KNN**, ya que presentó la mayor exactitud durante la fase de evaluación dentro del modelo CRISP-DM.

---

# 2. Modelado de Orange en Python

## Análisis del Dataset
El dataset contiene información demográfica y de comportamiento de 2627 clientes potenciales, incluyendo características como edad, género, estado civil, educación, profesión, experiencia laboral, puntaje de gasto, tamaño de familia y una variable categórica (`Var_1`).  

El objetivo es clasificar a estos clientes en uno de los cuatro segmentos existentes (A, B, C, D), basándose en patrones similares a un mercado actual, para personalizar estrategias de marketing.

---

## Limpieza de Datos
Los aspectos considerados para la limpieza de datos en este dataset fueron:
- **Campos vacíos:** Si la característica es cuantitativa, se asigna el valor `0`. Si es cualitativa, se asigna el valor `NaN`.
- **Eliminación de duplicados:** Se eliminaron duplicados con base en el campo `ID`.

---

## Matriz de Correlación
![Matriz de Correlación](https://github.com/user-attachments/assets/b812c64e-a3bd-4266-92be-53cf34518231)  
En la matriz de correlación se identifica una relación medianamente fuerte, que además es inversamente proporcional, entre las características **Spending_Score** y **Ever_Married**.

---

## Variables
- **Variable dependiente:** `Var_1`  
- **Variables independientes:** `ID`, `Gender`, `Ever_Married`, `Age`, `Graduated`, `Profession`, `Work_Experience`, `Spending_Score`, `Family_Size`.

---

## Identificación de Clases
Las clases encontradas dentro de la variable dependiente `Var_1` son:  
- `'Cat_6'`, `'Cat_4'`, `'Cat_3'`, `NaN`, `'Cat_1'`, `'Cat_2'`, `'Cat_5'`, `'Cat_7'`.

---

## Ingeniería de Características
Para mejorar la comprensión y normalizar los datos categóricos (nominales u ordinales), se implementó un **Label Encoding**.  

Esto permitió transformar características categóricas en valores numéricos, facilitando el análisis y la implementación de los modelos.

---

# Conclusiones Generales
Es imposible determinar si un modelo es mejor que otro sin implementarlo en un dataset y evaluar métricas como precisión, exactitud y sensibilidad.  

Para definir qué método es más adecuado, se debe reconocer que los modelos pueden complementarse entre sí.  

Orange es una herramienta eficiente para seleccionar modelos óptimos en una etapa inicial de análisis. Sin embargo, si se requiere modificar la lógica de los modelos, Orange tiene limitaciones significativas. En estos casos, Python es una mejor alternativa para un desarrollo más personalizado y avanzado.
