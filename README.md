# 🧠 Detección de Transacciones Bancarias Fraudulentas con Redes Neuronales

Este proyecto tiene como objetivo detectar transacciones bancarias fraudulentas utilizando redes neuronales implementadas en Python con TensorFlow y Keras. Se parte de un dataset real, preprocesado adecuadamente para entrenar y evaluar un modelo de clasificación binaria.

---

## 📁 Contenido

- 🔍 Exploración y preprocesamiento de datos  
- ⚖️ Balanceo de clases con sobremuestreo (`RandomOverSampler`)  
- 🧠 Construcción del modelo de red neuronal  
- 🏋️ Entrenamiento del modelo con validación cruzada  
- 📊 Evaluación del modelo con métricas (accuracy, precision, recall, F1-score y matriz de confusión)  
- 📉 Visualización de resultados  

---

## 🧠 Modelo

Se implementa una red neuronal **densa** con múltiples capas ocultas, funciones de activación ReLU y una capa de salida con activación **sigmoide**. Se utiliza `BinaryCrossentropy` como función de pérdida y el optimizador `Adam`.

---

## 📊 Dataset

El conjunto de datos representa transacciones financieras reales, con una fuerte desbalance entre clases:

- 🟢 **Clase 0**: Transacciones normales  
- 🔴 **Clase 1**: Transacciones fraudulentas  

Para mitigar el desbalance, se aplica **sobremuestreo** con `RandomOverSampler` de `imbalanced-learn`.

---

## ⚙️ Metodología

El flujo de trabajo seguido en el proyecto es el siguiente:

1.  **Análisis Exploratorio de Datos (EDA)**: Se realiza una investigación inicial para entender la distribución de las variables y la proporción de transacciones fraudulentas.
2.  **Preprocesamiento de Datos**:
    * Se escalan las características `Time` y `Amount` utilizando `StandardScaler`.
    * Los datos se dividen en conjuntos de entrenamiento, validación y prueba.
3.  **Selección de Características**:
    * Se utiliza `SelectKBest` para seleccionar las **20 características más relevantes** que ayudan a discriminar entre las clases.
4.  **Construcción del Modelo (Red Neuronal)**:
    * Se define un modelo secuencial utilizando Keras (TensorFlow).
    * La arquitectura de la red es la siguiente:
        * **Capa de Entrada**: `Dense` con 20 neuronas y activación `ReLU`.
        * **Capa de Regularización**: `Dropout` para prevenir el sobreajuste.
        * **Capa Oculta**: `Dense` con 10 neuronas y activación `ReLU`.
        * **Capa de Regularización**: `Dropout`.
        * **Capa de Salida**: `Dense` con 1 neurona y activación `Sigmoid` para clasificación binaria.
5.  **Entrenamiento y Evaluación**:
    * El modelo se compila con el optimizador `Adam` y la función de pérdida `binary_crossentropy`.
    * Se entrena durante 10 épocas.
    * Finalmente, se evalúa con métricas clave como la **matriz de confusión**, **precisión**, **recall** y **F1-score**.

---

* **Fuente del dataset**: [Credit Card Fraud Detection en Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)






