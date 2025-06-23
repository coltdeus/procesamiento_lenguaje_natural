# Procesamiento de Lenguaje Natural
Proyectos finales - UBA (Universidad de Buenos Aires) - Especialización en Inteligencia Artificial

![NLP Banner](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-blue?style=for-the-badge&logo=python)

## Descripción del Curso
Este repositorio contiene los desafíos desarrollados durante el curso de Procesamiento de Lenguaje Natural, donde se exploraron diferentes técnicas de NLP desde fundamentos básicos hasta modelos avanzados de deep learning.

---

## Contacto
**Daniel Fernando Peña Pinzon**  
📧 Email: [danielfer.colt01@gmail.com]  
🎓 Código: a1818

---

## Desafío 1: Fundamentos de NLP, Vectorización y Modelado

![Vectorización](https://img.shields.io/badge/Topic-Vectorization-green)

Este desafío aborda técnicas fundamentales de procesamiento de lenguaje natural, combinando el preprocesamiento de texto, la vectorización de documentos, el análisis de similitud semántica y la construcción de modelos de clasificación.

### 🔍 Objetivos principales:

* Vectorizar documentos y analizar la similitud entre ellos.
* Entrenar y optimizar modelos de Naïve Bayes para clasificación.
* Analizar la similitud entre palabras mediante la transposición de la matriz término-documento.

### 🧪 Técnicas implementadas:

* Preprocesamiento de texto: tokenización, lematización, limpieza y eliminación de stopwords.
* Vectorización con Bag of Words y TF-IDF.
* Cálculo de similitud entre documentos y palabras utilizando la similitud del coseno.
* Entrenamiento y comparación de modelos `MultinomialNB` y `ComplementNB`.
* Evaluación del desempeño con métricas como F1-score macro.

### 📈 Resultados destacados:

* Se identificaron relaciones semánticas coherentes entre documentos y palabras.
* Los modelos Naïve Bayes alcanzaron buenos niveles de precisión ajustando hiperparámetros y representación del texto.
* La similitud entre palabras capturó correctamente agrupamientos semánticos, demostrando la utilidad de la transposición vectorial.

**Notebook:** [Desafio\_1\_Solucion.ipynb](Desafio_1_Solucion.ipynb)

---

## Desafío 2: Análisis de Sentimientos en Texto

![Sentiment Analysis](https://img.shields.io/badge/Topic-Sentiment%20Analysis-orange)

Este desafío se enfocó en la construcción de modelos supervisados para análisis de sentimientos, utilizando un pipeline completo desde la vectorización hasta la evaluación comparativa de diferentes algoritmos de clasificación.

### 🎯 Objetivos principales:

* Predecir sentimientos a partir de textos utilizando técnicas clásicas de NLP y machine learning.
* Comparar el desempeño de distintos modelos con diferentes representaciones vectoriales.
* Optimizar los modelos con base en métricas como F1-score macro.

### ⚙️ Componentes implementados:

* **Preprocesamiento de texto**: limpieza, tokenización y lematización.
* **Vectorización**: uso de `CountVectorizer` y `TfidfVectorizer` con configuración de n-gramas.
* **Modelos entrenados**:

  * MultinomialNB y ComplementNB
  * Regresión Logística
  * Máquinas de Vectores de Soporte (SVM)
  * Random Forest
* **Optimización y evaluación**:

  * Ajuste de hiperparámetros
  * Evaluación con matriz de confusión y F1-score macro

### 📈 Resultados destacados:

* El modelo de **Logistic Regression** con TF-IDF y n-gramas obtuvo los mejores resultados globales.
* **ComplementNB** superó a MultinomialNB en escenarios con clases desbalanceadas.
* Se observaron patrones coherentes entre predicciones y polaridad del texto.

Este desafío permitió aplicar técnicas de clasificación textual sobre un problema real, mostrando cómo pequeños cambios en el preprocesamiento y representación pueden tener un gran impacto en el desempeño de los modelos.

**Notebook:** [Desafio\_2\_\_Solucion.ipynb](Desafio_2__Solucion.ipynb)

---

## Desafío 3: Modelo de Lenguaje a Nivel de Caracteres

![Language Model](https://img.shields.io/badge/Topic-Language%20Model-red)

En este desafío se desarrolló un modelo generativo de lenguaje entrenado sobre un corpus textual, utilizando redes neuronales recurrentes para predecir y generar texto carácter por carácter. Se exploraron distintas arquitecturas y estrategias de generación.

### 🎯 Objetivos principales:

* Entrenar un modelo de lenguaje utilizando secuencias de caracteres como entrada.
* Evaluar arquitecturas recurrentes (SimpleRNN, LSTM, GRU) en la tarea de predicción secuencial.
* Generar texto a partir de una semilla utilizando estrategias como greedy search, beam search y sampling con temperatura.

### ⚙️ Componentes implementados:

* **Preprocesamiento del corpus**:

  * Conversión a minúsculas, limpieza y tokenización carácter por carácter.
  * Construcción de secuencias de entrada/salida.
  * Codificación one-hot de caracteres.
* **Entrenamiento del modelo**:

  * Uso de `SimpleRNN`, `LSTM` y `GRU`.
  * Optimización con `RMSprop`.
  * Monitoreo de la **perplejidad** como métrica clave de desempeño.
* **Generación de texto**:

  * Greedy search.
  * Beam search (determinístico y estocástico).
  * Sampling con variación de **temperatura** para controlar la diversidad.

### 📈 Resultados destacados:

* El modelo fue capaz de generar texto con patrones sintácticos coherentes.
* **LSTM** superó a otras arquitecturas en estabilidad y calidad del texto.
* La **temperatura** demostró ser un parámetro crítico: a mayor temperatura, mayor creatividad; a menor, más precisión pero repetitividad.
* Se evidenció una mejora progresiva durante el entrenamiento, con reducción sostenida de la perplejidad.

Este desafío permitió entender cómo modelar dependencias secuenciales en lenguaje natural y cómo controlar la generación de texto a través de técnicas probabilísticas.

**Notebook:** [Desafio\_modelo\_lenguaje\_char\_solucion.ipynb](Desafio_modelo_lenguaje_char_solucion.ipynb)

---

## Desafío 4: Bot de Preguntas y Respuestas

![QA Bot](https://img.shields.io/badge/Topic-Question%20Answering-purple)

En este desafío se desarrolló un sistema de **preguntas y respuestas (QA)** que, a partir de un contexto textual, es capaz de extraer respuestas relevantes a preguntas formuladas por el usuario. El enfoque combinó modelos secuenciales, mecanismos de atención y embeddings contextuales.

### 🎯 Objetivos principales:

* Construir un modelo capaz de responder preguntas basadas en fragmentos de texto.
* Implementar arquitecturas de tipo **seq2seq** con mecanismos de atención.
* Evaluar la calidad de las respuestas generadas con métricas específicas de QA.
* Diseñar una interfaz interactiva para consultar al modelo de forma dinámica.

### ⚙️ Componentes implementados:

* **Preprocesamiento del corpus**: limpieza, tokenización y estructuración en pares `(pregunta, contexto)`.
* **Modelo de QA**:

  * Arquitectura seq2seq adaptada a la tarea extractiva.
  * Inclusión de **mecanismos de atención** para enfocar las respuestas.
  * Uso de embeddings contextuales para representar pregunta y contexto.
* **Entrenamiento y evaluación**:

  * Evaluación con métricas como *Exact Match* y *F1-score*.
  * Observación cualitativa de respuestas correctas e incorrectas.
* **Interfaz de usuario**:

  * Entrada manual para preguntas.
  * Salida textual automática del fragmento considerado como respuesta.

### 📈 Resultados destacados:

* El sistema fue capaz de identificar y extraer correctamente respuestas en la mayoría de los casos.
* El uso de atención y embeddings enriqueció la capacidad de comprender el contexto.
* Las respuestas obtenidas mostraron coherencia gramatical y proximidad semántica al contenido original.

Este desafío representa una síntesis de los conocimientos del curso, integrando representación del lenguaje, modelos neuronales y evaluación de tareas complejas de comprensión lectora.

**Notebook:** [Desafio\_4\_6\_bot\_qa\_Solucion.ipynb](Desafio_4_6_bot_qa_Solucion.ipynb)

---

## Tecnologías Utilizadas
- **Python** 🐍
- **TensorFlow/Keras** 🧠
- **NLTK** 📝
- **Scikit-learn** 🔬
- **Pandas & NumPy** 📊
- **Matplotlib & Seaborn** 📈

## Estructura del Repositorio
```
procesamiento_lenguaje_natural/
├── Desafio_1_Solucion.ipynb
├── Desafio_2__Solucion.ipynb
├── Desafio_modelo_lenguaje_char_solucion.ipynb
├── Desafio_4_6_bot_qa_Solucion.ipynb
└── README.md
```

---
