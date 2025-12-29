
# RESULTADOS Y ANÁLISIS DE LA COMPARATIVA

Durante la ejecución del pipeline completo, se entrenaron tres modelos durante 25 épocas cada uno:
1.  **CNN (Modelo customizado)**.
2.  **ResNet18**.
3.  **MobileNetV3 Small**.

### 1. Métricas de Rendimiento (Validación)

| Metric | CNN | ResNet18 | MobileNet V3 |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 59.81% | **85.76%** | **85.76%** |
| **Top-2 Acc** | 79.47% | 94.13% | **94.23%** |
| **F1 Score** | 0.4146 | 0.7731 | **0.7869%** |
| **Loss** | 1.1259 | 0.6106 | 0.7057 |
| **Parámetros** | **0.39 M** | 11.18 M | 1.52 M |

### 2. Análisis Comparativo

*   **Rendimiento General**: Ambos modelos pre-entrenados superan ampliamnete al modelo customizado en promedio más del 26% de precisión. Esto demuestra la ventaja del Transfer Learning.
*   **ResNet vs MobileNet**:
    *   Ambos alcanzaron una presición muy similar de entorno al 85.76%, sin embargo, MobileNetV3 tardo menos tiempo en entrenar y logró el mismo resultado, lamentablemente no tomé el tiempo exacto que tardo cada modelo ententrenamiento por lo que no puedo aportar datos exactos del tiempo de entrenamiento de cada red, solo puedo confirmar que en mi percepcción el modelo más rapido de entrenar fue MobileNetV3.

### 3. Pruebas con Imágenes Nuevas (Haar Cascade)
Se probaron 3 imágenes (`angry`, `happy`, `surprised`) fuera del dataset obtenidas gracias a nano banana, al comprobar este dataset se obtuvieron los siguientes resultados:
*   Tanto ResNet como MobileNet predijeron correctamente el 100% de las emociones de prueba. .
*   ResNet mostró una ligera tendencia a ser más preciso puesto que los resultados en las predicciones de las imagenes muestran que el modelo detecto la emoción con un valor muy cercano a 1 mientras que el resto de las emociones las mantiene en valores entorno a 0, mientras que MobileNet deja un poco más abierta la ventana de predicciones, de modo que cuando el modelo detecta la emoción acierta en su predicción pero deja un mayor margen de error tanto de la emoción corrrecta como el resto de las emociones, se podría decir que aunque acertó en todos los casos tiene un poco más de dudas, mientras que ResNet está 100% seguro de su respuesta.


### 4. Evidencia Visual Completa

#### A. Modelo 1: CNN custom

| Curvas de Entrenamiento | Matrices de Confusión |
| :---: | :---: |
| ![Curvas CNN](outputs/curves_scratch.png) | ![Matriz CNN](outputs/confusion_norm_scratch.png) |

**Predicciones (CNN):**
| Angry | Happy | Surprised |
| :---: | :---: | :---: |
| ![Angry](predictions/scratch/angry_pred.png) | ![Happy](predictions/scratch/happy_pred.png) | ![Surprised](predictions/scratch/surprised_pred.png) |

<details>
<summary><b>Ver Reporte de Clasificación Detallado (CNN)</b></summary>

```text
              precision    recall  f1-score   support

     alegria     0.7319    0.7764    0.7535      1185
    disgusto     0.4000    0.0125    0.0242       160
       enojo     0.5079    0.1975    0.2844       162
       miedo     0.8667    0.1757    0.2921        74
    seriedad     0.4357    0.8618    0.5788       680
    sorpresa     0.7796    0.4407    0.5631       329
    tristeza     0.6954    0.2866    0.4059       478

    accuracy                         0.5981      3068
   macro avg     0.6310    0.3930    0.4146      3068
weighted avg     0.6398    0.5981    0.5663      3068
```
</details>

---

#### B. Modelo 2: ResNet18

| Curvas de Entrenamiento | Matrices de Confusión |
| :---: | :---: |
| ![Curvas ResNet](outputs_resnet/curves_resnet18.png) | ![Matriz ResNet](outputs_resnet/confusion_norm_resnet18.png) |

**Predicciones (ResNet18):**
| Angry | Happy | Surprised |
| :---: | :---: | :---: |
| ![Angry](predictions/resnet/angry_pred.png) | ![Happy](predictions/resnet/happy_pred.png) | ![Surprised](predictions/resnet/surprised_pred.png) |

<details>
<summary><b>Ver Reporte de Clasificación Detallado (ResNet18)</b></summary>

```text
              precision    recall  f1-score   support

     alegria     0.9389    0.9333    0.9361      1185
    disgusto     0.6115    0.5312    0.5686       160
       enojo     0.7867    0.7284    0.7564       162
       miedo     0.6462    0.5676    0.6043        74
    seriedad     0.8177    0.8706    0.8433       680
    sorpresa     0.8559    0.8663    0.8610       329
    tristeza     0.8413    0.8431    0.8422       478

    accuracy                         0.8576      3068
   macro avg     0.7854    0.7629    0.7731      3068
weighted avg     0.8557    0.8576    0.8562      3068
```
</details>

---

#### C. Modelo 3: MobileNetV3

| Curvas de Entrenamiento | Matrices de Confusión |
| :---: | :---: |
| ![Curvas MobileNet](outputs_mobilenet/curves_mobilenet_v3_small.png) | ![Matriz MobileNet](outputs_mobilenet/confusion_norm_mobilenet_v3_small.png) |

**Predicciones (MobileNetV3):**
| Angry | Happy | Surprised |
| :---: | :---: | :---: |
| ![Angry](predictions/mobilenet/angry_pred.png) | ![Happy](predictions/mobilenet/happy_pred.png) | ![Surprised](predictions/mobilenet/surprised_pred.png) |

<details>
<summary><b>Ver Reporte de Clasificación Detallado (MobileNetV3)</b></summary>

```text
              precision    recall  f1-score   support

     alegria     0.9545    0.9215    0.9377      1185
    disgusto     0.5814    0.6250    0.6024       160
       enojo     0.7826    0.7778    0.7802       162
       miedo     0.7077    0.6216    0.6619        74
    seriedad     0.8093    0.8735    0.8402       680
    sorpresa     0.8598    0.8389    0.8492       329
    tristeza     0.8429    0.8305    0.8367       478

    accuracy                         0.8576      3068
   macro avg     0.7912    0.7841    0.7869      3068
weighted avg     0.8603    0.8576    0.8584      3068
```
</details>

---

### 5. Conclusiones finales

El modelo customizado de red neuronal al menos con 25 epocas de entrenamiento no logra una buena presición en la detección de emociones, tal vez con más entrenamiento o con una red más grande logré mejorar lo suficiente para poder ser usada en entornos reales, fue la de peor rendimiento no logrando una precisión del 50% pero tanmbién fue la más rapida de entrenar y la que menos recursos requiere, las dos redes preentrenadas lograron una presición similar con resultados correctos en los datos de prueba pero tambien consumen mucho más recursos y requirieron de un mayor tiempo de entrenamiento, en lo personal me iría por MobileNetV3 pues es más rapido en comparación con ResNet18.


To execute the entire project automatically, simply double-click the file:
`run_pipeline.bat` (Windows)

Or if you are on **Linux/Mac**, run in the terminal:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

This will install the libraries, download the images, and train the model using the configuration of `config.py`.


## COMPARATIVA DE MODELOS
Para cumplir con el requerimiento de comparar 2 modelos pre-entrenados (ResNet18 vs MobileNetV3), ejecuta el script de comparativa:

**Linux/Mac:**
```bash
chmod +x run_comparison.sh
./run_comparison.sh
```

**Windows:**
Ejecuta `run_comparison.bat`

Esto entrenará ambos modelos secuencialmente y guardará sus resultados en carpetas separadas (`outputs_resnet` y `outputs_mobilenet`) para que puedas analizarlos.

---
# CLASIFICADOR DE EMOCIONES
El objetivo de este trabajo es construir y comparar distintas arquitecturas de redes neuronales convolucionales (CNNs) utilizando Pytorch, capaces de clasificar emociones humanas a partir de imágenes faciales. El clasificador deberá identificar una de las 7 emociones básicas: alegría, tristeza, enojo, miedo, sorpresa, disgusto y seriedad. El dataset se encuentra en este link: https://drive.google.com/file/d/1aPHE00zkDhEV1waJKhaOJMdN6-lUc0iT/view?usp=sharing

Les recomiendo usar el siguiente código para poder obtener las imágenes fácilmente desde ese link. Pero son libres de descargar las imágenes como mejor les parezca.
import gdown
import zipfile
import os

url = "https://drive.google.com/uc?id=1aPHE00zkDhEV1waJKhaOJMdN6-lUc0iT"
output = "archivo.zip"

gdown.download(url, output, quiet=False)

destino = "datos_zip"
os.makedirs(destino, exist_ok=True)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(destino)

DATASET_ROOT_TRAIN = '/content/datos_zip/dataset_emociones/train'
DATASET_ROOT_VAL   = '/content/datos_zip/dataset_emociones/validation'

## 1. Preprocesamiento de Datos (2 puntos)

Antes de entrenar el modelo, se debe analizar qué tipo de preprocesamiento se debe aplicar a las imágenes. Para esto, se puede considerar uno o más aspectos como:

- Tamaño
- Ajuste de relación de aspecto
- Normalización
- Dejarlo en color o pasarlo a escala de grises

Y transformaciones de Data augmentation como:
- Reflejo horizontal
- Rotación
- Ajuste de brillo, contraste o saturación (si aplica)
- Etc.

Sean criteriosos y elijan solo las técnicas que consideren pertinentes para este caso de uso en específico, no usen todas solo porque sí y ya.

Recomendación: usar `torchvision.transforms` para facilitar el preprocesamiento. Lean su documentación si tienen dudas: https://docs.pytorch.org/vision/0.14/transforms.html

## 2. Construcción y entrenamiento del Modelo CNN (5 puntos)

- Construir una red neuronal convolucional desde cero, sin usar modelos pre-entrenados.
- Analizar correctamente qué funciones de activación se deben usar en cada etapa de la red, el learning rate a utilizar, la función de costo y el optimizador.
- Cosas como el número de capas, neuronas por capa, número y tamaño de los kernels, entre otros, queda a criterio de ustedes, nivel de pooling.

## 3. Modelos pre-entrenados (5 puntos)

- Seleccionar 2 modelos distintos del catálogo de torchvision: https://docs.pytorch.org/vision/main/models
- Realizar un reentrenamiento usando feature extraction o fine-tuning (a elección de ustedes) de la manera en como lo vimos en clase.

## 4. Evaluación y comparativa de modelos (3 puntos)

Cada uno de los 3 modelos deben ser evaluados utilizando las siguientes métricas:

- **Accuracy**:
  - Reportar el valor final en el conjunto de validación.
  - Incluir una gráfica de evolución por época para entrenamiento y validación.

- **F1 Score**:
  - Reportar el valor final en el conjunto de validación.
  - Incluir una gráfica de evolución por época para entrenamiento y validación.

- **Costo (Loss)**:
  - Mostrar una gráfica de evolución del costo por época para entrenamiento y validación.

- **Classification report**
  - Mostrar la precisión, recall y F1 score por cada clase usando `classification_report` de sklearn.

- **Matriz de confusión**:
  - Mostrar la matriz de confusión absoluta (valores enteros).
  - Mostrar la matriz de confusión normalizada (valores entre 0 y 1 por fila).
  - Para ambas usar annot=True, o sea, que se vean los valores numéricos.

Se recomienda utilizar `scikit-learn` para calcular métricas como accuracy, F1 score, el Classification report y las matrices de confusión. Las visualizaciones pueden realizarse con `matplotlib` o `seaborn`, separando claramente los datos de entrenamiento y validación en las gráficas.

 ## 5. Prueba de Imágenes Nuevas (2 punto)
Subir al menos 10 imágenes personales de cualquier relación de aspecto (pueden usar fotos del rostro de ustedes, rostros de personas generadas por IA o imágenes stock de internet), que no formen parte del dataset de entrenamiento ni de validación.

- Debe haber al menos una imagen para cada emoción.

- Aplicar el mismo pre-procesamiento que se usó para el dataset de validation durante el entrenamiento del modelo.

- Pasar las imágenes por el modelo entrenado y mostrar:

  - La imagen original
  - La imagen pre-procesada (mismas transformaciones que para validation durante el entrenamiento)
  - El score asignado a cada clase (normalizado de 0 a 1 o de 0% a 100%)
  - La clase ganadora inferida por el modelo

- Redactar **conclusiones preliminares**.

 ## 6. Prueba de Imágenes Nuevas con Pre-procesamiento Adicional (3 punto)
Las 10 imágenes del punto 5, ahora serán pasadas y recortadas por el algoritmo de detección de rostros **Haar Cascade**. Usen el siguiente código para realizar un pre-procesamiento inicial de la imagen y ya luego aplican el pre-procesamiento que usaron al momento de entrenar el modelo.

- Pasar las imágenes por el modelo entrenado y mostrar:
  - La imagen original
  - La imagen recortada por el algoritmo
  - La imagen pre-procesada (mismas transformaciones que para validation durante el entrenamiento)
  - El score asignado a cada clase (normalizado de 0 a 1 o de 0% a 100%)
  - La clase ganadora inferida por el modelo

- Comparar los resultados con el punto 5 y redactar **conclusiones finales**.

NOTA: Pueden adaptar el código y modificar el `scaleFactor` y el `minNeighbors` según crean conveniente para obtener mejores resultados.

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

image_path = ""

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

image_with_box = image.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

cropped_face_rgb = None
if len(faces) > 0:
    (x, y, w, h) = faces[0]
    center_x, center_y = x + w // 2, y + h // 2
    side = max(w, h)
    half_side = side // 2

    x1 = max(center_x - half_side, 0)
    y1 = max(center_y - half_side, 0)
    x2 = min(center_x + half_side, image.shape[1])
    y2 = min(center_y + half_side, image.shape[0])

    cropped_face = image[y1:y2, x1:x2]
    cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

image_with_box_rgb = cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image_with_box_rgb)
ax[0].set_title("Detección")
ax[0].axis('off')

if cropped_face_rgb is not None:
    ax[1].imshow(cropped_face_rgb)
    ax[1].set_title("Rostro recortado (relación aspecto 1:1)")
    ax[1].axis('off')
else:
    ax[1].text(0.5, 0.5, 'No se detectó rostro', horizontalalignment='center', verticalalignment='center')
    ax[1].axis('off')

plt.tight_layout()
plt.show()
