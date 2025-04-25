# Documentación del Modelo CNN para MNIST

## Introducción

Este documento describe la arquitectura, entrenamiento y despliegue de una Red Neuronal Convolucional (CNN) para clasificar dígitos manuscritos utilizando el dataset MNIST. La solución incluye un modelo de deep learning y una API web que permite consumir el modelo desde cualquier frontend.

## Dataset: MNIST

El dataset MNIST (Modified National Institute of Standards and Technology) es un conjunto de datos estándar en el campo de la visión por computadora, que contiene 70,000 imágenes de dígitos manuscritos (60,000 para entrenamiento y 10,000 para pruebas). Cada imagen es de 28x28 píxeles en escala de grises.

**Características:**
- 70,000 imágenes de dígitos manuscritos (0-9)
- Imágenes de 28x28 píxeles en escala de grises
- Valores de píxeles entre 0 (blanco) y 255 (negro)
- 10 clases (dígitos del 0 al 9)

## Arquitectura del Modelo

El modelo utiliza una arquitectura de Red Neuronal Convolucional (CNN) profunda optimizada para el reconocimiento de patrones en imágenes. La arquitectura consta de varias capas convolucionales seguidas de capas de agrupación (pooling) y finalmente capas totalmente conectadas (densas).

### Detalles de la Arquitectura

```
Modelo: Sequential
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d (MaxPooling2D)  (None, 13, 13, 32)      0         
                                                                 
 conv2d_1 (Conv2D)           (None, 11, 11, 64)        18,496    
                                                                 
 max_pooling2d_1 (MaxPooling2  (None, 5, 5, 64)        0         
 D)                                                              
                                                                 
 conv2d_2 (Conv2D)           (None, 3, 3, 128)         73,856    
                                                                 
 max_pooling2d_2 (MaxPooling2  (None, 1, 1, 128)       0         
 D)                                                              
                                                                 
 flatten (Flatten)           (None, 128)               0         
                                                                 
 dense (Dense)               (None, 128)               16,512    
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 10)                1,290     
                                                                 
=================================================================
Total params: 110,474
Trainable params: 110,474
Non-trainable params: 0
_________________________________________________________________
```

### Componentes clave:

1. **Capas Convolucionales:**
   - Primera capa: 32 filtros de tamaño 3x3
   - Segunda capa: 64 filtros de tamaño 3x3
   - Tercera capa: 128 filtros de tamaño 3x3
   - Cada capa convolucional utiliza la función de activación ReLU

2. **Capas de Agrupación (MaxPooling):**
   - Reducen dimensionalidad espacial con tamaño de pool 2x2
   - Ayudan a la invarianza a pequeñas transformaciones

3. **Capas Totalmente Conectadas:**
   - Una capa oculta de 128 neuronas con activación ReLU
   - Capa Dropout (0.5) para evitar sobreajuste
   - Capa de salida de 10 neuronas con activación softmax (una por dígito)

4. **Optimizador y Función de Pérdida:**
   - Optimizador: Adam
   - Función de pérdida: Entropía cruzada categórica

## Justificación de la Arquitectura

La arquitectura fue diseñada considerando los siguientes factores:

1. **Múltiples Capas Convolucionales**: Las capas convolucionales permiten extraer características de bajo nivel (bordes, líneas) en las primeras capas y características más complejas (formas, patrones) en las capas más profundas. Tres capas convolucionales proporcionan suficiente profundidad para capturar las características relevantes en los dígitos manuscritos.

2. **Aumento progresivo de filtros**: El número de filtros aumenta progresivamente (32 → 64 → 128) para permitir que el modelo aprenda un conjunto más rico de características conforme avanza en profundidad.

3. **MaxPooling**: Después de cada capa convolucional, se utiliza MaxPooling para reducir la dimensionalidad espacial, lo que ayuda a:
   - Reducir el número de parámetros y costo computacional
   - Hacer el modelo más robusto a pequeñas variaciones en la posición de las características

4. **Dropout**: Se incluye una capa de Dropout (0.5) antes de la capa de salida para prevenir el sobreajuste, lo que mejora la generalización del modelo.

5. **Parámetros totales**: El modelo tiene aproximadamente 110,474 parámetros, lo que proporciona suficiente capacidad para aprender las características distintivas de los dígitos manuscritos sin ser excesivamente complejo.

## Rendimiento del Modelo

Tras entrenar por 10 épocas con un tamaño de lote de 128:

- **Precisión en entrenamiento**: ~99.5%
- **Precisión en validación**: ~99.2%
- **Pérdida en validación**: ~0.03

La alta precisión y baja pérdida indican que el modelo ha aprendido efectivamente a reconocer dígitos manuscritos. El pequeño gap entre la precisión de entrenamiento y validación sugiere que el modelo no presenta sobreajuste significativo.

## API Predictiva

Se ha implementado una API REST utilizando Flask que expone el endpoint `/predict-image` para recibir imágenes en formato base64 y devolver la predicción. El servicio se expone a Internet utilizando Ngrok para facilitar el acceso desde cualquier frontend.

### Endpoints

- **GET /**: Página de inicio con instrucciones básicas
- **POST /predict-image**: Endpoint principal para predicciones
  - Recibe: JSON con campo `image` conteniendo la imagen en base64
  - Devuelve: JSON con `predicted_digit`, `confidence` y `probabilities` para cada clase

### Formato de respuesta

```json
{
  "predicted_digit": 5,
  "confidence": 0.9987,
  "probabilities": {
    "0": 0.0001,
    "1": 0.0002,
    "2": 0.0001,
    "3": 0.0003,
    "4": 0.0001,
    "5": 0.9987,
    "6": 0.0002,
    "7": 0.0001,
    "8": 0.0001,
    "9": 0.0001
  }
}
```

## Instrucciones de Implementación

### Requisitos
- Google Colab (cuenta de Google)
- Navegador web moderno

### Pasos para implementar
1. Subir el notebook a Google Colab
2. Ejecutar todas las celdas
3. Obtener la URL de Ngrok que se muestra en la consola
4. Utilizar el frontend proporcionado, ingresando la URL de Ngrok en el campo correspondiente

### Consideraciones técnicas
- El modelo se guarda en el entorno de Colab, por lo que no es necesario reentrenarlo en cada ejecución
- La sesión de Colab se cerrará después de cierto tiempo de inactividad, lo que requerirá reiniciar el servicio
- La URL de Ngrok cambia en cada ejecución, por lo que deberá actualizarse en el frontend

## Limitaciones y Mejoras Futuras

- **Persistencia**: Implementar un sistema de almacenamiento persistente para el modelo
- **Preprocesamiento mejorado**: Añadir más técnicas de preprocesamiento para mejorar la robustez ante diferentes estilos de escritura
- **Aumento de datos**: Utilizar técnicas de aumento de datos para mejorar la generalización
- **Despliegue en plataforma permanente**: Migrar de Colab+Ngrok a una plataforma gratuita más estable como Heroku o Render

## Conclusión

Este proyecto implementa satisfactoriamente una solución completa para clasificación de dígitos manuscritos, desde el entrenamiento del modelo hasta su exposición como servicio web. La arquitectura CNN elegida proporciona un excelente equilibrio entre complejidad y rendimiento, logrando una alta precisión en el dataset MNIST. La API implementada permite consumir fácilmente el modelo desde cualquier frontend, facilitando la integración en aplicaciones web.