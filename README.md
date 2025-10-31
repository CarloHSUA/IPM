# IPM mediante visión por computador

[Página oficial de MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=es-419)

Descargar conda o anaconda 

## Prerrequisitos
Crear un entorno de conda
```bash
conda create -n IPM python=3.12
conda activate IPM
```

## Instalación
Se instalan las siguientes dependencias:
```bash
pip install -q mediapipe requests tqdm
```

## Descargar pesos
```bash
python download_models
```













# APLN

![download](https://github.com/CarloHSUA/APLN/assets/99215566/99141e7b-fab4-4259-a40d-115d6ce09e36)


# Resumen

Se presenta un __sistema de búsqueda de respuestas__ que combina técnicas de recuperación de información con modelos de texto generativo, aprovechando un corpus de artículos de __[CNN](https://edition.cnn.com/)__ y técnicas de web scrapping para crear un conjunto de datos propio. Este corpus, denominado NewsQASum, abarca diversos temas y se utiliza para tareas clave como Question Answering, Summarization y Text Retrieval. Se detalla el proceso de web scrapping para recolectar artículos del año 2024 del sitio web de CNN, junto con el uso de modelos de HuggingFace para generar resúmenes, preguntas y respuestas para cada artículo. La arquitectura del sistema se explora en profundidad, desde el procesamiento del corpus hasta el almacenamiento de embeddings y el flujo de la aplicación para la búsqueda y presentación de respuestas. Se evalúa la calidad de las respuestas generadas mediante métricas como BLEU, ROUGE, METEOR y DeBERTa, mostrando resultados que, aunque prometedores, no alcanzan completamente los objetivos debido a la generalidad de las noticias y preguntas en el dataset.

# Arquitectura RAG
![image](figures/architecture.png)

En esta sección se examina la arquitectura empleada en la aplicación, la cual sigue el esquema típico utilizado por otros modelos RAG. Como se puede observar en la figura, se puede hacer una clara distinción en dos partes: el procesamiento previo del corpus y su almacenamiento, y el flujo normal de la aplicación. Esto da como resultado la generación de una respuesta a una pregunta formulada por el usuario, donde dicha respuesta se basa en el documento recuperado gracias a la base de datos vectorial.

# Requisitos
1. Instalar dependencias:
```
pip install -r requirements.txt
```