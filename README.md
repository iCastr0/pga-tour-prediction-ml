# PGA Tour Winner Prediction

Este proyecto utiliza técnicas de Machine Learning para predecir ganadores de torneos del PGA Tour, basado en estadísticas históricas de los jugadores.

## Modelos utilizados

- Regresión Logística
- Regresión Logística Regularizada (GLMNET)
- Random Forest (ajustado con hiperparámetros)

## Resultados

| Modelo                         | Precisión | AUC    |
|--------------------------------|-----------|--------|
| Regresión Logística            | 0.9877    | 0.8872 |
| GLMNET                         | 0.9847    | 0.9819 |
| Random Forest (ajustado)       | 0.9859    | 0.9858 |

## Requisitos

- R
- Librerías: tidyverse, caret, glmnet, randomForest, pROC, etc.

## Cómo correr el modelo

1. Clona este repositorio
2. Abre `modelo_pga.R` en RStudio
3. Ejecuta el script completo

## Autor

[Nicolas Castro](https://github.com/iCastr0)  
[Linkedln](https://www.linkedin.com/in/nicol%C3%A1s-castro-palma-324071274/)

