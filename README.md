# Predicción de Rotación de Empleados (Employee Churn)

Proyecto académico individual — Licenciatura en Ciencia de Datos (UNGB)

## Objetivo

Predecir si un empleado va a dejar la empresa (`left`) a partir de variables como
satisfacción laboral, evaluación de desempeño, cantidad de proyectos, horas
mensuales trabajadas, antigüedad, accidentes laborales, promociones, salario y
departamento.

## Dataset

Base de RR.HH. con ~15.000 registros y 10 variables (numéricas y categóricas).

## Flujo de trabajo

1. **Exploración inicial (EDA):** revisión de tipos de datos, estadísticas
   descriptivas y valores faltantes.
2. **Preprocesamiento:**
   - Codificación ordinal de `salary` (low / medium / high).
   - One-Hot Encoding del departamento (`sales`), evitando multicolinealidad
     (`drop_first=True`).
   - Escalado de variables numéricas con `StandardScaler`.
3. **Análisis de correlación:** matriz de correlación entre variables para
   detectar relaciones relevantes con la variable objetivo.
4. **Reducción de dimensionalidad:** PCA sobre las variables preprocesadas,
   con análisis de varianza explicada acumulada y de los pesos (*loadings*)
   del primer componente principal.
5. **Modelado:** entrenamiento y comparación de 4 algoritmos de clasificación
   sobre el conjunto reducido por PCA:
   - Naive Bayes (Gaussiano)
   - LDA (Análisis Discriminante Lineal)
   - QDA (Análisis Discriminante Cuadrático)
   - SVM con kernel RBF
6. **Evaluación:** Accuracy, Precisión, Recall y matriz de confusión para
   cada modelo, con gráfico comparativo final.

## Resultados

| Modelo         | Accuracy | Precisión | Recall |
|----------------|---------:|----------:|-------:|
| Naive Bayes    | 0.862    | 0.693     | 0.758  |
| LDA            | 0.782    | 0.571     | 0.340  |
| QDA            | 0.912    | 0.779     | 0.881  |
| **SVM (RBF)**  | **0.965**| **0.938** | **0.912** |

El modelo **SVM con kernel RBF** obtuvo el mejor desempeño en las tres
métricas, a costa de un mayor consumo computacional respecto a los modelos
lineales/paramétricos (Naive Bayes, LDA, QDA).

## Herramientas

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn

## Próximos pasos

- Ajuste de hiperparámetros del SVM (`GridSearchCV`) para optimizar aún más
  el recall, priorizando la detección de empleados en riesgo de irse.
- Evaluar interpretabilidad (SHAP) para identificar los principales
  impulsores de la rotación.
