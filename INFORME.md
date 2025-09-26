# Informe del proyecto

## Planteamiento
Se evalua el impacto de habitos de estudio y metodo pedagogico (A/B) sobre la puntuacion (0-100) y la probabilidad de aprobar (>=60).

## Datos
Datos sinteticos (n=300) generados conforme al documento base. Variables: metodo, horas_estudio, horas_suenio, indice_socioeco, gpa_previo, puntuacion, aprueba.

## Metodologia
- Contraste: Welch t-test para comparar medias entre metodos.
- Incertidumbre: Bootstrap (2000) de delta de medias A-B para IC95%.
- Modelo lineal: OLS y Ridge (alpha) para predecir puntuacion. Metricas: MSE, R2.
- Modelo logistico: Clasificacion de aprueba (0/1) con control de C y umbral. Metricas: accuracy, sensibilidad, especificidad, ROC-AUC.
- DOE computacional: se varian hiperparametros (train size, alpha, C, umbral y features) de forma controlada y se observan metricas en tiempo real.
- Regularizacion: alpha>0 activa Ridge para mitigar sobreajuste.
- Mejora iterativa: los controles permiten explorar combinaciones hasta alcanzar desempeno estable.

## Decisiones tecnicas
- Welch por posibles varianzas distintas entre grupos.
- Bootstrap para no depender de supuestos parametricos estrictos en IC de la diferencia de medias.
- Ridge para estabilizar coeficientes.
- Umbral en logistica para alinear el clasificador con costos/beneficios.

## Resultados (ejemplo con parametros por defecto)
- Las metricas se muestran en el dashboard (tarjetas y graficas). R2 moderado en el modelo lineal; ROC-AUC informativa para la logistica.
- El IC95% del bootstrap de delta medias muestra magnitud e incertidumbre del efecto del metodo.

## Conclusiones
- El pipeline permite diagnosticar, ajustar y justificar el modelo con evidencia cuantitativa.
- La interfaz guia una narrativa reproducible de como los cambios de hiperparametros afectan el desempeno.
- Recomendacion: si hay sobreajuste, incrementar alpha (Ridge) y/o simplificar features; para clasificacion desbalanceada, ajustar umbral o ponderar clases.

## Reproducibilidad
- dataset.py fija random_state.
- Requisitos y comandos estan documentados en README.md.
