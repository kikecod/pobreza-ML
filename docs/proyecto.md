# UNIVERSIDAD MAYOR DE SAN ANDRÉS
### FACULTAD DE CIENCIAS PURAS Y NATURALES - CARRERA DE INFORMÁTICA
**PACENSIS DIVI ANDRE Æ** | **UNIVERSITAS MAJOR**

## "Predicción de la Condición de Pobreza (p0) en Bolivia utilizando Aprendizaje Supervisado"

* **MATERIA:** APRENDIZAJE AUTOMÁTICO
* **PARALELO:** A
* **APELLIDOS:** FERNANDEZ CHIRI
* **NOMBRES:** ENRIQUE RAFAEL (C.I.: 10900348)
* **FECHA:** 24 DE MARZO DEL 2026
* **LUGAR:** LA PAZ BOLIVIA

---

## 1. Introducción
El presente proyecto propone el diseño, desarrollo y despliegue de un sistema de aprendizaje automático supervisado de alto impacto social, orientado a predecir la condición de pobreza de los hogares bolivianos (etiqueta `target_pobreza`, derivada de `p0`) sin requerir el ingreso monetario como variable de entrada (feature).

* **¿Qué se hará?** Se construirá un modelo predictivo basado en el algoritmo Extreme Gradient Boosting (XGBoost), integrado dentro de una arquitectura de Operaciones de Machine Learning (MLOps) y desplegado en una aplicación web interactiva.
* **¿Por qué se hace?** Las mediciones tradicionales de pobreza basadas en ingresos enfrentan severos problemas de subdeclaración y alta volatilidad, lo que genera errores de inclusión y exclusión en la asignación de programas sociales.
* **¿Para qué se hace?** Para dotar al Estado y a los tomadores de decisiones de una herramienta analítica precisa, auditable y en tiempo real que optimice la focalización de recursos y maximice el impacto de las políticas públicas.
* **¿Con qué se hará?** Se utilizarán los microdatos oficiales de la Encuesta de Hogares 2023 (BOL-INE-EH-2023) del INE, procesados mediante Python. El seguimiento y versionado de experimentos se realiza con MLflow (tracking local), el modelo se serializa para consumo en producción y se expone mediante una API REST con FastAPI junto a un frontend web (HTML/CSS/JavaScript). La solución está contenerizada con Docker y preparada para desplegarse como contenedor (por ejemplo, en AWS ECS/Fargate).

## 2. Antecedentes
La medición y mitigación de la pobreza ha sido, históricamente, uno de los ejes centrales de la política pública en el Estado Plurinacional de Bolivia. Durante las últimas dos décadas, el panorama social boliviano ha mostrado avances sumamente alentadores en términos de indicadores macroeconómicos. De acuerdo con las estadísticas oficiales, la pobreza monetaria, medida a través de la línea de pobreza absoluta, experimentó una reducción drástica a nivel nacional, cayendo del 66% a principios de la década de los 2000 a un 39%. De manera aún más notable, la pobreza extrema logró reducirse a un 11,1% para el año 2021, impulsada principalmente por políticas de protección social y transferencias no contributivas como la Renta Dignidad y el Bono Juana Azurduy. Estos programas han sido fundamentales para garantizar un ingreso mínimo y cerrar brechas en salud y educación entre áreas urbanas y rurales.

Sin embargo, a pesar de estos logros cuantitativos, la persistencia de la pobreza estructural revela las profundas limitaciones de las metodologías tradicionales de medición. El análisis de la pobreza basado exclusivamente en el ingreso de los hogares es insuficiente e inestable. Los ingresos monetarios son altamente susceptibles a choques económicos a corto plazo, la informalidad laboral predominante en la región y el fenómeno crónico de la subdeclaración en las encuestas. Como respuesta a estas deficiencias, instituciones investigativas y el propio Estado han comenzado a transitar hacia enfoques de vulnerabilidad multidimensional, los cuales consideran que la pobreza es una situación en la que las personas carecen de garantías para el ejercicio de derechos fundamentales en dimensiones económicas, sociales y ambientales.

En este contexto de transformación analítica, la revolución de la ciencia de datos y el aprendizaje automático (Machine Learning) ha comenzado a permear las estructuras de gobernanza en América Latina. Específicamente en Bolivia, investigaciones recientes han demostrado la viabilidad técnica de predecir la pobreza utilizando exclusivamente variables no monetarias estructurales, tales como las características de la vivienda, el acceso a servicios básicos y el nivel educativo. Estudios pioneros han aplicado algoritmos de clasificación de aprendizaje supervisado, como Bosques Aleatorios (Random Forest) y Extreme Gradient Boosting (XGBoost), sobre los datos de las Encuestas de Hogares del Instituto Nacional de Estadística (INE), logrando tasas de exactitud superiores al 84% al clasificar a los hogares sin necesidad de preguntar directamente por sus ingresos.

El principal desafío actual, no obstante, trasciende la mera precisión del modelo en un entorno de laboratorio. A nivel gubernamental y de políticas públicas, los modelos de aprendizaje automático enfrentan barreras organizacionales significativas que impiden su adopción sostenible, tales como el acceso burocrático a los datos, la carencia de conjuntos de datos versionados y la falta de infraestructuras cívicas que garanticen la reproducibilidad y transparencia. La literatura especializada subraya que, en el sector público, el aprendizaje automático responsable no es únicamente un problema de modelado matemático, sino un desafío de ingeniería institucional. Por ende, surge la necesidad inminente de transitar de experimentos aislados a sistemas robustos basados en prácticas de Operaciones de Machine Learning (MLOps), que permitan automatizar la actualización, el monitoreo y el despliegue de estos modelos predictivos para generar un impacto social continuo y auditable.

## 3. Problemática
La identificación precisa de las poblaciones vulnerables constituye el pilar fundamental para la eficacia de cualquier sistema de protección social. El problema central que aborda este proyecto es la alta incidencia de errores de focalización en los programas de asistencia gubernamental debido a la utilización de mecanismos de evaluación estáticos, costosos y metodológicamente limitados. Tradicionalmente, los Estados utilizan pruebas de comprobación de medios indirectos (Proxy Means Tests) o encuestas extensas que dependen de umbrales rígidos para determinar la elegibilidad de un hogar.

Este enfoque genera dos consecuencias perjudiciales de alto impacto:
* En primer lugar, los errores de inclusión, donde hogares que no se encuentran en situación de pobreza estructural logran acceder a subsidios debido a fluctuaciones temporales en sus ingresos o a la manipulación de sus declaraciones, lo que resulta en un desperdicio significativo de los limitados recursos fiscales del Estado.
* En segundo lugar, y de manera más crítica desde una perspectiva de derechos humanos, se producen errores de exclusión, donde familias con carencias estructurales severas quedan fuera de la red de protección social debido a umbrales inflexibles o a la lentitud en la actualización de los censos y encuestas.

La raíz metodológica de este problema es la dependencia de modelos lineales estáticos que no logran capturar las interacciones complejas y no lineales entre las múltiples dimensiones de la vulnerabilidad (educación, infraestructura, demografía). Sumado a esto, la ausencia de una infraestructura tecnológica moderna (MLOps) en las instituciones públicas impide que los modelos de evaluación se actualicen y reentrenen automáticamente a medida que emergen nuevos datos demográficos, condenando a las políticas públicas a operar con información obsoleta.

### Resumen del Árbol de Problemas
* **Problema Central:** Alta incidencia de errores de inclusión y exclusión en la focalización de programas sociales por el uso de modelos de medición estáticos y dependientes del ingreso monetario en Bolivia.
* **Causas Raíz:**
    * Limitación de los modelos lineales para capturar la complejidad multidimensional, lo que lleva a la carencia de herramientas analíticas predictivas en el sector público.
    * Alta subdeclaración y volatilidad del ingreso monetario en encuestas (INE).
    * Ausencia de arquitecturas tecnológicas (MLOps) para actualizar modelos en tiempo real, resultando en costos elevados de actualización de censos y encuestas tradicionales.
* **Efectos Principales:**
    * Exclusión sistemática de poblaciones marginadas de la red de protección.
    * Ineficiencia fiscal y desperdicio de recursos del Estado en subsidios.
    * Perpetuación de la desigualdad estructural y vulneración de derechos sociales.

## 4. Objetivo
El propósito general de esta investigación aplicada es diseñar, entrenar y desplegar en la nube un sistema de aprendizaje automático supervisado, basado en un modelo XGBoost optimizado e integrado en una arquitectura MLOps, para predecir la condición de pobreza (etiqueta `target_pobreza`, derivada de `p0`) en Bolivia a partir de variables no monetarias de la Encuesta de Hogares del INE, permitiendo a los formuladores de políticas públicas focalizar la asistencia social mediante una interfaz web interactiva.

Para alcanzar esta meta, se establecen los siguientes **objetivos específicos**:
1.  Extraer y procesar los microdatos oficiales de la Encuesta de Hogares 2023 del INE, aplicando técnicas de ingeniería de características y remuestreo sintético (SMOTE) para corregir el desequilibrio de clases propio de los indicadores de pobreza.
2.  Entrenar un clasificador XGBoost con hiperparámetros robustos y evaluar su desempeño con validación cruzada estratificada y métricas AUC-ROC, priorizando la reducción del error de exclusión (falsos negativos).
3.  Incorporar módulos de interpretabilidad del modelo (SHAP) para garantizar que las predicciones sean transparentes, explicables y éticamente justificables ante los organismos gubernamentales.
4.  Implementar prácticas MLOps para asegurar reproducibilidad, trazabilidad y versionado del modelo (métricas, parámetros, artefactos), habilitando reentrenamiento controlado.
5.  Desplegar la solución predictiva mediante una aplicación web (FastAPI + frontend), incluyendo un panel MLOps para visualizar versiones (hash del artefacto) y para lanzar entrenamiento/reentrenamiento desde la web.

### Resumen del Árbol de Objetivos
* **Objetivo Central:** Desplegar un sistema web predictivo con MLOps y algoritmos de aprendizaje supervisado para focalizar la condición de pobreza (según `p0`) en Bolivia sin requerir el ingreso como variable de entrada.
* **Medios y Acciones:**
    * Entrenamiento de algoritmos de Ensamble (XGBoost) para capturar no linealidades, incorporando valores SHAP para garantizar la explicabilidad del modelo.
    * Uso exclusivo de variables proxy no monetarias de la Encuesta de Hogares.
    * Implementación de arquitectura MLOps para despliegue y monitoreo web (FastAPI + Docker), y preparación de CI/CD con GitHub Actions para despliegue como contenedor (p. ej., AWS ECS).
* **Fines Perseguidos:**
    * Cobertura universal y precisa de las familias bolivianas con mayor vulnerabilidad.
    * Optimización del presupuesto estatal minimizando el error de inclusión.
    * Reducción sostenida de la vulnerabilidad mediante políticas públicas inclusivas.

## 5. Justificación
La implementación de este proyecto posee una fundamentación multidimensional que abarca esferas sociales, académicas, técnicas y científicas, respondiendo de manera directa a los retos actuales del Estado boliviano.

### Justificación Social
La política pública moderna debe transitar hacia enfoques centrados en el ciudadano que "no dejen a nadie atrás". El impacto social de un modelo predictivo altamente exacto radica en su capacidad para garantizar el ejercicio de derechos sociales. Al reducir el error de exclusión metodológica, el sistema propuesto asegura que transferencias clave -como el Bono Juana Azurduy o la Renta Dignidad- lleguen efectivamente a las poblaciones geográficamente aisladas y estructuralmente vulnerables. Además, independizar la medición del ingreso monetario protege a las familias cuyas carencias son de infraestructura y servicios básicos.

### Justificación Académica
Desde la perspectiva académica, el proyecto contribuye a la literatura de la economía computacional aplicada y la ciencia de datos en contextos de países en vías de desarrollo. El estudio proporciona un caso de uso empírico y documentado sobre cómo los datos abiertos del Instituto Nacional de Estadística pueden ser explotados más allá de la estadística descriptiva.

### Justificación Técnica
Técnicamente, el proyecto aborda uno de los cuellos de botella más severos en la aplicación de la IA: el despliegue en producción. Aproximadamente el 80% de los modelos creados en entornos de investigación nunca se integran a procesos operativos. Al orquestar el ciclo de vida del modelo mediante MLOps (MLflow, GitHub Actions y Docker), se entrega un producto de software continuo, escalable, versionado y resistente a la deriva de datos.

### Justificación Científica
El enfoque científico se sustenta en la adopción de algoritmos de Ensamble de Árboles con Gradiente (XGBoost), que han demostrado superioridad matemática para modelar distribuciones complejas. Asimismo, la integración del marco SHAP otorga una interpretación de la contribución marginal exacta de cada variable, garantizando que el modelo sea auditable y libre de sesgos ocultos.

## 6. Metodología
Para garantizar el rigor procedimental, el desarrollo se rige estrictamente por la metodología CRISP-DM (Cross-Industry Standard Process for Data Mining).

* **Fase 1: Comprensión del Negocio:** Se define que el sistema no requiere el ingreso monetario como variable de entrada y que la salida es una clasificación binaria `target_pobreza` (derivada de `p0`).
* **Fase 2: Comprensión de los Datos:** Se utilizarán los Microdatos de la Encuesta de Hogares 2023 (Identificador: BOL-INE-EH-2023) del INE. Requiere exploración de módulos, entendiendo codificación, valores atípicos y metadatos.
* **Fase 3: Preparación de los Datos:** Se implementa un Data Pipeline para integrar archivos usando la llave "folio". Incluye Ingeniería de Características (tasa de hacinamiento, años de educación, índice de equipamiento) y el manejo de desequilibrio mediante SMOTE.
* **Fase 4: Modelado:** Se utiliza XGBoost como modelo principal, configurando hiperparámetros robustos para evitar sobreajuste y maximizar capacidad de generalización en datos de encuesta.
* **Fase 5: Evaluación:** Se utiliza validación cruzada estratificada (AUC-ROC) y evaluación en conjunto de prueba. Se integra SHAP para interpretabilidad global e individual.
* **Fase 6: Despliegue:** El modelo se serializa como artefacto para producción y se expone mediante una API REST con FastAPI junto a un frontend web interactivo. Se añade un panel MLOps para visualizar versión del modelo y lanzar reentrenamiento desde la web.

## 7. Desarrollo Analítico y Modelos de Aprendizaje Supervisado
El desarrollo se sustenta en el conjunto de datos BOL-INE-EH-2023. 

### Diccionario y Selección Exacta del Dataset
Las siguientes variables estructurales son seleccionadas como el vector de características:

| Variable Derivada | Origen Módulo INE 2023 | Descripción Estadística |
| :--- | :--- | :--- |
| `anios_educ_jefe` | EH2023_Persona | Años de escolaridad formal aprobados por el jefe del hogar. Variable continua con alta correlación inversa a la pobreza. |
| `hacinamiento` | EH2023_Vivienda | Ratio calculado entre total de miembros del hogar (`totper`) y número de habitaciones para dormir (`s06a_17`). |
| `indice_equipamiento` | EH2023_Equipamiento | Índice en rango [0,1] calculado como suma normalizada de bienes duraderos reportados (pivot por `item` usando `s08b_1`). |
| `afiliacion_afp` | EH2023_Persona | Variable binaria. 1 si el principal proveedor aporta a pensiones de largo plazo, 0 si pertenece a economía informal. |
| `material_vivienda` | EH2023_Vivienda | Material de paredes (`s06a_03`) mapeado a categorías y codificado con One-Hot Encoding. |
| `area` | EH2023_Vivienda | Área de residencia (`area`: urbana/rural) tratada como categórica (One-Hot Encoding). |
| `target_pobreza` | EH2023_Persona | Variable objetivo derivada de `p0` (indicador de pobreza por ingreso, 0/1) para el jefe del hogar. |

### Arquitectura del Algoritmo Supervisado: XGBoost
XGBoost es un modelo supervisado de ensamble que construye árboles de decisión débiles secuencialmente. La predicción final se calcula aditivamente:

$$\tilde{y}_{i}=\sum_{k=1}^{K}f_{k}(x_{i}), f_{k}\in\mathcal{F}$$

A diferencia de otros algoritmos, XGBoost optimiza directamente una función objetivo formal:

$$Obj=\sum_{i=1}^{n}L(y_{i},\hat{y}_{i})+\sum_{k=1}^{K}\Omega(f_{k})$$

$$\Omega(f)=\gamma T+\frac{1}{2}\lambda\sum_{j=1}^{T}w_{j}^{2}$$

El término penaliza los nodos hojas, exigiendo que cada nueva división aporte ganancia, mientras que aplica contracción para garantizar predicciones estables.

### Análisis de la Salida Gráfica
El desarrollo en Python proporciona dos salidas visuales fundamentales:
1.  **Gráfico de la Curva ROC:** Expone el balance estadístico entre sensibilidad y especificidad. Un área bajo la curva (AUC) más cercana a 1 indica mayor poder de discriminación. En el proyecto, AUC-ROC es la métrica principal para comparar desempeño en validación cruzada y en el conjunto de prueba.
2.  **Gráfico de Resumen de Impacto SHAP ("beeswarm"):** Representa a cada hogar de prueba. El eje Y ordena características por importancia y el eje X muestra el impacto numérico en la probabilidad de ser clasificado como "pobre". Las gradaciones de colores revelan relaciones causales, respaldando que la IA emula el razonamiento socioeconómico estructurado.

## 8. Herramientas y Técnicas: Arquitectura MLOps
Este proyecto se rige por buenas prácticas de Operaciones de Machine Learning (MLOps), unificando desarrollo y operaciones para asegurar trazabilidad, reproducibilidad y preparación de CI/CD del servicio de predicción.

| Herramienta | Capa Arquitectónica | Descripción y Uso Específico en el Proyecto |
| :--- | :--- | :--- |
| **Git/GitHub** | Control de Versiones | Repositorio centralizado que aloja todo el código fuente. Es la fuente de la verdad para el trabajo colaborativo. |
| **MLflow** | Tracking de Experimentos | Tracking local (`mlruns/`) para registrar parámetros, métricas y artefactos (modelo). Los runs pueden visualizarse desde el panel web MLOps. |
| **GitHub Actions** | Automatización (CI/CD) | Orquestador de integración y despliegue continuo. Ejecuta validaciones de calidad (quality gate por AUC) y construye la imagen Docker para despliegue como contenedor. |
| **Docker** | Contenerización | Empaqueta modelo, bibliotecas y entorno en contenedores inmutables, asegurando que se ejecute de manera idéntica en cualquier servidor. |
| **FastAPI + Frontend Web** | Serving e Interfaz | API REST para predicción y endpoints MLOps; frontend estático (HTML/CSS/JS) con páginas de predictor, informe, gráficos y panel MLOps. |

### Mecánica del Flujo de Trabajo (Workflow)
El flujo de trabajo contempla trazabilidad del entrenamiento (MLflow) y un *quality gate* para el despliegue. Las métricas del modelo se exportan a `output/metrics.json` y el workflow de GitHub Actions valida que `auc_test` cumpla el umbral configurado (por defecto 0.84). Si el umbral no se cumple, el despliegue se bloquea; si se cumple, se construye la imagen Docker y se habilita el despliegue como contenedor (por ejemplo, en AWS ECS/Fargate).

## 9. Conclusiones
La elaboración del proyecto evidencia una convergencia crítica entre la innovación algorítmica y la resolución de urgencias sociales estructurales. Al reemplazar la dependencia de variables monetarias como *features* por variables no monetarias (vivienda, educación, equipamiento y afiliación), el sistema busca mejorar la focalización de programas sociales sin depender de la declaración directa de ingresos.

En la versión actual del artefacto del modelo, las métricas reportadas en `output/metrics.json` son: AUC en validación cruzada $\approx 0.8182$ y AUC en prueba $\approx 0.7598$. Estas métricas se exponen en la aplicación web y forman parte del control de calidad del pipeline. Asimismo, el avance técnico mediante CRISP-DM y MLOps sienta un precedente tecnológico para el sector público al incorporar trazabilidad (MLflow), versionado del artefacto y despliegue como aplicación web (FastAPI + frontend) preparada para contenerización y despliegue como contenedor.

## Anexos
**Tabla Anexo 1: Variables técnicas crudas utilizadas en el pipeline (BOL-INE-EH-2023)**
La siguiente tabla lista las variables originales utilizadas por el pipeline para construir las features finales y la etiqueta.

| Variable Técnica | Módulo INE 2023 | Tipo | Uso en el pipeline |
| :--- | :--- | :--- | :--- |
| `folio` | Todos | Llave primaria | Llave para uniones relacionales y agregaciones a nivel hogar. |
| `s01a_05` | EH2023_Persona | Discreta | Filtro para jefe/jefa del hogar (`s01a_05 == 1`). |
| `aestudio` | EH2023_Persona | Discreta/Continua | Base para construir `anios_educ_jefe`. |
| `s04f_36` | EH2023_Persona | Discreta | Base para construir `afiliacion_afp` (aporta actualmente). |
| `p0` | EH2023_Persona | Binaria | Etiqueta usada para construir `target_pobreza`. |
| `totper` | EH2023_Vivienda | Discreta | Numerador para calcular `hacinamiento`. |
| `s06a_17` | EH2023_Vivienda | Discreta | Denominador para calcular `hacinamiento` (habitaciones para dormir). |
| `s06a_03` | EH2023_Vivienda | Categórica | Base para construir `material_vivienda` (mapeo + One-Hot Encoding). |
| `area` | EH2023_Vivienda | Categórica | Base para `area` (urbana/rural, One-Hot Encoding). |
| `item` | EH2023_Equipamiento | Discreta | Identifica el bien duradero para el pivot. |
| `s08b_1` | EH2023_Equipamiento | Discreta | Base para `indice_equipamiento` (1=posee, 2=no posee). |

## Fuentes citadas
1. Pobreza y movilidad social en Bolivia en la última década - IADB
2. Bolivia reduce la pobreza extrema a 11,1% con políticas de protección social - MEFP
3. Desafíos del desarrollo: Vulnerabilidad multidimensional - PNUD
4. CEDLA revela crecimiento de la pobreza multidimensional en ciudades del eje
5. Predicción de la pobreza en Bolivia usando Machine Learning - MEFP
6. ¿Qué puede esperar ALC de la revolución del "machine learning"?
7. Clasificación de la pobreza en Bolivia utilizando Random Forest y XGBoost
8. Clasificación de la Pobreza en Bolivia, Utilizando Random Forest y XGBoost
9. From Pre-labeling to Production: Engineering Lessons from a Machine Learning Pipeline
10. ¿Qué son las MLOps?: explicación - AWS
11. What is MLOps? - Databricks
12. A machine learning proposal to predict poverty - Dialnet
13. IMPORTANCIA DE LA CIENCIA DE LOS DATOS EN LA SOCIEDAD
14. La Generación de Datos Abiertos en Bolivia - UMSA
15. MLOps: Continuous delivery and automation pipelines - Cloud Architecture Center
16. Evolution of poverty in Bolivian communities 2012-2022 - MPRA
17. Predicción de la pobreza en Bolivia usando Machine Learning - MEFP
18-22. Encuesta de Hogares 2023 - ANDA (INE)
23. Clasificación de la pobreza en Bolivia - ResearchGate
24. Despliegue de contenedores en la nube (AWS)
25. How to build an MLOps pipeline? - LeewayHertz
26. Encuesta de Hogares - INE
27. Census income classification with XGBoost - SHAP documentation
28. MLOps - Best Practices for Public Sector Organizations
29. MLOps Best Practices - Databricks
30-32. MLOps Pipelines and CI/CD with GitHub Actions
33. Despliegue de modelos de machine learning en entornos de producción
34-35. Despliegue de APIs con FastAPI y contenedores (Docker)
36. End-to-End MLOps Architecture Design Guide - AWS
37. Bolivia en la era de la IA