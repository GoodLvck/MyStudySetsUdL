# Aprendizaje Bayesiano — Resumen del Temario Completo

> **Fuente:** Tema 2 — Aprendizaje Bayesiano (ARA, Aprendizaje, Razonamiento y Agentes)

---

## Tabla de Contenidos

1. [Introducción al Aprendizaje Bayesiano](#1-introducción-al-aprendizaje-bayesiano)
2. [Aprendizaje de Parámetros](#2-aprendizaje-de-parámetros)
   - 2.1 [Estimación por Máxima Verosimilitud (MLE)](#21-estimación-por-máxima-verosimilitud-mle)
   - 2.2 [El Problema de las Probabilidades Cero](#22-el-problema-de-las-probabilidades-cero)
3. [Aprendizaje de Estructura](#3-aprendizaje-de-estructura)
   - 3.1 [Espacio de Búsqueda](#31-espacio-de-búsqueda)
   - 3.2 [Función de Scoring](#32-función-de-scoring)
4. [Puntuación UPSM (Bayesian Scoring)](#4-puntuación-upsm-bayesian-scoring)
   - 4.1 [Derivación mediante Integrales de Dirichlet](#41-derivación-mediante-integrales-de-dirichlet)
   - 4.2 [Propiedad de Scoring Local](#42-propiedad-de-scoring-local)
   - 4.3 [Mejor y Peor Caso](#43-mejor-y-peor-caso)
5. [Penalización BIC/MDL](#5-penalización-bicmdl)
6. [Algoritmo K2](#6-algoritmo-k2)
   - 6.1 [Pseudocódigo](#61-pseudocódigo)
   - 6.2 [Limitaciones del Greedy](#62-limitaciones-del-greedy)
7. [Overfitting y Regularización](#7-overfitting-y-regularización)
8. [Ejemplo Práctico: Speed Dating](#8-ejemplo-práctico-speed-dating)
9. [Bibliografía](#9-bibliografía)

---

## 1. Introducción al Aprendizaje Bayesiano

El **aprendizaje bayesiano** tiene como objetivo **estimar** la distribución de probabilidad subyacente a partir de un **conjunto de datos observados** (base de datos D).

### Dos niveles de aprendizaje

| Nivel | Objetivo | Supuesto |
|---|---|---|
| **Aprendizaje de parámetros** | Estimar las CPTs dado el grafo G | El grafo G es conocido |
| **Aprendizaje de estructura** | Estimar el grafo G a partir de D | Todo es desconocido |

En la práctica, el aprendizaje de estructura es mucho más difícil: el espacio de DAGs sobre `n` nodos es superexponencial en n.

### Notación

- **D** = base de datos con N registros (ejemplos i.i.d.)
- **Xᵢ** = variable aleatoria con `rᵢ` valores posibles: {xᵢ₁, ..., xᵢrᵢ}
- **πᵢ** = conjunto de padres de Xᵢ en el DAG G
- **qᵢ** = número de configuraciones posibles de los padres πᵢ = ∏ⱼ∈πᵢ rⱼ
- **Nᵢⱼₖ** = número de registros en D donde Xᵢ = xᵢₖ y πᵢ tiene la j-ésima configuración
- **Nᵢⱼ** = Σₖ Nᵢⱼₖ = número total de registros con la j-ésima configuración de padres

---

## 2. Aprendizaje de Parámetros

### 2.1 Estimación por Máxima Verosimilitud (MLE)

Dado el grafo G (estructura conocida), queremos estimar los parámetros **θᵢⱼₖ = P(Xᵢ = xᵢₖ | πᵢ = wᵢⱼ)**.

La **estimación por máxima verosimilitud** elige los parámetros que maximizan la probabilidad de los datos observados:

> **θ̂ᵢⱼₖ = Nᵢⱼₖ / Nᵢⱼ**

Es decir, simplemente la **frecuencia relativa** observada en los datos.

**Derivación:** La log-verosimilitud del conjunto de datos se descompone en contribuciones independientes por nodo (gracias a la factorización de la BN):

```
log P(D | G, θ) = Σᵢ Σⱼ Σₖ Nᵢⱼₖ · log θᵢⱼₖ
```

Maximizando respecto a θᵢⱼₖ con la restricción Σₖ θᵢⱼₖ = 1 (usando multiplicadores de Lagrange):

```
∂/∂θᵢⱼₖ [Σₖ Nᵢⱼₖ · log θᵢⱼₖ - λ(Σₖ θᵢⱼₖ - 1)] = 0

⟹ Nᵢⱼₖ / θᵢⱼₖ = λ    ⟹    θ̂ᵢⱼₖ = Nᵢⱼₖ / Nᵢⱼ
```

**Ejemplo:**

| X₁ | X₂ | count |
|---|---|---|
| 0 | 0 | 3 |
| 0 | 1 | 1 |
| 1 | 0 | 2 |
| 1 | 1 | 4 |

Con la red X₁ → X₂:
- θ̂(X₁=0) = 4/10 = 0.4, θ̂(X₁=1) = 6/10 = 0.6
- θ̂(X₂=0|X₁=0) = 3/4 = 0.75, θ̂(X₂=1|X₁=0) = 1/4 = 0.25
- θ̂(X₂=0|X₁=1) = 2/6 ≈ 0.33, θ̂(X₂=1|X₁=1) = 4/6 ≈ 0.67

---

### 2.2 El Problema de las Probabilidades Cero

Si **ningún registro** en D tiene la combinación {Xᵢ = xᵢₖ, πᵢ = wᵢⱼ}, entonces:

> **Nᵢⱼₖ = 0 ⟹ θ̂ᵢⱼₖ = 0/Nᵢⱼ = 0**

**Consecuencia grave:** Una probabilidad estimada a 0 hace que **cualquier asignación que contenga ese evento tenga probabilidad 0**, independientemente del resto de la evidencia. Esto invalida el modelo para esos casos.

**Causas principales:**
- Base de datos pequeña: combinaciones raras nunca aparecen.
- Variables con muchos valores: el espacio de combinaciones crece exponencialmente.

**Soluciones:**

| Solución | Descripción |
|---|---|
| **Suavizado de Laplace (Laplace Smoothing)** | Añadir 1 (o α) a todos los conteos: θ̂ᵢⱼₖ = (Nᵢⱼₖ + α) / (Nᵢⱼ + α·rᵢ) |
| **Estimación Bayesiana** | Usar una distribución a priori de Dirichlet sobre los parámetros |
| **Redes con estructura más simple** | Reducir el número de padres para evitar combinaciones raras |

---

## 3. Aprendizaje de Estructura

### 3.1 Espacio de Búsqueda

El número de DAGs posibles sobre `n` nodos crece **superexponencialmente**:

| n | DAGs posibles |
|---|---|
| 1 | 1 |
| 2 | 3 |
| 3 | 25 |
| 4 | 543 |
| 5 | 29.281 |
| 10 | ~4.2 × 10¹⁸ |

Para n = 10, ya es imposible enumerar todos los DAGs posibles. Se necesitan **algoritmos heurísticos**.

### Operaciones de búsqueda en el espacio de DAGs

Los algoritmos de búsqueda se mueven por el espacio de DAGs mediante tres operaciones:

| Operación | Descripción |
|---|---|
| **Añadir arco** | Xᵢ → Xⱼ (si no genera ciclo) |
| **Eliminar arco** | Quitar Xᵢ → Xⱼ |
| **Invertir arco** | Xᵢ → Xⱼ pasa a ser Xⱼ → Xᵢ (si no genera ciclo) |

---

### 3.2 Función de Scoring

Para comparar estructuras candidatas, necesitamos una **función de puntuación** S(G, D) que mida qué tan bien el DAG G explica los datos D.

**Requisitos de una buena función de scoring:**
1. **Consistencia:** Con datos suficientes, identifica el grafo verdadero.
2. **Descomponibilidad local:** S(G, D) = Σᵢ s(Xᵢ, πᵢ, D) — se puede calcular nodo a nodo.
3. **Invariancia al reordenamiento:** El score no depende del orden de los datos.
4. **Penalización de complejidad:** Evita sobreajuste favoreciendo estructuras simples.

---

## 4. Puntuación UPSM (Bayesian Scoring)

La puntuación **UPSM** (también llamada **BD score** o puntuación de Dirichlet) es el logaritmo de la probabilidad marginal de los datos dado el grafo:

> **UPSM(G, D) = log P(D | G)**

### 4.1 Derivación mediante Integrales de Dirichlet

Bajo la hipótesis de que los parámetros de cada nodo siguen una **distribución a priori de Dirichlet** con hiperparámetros uniformes (αᵢⱼₖ = 1 para todos i, j, k), la integral de los parámetros se resuelve analíticamente:

```
P(D | G) = ∏ᵢ ∏ⱼ [ Γ(qᵢⱼ·rᵢ) / Γ(qᵢⱼ·rᵢ + Nᵢⱼ) · ∏ₖ Γ(1 + Nᵢⱼₖ) / Γ(1) ]
```

Simplificando con αᵢⱼₖ = 1 y usando Γ(n) = (n−1)!:

> **UPSM(Xᵢ, πᵢ, D) = Σⱼ [ log (rᵢ−1)! / (rᵢ−1+Nᵢⱼ)! + Σₖ log Nᵢⱼₖ! ]**

O equivalentemente:

```
UPSM(Xᵢ, πᵢ, D) = Σⱼ [ (rᵢ-1)!·(Nᵢⱼ-rᵢ)! / (Nᵢⱼ+rᵢ-1)! · ∏ₖ Nᵢⱼₖ! ]
```

**Versión logarítmica (más estable numéricamente):**

```
log UPSM(Xᵢ, πᵢ, D) = Σⱼ [ log(rᵢ-1)! - log(Nᵢⱼ+rᵢ-1)! + Σₖ log Nᵢⱼₖ! ]
```

---

### 4.2 Propiedad de Scoring Local

Una propiedad crucial: la puntuación UPSM es **descomponible localmente**:

> **UPSM(G, D) = Σᵢ UPSM(Xᵢ, πᵢ, D)**

Esto significa que el score global del grafo es la **suma de scores locales** de cada nodo. Cuando añadimos/eliminamos un arco, solo necesitamos **recalcular el score del nodo afectado**, no toda la red.

**Implicación algorítmica:** Hace que la búsqueda greedy sea eficiente — cada operación solo requiere recalcular el score de 1 ó 2 nodos.

---

### 4.3 Mejor y Peor Caso

**Peor caso:** Todos los registros son distintos — máxima dispersión de los datos.
```
Nᵢⱼₖ ∈ {0, 1} para todo i, j, k
UPSM → mínimo valor (penaliza la falta de datos repetidos)
```

**Mejor caso:** Todos los registros tienen los mismos valores para la configuración (i, j):
```
Nᵢⱼₖ = Nᵢⱼ para algún k, y Nᵢⱼₖ' = 0 para k' ≠ k
UPSM → máximo valor (evidencia perfectamente concentrada)
```

**Efecto del tamaño de la base de datos:**
- Con pocos datos: el UPSM favorece estructuras simples (menos parámetros).
- Con muchos datos: el UPSM puede sobreajustar (añadir arcos innecesarios).

---

## 5. Penalización BIC/MDL

Para evitar el **sobreajuste** (overfitting), se añade una penalización por complejidad del modelo.

### BIC (Bayesian Information Criterion)

> **BIC(G, D) = log P(D | G, θ̂) − (|G| / 2) · log N**

Donde:
- `log P(D | G, θ̂)` = log-verosimilitud maximizada (con parámetros MLE)
- `|G|` = número de parámetros libres del modelo
- `N` = tamaño de la base de datos
- El término `(|G| / 2) · log N` **penaliza** la complejidad

### MDL (Minimum Description Length)

El criterio **MDL** es equivalente al BIC. Proviene de la teoría de la información:

> **MDL(G, D) = −log P(D | G, θ̂) + (|G| / 2) · log N**

Minimizar MDL equivale a maximizar BIC. La interpretación es: el modelo que mejor comprime los datos es el mejor modelo.

### Número de parámetros libres

El número de parámetros libres de un nodo Xᵢ con padres πᵢ es:

> **|Xᵢ, πᵢ| = (rᵢ − 1) · qᵢ**

Donde `qᵢ` = número de configuraciones de padres = ∏ⱼ∈πᵢ rⱼ.

**Total del modelo:**
```
|G| = Σᵢ (rᵢ − 1) · qᵢ
```

### Trade-off ajuste vs. complejidad

| Situación | Efecto del término de penalización |
|---|---|
| N pequeño | Penalización grande → favorece modelos simples |
| N grande | Penalización relativa pequeña → permite modelos más complejos |
| Añadir arco innecesario | Mejora verosimilitud pero penalización supera la ganancia |
| Arco necesario | Ganancia en verosimilitud supera la penalización |

---

## 6. Algoritmo K2

El **algoritmo K2** (Cooper & Herskovits, 1992) es un algoritmo **greedy** para el aprendizaje de estructura de Redes Bayesianas.

### Supuesto clave

K2 requiere un **orden topológico** fijo sobre las variables: X₁, X₂, ..., Xₙ. Un arco Xᵢ → Xⱼ solo se permite si i < j (los padres siempre preceden a los hijos en el orden).

### 6.1 Pseudocódigo

```
function K2(variables, ordering, maxParents, D):
    G := empty DAG
    for i := 1 to n do
        πᵢ := {}                              // start with no parents
        score_old := UPSM(Xᵢ, πᵢ, D)
        improved := true
        while improved and |πᵢ| < maxParents do
            improved := false
            best_score := score_old
            best_parent := null
            for each Xⱼ with j < i and Xⱼ ∉ πᵢ do
                score_new := UPSM(Xᵢ, πᵢ ∪ {Xⱼ}, D)
                if score_new > best_score then
                    best_score := score_new
                    best_parent := Xⱼ
                end if
            end for
            if best_parent ≠ null then
                πᵢ := πᵢ ∪ {best_parent}
                score_old := best_score
                improved := true
            end if
        end while
        G := G with parents(Xᵢ) = πᵢ
    end for
    return G
```

### Complejidad

- **Por nodo:** O(n · maxParents) evaluaciones de UPSM.
- **Total:** O(n² · maxParents) evaluaciones de UPSM.
- Cada evaluación de UPSM(Xᵢ, πᵢ, D) es O(N · |πᵢ|) sobre la base de datos.

**Complejidad global:** O(n³ · N) en el peor caso.

---

### 6.2 Limitaciones del Greedy

K2 y los algoritmos greedy de búsqueda de estructura tienen limitaciones importantes:

| Limitación | Descripción |
|---|---|
| **Óptimo local** | Puede quedar atrapado en un DAG subóptimo |
| **Sensibilidad al orden** | El resultado depende fuertemente del orden inicial de variables |
| **No explora inversiones** | No puede deshacer decisiones anteriores |
| **maxParents** | Si está mal ajustado, corta búsqueda prematuramente |
| **Escala** | Para n > 20, el espacio de búsqueda es enorme |

**Estrategias para mejorar:**
- Múltiples reinicios con distintos órdenes.
- Operadores de búsqueda adicionales (añadir, eliminar, invertir).
- Algoritmos evolutivos o MCMC sobre el espacio de DAGs.

---

## 7. Overfitting y Regularización

### El problema del sobreajuste

Con MLE puro y sin penalización, el algoritmo tenderá a añadir arcos hasta que cada nodo tenga prácticamente todos los demás como padres. Esto produce un modelo que:

- Tiene un score de verosimilitud muy alto en los datos de entrenamiento.
- **Generaliza mal** a datos nuevos.
- Es computacionalmente costoso para la inferencia.

**Indicador de overfitting:** Un modelo con muchos arcos que no aportan información real — el "ruido" del conjunto de entrenamiento se interpreta como estructura.

### Regularización mediante penalización

La penalización BIC/MDL actúa como **regularizador**:

```
score_penalizado = log-verosimilitud − λ · complejidad
```

Donde λ controla el balance entre ajuste y simplicidad:
- λ pequeño → más arcos, posible sobreajuste.
- λ grande → menos arcos, posible subajuste.

Con BIC: λ = log(N)/2, que **crece con el tamaño de la base de datos** — con más datos, se puede justificar más complejidad.

---

## 8. Ejemplo Práctico: Speed Dating

El **experimento Speed Dating** de Columbia University es un caso de estudio para aprendizaje bayesiano.

### Descripción del experimento

En una sesión de speed dating, participantes (hombres y mujeres) tienen breves conversaciones de 4 minutos. Después, cada participante indica si le gustaría volver a ver a su pareja.

### Variables del modelo

| Variable | Tipo | Descripción |
|---|---|---|
| **ideo** | Binaria | Ideología política (liberal/conservador) |
| **sex** | Binaria | Sexo del participante |
| **talks** | Binaria | ¿Le gusta hablar en las citas? |
| **vege** | Binaria | ¿Es vegetariano/a? |
| **blade** | Binaria | ¿Le gusta la película Blade Runner? |
| **youlike** | Binaria | ¿Te ha gustado esta persona? (variable objetivo) |

### Objetivo del aprendizaje

Aprender la estructura de una Red Bayesiana que modele las **dependencias entre variables** y permita predecir `youlike` en función de los atributos del participante.

### Proceso con K2

1. **Preprocesamiento:** discretizar variables continuas, tratar valores perdidos.
2. **Fijar orden:** por ejemplo, {ideo, sex, talks, vege, blade, youlike}.
3. **Ejecutar K2** con maxParents = 3 (por ejemplo).
4. **Evaluar** con BIC o holdout.

### Resultado típico

La BN aprendida puede revelar relaciones como:
- `youlike` depende de `talks` y `blade` (quizás: intereses comunes en conversación y cultura).
- `ideo` y `vege` pueden estar correlacionadas (tendencias ideológicas).
- `sex` puede influir en la probabilidad de `youlike` (asimetría en preferencias).

### Evaluación del modelo

Para evaluar la BN aprendida se usa **validación cruzada** o un conjunto de test independiente:

```
Accuracy = (TP + TN) / N

o bien:

Log-loss = −Σᵢ [yᵢ · log P(youlike=1 | xᵢ) + (1−yᵢ) · log P(youlike=0 | xᵢ)]
```

---

## 9. Bibliografía

- **Cooper, G.F. & Herskovits, E.** — *A Bayesian Method for the Induction of Probabilistic Networks from Data*, Machine Learning 9, 1992. (Artículo original del algoritmo K2 y la puntuación BD.)
- **Heckerman, D., Geiger, D. & Chickering, D.M.** — *Learning Bayesian Networks: The Combination of Knowledge and Statistical Data*, Machine Learning, 1995.
- **Schwarz, G.** — *Estimating the Dimension of a Model*, Annals of Statistics, 1978. (Criterio BIC.)
- **Rissanen, J.** — *Stochastic Complexity in Statistical Inquiry*, World Scientific, 1989. (Principio MDL.)
- **Koller, D. & Friedman, N.** — *Probabilistic Graphical Models*, MIT Press, 2009. Capítulo 18 (Structure Learning).
- **Fisman, R. & Iyengar, S.** — *Gender Differences in Mate Selection: Evidence from a Speed Dating Experiment*, Quarterly Journal of Economics, 2006. (Datos del experimento.)

---

*Resumen generado a partir de las diapositivas del curso ARA. Para mayor detalle sobre demostraciones matemáticas y código de implementación, consultar los PDFs originales y la bibliografía citada.*
