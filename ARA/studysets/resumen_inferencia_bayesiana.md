# Inferencia Bayesiana — Resumen del Temario Completo

> **Fuente:** Tema 1 — Inferencia Bayesiana (ARA, Aprendizaje, Razonamiento y Agentes)

---

## Tabla de Contenidos

1. [Motivación: razonamiento bajo incertidumbre](#1-motivación-razonamiento-bajo-incertidumbre)
2. [Distribución Conjunta y Problema FO/DO](#2-distribución-conjunta-y-problema-fodo)
   - 2.1 [Regla de la Suma — Probabilidades Marginales](#21-regla-de-la-suma--probabilidades-marginales)
   - 2.2 [Probabilidad Condicional](#22-probabilidad-condicional)
   - 2.3 [El Problema de la Complejidad](#23-el-problema-de-la-complejidad)
3. [Teorema de Bayes](#3-teorema-de-bayes)
4. [Redes Bayesianas](#4-redes-bayesianas)
   - 4.1 [Definición y Estructura](#41-definición-y-estructura)
   - 4.2 [Factorización de la Distribución Conjunta](#42-factorización-de-la-distribución-conjunta)
   - 4.3 [Tablas de Probabilidad Condicional (CPT)](#43-tablas-de-probabilidad-condicional-cpt)
5. [Independencia Condicional](#5-independencia-condicional)
   - 5.1 [Tipos de Conexión](#51-tipos-de-conexión)
   - 5.2 [D-separación](#52-d-separación)
6. [Inferencia en Redes Bayesianas](#6-inferencia-en-redes-bayesianas)
   - 6.1 [Algoritmo ExpandProb](#61-algoritmo-expandprob)
   - 6.2 [Algoritmo QueryVar](#62-algoritmo-queryvar)
   - 6.3 [Complejidad](#63-complejidad)
7. [Inferencia Aproximada](#7-inferencia-aproximada)
8. [Ejemplos Completos](#8-ejemplos-completos)
9. [Bibliografía](#9-bibliografía)

---

## 1. Motivación: razonamiento bajo incertidumbre

Los agentes inteligentes deben tomar decisiones con **información incompleta o incierta**. La incertidumbre proviene de:

- **Parcialidad de la observación**: no se observan todas las variables del mundo.
- **Ruido en los sensores**: las medidas del entorno son imperfectas.
- **Incertidumbre inherente**: el mundo es estocástico por naturaleza.

### ¿Por qué probabilidad?

La **teoría de la probabilidad** proporciona una forma rigurosa de representar y razonar con incertidumbre. En lugar de afirmar verdad/falsedad, asignamos **grados de creencia** a proposiciones.

| Enfoque | Característica | Limitación |
|---|---|---|
| Lógica clásica | Certeza total, monótona | No maneja incertidumbre |
| Lógica difusa | Grados de verdad | No es probabilística |
| **Probabilidad** | Grados de creencia, coherente | Requiere datos o estimaciones |

### Probabilidad de Prior y Posterior

- **P(H)**: probabilidad a priori de una hipótesis antes de observar evidencia.
- **P(H | E)**: probabilidad a posteriori después de observar evidencia E.

---

## 2. Distribución Conjunta y Problema FO/DO

### El Problema FO (Full Observability)

Con **observabilidad completa**, conocemos el estado exacto del mundo. La probabilidad de cualquier evento es determinista (0 o 1). No es el caso habitual en IA.

### El Problema DO (Distribución Conjunta)

Con **observabilidad parcial**, definimos la **distribución de probabilidad conjunta** sobre todas las variables del dominio. Para `n` variables booleanas, esto requiere una tabla con **2ⁿ entradas**.

**Ejemplo** con 3 variables: *Caries* (C), *Dolor* (D), *Tiempo* (T):

| C | D | T | P(C, D, T) |
|---|---|---|---|
| T | T | T | 0.108 |
| T | T | F | 0.012 |
| T | F | T | 0.072 |
| T | F | F | 0.008 |
| F | T | T | 0.016 |
| F | T | F | 0.064 |
| F | F | T | 0.144 |
| F | F | F | 0.576 |

La suma de todas las entradas debe ser **1**.

---

### 2.1 Regla de la Suma — Probabilidades Marginales

Para obtener la probabilidad de un subconjunto de variables, **sumamos** (marginalizamos) sobre los valores de las variables restantes:

> **P(X) = Σ_y P(X, Y=y)**

**Ejemplo:** P(Caries = True):

```
P(C=T) = P(C=T,D=T,T=T) + P(C=T,D=T,T=F) + P(C=T,D=F,T=T) + P(C=T,D=F,T=F)
       = 0.108 + 0.012 + 0.072 + 0.008 = 0.200
```

**Generalización:** Para un conjunto de variables de consulta Q y variables de evidencia E:

```
P(Q | E) = α · P(Q, E)    donde α = 1 / P(E) es la constante de normalización
```

---

### 2.2 Probabilidad Condicional

La **probabilidad condicional** de X dado Y se define como:

> **P(X | Y) = P(X, Y) / P(Y)**      si P(Y) > 0

Equivalentemente: **P(X, Y) = P(X | Y) · P(Y)**

**Regla de la cadena (Chain Rule):**

> **P(X₁, X₂, ..., Xₙ) = P(X₁) · P(X₂|X₁) · P(X₃|X₁,X₂) · ... · P(Xₙ|X₁,...,Xₙ₋₁)**

Esto permite descomponer distribuciones conjuntas en factores condicionales más manejables.

**Ejemplo:**

```
P(Caries | Dolor) = P(Caries, Dolor) / P(Dolor)
                  = (0.108 + 0.012 + 0.072 + 0.008) / (P(D=T))
                  = 0.200 / 0.200 = 1.0  (si calculamos bien)
```

---

### 2.3 El Problema de la Complejidad

Especificar la distribución conjunta completa requiere **2ⁿ − 1** parámetros (uno menos por la restricción de suma a 1). Esto es **exponencial** en el número de variables:

| Variables | Entradas necesarias |
|---|---|
| 10 | 1.023 |
| 20 | 1.048.575 |
| 30 | >1.000.000.000 |

Este problema de escala motiva el uso de **Redes Bayesianas**, que aprovechan las independencias condicionales para reducir drásticamente el número de parámetros.

---

## 3. Teorema de Bayes

El **Teorema de Bayes** permite invertir la dirección de la probabilidad condicional: calcular P(causa | efecto) a partir de P(efecto | causa).

> **P(H | E) = P(E | H) · P(H) / P(E)**

Donde:
- **P(H)**: probabilidad a priori de la hipótesis H
- **P(E | H)**: verosimilitud — probabilidad de observar E si H es verdad
- **P(E)**: evidencia — constante de normalización = Σ_h P(E | H=h) · P(H=h)
- **P(H | E)**: probabilidad a posteriori de H dado E

### Formulación con múltiples hipótesis

Para un conjunto de hipótesis mutuamente exclusivas y exhaustivas:

```
P(Hᵢ | E) = P(E | Hᵢ) · P(Hᵢ) / Σⱼ P(E | Hⱼ) · P(Hⱼ)
```

### Ejemplo: Diagnóstico de caries

- P(Dolor | Caries) = 0.6 (si hay caries, 60% de probabilidad de dolor)
- P(Dolor | ¬Caries) = 0.1 (sin caries, solo 10% de probabilidad de dolor)
- P(Caries) = 0.2 (prevalencia de caries)

```
P(Caries | Dolor) = P(Dolor | Caries) · P(Caries) / P(Dolor)
                  = 0.6 · 0.2 / (0.6·0.2 + 0.1·0.8)
                  = 0.12 / (0.12 + 0.08) = 0.12 / 0.20 = 0.60
```

### Bayes con múltiple evidencia (Naive Bayes)

Si se asume **independencia condicional** entre las evidencias dada la hipótesis:

> **P(H | E₁, E₂) ∝ P(E₁ | H) · P(E₂ | H) · P(H)**

---

## 4. Redes Bayesianas

### 4.1 Definición y Estructura

Una **Red Bayesiana** (Bayesian Network, BN) es un **grafo dirigido acíclico (DAG)** donde:
- Cada **nodo** representa una variable aleatoria.
- Cada **arco** representa una dependencia probabilística directa entre variables.
- La ausencia de arco indica **independencia condicional**.

Formalmente: una BN sobre variables {X₁, ..., Xₙ} es un par (**G**, **θ**) donde:
- **G** = DAG con nodos X₁, ..., Xₙ
- **θ** = parámetros: para cada nodo Xᵢ, la distribución P(Xᵢ | pa(Xᵢ)) donde pa(Xᵢ) son sus padres en G

### Propiedades fundamentales

| Propiedad | Descripción |
|---|---|
| **Representación compacta** | Usa independencias para reducir parámetros |
| **Factorización** | Conjunta = producto de probabilidades condicionales locales |
| **Inferencia** | Responde consultas probabilísticas arbitrarias |
| **Aprendizaje** | Puede estimarse de datos |

---

### 4.2 Factorización de la Distribución Conjunta

La **propiedad fundamental** de las BN es que la distribución conjunta se factoriza como:

> **P(X₁, X₂, ..., Xₙ) = ∏ᵢ P(Xᵢ | pa(Xᵢ))**

Donde pa(Xᵢ) son los **padres** del nodo Xᵢ en el DAG.

**Ejemplo — Red de tres nodos:**

```
   A → B → C
```

```
P(A, B, C) = P(A) · P(B | A) · P(C | B)
```

**Ejemplo — Red de cuatro nodos (Gemelos):**

```
   G → H₁
   G → H₂
```

```
P(G, H₁, H₂) = P(G) · P(H₁ | G) · P(H₂ | G)
```

**Ventaja de la factorización:** En vez de 2³ − 1 = 7 parámetros para la conjunta, necesitamos solo:
- P(A): 1 parámetro
- P(B|A): 2 parámetros
- P(C|B): 2 parámetros
→ **Total: 5 parámetros** (ahorro del 29%)

Con redes más grandes y más independencias, el ahorro es dramático.

---

### 4.3 Tablas de Probabilidad Condicional (CPT)

Cada nodo Xᵢ tiene asociada una **Conditional Probability Table (CPT)** que especifica P(Xᵢ | pa(Xᵢ)) para cada combinación de valores de sus padres.

**Ejemplo — Nodo con 2 padres booleanos (A, B) y variable Cᵢ booleana:**

| A | B | P(Cᵢ = T) | P(Cᵢ = F) |
|---|---|---|---|
| T | T | 0.95 | 0.05 |
| T | F | 0.80 | 0.20 |
| F | T | 0.60 | 0.40 |
| F | F | 0.10 | 0.90 |

**Número de entradas en la CPT:** Si Xᵢ tiene `k` valores posibles y sus padres tienen configuraciones `q`, la CPT tiene `q × (k−1)` entradas libres.

---

## 5. Independencia Condicional

### Definición

Las variables X e Y son **condicionalmente independientes** dado Z si:

> **P(X | Y, Z) = P(X | Z)**

Equivalentemente: **P(X, Y | Z) = P(X | Z) · P(Y | Z)**

Notación: **X ⊥ Y | Z**

En términos prácticos: conocer Y no aporta información adicional sobre X cuando ya conocemos Z.

---

### 5.1 Tipos de Conexión

En una Red Bayesiana, la información fluye (o no) a través de los nodos según el patrón de conexión:

#### Cadena (Chain / Causa-Efecto)

```
A → B → C
```

- **Activo** (si B no está observado): A influye en C a través de B.
- **Bloqueado** (si B está observado): A ⊥ C | B

> Observar B **bloquea** el flujo de información entre A y C.

#### Fork (Causa Común / V invertida)

```
A ← B → C
```

- **Activo** (si B no está observado): A y C están correlacionados (tienen causa común).
- **Bloqueado** (si B está observado): A ⊥ C | B

> Conocer la causa común B hace que sus efectos A y C sean independientes.

#### Collider (Efecto Común / V)

```
A → B ← C
```

- **Bloqueado** (si B no está observado): A ⊥ C (¡son independientes marginalmente!)
- **Activo** (si B o un descendiente de B está observado): A y C se vuelven **dependientes**.

> Observar el efecto común B **activa** la dependencia entre sus causas A y C. También llamado **"explaining away"**.

---

### 5.2 D-separación

**D-separación** (Directional Separation) es el criterio formal para determinar si un conjunto de variables X es independiente de Y dado Z en un DAG.

**Definición:** Un camino entre X e Y está **bloqueado** por un conjunto de nodos Z si existe algún nodo B en el camino tal que:
1. B es un nodo **chain** o **fork**, y B ∈ Z (está observado).
2. B es un nodo **collider**, y B ∉ Z **y** ningún descendiente de B está en Z (no observado).

Si **todos** los caminos entre X e Y están bloqueados por Z, entonces X y Y están **d-separados** por Z, lo que implica X ⊥ Y | Z.

**Ejemplo:**

```
Lluvia → CéspedMojado ← Aspersor
             ↓
          Resbaladizo
```

- ¿Lluvia ⊥ Aspersor? Sí (CéspedMojado es collider no observado → camino bloqueado).
- ¿Lluvia ⊥ Aspersor | CéspedMojado? No (observar el collider activa la dependencia).

---

## 6. Inferencia en Redes Bayesianas

El objetivo de la **inferencia** en una BN es calcular:

> **P(Q | E = e)**

Donde **Q** son las variables de consulta (query) y **E** es el conjunto de variables de evidencia observadas.

La idea básica: usar la factorización de la conjunta y la regla de la suma.

```
P(Q | E = e) = α · P(Q, E = e) = α · Σ_H P(Q, H, E = e)
```

Donde **H** son las variables ocultas (hidden), y α es la constante de normalización.

---

### 6.1 Algoritmo ExpandProb

`ExpandProb` calcula P(X₁=v₁, ..., Xₙ=vₙ) para una asignación completa de variables, usando la factorización de la BN.

```
function ExpandProb(i, n, BN, assignment):
    if i > n then
        return 1
    end if
    Xᵢ := variable i
    vᵢ := assignment[i]
    pa := parents(Xᵢ) in BN with values from assignment
    return P(Xᵢ=vᵢ | pa) · ExpandProb(i+1, n, BN, assignment)
```

**Funcionamiento:** Multiplica recursivamente los factores de la CPT de cada nodo dada la asignación parcial acumulada.

---

### 6.2 Algoritmo QueryVar

`QueryVar` calcula P(Xq = v | E = e) para una variable de consulta Xq.

```
function QueryVar(q, BN, e):
    result := array of zeros, length = |domain(Xq)|
    for each value v in domain(Xq) do
        assignment := e ∪ {Xq = v}
        result[v] := SumOut(assignment, hidden_variables, BN)
    end for
    return Normalize(result)

function SumOut(assignment, hidden, BN):
    if hidden is empty then
        return ExpandProb(1, n, BN, assignment)
    end if
    H := first variable in hidden
    total := 0
    for each value w in domain(H) do
        assignment[H] := w
        total += SumOut(assignment, hidden \ {H}, BN)
    end for
    return total
```

**Complejidad:** Exponencial en el número de variables ocultas. Para `h` variables ocultas booleanas: O(2ʰ) operaciones.

---

### 6.3 Complejidad

La inferencia exacta en Redes Bayesianas es **NP-hard** en el caso general.

| Aspecto | Detalle |
|---|---|
| **Inferencia exacta** | NP-hard (Cooper, 1990) |
| **Causa principal** | Exponencial en el ancho del árbol del grafo (treewidth) |
| **Grafos con treewidth pequeño** | Eficiente con algoritmos como Variable Elimination o Junction Tree |
| **Grafos densamente conectados** | Intractable en la práctica |

**Intuición:** Para calcular P(Xq | E), hay que sumar sobre todas las combinaciones de valores de las variables ocultas — hay 2ʰ si son booleanas.

---

## 7. Inferencia Aproximada

Cuando la inferencia exacta es intractable, se recurren a métodos **aproximados** basados en muestreo.

### Gibbs Sampling

**Gibbs Sampling** es un método de Monte Carlo por Cadenas de Markov (MCMC) adaptado a BN:

1. Inicializar aleatoriamente los valores de todas las variables no observadas.
2. Repetir N veces:
   - Para cada variable oculta Xᵢ: muestrear P(Xᵢ | todos los demás nodos = sus valores actuales).
3. Contar las muestras para estimar la distribución deseada.

**Propiedades:**
- Converge a la distribución correcta conforme N → ∞.
- El tiempo de convergencia (mixing time) puede ser lento en distribuciones multimodales.
- Aplicable cuando la distribución condicional de cada nodo es fácil de muestrear.

```
function GibbsSampling(BN, evidence, N):
    x := random initialization consistent with evidence
    counts := {}
    for t := 1 to N do
        for each non-evidence variable Xᵢ do
            xᵢ := sample from P(Xᵢ | Markov_blanket(Xᵢ) = x)
            x[Xᵢ] := xᵢ
        end for
        counts[x[query]] += 1
    end for
    return Normalize(counts)
```

### Particle Filters (Filtros de Partículas)

Los **filtros de partículas** aproximan la distribución de probabilidad mediante un conjunto de **muestras ponderadas** (partículas):

1. Generar N partículas muestreando desde la distribución a priori.
2. Para cada nueva observación:
   - Puntuar cada partícula según la verosimilitud de la observación.
   - Re-muestrear con reemplazo proporcional a los pesos.
3. Las partículas supervivientes aproximan la distribución posterior.

**Uso principal:** modelos dinámicos (localización de robots, tracking, filtrado temporal).

---

## 8. Ejemplos Completos

### Ejemplo 1: Red del Cáncer (4 nodos)

```
Polución (P) → Cáncer (C) ← Fumador (F)
                 ↓
              Test Positivo (T)    Disnea (D)
```

Variables: P, F, C, T, D (todas booleanas)

**Factorización:**
```
P(P, F, C, T, D) = P(P) · P(F) · P(C | P, F) · P(T | C) · P(D | C)
```

**Parámetros necesarios:**
- P(P): 1 parámetro
- P(F): 1 parámetro
- P(C | P, F): 4 parámetros (2 padres booleanos)
- P(T | C): 2 parámetros
- P(D | C): 2 parámetros
- **Total: 10 parámetros** (vs. 2⁵ − 1 = 31 sin BN)

---

### Ejemplo 2: Red de los Gemelos

Un experimento clásico sobre gemelos: se observa si ambos gemelos tienen la misma característica fenotípica.

```
Genoma (G) → Fenotipo Gemelo 1 (H₁)
Genoma (G) → Fenotipo Gemelo 2 (H₂)
```

**Distribuciones:**
- G ∈ {AA, Aa, aa}: P(G) = {0.25, 0.50, 0.25} (distribución Hardy-Weinberg)
- P(H = 1 | G = AA) = 1.0; P(H = 1 | G = Aa) = 0.5; P(H = 1 | G = aa) = 0.0

**Consulta:** P(H₂ = 1 | H₁ = 1)?

```
P(H₂=1 | H₁=1) = P(H₂=1, H₁=1) / P(H₁=1)

P(H₁=1) = Σ_g P(H₁=1|G=g)·P(G=g)
         = 1.0·0.25 + 0.5·0.50 + 0.0·0.25 = 0.50

P(H₁=1, H₂=1) = Σ_g P(H₁=1|G=g)·P(H₂=1|G=g)·P(G=g)
               = 1·1·0.25 + 0.5·0.5·0.50 + 0·0·0.25
               = 0.25 + 0.125 = 0.375

P(H₂=1 | H₁=1) = 0.375 / 0.50 = 0.75
```

> Si el primer gemelo tiene la característica, hay un 75% de probabilidad de que el segundo también la tenga (frente al 50% a priori).

---

### Ejemplo 3: Problema de la Puerta (Monty Hall)

Variables: **P** = puerta con premio (1,2,3), **A** = puerta abierta por el presentador, **D** = decisión del concursante.

**Consulta clave:** P(premio detrás de puerta 3 | concursante elige puerta 1, presentador abre puerta 2)?

Usando la BN con factorización y QueryVar:
- P(P=3 | A=2, D=1) = 2/3
- P(P=1 | A=2, D=1) = 1/3

> Siempre conviene **cambiar de puerta**: la probabilidad de ganar pasa de 1/3 a 2/3.

---

## 9. Bibliografía

- **Russell, S. & Norvig, P.** — *Artificial Intelligence: A Modern Approach* (4ª ed.), capítulos 12-13 (Probabilidad y Redes Bayesianas).
- **Pearl, J.** — *Probabilistic Reasoning in Intelligent Systems*, Morgan Kaufmann, 1988. (Obra fundacional de las BN.)
- **Koller, D. & Friedman, N.** — *Probabilistic Graphical Models: Principles and Techniques*, MIT Press, 2009.
- **Cooper, G.** — *The Computational Complexity of Probabilistic Inference Using Bayesian Belief Networks*, Artificial Intelligence, 1990.
- **Jensen, F.V. & Nielsen, T.D.** — *Bayesian Networks and Decision Graphs*, Springer, 2007.

---

*Resumen generado a partir de las diapositivas del curso ARA. Para mayor detalle sobre demostraciones formales y ejemplos adicionales, consultar los PDFs originales.*
