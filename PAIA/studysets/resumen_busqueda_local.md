# Búsqueda Local — Resumen del Temario Completo

> **Fuentes:** Tema 1 — Búsqueda Local (PDF 01) · Tema 2 — Búsqueda Local y SAT (PDF 02)

---

## Tabla de Contenidos

1. [Introducción](#1-introducción)
2. [Preliminares](#2-preliminares)
3. [Búsqueda por Entornos (Neighborhood Search)](#3-búsqueda-por-entornos-neighborhood-search)
   - 3.1 [Algoritmo Básico NS](#31-algoritmo-básico-ns)
   - 3.2 [Variantes de la Función de Selección](#32-variantes-de-la-función-de-selección)
4. [Simulated Annealing](#4-simulated-annealing)
5. [Búsqueda Tabú (Tabu Search)](#5-búsqueda-tabú-tabu-search)
6. [Algoritmos Genéticos](#6-algoritmos-genéticos)
7. [SAT y Búsqueda Local](#7-sat-y-búsqueda-local)
   - 7.1 [El Problema SAT](#71-el-problema-sat)
   - 7.2 [Generador Random k-SAT](#72-generador-random-k-sat)
   - 7.3 [GSAT](#73-gsat)
   - 7.4 [WalkSAT](#74-walksat)
   - 7.5 [UBCSAT](#75-ubcsat)
8. [Bibliografía](#8-bibliografía)

---

## 1. Introducción

La **búsqueda local** (o búsqueda incompleta/no sistemática) es una familia de algoritmos diseñados para resolver problemas de **satisfacción de restricciones (CSP)** y problemas de **optimización** donde el espacio de búsqueda es demasiado grande para métodos exhaustivos.

### Clasificación de algoritmos de búsqueda

| Tipo | Características | Ejemplos |
|---|---|---|
| **Sistemáticos / Completos** | Garantizan encontrar solución si existe; exploran el espacio entero | Backtracking, BFS, A* |
| **No sistemáticos / Incompletos** | No garantizan solución óptima; muy eficientes en problemas grandes | Hill Climbing, SA, Tabu, GA |

### Áreas de aplicación

- Scheduling y planificación de recursos
- Diseño de circuitos y VLSI
- Satisfacción de restricciones (SAT, CSP)
- Optimización combinatoria (TSP, N-Queens, etc.)
- Problemas de aprendizaje automático y entrenamiento de redes

### ¿Por qué búsqueda local?

- Los problemas reales (NP-hard) no tienen solución en tiempo polinomial con métodos exactos.
- La búsqueda local encuentra **soluciones de buena calidad** en tiempo razonable.
- Trabaja directamente sobre el **espacio de soluciones completas**, no sobre estados parciales.

---

## 2. Preliminares

### Formulación del problema

Un problema de búsqueda local se define como una terna **(X, D, F)**:

| Componente | Descripción |
|---|---|
| **X = {x₁, x₂, ..., xₙ}** | Conjunto de variables del problema |
| **D = D₁ × D₂ × ... × Dₙ** | Dominio de cada variable (espacio de búsqueda) |
| **F: D → ℝ** | Función de coste o calidad a minimizar/maximizar |

Un **punto** en el espacio de búsqueda es una asignación completa de valores a todas las variables: `I = (v₁, v₂, ..., vₙ) ∈ D`.

### Función de entorno N(X)

La **función de vecindad** (neighborhood function) `N: D → 2^D` asigna a cada punto del espacio un conjunto de **puntos vecinos** que se pueden explorar desde él.

**Propiedad de accesibilidad:** Para cualquier par de puntos `I, J ∈ D`, existe una secuencia finita `I = I₀, I₁, ..., Iₖ = J` tal que `Iⱼ₊₁ ∈ N(Iⱼ)` para todo j. Esto garantiza que el espacio es completamente explorable.

**Vecindad de orden i — Nᵢ(X):**

> `Nᵢ(X)` = conjunto de puntos que difieren de X en **a lo sumo i variables**.

- `N₁(X)`: vecinos que cambian exactamente 1 variable (vecindad de Hamming).
- `N₂(X)`: vecinos que cambian hasta 2 variables → |N₂| = O(n²).
- En general, `|Nᵢ(X)| = O(nⁱ)`.

**Trade-off:** Vecindades más grandes permiten explorar más pero son más costosas de evaluar.

---

## 3. Búsqueda por Entornos (Neighborhood Search)

### 3.1 Algoritmo Básico NS

El esquema general de la búsqueda por entornos es:

```
function NS(X, D, F, N):
    I := InitialState(X, D)
    while not StopCondition() do
        I' := Select(N(I))
        if AcceptanceCriterion(I, I') then
            I := I'
        end if
    end while
    return best solution found
```

Los tres elementos configurables son:
- **`InitialState`**: cómo se genera el estado inicial (aleatoria, greedy, etc.)
- **`Select`**: qué vecino se elige en cada iteración (define la variante del algoritmo)
- **`AcceptanceCriterion`**: cuándo se acepta el movimiento

#### Ejemplo ilustrativo: N-Queens (N-Reinas)

- **Variables**: posición de fila de cada reina en columna i → `xᵢ ∈ {1..N}`
- **Coste F(I)**: número de pares de reinas que se atacan mutuamente
- **Vecindad N₁**: mover una reina a otra fila en su misma columna → `|N₁| = N(N-1)`
- **Objetivo**: encontrar `I` con `F(I) = 0`

---

### 3.2 Variantes de la Función de Selección

Todas las variantes comparten la misma estructura NS básica pero difieren en cómo eligen el vecino:

#### Random Search (Búsqueda Aleatoria)

- Elige un vecino **completamente al azar** de `N(I)`.
- No guía la búsqueda → sirve de baseline/referencia.
- Pseudocódigo: `I' := Random(N(I))`

#### Hill Climbing (HC)

- Elige un vecino que **mejore** la solución actual.
- Si no hay mejora posible, la búsqueda se detiene (**óptimo local**).
- Problema: queda atrapado en mínimos locales.

```
I' := any I'' ∈ N(I) such that F(I'') < F(I)
```

#### Steepest Ascent Hill Climbing (SAHC)

- Variante del HC que elige el **mejor vecino** de todo `N(I)`.
- Más costoso por iteración pero mejora más agresivamente.

```
I' := argmin_{I'' ∈ N(I)} F(I'')
```

#### Restarts (Reinicios)

- Cuando HC queda atrapado en un óptimo local, se **reinicia desde un nuevo estado aleatorio**.
- Se conserva la mejor solución global encontrada.
- Permite escapar de óptimos locales sin modificar el criterio de selección.

```
if StuckInLocalOptimum() then
    I := Random(D)
    // preserve global best
end if
```

#### Random Walk (Paseo Aleatorio)

- Con probabilidad `ω` se realiza un **movimiento aleatorio** (aunque empeore).
- Con probabilidad `1-ω` se aplica HC normal.
- Evita quedar atrapado indefinidamente en un óptimo local.

```
if Random() < ω then
    I' := Random(N(I))      // random walk step
else
    I' := BestImproving(N(I))  // HC step
end if
```

#### Mildest Descent (Descenso más suave)

- Cuando no hay mejora disponible, elige el vecino que **empeora menos** (mínimo incremento de coste).
- También llamado "sideways moves" cuando el coste es igual.

```
I' := argmin_{I'' ∈ N(I)} F(I'')   // even if F(I'') >= F(I)
```

#### Mildest Descent + Random Walk

- Combina ambas estrategias: primero intenta la mejora más suave, y ocasionalmente hace un paso aleatorio.
- Mayor diversificación sin perder completamente la guía.

#### Breakout

- Aplica perturbaciones sistemáticas cuando se detecta que la búsqueda está atascada.
- La perturbación es **más fuerte** que random walk: cambia varios valores a la vez.
- Útil para escapar de cuencas de atracción profundas.

#### Local Beam Search

- Mantiene **k soluciones** en paralelo en lugar de una sola.
- En cada iteración genera todos los vecinos de las k soluciones y selecciona los k mejores.
- Versión **estocástica**: selecciona los k siguientes de forma ponderada por calidad (similar a selección en GA).

```
Beam := {k random initial states}
while not StopCondition() do
    Successors := ∅
    for each I in Beam do
        Successors := Successors ∪ N(I)
    end for
    Beam := SelectBest(k, Successors)
end while
```

---

## 4. Simulated Annealing

### Motivación

Inspirado en el proceso físico de **recocido** (annealing) de metales: calentar y enfriar lentamente para alcanzar el estado de mínima energía. Permite **aceptar movimientos que empeoran** con una probabilidad que disminuye con el tiempo.

### Algoritmo

```
function SimulatedAnnealing(X, D, F, N):
    current := InitialState(X, D)    // initial state, energy
    T := T_initial                   // initial temperature
    while not StopCondition() do
        next := Random(N(current))   // pick random neighbor
        ΔE := F(next) - F(current)
        if ΔE < 0 then               // improvement: always accept
            current := next
        else
            if Random() < P(ΔE, T) then   // worsening: accept with probability
                current := next
            end if
        end if
        T := CoolingSchedule(T)      // decrease temperature
    end while
    return best solution found
```

### Probabilidad de aceptación

La probabilidad de aceptar una solución **peor** viene dada por la distribución de Boltzmann:

> **P(e, e', T) = exp(-(e' - e) / T)**

Donde:
- `e = F(current)`, `e' = F(next)` (e' > e, empeora)
- `T` = temperatura actual
- Con T alta → P ≈ 1 (acepta casi todo, exploración)
- Con T baja → P ≈ 0 (solo acepta mejoras, explotación)

### Schedule de temperatura (Cooling Schedule)

| Parámetro | Descripción |
|---|---|
| **T₀** | Temperatura inicial (alta) |
| **α (alpha)** | Factor de enfriamiento (típico: 0.95–0.99) |
| **Tₘᵢₙ** | Temperatura mínima (criterio de parada) |
| **Actualización** | `T := α · T` (enfriamiento geométrico) |

**Propiedades teóricas:** Con un schedule de enfriamiento suficientemente lento, SA converge al óptimo global con probabilidad 1 (pero esto requiere tiempo exponencial en la práctica).

---

## 5. Búsqueda Tabú (Tabu Search)

### Motivación

La búsqueda tabú extiende el hill climbing permitiendo movimientos que empeoran, pero **prohíbe volver a visitar** soluciones recientes mediante una **lista tabú**. Fue propuesta por Fred Glover (1989).

### Estructura de la lista tabú

| Tipo de memoria | Duración | Función |
|---|---|---|
| **Corto plazo** | Últimas k iteraciones | Evita ciclos inmediatos |
| **Intermedio plazo** | Decenas de iteraciones | Intensificación en regiones prometedoras |
| **Largo plazo** | Toda la búsqueda | Diversificación hacia zonas no exploradas |

### Algoritmo

```
function TabuSearch(X, D, F, N):
    current := InitialState(X, D)
    best := current
    TabuList := null              // empty tabu list
    while not StopCondition() do
        Neighbors := N(current) \ TabuList   // exclude tabu moves
        current := argmin_{I ∈ Neighbors} F(I)   // best non-tabu neighbor
        if F(current) < F(best) then
            best := current
        end if
        TabuList := TabuList + current    // add to tabu list
        ExpireFeatures(TabuList)          // remove old entries
    end while
    return best
```

### Función ExpireFeatures

Elimina de la lista tabú las entradas que han superado su **tiempo de permanencia** (tenure). El tenure puede ser fijo o adaptativo.

### Criterio de aspiración

Excepción a la lista tabú: un movimiento tabú se **acepta igualmente** si produce una solución **mejor que la mejor global** conocida. Evita rechazar mejoras óptimas por la restricción tabú.

### Parámetros clave

- **Tamaño de la lista tabú**: pequeño → intensificación; grande → diversificación.
- **Tenure**: número de iteraciones que un movimiento permanece prohibido.
- **Criterio de aspiración**: condición bajo la cual se ignora la restricción tabú.

---

## 6. Algoritmos Genéticos

### Inspiración biológica

Inspirados en la evolución natural (Darwin): una **población** de individuos evoluciona generación a generación mediante **selección**, **cruce** y **mutación**, favoreciendo a los más aptos.

### Componentes principales

#### Representación

Cada **individuo** es una codificación (generalmente binaria o entera) de una solución. Llamado también **cromosoma** o **genotipo**.

#### Inicialización

La población inicial se genera normalmente de forma **aleatoria** (o con heurísticas). Tamaño de población típico: 20–200 individuos.

#### Función de aptitud (Fitness)

`fitness(I)`: mide la calidad de un individuo. Equivale a `F(I)` pero orientada a **maximización**.

#### Selección

Elige qué individuos se reproducen. Tres métodos principales:

| Método | Descripción |
|---|---|
| **Ruleta (Roulette Wheel)** | Probabilidad de selección proporcional al fitness: `P(i) = f(i) / Σf` |
| **Torneo (Tournament)** | Se eligen k individuos al azar y gana el de mayor fitness |
| **Ranking** | Se ordenan individuos y la probabilidad depende del rango, no del valor absoluto |

#### Cruce (Crossover)

Combina el material genético de dos **padres** para producir uno o más **hijos**:

- **Cruce de un punto**: se elige un punto de corte; el hijo hereda la primera parte del padre 1 y la segunda del padre 2.
- **Cruce de dos puntos**: dos puntos de corte; alternancia de segmentos.
- **Cruce uniforme**: cada gen del hijo se copia del padre 1 o del padre 2 con probabilidad 0.5.

```
// One-point crossover example
Parent1: [A B C | D E F G]
Parent2: [a b c | d e f g]
Child1:  [A B C   d e f g]
Child2:  [a b c   D E F G]
```

#### Mutación

Pequeña perturbación aleatoria de un individuo. En representación binaria: **flip de un bit** con probabilidad `pₘ` (típico: 1/n).

- Mantiene diversidad genética.
- Evita convergencia prematura.

### Pseudocódigo general

```
function GeneticAlgorithm(population_size, max_generations):
    P := InitPopulation(population_size)
    Evaluate(P)
    while not StopCondition() do
        P' := Selection(P)
        P'' := Crossover(P')
        P''' := Mutation(P'')
        Evaluate(P''')
        P := Replace(P, P''')
    end while
    return BestIndividual(P)
```

### Ejemplos de aplicación

- **BoxCar2D**: evolución de vehículos 2D con ruedas y chasis optimizados para recorrer terreno irregular. Los individuos son diseños de vehículos; el fitness es la distancia recorrida.
- **Genetic Walkers**: evolución de robots bípedos o cuadrúpedos para caminar eficientemente. Cromosomas codifican ángulos y fuerzas en las articulaciones.

### Parámetros clave

| Parámetro | Efecto |
|---|---|
| Tamaño de población | Mayor → mejor exploración, más coste |
| Tasa de cruce | Alta → mezcla rápida; muy alta → destruye buenas soluciones |
| Tasa de mutación | Baja → explotación; muy alta → búsqueda aleatoria |
| Criterio de reemplazo | Generacional (todos), steady-state (uno a uno), elitismo |

---

## 7. SAT y Búsqueda Local

### 7.1 El Problema SAT

**SAT (Boolean Satisfiability Problem)** es el primer problema demostrado NP-completo (Cook, 1971). Consiste en determinar si existe una asignación de verdad que satisfaga una fórmula booleana.

#### Forma Normal Conjuntiva (CNF)

Una fórmula en **CNF** es una conjunción (AND) de **cláusulas**, donde cada cláusula es una disyunción (OR) de **literales**:

> **F = C₁ ∧ C₂ ∧ ... ∧ Cₘ**
> **Cᵢ = (l₁ ∨ l₂ ∨ ... ∨ lₖ)**

Donde un **literal** es una variable `xᵢ` o su negación `¬xᵢ`.

#### Ejemplo

```
F = (x₁ ∨ ¬x₂ ∨ x₃) ∧ (¬x₁ ∨ x₂) ∧ (x₂ ∨ ¬x₃)
```

#### Variantes

| Variante | Descripción |
|---|---|
| **k-SAT** | Todas las cláusulas tienen exactamente k literales |
| **3-SAT** | Caso más estudiado; NP-completo para k ≥ 3 |
| **2-SAT** | Resoluble en tiempo polinomial |
| **MAX-SAT** | Maximizar el número de cláusulas satisfechas |

#### Formulación como búsqueda local

- **Variables**: `x₁, ..., xₙ` ∈ {True, False}
- **Espacio**: 2ⁿ asignaciones posibles
- **Coste F(I)**: número de cláusulas **no satisfechas** por la asignación I
- **Objetivo**: encontrar I con F(I) = 0

---

### 7.2 Generador Random k-SAT

Para evaluar y comparar algoritmos SAT se usan **instancias aleatorias** generadas por el modelo **Random k-SAT**:

- Se tienen `n` variables y se generan `m` cláusulas de k literales cada una.
- Cada cláusula se forma eligiendo k variables al azar y negando cada una con probabilidad 0.5.

#### Ratio de cláusulas/variables (r = m/n)

Este parámetro determina la **dificultad** de la instancia:

| Valor de r | Comportamiento |
|---|---|
| r bajo (< 4.25) | Instancias **satisfacibles** y fáciles (muchas soluciones) |
| r ≈ **4.25** | **Transición de fase** — instancias más difíciles |
| r alto (> 4.25) | Instancias insatisfacibles (muy pocas o ninguna solución) |

La **transición de fase** en r ≈ 4.25 (para 3-SAT) es el punto donde las instancias son máximamente difíciles tanto para algoritmos completos como incompletos.

---

### 7.3 GSAT

**GSAT** (Greedy SAT) es un algoritmo de búsqueda local voraz para SAT propuesto por Selman, Levesque y Mitchell (1992).

#### Idea principal

En cada paso, hace **flip** (invierte el valor) de la variable que **reduce más el número de cláusulas insatisfechas** (o lo aumenta menos).

#### Pseudocódigo

```
function GSAT(F, max_tries, max_flips):
    for i := 1 to max_tries do
        I := Random assignment of variables
        for j := 1 to max_flips do
            if F is satisfied by I then
                return I                  // solution found
            end if
            x := variable whose flip minimizes unsat(F, I)
            I := I with x flipped
        end for
    end for
    return failure
```

#### Parámetros

| Parámetro | Descripción |
|---|---|
| `max_tries` | Número de reinicios desde estado aleatorio |
| `max_flips` | Número máximo de flips por intento |

#### Análisis

- Es esencialmente **Steepest Ascent Hill Climbing** con reinicios.
- Puede quedar atrapado en **plateau** (región donde todos los movimientos tienen igual coste).
- No garantiza encontrar solución aunque exista.
- Muy eficaz en la práctica para instancias de tamaño moderado.

---

### 7.4 WalkSAT

**WalkSAT** extiende GSAT combinando la selección greedy con un **paseo aleatorio** dentro de cláusulas insatisfechas. Propuesto por Selman, Kautz y Cohen (1994).

#### Idea principal

En cada paso:
1. Elige aleatoriamente una **cláusula insatisfecha**.
2. Con probabilidad `1 - ω`: hace flip de la variable que **menos rompe** cláusulas satisfechas (heurística `broken`).
3. Con probabilidad `ω`: hace flip de una variable **aleatoria** de esa cláusula (random walk).

#### Función broken(p, F, I)

> `broken(p, F, I)` = número de cláusulas satisfechas en I que dejan de estarlo si se invierte la variable p.

Elegir la variable con **menor broken** es equivalente a causar el mínimo daño colateral.

#### Pseudocódigo

```
function WalkSAT(F, max_tries, max_flips, ω):
    for i := 1 to max_tries do
        I := Random assignment of variables
        for j := 1 to max_flips do
            if F is satisfied by I then
                return I
            end if
            C := Random unsatisfied clause in F
            if Random() < ω then
                x := Random variable in C       // random walk
            else
                x := variable in C with min broken(x, F, I)  // greedy
            end if
            I := I with x flipped
        end for
    end for
    return failure
```

#### Comparación GSAT vs WalkSAT

| Característica | GSAT | WalkSAT |
|---|---|---|
| Selección de variable | Global (todas las variables) | Local (solo variables en cláusula insatisfecha) |
| Estrategia | Greedy puro | Greedy + Random Walk |
| Eficiencia | O(n) por flip | O(k) por flip (k = tamaño de cláusula) |
| Rendimiento empírico | Bueno en instancias pequeñas | Mejor en instancias grandes |
| Tendencia al plateau | Alta | Reducida (por RW) |

#### Parámetro ω (omega)

- `ω = 0`: WalkSAT es idéntico a GSAT local.
- `ω = 1`: WalkSAT es búsqueda aleatoria dentro de cláusulas.
- Valor típico: `ω = 0.5` (equilibrio exploración/explotación).

---

### 7.5 UBCSAT

**UBCSAT** (University of British Columbia SAT) es una **biblioteca de código abierto** que implementa y permite comparar múltiples algoritmos de búsqueda local para SAT. Desarrollada por Dave Tompkins y Holger Hoos.

#### Características principales

- Implementa decenas de variantes: GSAT, WalkSAT, SAPS, PAWS, etc.
- Interfaz de **línea de comandos** unificada.
- Facilita la experimentación y benchmarking reproducible.

#### Uso básico

```bash
ubcsat -alg walksat -i instancia.cnf -runs 100 -cutoff 100000
```

#### Parámetros comunes

| Parámetro | Descripción |
|---|---|
| `-alg <nombre>` | Algoritmo a ejecutar (walksat, gsat, saps, ...) |
| `-i <archivo>` | Archivo de instancia en formato **DIMACS CNF** |
| `-runs <n>` | Número de ejecuciones independientes |
| `-cutoff <n>` | Número máximo de flips por ejecución |
| `-seed <n>` | Semilla para reproducibilidad |
| `-noise <p>` | Parámetro de ruido/aleatoriedad (análogo a ω) |

#### Formato DIMACS CNF

Formato estándar para codificar instancias SAT:

```
c Comentario
p cnf <num_vars> <num_clauses>
1 -2 3 0
-1 2 0
2 -3 0
```

Cada línea de cláusula termina en `0`. Los literales negativos representan variables negadas.

---

## 8. Bibliografía

### Búsqueda Local (PDF 01)

- **Russell & Norvig** — *Artificial Intelligence: A Modern Approach*, capítulos sobre búsqueda local y algoritmos genéticos.
- **Hoos & Stützle** — *Stochastic Local Search: Foundations and Applications*, Morgan Kaufmann, 2004.
- **Glover, F.** — *Tabu Search*, ORSA Journal on Computing, 1989.
- **Kirkpatrick et al.** — *Optimization by Simulated Annealing*, Science, 1983.
- **Holland, J.** — *Adaptation in Natural and Artificial Systems*, 1975.
- **Goldberg, D.** — *Genetic Algorithms in Search, Optimization, and Machine Learning*, 1989.

### SAT y Búsqueda Local (PDF 02)

- **Cook, S.** — *The Complexity of Theorem Proving Procedures* (primer problema NP-completo), STOC 1971.
- **Selman, B., Levesque, H., Mitchell, D.** — *A New Method for Solving Hard Satisfiability Problems* (GSAT), AAAI 1992.
- **Selman, B., Kautz, H., Cohen, B.** — *Noise Strategies for Improving Local Search* (WalkSAT), AAAI 1994.
- **Tompkins, D., Hoos, H.** — *UBCSAT: An Implementation and Experimentation Environment for SLS Algorithms for SAT and MAX-SAT*, SAT 2004.
- **Mitchell, D., Selman, B., Levesque, H.** — *Hard and Easy Distributions of SAT Problems* (transición de fase), AAAI 1992.

---

*Resumen generado a partir de las diapositivas del curso. Para mayor detalle sobre implementaciones y experimentos, consultar los PDFs originales y la bibliografía citada.*
