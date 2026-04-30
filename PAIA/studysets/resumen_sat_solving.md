# Tema 2: SAT Solving Completo

## 1. Introducción a SAT

### ¿Qué es SAT?
El problema de **Satisfacibilidad Booleana (SAT)** consiste en determinar si existe una asignación de valores de verdad a las variables de una fórmula proposicional que la haga verdadera.

- SAT fue el **primer problema demostrado NP-Completo** (Cook, 1971)
- Todo problema en NP puede reducirse polinomialmente a SAT
- Si SAT ∈ P, entonces P = NP

### Forma Normal Conjuntiva (CNF)
Una fórmula está en **CNF** cuando es una conjunción de cláusulas, donde cada cláusula es una disyunción de literales:

```
F = (l₁₁ ∨ l₁₂ ∨ ...) ∧ (l₂₁ ∨ l₂₂ ∨ ...) ∧ ...
```

Cualquier fórmula proposicional puede convertirse a CNF, aunque puede tener coste exponencial en el tamaño.

### Codificación de Tseitin
La **transformación de Tseitin** convierte cualquier fórmula a CNF en tiempo y espacio **polinomiales**, introduciendo variables auxiliares:

- Para cada subfórmula φ, se introduce una variable nueva `z`
- Se añaden cláusulas que codifican la equivalencia `z ↔ φ`
- La fórmula original es equisatisfacible (no equivalente) con el resultado

**Ejemplo:** Para `z ↔ (a ∧ b)`:
- `¬z ∨ a`, `¬z ∨ b`, `z ∨ ¬a ∨ ¬b`

---

## 2. Casos Especiales en Tiempo Polinomial

### Propagación de Unidades (Unit Propagation)
Si una cláusula tiene un **único literal sin asignar**, ese literal debe ser verdadero:

```
Algoritmo UP:
1. Si existe cláusula unitaria {l}: asignar l = True
2. Propagar: eliminar cláusulas satisfechas, eliminar ¬l de otras
3. Si aparece cláusula vacía: UNSAT
4. Si no quedan cláusulas: SAT
5. Si no hay cláusulas unitarias: devolver la fórmula simplificada
```

### LTUR para Fórmulas de Horn
Una fórmula de Horn tiene como máximo **un literal positivo por cláusula**.

El algoritmo **LTUR** (Linear Time Unit Resolution) decide satisfacibilidad de Horn en tiempo O(n·m):
1. Si no hay cláusula vacía → SAT (asignar todo a False)
2. Si existe cláusula positiva unitaria `{p}` → asignar p = True, propagar
3. Repetir hasta SAT o encontrar cláusula vacía (UNSAT)

### 2-SAT mediante SCCs
Las fórmulas con **exactamente 2 literales por cláusula** son resolubles en tiempo lineal:

1. Construir grafo de implicaciones: `(a ∨ b)` → `¬a → b` y `¬b → a`
2. Calcular las **Componentes Fuertemente Conexas (SCCs)** con Tarjan o Kosaraju
3. **UNSAT** si y solo si existe una variable x tal que x y ¬x están en la misma SCC
4. Asignación satisfacible: si SCC(¬x) se descubre antes que SCC(x) en el orden topológico inverso, asignar x = True

---

## 3. Métodos de Inferencia

### Davis-Putnam (DP) — Basado en Resolución
El algoritmo original de Davis-Putnam (1960) aplica **resolución** para eliminar variables:

```
Para cada variable p:
  Combinar cada cláusula con p con cada cláusula con ¬p
  → genera el resolvente
  Eliminar todas las cláusulas con p o ¬p
```

**Problema:** Explosión de memoria — el número de cláusulas puede crecer exponencialmente.

**Ejemplo:**
- `(a ∨ b) ∧ (¬a ∨ c)` → resolución sobre a → `(b ∨ c)`

### Eliminación por Cubos (Bucket Elimination)
Generalización de DP que organiza las cláusulas en **cubos** (uno por variable en orden de eliminación):

1. Ordenar variables: x₁, x₂, ..., xₙ
2. Cubo(xᵢ) = cláusulas con xᵢ como variable de mayor índice
3. Procesar cada cubo: resolver sobre xᵢ, añadir resolventes al cubo anterior
4. Si aparece cláusula vacía → UNSAT

**Ventaja:** Más organizado que DP puro.  
**Complejidad:** Exponencial en el worst-case (anchura del árbol del grafo de interacción).

### Tableaux de Cláusulas
Método basado en la **expansión de cláusulas** en un árbol:

- Cada nodo del árbol representa una posible extensión de la asignación parcial
- Un tableau está **cerrado** si cada rama contiene una cláusula vacía
- Una fórmula es UNSAT si y solo si su tableau está cerrado

### Método de Stålmarck
Algoritmo para razonamiento de equivalencias proposicionales:

**Reglas de saturación (A9+B12):**
- Procesamiento de implicaciones y equivalencias
- Propagación de valores conocidos

**Regla del dilema:**
- Si se puede demostrar que tanto asignar x=True como x=False lleva a la misma conclusión C, entonces C es válida
- Permite razonamiento sin enumerar asignaciones

---

## 4. Algoritmo DPLL (Davis-Putnam-Logemann-Loveland)

### Estructura del Algoritmo
DPLL (1962) es el algoritmo de **búsqueda con backtracking** más influyente para SAT:

```
DPLL(F):
  F = UnitPropagation(F)
  si F = ∅ → return SAT
  si □ ∈ F → return UNSAT
  
  l = SeleccionarVariable(F)
  
  si DPLL(F ∧ {l}) = SAT → return SAT
  si DPLL(F ∧ {¬l}) = SAT → return SAT
  return UNSAT
```

### Árbol de Búsqueda
- **Líneas sólidas:** decisiones de splitting (heurísticas)
- **Líneas punteadas:** propagaciones por UP

### Optimizaciones de DPLL
- **Unit Propagation (UP):** reducir antes de bifurcar
- **Literal Puro:** si un literal aparece solo en polaridad positiva, asignarlo a True sin pérdida
- **Backtracking:** retroceder al punto de decisión anterior cuando se detecta conflicto

---

## 5. Heurísticas de Selección de Variables

### MAX
Elegir la variable que aparece en el **mayor número de cláusulas** (máxima frecuencia).

### MOMS (Maximum Occurrence in Minimum Size clauses)
**Pretolani (1993):** Priorizar variables en las **cláusulas más pequeñas** (más restringidas):
- Mayor impacto por propagación
- Detecta conflictos antes

### VSIDS (Variable State Independent Decaying Sum)
**Chaff (2001):** Sistema de **puntajes dinámicos**:
- Cada variable tiene un contador de actividad
- Al aprender una cláusula, incrementar los contadores de las variables involucradas
- Periódicamente dividir todos los contadores (decaimiento)
- Elegir variable con mayor actividad

VSIDS prioriza variables que han aparecido recientemente en conflictos.

### Jeroslow-Wang
Función de peso que pondera variables según el tamaño de las cláusulas:
```
J(l) = Σ_{cláusula c con l} 2^(-|c|)
```
Prioriza variables en cláusulas cortas con penalización exponencial.

### Satz
Combina heurísticas: evalúa el impacto de asignar x=True y x=False mediante look-ahead con UP.

---

## 6. CDCL — Conflict-Driven Clause Learning

### Historia
- **GRASP (1996):** Primer solver CDCL (Marques-Silva & Sakallah)
- **Chaff (2001):** Implementación industrial con VSIDS y 2-watch literals

### Grafo de Implicación
Grafo dirigido que registra las **razones de las asignaciones**:
- Nodo con nivel de decisión 0: hechos derivados sin decisiones
- Nodo de nivel k: variable asignada en la decisión k
- Aristas: de las causas al efecto por UP

**Nodo de conflicto (κ):** aparece cuando UP deriva ⊥.

### Aprendizaje de Cláusulas (Clause Learning)
Al detectar un conflicto:
1. Analizar el grafo de implicación
2. Encontrar un **corte** que separe el nodo de conflicto del nivel 0
3. La cláusula aprendida corresponde al corte

**UIP (Unique Implication Point):**
- Punto del grafo de implicación donde todas las rutas al conflicto pasan
- La **primera UIP** (más cercana al conflicto) da cláusulas más compactas
- Cláusula aprendida: negar todos los literales del lado del conflicto del corte UIP

### Non-Chronological Backtracking (Backjumping)
En lugar de retroceder al nivel inmediatamente anterior:
1. Analizar la cláusula aprendida
2. Encontrar el **segundo nivel de decisión más alto** en la cláusula
3. Saltar directamente a ese nivel

Esto evita re-explorar subárboles que inevitablemente llevarían al mismo conflicto.

---

## 7. Técnicas de Implementación

### 2-Watch Literals (Literales Vigilados)
**Estructura de datos lazy** para propagación eficiente:

- Para cada cláusula, mantener **2 literales no falsificados** como "vigías"
- Al falsearse un vigía, buscar otro literal no falsificado
- Solo propagar (hacer unitaria) cuando no queda ningún literal libre

**Ventaja:** O(1) amortizado por actualización; permite backtracking en O(1) sin restaurar estructuras.

### Restarts
Observación de **Gomes et al. (1997):** las distribuciones de tiempo de DPLL tienen colas pesadas (heavy-tailed):
- Algunos intentos son muy largos pero la mayoría son cortos
- Recomenzar la búsqueda desde cero (pero **conservando las cláusulas aprendidas**) puede ser beneficioso

**Políticas de restart:**
- Geometricas: duplicar el límite cada vez
- Luby: secuencia 1,1,2,1,1,2,4,1,1,2,1,1,2,4,8,...

### Núcleos Insatisfacibles (Unsatisfiable Cores)
Cuando F es UNSAT, se puede extraer un **subconjunto mínimo** de cláusulas que también sea UNSAT.

Útil para:
- Depuración de especificaciones
- Optimización de problemas de satisfacibilidad máxima

---

## 8. SAT Competition y Solvers Modernos

### Historia
- **GRASP (1996):** Primer CDCL, backjumping, clause learning
- **SATO (1995):** DPLL eficiente
- **Chaff (2001):** VSIDS + 2-watch literals, dominó la competición
- **MiniSAT (2003):** Implementación limpia y educativa
- **Lingeling, CaDiCaL, Kissat:** Solvers modernos con optimizaciones avanzadas

### Características de los Mejores Solvers
1. CDCL con aprendizaje de cláusulas (primera UIP)
2. VSIDS o variantes para selección de variables
3. 2-watch literals para propagación lazy
4. Restarts con política de Luby
5. Eliminación de variables (SatELite)
6. Simplificación de cláusulas

---

## Resumen de Complejidad

| Algoritmo | Tiempo | Notas |
|-----------|--------|-------|
| Unit Propagation | O(n·m) | Solo fórmulas Horn |
| LTUR | O(n·m) | Solo fórmulas Horn |
| 2-SAT | O(n+m) | Solo 2-CNF |
| Davis-Putnam | Exp | Explosión de memoria |
| Bucket Elimination | Exp | Exp en anchura del árbol |
| DPLL | Exp | Worst-case |
| CDCL | Exp | Muy eficiente en práctica |

> **SAT es NP-Completo:** no se conoce algoritmo polinomial general.  
> Los solvers CDCL modernos resuelven instancias industriales con millones de variables.
