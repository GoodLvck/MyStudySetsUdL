# Tema 3: Machine Learning y Redes Neuronales

## 1. Historia y Contexto

### Línea Temporal
| Año | Hito |
|-----|------|
| **1958** | **Perceptrón** — Frank Rosenblatt. Primer modelo de neurona artificial entrenada |
| **1986** | **Backpropagation** — Rumelhart, Hinton, Williams. Permite entrenar redes multicapa |
| **2012** | **AlexNet** — Krizhevsky et al. CNN profunda que gana ImageNet (error top-5: 15.3%) |
| **2017** | **Transformer** — Vaswani et al. "Attention is All You Need" |
| **2020** | **GPT-3** — OpenAI. 175B parámetros, pocos ejemplos suficientes (few-shot learning) |
| **2022** | **AlphaFold 2** — DeepMind. Resuelve el plegamiento de proteínas |
| **2024** | **Premios Nobel** de Física y Química concedidos a investigadores de IA/ML |

### ¿Por qué ahora?
1. **Datos:** Internet genera cantidades masivas de datos etiquetados
2. **Computación:** GPUs y TPUs permiten entrenamiento paralelo masivo
3. **Algoritmos:** Mejoras en arquitecturas, optimizadores y técnicas de regularización

---

## 2. Modelo de Neurona Artificial

### Neurona Básica
Una neurona artificial calcula:

```
y = f(Σᵢ wᵢ·xᵢ + b)
```

Donde:
- **xᵢ:** entradas (features)
- **wᵢ:** pesos (parámetros aprendibles)
- **b:** bias (desplazamiento, también aprendible)
- **f:** función de activación (introduce no-linealidad)
- **y:** salida de la neurona

### Interpretación
- Los pesos determinan la importancia de cada entrada
- El bias permite que la neurona se active incluso con entradas nulas
- La función de activación decide si la neurona "dispara" y con qué intensidad

---

## 3. Tipos de Redes Neuronales

### Feedforward Neural Networks (FFNN)
- Información fluye en **una sola dirección**: entrada → capas ocultas → salida
- Sin ciclos ni conexiones hacia atrás
- Cada capa aplica: `h = f(W·h_prev + b)`

### Convolutional Neural Networks (CNN)
- Diseñadas para datos con **estructura espacial** (imágenes, señales)
- Usan **filtros convolucionales** compartidos → muchos menos parámetros
- Capas: Convolución → Activación → Pooling → Fully Connected
- Ejemplo: AlexNet, ResNet, VGG

### Recurrent Neural Networks (RNN)
- Conexiones cíclicas permiten procesar **secuencias de longitud variable**
- Estado oculto hₜ = f(W·hₜ₋₁ + U·xₜ + b) — memoria del pasado
- Variantes: LSTM (Long Short-Term Memory), GRU (Gated Recurrent Unit)
- Aplicaciones: texto, audio, series temporales

### Graph Neural Networks (GNN)
- Operan sobre **datos estructurados en grafos** (moléculas, redes sociales)
- Cada nodo agrega información de sus vecinos en capas sucesivas
- Aplicaciones: química computacional, recomendación, análisis de redes

---

## 4. Machine Learning Clásico

### Aprendizaje Supervisado
Aprender f: X → Y a partir de pares etiquetados {(xᵢ, yᵢ)}

**Algoritmos principales:**

| Modelo | Tipo | Descripción |
|--------|------|-------------|
| **Regresión Lineal** | Regresión | Ajusta hiperplano: ŷ = wᵀx + b |
| **Regresión Logística** | Clasificación | Probabilidad con sigmoide: P(y=1) = σ(wᵀx + b) |
| **Árbol de Decisión** | Ambos | Reglas jerárquicas: divide el espacio de features |
| **Random Forest** | Ambos | Ensemble de árboles con bagging y votación |
| **SVM** | Clasificación | Maximiza el margen entre clases |

### Aprendizaje No Supervisado
Descubrir estructura en datos **sin etiquetas**

| Algoritmo | Tipo | Descripción |
|-----------|------|-------------|
| **k-Means** | Clustering | Agrupa en k clusters minimizando distancia al centroide |
| **Clustering Jerárquico** | Clustering | Construye dendrograma (aglomerativo o divisivo) |
| **PCA** | Reducción de dim. | Proyecta en direcciones de máxima varianza |

---

## 5. Fundamentos Matemáticos

### Derivadas Parciales
La derivada parcial ∂f/∂xᵢ mide la sensibilidad de f a cambios en xᵢ, manteniendo el resto constante:

```
∂f/∂xᵢ = lím_{h→0} [f(x₁,...,xᵢ+h,...,xₙ) − f(x₁,...,xₙ)] / h
```

**Ejemplo:** f(x,y) = x²y + 3y
- ∂f/∂x = 2xy
- ∂f/∂y = x² + 3

### Vector Gradiente
El gradiente reúne **todas las derivadas parciales** en un vector:

```
∇f(x) = (∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ)
```

Propiedades:
- **Apunta** en la dirección de máximo crecimiento de f
- **-∇f** apunta en la dirección de máximo descenso
- |∇f| mide la tasa de cambio en esa dirección

### Regla de la Cadena
Para funciones compuestas f(g(x)):
```
d/dx f(g(x)) = f'(g(x)) · g'(x)
```

Es la base del algoritmo de backpropagation.

### Teorema de Weierstrass (Aproximación Universal)
"Toda función continua en un intervalo compacto puede aproximarse arbitrariamente bien por un polinomio."

**Implicación para ML:**
- Si los polinomios pueden aproximar cualquier función → los polinomios son **aproximadores universales**
- Las redes neuronales con activaciones no lineales también son aproximadores universales (Hornik, 1989)
- Esto justifica teóricamente la capacidad de las redes de aprender cualquier función

---

## 6. Regresión Lineal

### Modelo
```
ŷ = wᵀx + b = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

### Función de Pérdida: MSE
```
L(w,b) = (1/n) Σᵢ (yᵢ − ŷᵢ)²
```

### Solución Analítica (Ecuación Normal)
Derivando e igualando a cero: ∂L/∂w = 0

```
w* = (XᵀX)⁻¹ Xᵀy
```

**Ventaja:** Solución exacta en un paso.  
**Desventaja:** Invertir XᵀX es O(d³), imposible para d grande.

### Solución Iterativa: Gradiente Descendente
El gradiente del MSE respecto a w:
```
∇L = (2/n) Xᵀ(Xw − y)
```

Actualización:
```
w ← w − α · ∇L
```

---

## 7. Descenso de Gradiente

### Algoritmo General
```
Inicializar parámetros θ₀
Para k = 1, 2, ..., max_iter:
    g = ∇L(θₖ)       # calcular gradiente
    θₖ₊₁ = θₖ − α·g  # actualizar
```

### Variantes

| Variante | Datos por iteración | Características |
|----------|---------------------|-----------------|
| **Batch GD** | Todo el dataset | Estable, lento |
| **SGD** | 1 ejemplo | Rápido, ruidoso |
| **Mini-batch SGD** | k ejemplos (32-512) | Equilibrio práctico |

### Learning Rate α
- **Muy alto:** oscilaciones, divergencia
- **Muy bajo:** convergencia muy lenta
- **Técnicas adaptativas:** Adam, RMSProp, AdaGrad — ajustan α automáticamente

### Problemas
- **Mínimos locales:** GD puede quedar atrapado (raro en alta dimensionalidad)
- **Mesetas:** ∇f ≈ 0 sin ser mínimo
- **Vanishing gradient:** gradientes se vuelven muy pequeños en capas profundas

---

## 8. Funciones de Activación

### Sigmoide
```
σ(x) = 1 / (1 + e^(−x))
```
- Rango: (0, 1) — ideal para probabilidades
- **Problema:** saturación en extremos → vanishing gradient

### Tanh
```
tanh(x) = (eˣ − e^(−x)) / (eˣ + e^(−x))
```
- Rango: (-1, 1) — centrada en cero
- Mejor que sigmoide para capas ocultas

### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x)
```
- **La más usada** en capas ocultas de redes profundas
- Computacionalmente eficiente
- No saturación en x > 0 → gradientes fluyen mejor
- **Dying ReLU problem:** neuronas con salida siempre 0

### Softmax (para salida de clasificación multiclase)
```
Softmax(xᵢ) = e^(xᵢ) / Σⱼ e^(xⱼ)
```
- Convierte un vector en distribución de probabilidad (suma = 1)

---

## 9. Entrenamiento de Redes

### Forward Pass
Calcular predicciones capa a capa:
```
h⁰ = x
h^l = f^l(W^l · h^(l-1) + b^l)   para l = 1,...,L
ŷ = h^L
```

### Cálculo de Pérdida

**MSE (Regresión):**
```
L = (1/n) Σᵢ (yᵢ − ŷᵢ)²
```

**Cross-Entropy (Clasificación):**
```
L = −(1/n) Σᵢ Σₖ yᵢₖ · log(ŷᵢₖ)
```

### Backward Pass (Backpropagation)
Aplicar la regla de la cadena de salida a entrada:
```
δ^L = ∂L/∂z^L                          # gradiente capa salida
δ^l = (W^(l+1))ᵀ · δ^(l+1) · f'^l(z^l) # propagar hacia atrás
```

Gradientes de los parámetros:
```
∂L/∂W^l = δ^l · (h^(l-1))ᵀ
∂L/∂b^l = δ^l
```

### Train / Validation / Test Split
- **Training set (~70%):** para ajustar los parámetros (w, b)
- **Validation set (~15%):** para seleccionar hiperparámetros y arquitectura
- **Test set (~15%):** evaluación final sin sesgo — solo usar una vez

---

## 10. Normalización de Datos

### ¿Por qué normalizar?
- Features con escalas distintas → gradientes desequilibrados
- Convergencia más lenta o inestable
- Algunos algoritmos (k-NN, SVM) muy sensibles a la escala

### Min-Max Scaling
```
x' = (x − x_min) / (x_max − x_min)
```
- Rango de salida: [0, 1]
- Ideal cuando se conocen los límites del dominio

### Z-Score (Estandarización)
```
x' = (x − μ) / σ
```
- Media 0, desviación estándar 1
- Ideal para distribuciones gaussianas
- Más robusta a outliers que Min-Max

---

## 11. PyTorch y Tensores

### Tensores
Arrays multidimensionales con soporte para GPU y diferenciación automática:

```python
import torch

# Crear tensor
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Operaciones
y = x ** 2 + 3 * x

# Calcular gradientes
y.sum().backward()  # backward
print(x.grad)       # dy/dx = 2x + 3 = [5, 7, 9]
```

### Autograd
Sistema de **diferenciación automática** de PyTorch:
- Construye un grafo computacional dinámico durante el forward pass
- `backward()` recorre el grafo en sentido inverso aplicando la regla de la cadena
- Los gradientes se acumulan en `.grad` de cada tensor con `requires_grad=True`

### Bucle de Entrenamiento Estándar
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()       # 1. limpiar gradientes
        y_pred = model(x_batch)     # 2. forward pass
        loss = criterion(y_pred, y_batch)  # 3. pérdida
        loss.backward()             # 4. backward (calcular gradientes)
        optimizer.step()            # 5. actualizar parámetros
```

---

## 12. Arquitectura MNIST (Ejemplo Práctico)

Para clasificar dígitos escritos a mano (28×28 pixels, 10 clases):

```
Input: 784 neuronas (28×28 aplanado)
   ↓
Capa oculta 1: 128 neuronas + ReLU
   ↓
Capa oculta 2: 64 neuronas + ReLU
   ↓
Capa salida: 10 neuronas + Softmax
   ↓
Output: probabilidad de cada dígito (0-9)
```

**Parámetros totales:**
- W1: 784×128 = 100,352
- W2: 128×64 = 8,192
- W3: 64×10 = 640
- Total ≈ 109K parámetros

---

## Resumen: Pipeline Completo de ML

```
1. Datos: recopilar, limpiar, etiquetar
      ↓
2. Preprocesamiento: normalización, codificación
      ↓
3. División: train / val / test
      ↓
4. Arquitectura: elegir modelo y estructura
      ↓
5. Entrenamiento: forward → loss → backward → update
      ↓
6. Validación: ajustar hiperparámetros (α, capas, neuronas)
      ↓
7. Evaluación: métricas finales en test set
      ↓
8. Despliegue: producción / inferencia
```

> **Nota:** El test set se usa **solo una vez** al final. Usarlo para decisiones de diseño introduce sesgo de optimismo.
