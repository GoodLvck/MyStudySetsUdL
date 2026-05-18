# Modern Java — Resumen Completo (Tema 6)

**Asignatura:** AEPBD — Ampliació de BBDD i Enginyeria del Software  
**Profesor:** Juan Manuel Gimeno Illa  
**Universidad:** Universitat de Lleida (UdL)

---

## Índice

1. [Evolución de Java](#1-evolución-de-java)
2. [Default Methods — Permitir la Evolución](#2-default-methods--permitir-la-evolución)
3. [Expresiones Lambda](#3-expresiones-lambda)
4. [Interfaces Funcionales](#4-interfaces-funcionales)
5. [Referencias a Método](#5-referencias-a-método)
6. [Optional\<T\>](#6-optionalt)
7. [Streams](#7-streams)

---

## 1. Evolución de Java

### Línea temporal de versiones

| Versión | Año | Tipo | Novedades destacadas |
|---------|-----|------|----------------------|
| Java 1.0 | 1996 | — | JVM, applets, AWT |
| Java 5 | 2004 | — | Generics, Annotations, Autoboxing, for-each |
| Java 7 | 2011 | — | Diamond operator, try-with-resources, Fork/Join |
| Java 8 | 2014 | LTS | **Lambdas, Streams, Default methods**, Optional |
| Java 9 | 2017 | — | Módulos (Project Jigsaw), jshell (REPL) |
| Java 11 | 2018 | **LTS** | `var` en lambdas, métodos en String/Files |
| Java 17 | 2021 | **LTS** | Sealed classes, pattern matching para `instanceof` |
| Java 21 | 2023 | **LTS** | **Virtual Threads**, pattern matching para switch, record patterns |
| Java 25 | 2025 | **LTS** | — |

### Ciclo de lanzamiento

- **Cada 6 meses** (marzo y septiembre) desde Java 9
- **LTS cada ~2 años**: Java 11, 17, 21, 25
- LTS = soporte extendido (varios años de actualizaciones de seguridad)

### Evolución de la concurrencia

```
Java 1.0  →  Thread
Java 5/6  →  java.util.concurrent (Executor, BlockingQueue...)
Java 7    →  Fork/Join framework
Java 8    →  Parallel Streams + CompletableFuture
Java 21   →  Virtual Threads (Project Loom)
```

Los **Virtual Threads** (Java 21) permiten millones de hilos concurrentes con overhead mínimo, gestionados por la JVM en lugar del sistema operativo.

---

## 2. Default Methods — Permitir la Evolución

### El problema

Antes de Java 8: añadir un nuevo método a una interfaz **rompía TODAS sus implementaciones** existentes (error de compilación en las clases que no implementaban el nuevo método).

### La solución: default methods

```java
public interface Collection<E> {
    // Nuevo método en Java 8 — sin romper implementaciones
    default boolean removeIf(Predicate<? super E> filter) {
        Objects.requireNonNull(filter);
        boolean removed = false;
        Iterator<E> each = iterator();
        while (each.hasNext()) {
            if (filter.test(each.next())) {
                each.remove();
                removed = true;
            }
        }
        return removed;
    }
}
```

La clase que implementa la interfaz puede:
- **Heredar** el default method tal cual
- **Sobrescribir** el default method con su propia implementación

### Métodos estáticos en interfaces

Java 8 también permite **métodos estáticos** en interfaces, lo que permite mover los métodos utilitarios de la clase compañera (`Collections`) a la propia interfaz (`Collection`):

```java
// Antes: Collections.unmodifiableList(list)
// Ahora podría ser: Collection.unmodifiableList(list)
```

### Clases abstractas vs. Interfaces en Java 8+

| Característica | Clase abstracta | Interfaz (Java 8+) |
|---|---|---|
| Herencia | Solo UNA | Múltiples |
| Variables de instancia | Sí | No |
| Constructores | Sí | No |
| Métodos con implementación | Sí | Sí (default) |
| Visibilidad métodos | Cualquiera | public (implícito) |

### Optional methods

Default methods que lanzan `UnsupportedOperationException`:

```java
// Iterator.remove() en Java
default void remove() {
    throw new UnsupportedOperationException("remove");
}
```

### Herencia múltiple de comportamiento

```java
public interface Rotatable {
    void setRotationAngle(int angle);
    int getRotationAngle();
    default void rotateBy(int angle) {
        setRotationAngle((getRotationAngle() + angle) % 360);
    }
}

public interface Movable {
    int getX(); int getY();
    void setX(int x); void setY(int y);
    default void moveHorizontally(int distance) { setX(getX() + distance); }
    default void moveVertically(int distance) { setY(getY() + distance); }
}

public class Monster implements Rotatable, Movable, Resizable {
    // Monster hereda todos los comportamientos default
}
```

### Reglas de resolución de conflictos

Cuando una clase hereda el mismo default method de múltiples fuentes:

**Regla 1 — Las clases ganan:**
```java
public class C extends B implements A {
    // Gana el método de B (clase), aunque A sea interfaz con default method
}
```

**Regla 2 — Sub-interfaces ganan (más específico gana):**
```java
interface A { default void hello() { System.out.println("A"); } }
interface B extends A { default void hello() { System.out.println("B"); } }
class C implements A, B {
    // Gana B.hello() porque B es más específica (sub-interfaz de A)
}
```

**Regla 3 — Ambigüedad: la clase debe resolver explícitamente:**
```java
interface A { default void hello() { System.out.println("A"); } }
interface B { default void hello() { System.out.println("B"); } }
class C implements A, B {
    @Override
    public void hello() {
        B.super.hello();  // Explícitamente elige B
    }
}
```

---

## 3. Expresiones Lambda

### ¿Qué es una lambda?

Una **representación concisa de una función anónima** que puede pasarse como argumento. Características:
- Sin nombre
- Tiene lista de parámetros
- Tiene cuerpo
- Tiene tipo de retorno (inferido)
- Opcionalmente puede lanzar excepciones

### Sintaxis

```
(parámetros) -> expresión
(parámetros) -> { sentencias; }
```

Ejemplos válidos:
```java
(String s) -> s.length()
(Apple a) -> a.getWeight() > 150
(int x, int y) -> { System.out.print(x); System.out.println(y); }
() -> 42
(Apple a1, Apple a2) -> Integer.compare(a1.getWeight(), a2.getWeight())
```

### Behaviour Parameterization (Parametrización del comportamiento)

La capacidad de pasar múltiples comportamientos como parámetros a un método. Evolución del ejemplo de filtrar manzanas:

```java
// 1. Específico — no reutilizable
List<Apple> filterGreenApples(List<Apple> inventory) {
    return inventory.stream()
        .filter(a -> "green".equals(a.getColor()))
        .collect(toList());
}

// 2. Parcialmente genérico
List<Apple> filterApplesByColor(List<Apple> inventory, String color) { ... }

// 3. Genérico con interfaz
List<Apple> filter(List<Apple> inv, ApplePredicate p) {
    return inv.stream().filter(p::test).collect(toList());
}

// 4. Con lambda (Java 8)
filter(inventory, a -> "green".equals(a.getColor()));
filter(inventory, a -> a.getWeight() > 150);
```

### Ámbito léxico (Lexical Scoping)

Las lambdas pueden capturar variables de su entorno, pero solo variables locales **effectively final**:

```java
int portNumber = 1337; // effectively final (no se reasigna)
Runnable r = () -> System.out.println(portNumber); // OK

// portNumber = 999; // Esto haría que la captura falle
```

Las variables de instancia y estáticas se pueden usar libremente.

### Inferencia de tipos

El compilador infiere los tipos de los parámetros lambda:

```java
// Con tipos explícitos
Comparator<Apple> c1 = (Apple a1, Apple a2) -> 
    Integer.compare(a1.getWeight(), a2.getWeight());

// Con inferencia (equivalente)
Comparator<Apple> c2 = (a1, a2) -> 
    Integer.compare(a1.getWeight(), a2.getWeight());
```

---

## 4. Interfaces Funcionales

### Definición

Una interfaz funcional tiene **exactamente UN método abstracto** (SAM — Single Abstract Method). Puede tener default methods y métodos estáticos adicionales.

```java
@FunctionalInterface
public interface Predicate<T> {
    boolean test(T t);  // único método abstracto
    
    default Predicate<T> and(Predicate<? super T> other) { ... }
    default Predicate<T> or(Predicate<? super T> other) { ... }
}
```

La anotación `@FunctionalInterface` es opcional pero recomendada: el compilador verificará que haya exactamente un método abstracto.

### Functional Descriptor

La firma del método abstracto describe qué tipo de lambda es compatible:

| Interfaz | Descriptor | Método abstracto |
|----------|-----------|-----------------|
| `Predicate<T>` | `T → boolean` | `boolean test(T t)` |
| `Consumer<T>` | `T → void` | `void accept(T t)` |
| `Function<T,R>` | `T → R` | `R apply(T t)` |
| `Supplier<T>` | `() → T` | `T get()` |
| `UnaryOperator<T>` | `T → T` | `T apply(T t)` |
| `BinaryOperator<T>` | `(T,T) → T` | `T apply(T t1, T t2)` |
| `BiPredicate<L,R>` | `(L,R) → boolean` | `boolean test(L l, R r)` |
| `BiConsumer<T,U>` | `(T,U) → void` | `void accept(T t, U u)` |
| `BiFunction<T,U,R>` | `(T,U) → R` | `R apply(T t, U u)` |
| `Runnable` | `() → void` | `void run()` |
| `Comparator<T>` | `(T,T) → int` | `int compare(T o1, T o2)` |

### Especializaciones primitivas

Para evitar el autoboxing (int ↔ Integer), existen versiones primitivas:

```java
// Con boxing: int → Integer → se crea objeto
Predicate<Integer> isPrime = n -> n % 2 != 0;

// Sin boxing: directo sobre int
IntPredicate isPrime = n -> n % 2 != 0;
```

Principales especializaciones: `IntPredicate`, `LongPredicate`, `DoublePredicate`, `IntFunction<R>`, `ToIntFunction<T>`, `IntUnaryOperator`, `IntBinaryOperator`, etc.

### Excepciones comprobadas

Ninguna interfaz funcional estándar declara excepciones comprobadas. Dos soluciones:

```java
// Solución 1: Interfaz funcional propia
@FunctionalInterface
interface CheckedFunction<T, R> {
    R apply(T t) throws Exception;
}

// Solución 2: Envolver en try/catch
Function<String, Integer> safe = s -> {
    try { return Integer.parseInt(s); }
    catch (NumberFormatException e) { throw new RuntimeException(e); }
};
```

---

## 5. Referencias a Método

### Concepto

Una forma más concisa de escribir una lambda cuando se limita a invocar un método existente.

### Los 3 tipos

**Tipo 1: Referencia a método estático**
```java
// Lambda
Function<String, Integer> f = s -> Integer.parseInt(s);
// Method reference
Function<String, Integer> f = Integer::parseInt;
```

**Tipo 2: Referencia a método de instancia de tipo arbitrario**
```java
// Lambda: primer parámetro es el receptor
Function<String, String> f = s -> s.toUpperCase();
Function<String, String> f = String::toUpperCase;

// Lambda con dos parámetros: primero es receptor
BiPredicate<List<String>, String> f = (list, elem) -> list.contains(elem);
BiPredicate<List<String>, String> f = List::contains;
```

**Tipo 3: Referencia a método de instancia de objeto concreto**
```java
// Lambda con objeto externo
Consumer<String> f = s -> System.out.println(s);
Consumer<String> f = System.out::println;

// Con objeto en variable
String prefix = "Hello: ";
Consumer<String> f = s -> prefix.concat(s);
// No aplica method reference aquí
```

### Referencias a constructor

```java
// Constructor sin argumentos
Supplier<Apple> s = () -> new Apple();
Supplier<Apple> s = Apple::new;

// Constructor con un argumento
Function<Integer, Apple> f = weight -> new Apple(weight);
Function<Integer, Apple> f = Apple::new;

// Constructor con dos argumentos
BiFunction<String, Integer, Apple> bf = (color, weight) -> new Apple(color, weight);
BiFunction<String, Integer, Apple> bf = Apple::new;
```

### Tabla de conversiones

| Lambda | Method Reference |
|--------|-----------------|
| `(Apple a) -> a.getWeight()` | `Apple::getWeight` |
| `() -> Thread.currentThread().getPriority()` | `Thread.currentThread()::getPriority` |
| `(str, i) -> str.substring(i)` | `String::substring` |
| `(String s) -> System.out.println(s)` | `System.out::println` |
| `(String s) -> Integer.parseInt(s)` | `Integer::parseInt` |
| `() -> new Apple()` | `Apple::new` |
| `(w) -> new Apple(w)` | `Apple::new` |

---

## 6. Optional\<T\>

### La motivación: el problema de null

Tony Hoare introdujo `null` en 1965 en ALGOL W. Más tarde lo llamó **"my billion dollar mistake"**.

Problemas de null en Java:
- Es fuente de `NullPointerException`
- Engrosa el código con comprobaciones
- No tiene significado semántico (¿ausencia de valor? ¿error? ¿no inicializado?)
- Rompe la filosofía Java (null no es un objeto)
- Es un hueco en el sistema de tipos

### Optional\<T\>

Clase que encapsula un valor opcional (inspirada en el tipo `Maybe` de Haskell):

```java
// Creación
Optional<Car> opt1 = Optional.empty();           // vacío
Optional<Car> opt2 = Optional.of(car);           // car != null (NPE si null)
Optional<Car> opt3 = Optional.ofNullable(car);  // puede ser null
```

### Operaciones principales

```java
// Transformación
Optional<String> name = optCar.map(Car::getBrand);
Optional<Insurance> ins = optCar.flatMap(Car::getInsurance);
Optional<Car> heavy = optCar.filter(c -> c.getWeight() > 1000);

// Extracción
String brand = optCar.orElse("Unknown");                      // valor o default
String brand = optCar.orElseGet(() -> fetchDefault());         // valor o supplier
String brand = optCar.orElseThrow(NotFoundException::new);     // valor o excepción
String brand = optCar.get();                                    // valor o NoSuchElementException

// Consulta
boolean present = optCar.isPresent();   // Java 8+
boolean empty = optCar.isEmpty();       // Java 11+
optCar.ifPresent(c -> System.out.println(c));  // ejecuta si hay valor
```

### Por qué flatMap en lugar de map

```java
// Model:
class Person { Optional<Car> getCar() {...} }
class Car { Optional<Insurance> getInsurance() {...} }
class Insurance { String getName() {...} }

// Con map — NO funciona: devuelve Optional<Optional<Car>>
Optional<Optional<Car>> bad = optPerson.map(Person::getCar);

// Con flatMap — aplana el resultado
Optional<Car> good = optPerson.flatMap(Person::getCar);
```

### Cadena completa

```java
String carInsuranceName = optPerson
    .flatMap(Person::getCar)          // Optional<Car>
    .flatMap(Car::getInsurance)       // Optional<Insurance>
    .map(Insurance::getName)          // Optional<String>
    .orElse("Unknown");               // String
```

### Modelado con Optional

```java
// La propiedad Optional<Car> indica que el coche es OPCIONAL
class Person {
    private Optional<Car> car;
    public Optional<Car> getCar() { return car; }
}

// El nombre String es REQUERIDO (siempre presente)
class Insurance {
    private String name;
    public String getName() { return name; }
}
```

> **Nota importante:** No usar `Optional` como campo de clase (no implementa `Serializable`). Se diseñó para ser tipo de retorno.

---

## 7. Streams

### Limitaciones de Collections

```java
// Java 7: imperativo, difícil de paralelizar
List<Dish> lowCalDishes = new ArrayList<>();
for (Dish d : dishes) {
    if (d.getCalories() < 400) lowCalDishes.add(d);
}
Collections.sort(lowCalDishes, Comparator.comparingInt(Dish::getCalories));
List<String> names = new ArrayList<>();
for (Dish d : lowCalDishes) names.add(d.getName());
```

```java
// Java 8: declarativo, fácilmente paralelizable
List<String> names = dishes.stream()
    .filter(d -> d.getCalories() < 400)
    .sorted(comparing(Dish::getCalories))
    .map(Dish::getName)
    .collect(toList());
```

### Definición

Un Stream es una **secuencia de elementos de una fuente** que soporta operaciones de procesamiento de datos.

Características:
- **Secuencia de elementos**: tipos declarados, no almacena datos
- **Fuente**: colección, array, I/O, generación
- **Lazy**: operaciones intermedias no se ejecutan hasta la terminal
- **Pipelining**: operaciones intermedias devuelven Stream (encadenables)
- **Iteración interna**: la biblioteca gestiona la iteración

### Operaciones intermedias vs. terminales

**Operaciones intermedias** (devuelven Stream, son lazy):

| Operación | Descripción |
|-----------|-------------|
| `filter(Predicate<T>)` | Filtra elementos por predicado |
| `map(Function<T,R>)` | Transforma cada elemento |
| `flatMap(Function<T,Stream<R>>)` | Transforma y aplana |
| `sorted()` / `sorted(Comparator)` | Ordena elementos |
| `distinct()` | Elimina duplicados |
| `limit(n)` | Primeros n elementos |
| `skip(n)` | Salta los primeros n |
| `peek(Consumer)` | Ejecuta sin consumir (debug) |
| `takeWhile(Predicate)` | Mientras se cumple (Java 9+) |
| `dropWhile(Predicate)` | Descarta mientras se cumple (Java 9+) |

**Operaciones terminales** (producen resultado, ejecutan el pipeline):

| Operación | Resultado |
|-----------|-----------|
| `collect(Collector)` | Colección u otra estructura |
| `forEach(Consumer)` | void |
| `count()` | long |
| `reduce(BinaryOperator)` | Optional<T> |
| `reduce(T, BinaryOperator)` | T |
| `findFirst()` / `findAny()` | Optional<T> |
| `anyMatch(Predicate)` | boolean |
| `allMatch(Predicate)` | boolean |
| `noneMatch(Predicate)` | boolean |
| `min(Comparator)` / `max(Comparator)` | Optional<T> |

### Ejemplos comunes

**Filtrar y transformar:**
```java
List<String> heavyAppleNames = apples.stream()
    .filter(a -> a.getWeight() > 150)
    .map(Apple::getName)
    .collect(toList());
```

**Reducción numérica:**
```java
// Suma con reduce
int total = numbers.stream().reduce(0, Integer::sum);

// Suma con IntStream (sin boxing)
int total = names.stream().mapToInt(String::length).sum();

// Estadísticas
IntSummaryStatistics stats = numbers.stream()
    .mapToInt(Integer::intValue)
    .summaryStatistics();
```

**flatMap para aplanar:**
```java
List<String> uniqueLetters = words.stream()
    .map(word -> word.split(""))    // Stream<String[]>
    .flatMap(Arrays::stream)         // Stream<String>
    .distinct()
    .collect(toList());
```

**Ordenación compuesta:**
```java
persons.stream()
    .sorted(comparing(Person::age).reversed()
        .thenComparing(Person::name))
    .collect(toList());
```

**Joining:**
```java
String result = names.stream()
    .collect(Collectors.joining(", ", "[", "]"));
// "[Ana, Bob, Carlos]"
```

**Leer archivo como Stream:**
```java
try (Stream<String> lines = Files.lines(Paths.get("data.txt"))) {
    long count = lines
        .flatMap(line -> Arrays.stream(line.split(" ")))
        .distinct()
        .count();
}
```

### Parallel Streams

```java
// Activar paralelismo
list.parallelStream()
list.stream().parallel()

// De vuelta a secuencial
stream.sequential()
```

**Internamente** usa Fork/Join framework y `Spliterator` (interfaz que permite dividir iteradores).

**Descomposabilidad de fuentes:**

| Fuente | Descomposabilidad |
|--------|-------------------|
| `ArrayList` | Excelente |
| `IntStream.range` | Excelente |
| `HashSet` / `TreeSet` | Buena |
| `LinkedList` | Pobre |
| `Stream.iterate` | Pobre |

> No siempre vale la pena paralelizar: para datos pequeños o operaciones baratas, el overhead del ForkJoin puede ser mayor que el beneficio.

### Streams vs. Colecciones

| Aspecto | Colecciones | Streams |
|---------|------------|---------|
| Cuándo se calculan | En memoria | Bajo demanda (lazy) |
| Iteración | Externa (el programador) | Interna (la biblioteca) |
| Reutilización | Sí | Solo una vez |
| Paralelismo | Manual | `.parallel()` |
| Estilo | Imperativo | Declarativo |

---

## Resumen de Java 8: el gran salto

Java 8 fuerza el **pensamiento declarativo**: expresar *qué* se quiere, no *cómo* conseguirlo.

Los tres pilares de Java 8:
1. **Lambdas** — comportamiento como dato, reducen el boilerplate
2. **Streams** — procesar colecciones de forma funcional y potencialmente paralela
3. **Default methods** — evolución de APIs sin romper implementaciones existentes

Con el ciclo de 6 meses, Java sigue evolucionando: Java 17 trajo sealed classes y pattern matching, Java 21 trajo virtual threads. La base funcional de Java 8 es el cimiento sobre el que se construyen estas mejoras.
