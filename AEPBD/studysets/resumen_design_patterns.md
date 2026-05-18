# Patrones de Diseño (GoF) — AEPBD

## 1. Introducción

Un **patrón de diseño** describe un problema que ocurre repetidamente en nuestro entorno y su solución, de manera que pueda aplicarse indefinidamente sin hacer lo mismo dos veces (Christopher Alexander).

### Gang of Four (GoF)
- Gamma, Helm, Johnson & Vlissides — *"Design Patterns"* (1995)
- 23 patrones basados en el estudio de sistemas existentes (no se inventan, se sintetizan)
- Objetivo: mayor reutilización y flexibilidad mediante patrones simples

### Clasificación GoF clásica

| Propósito \ Ámbito | Clase (herencia, estático) | Objeto (polimorfismo, composición, dinámico) |
|---|---|---|
| **Creación** | Factory Method | Abstract Factory, Builder, Prototype, Singleton |
| **Estructural** | Class Adapter | Object Adapter, Bridge, Composite, Decorator, Façade, Flyweight, Proxy |
| **Comportamiento** | Interpreter, Template Method | Chain of Responsibility, Command, Iterator, Mediator, Memento, Observer, State, Strategy, Visitor |

### Clasificación por Intención (Metsker & Wake)

| Intención | Patrones |
|---|---|
| **Interfaces** | Adapter, Façade, Composite, Bridge |
| **Responsabilidad** | Singleton, Observer, Mediator, Proxy, Chain of Responsibility, Flyweight |
| **Construcción** | Builder, Factory Method, Abstract Factory, Prototype, Memento |
| **Operaciones** | Template Method, State, Strategy, Command, Interpreter |
| **Extensiones** | Decorator, Iterator, Visitor |

---

## 2. Patrones de Interfaz

> Intención: ir más allá de las facilidades que existen en Java para manejar interfaces.

| Si la intención es | Patrón |
|---|---|
| Adaptar la interfaz de una clase para que se corresponda con la que espera un cliente | **Adapter** |
| Proporcionar una interfaz simple para interactuar con un conjunto de clases | **Façade** |
| Definir una interfaz que se aplique tanto a objetos individuales como a grupos | **Composite** |
| Desacoplar una abstracción de su implementación | **Bridge** |

### Adapter (Adaptador)

**Intención:** proporcionar la interfaz que un cliente espera usando los servicios de una clase con un interfaz diferente.

**Dos variantes:**

*Class Adapter* — usa herencia:
```java
public class OozinozRocket extends PhysicalRocket implements RocketSim {
    public double getMass() { return getMass(time); }
    public double getThrust() { return getThrust(time); }
}
```

*Object Adapter* — usa delegación (más flexible):
```java
public class OozinozSkyrocket extends Skyrocket {
    private PhysicalRocket rocket;  // delegación
    public double getMass() { return rocket.getMass(simTime); }
}
```

**Ayudando a la adaptación:** cuando una interfaz es muy grande, combinar interfaz + clase abstracta con implementaciones por defecto (cliente solo redefine lo que necesita).

### Composite (Compuesto)

**Intención:** permitir que los clientes utilicen de forma uniforme tanto objetos individuales como sus composiciones.

**Estructura:**
- `Component` (interfaz/abstracta) — define la operación
- `Composite` (tiene hijos: `List<Component>`) — delega recursivamente
- `Leaf` — caso base

```java
// En MachineComposite:
public int getMachineCount() {
    int count = 0;
    for (MachineComponent c : components) count += c.getMachineCount();
    return count;
}
// En Machine (leaf):
public int getMachineCount() { return 1; }
```

**Árboles vs ciclos:**
- Grafo dirigido es árbol si: nodo raíz único no referenciado + cada nodo tiene un único padre
- Con ciclos: detectar nodos visitados para evitar bucles infinitos
- Usar `Collections.newSetFromMap(new IdentityHashMap<>())` para detección por identidad

**Copia:** superficial (se comparten componentes) vs profunda (se copia toda la estructura recursivamente).

---

## 3. Patrones de Responsabilidad

> Centralizar, aislar o escalar responsabilidades entre objetos.

### Singleton

**Intención:** asegurar que una clase tiene una única instancia y proporcionar un punto global de acceso.

```java
// Eager (no lazy, siempre seguro):
public class Factory {
    public static final Factory INSTANCE = new Factory();
    private Factory() { }
}

// Lazy (NO thread-safe):
public class Factory {
    private static Factory factory;
    private Factory() { }
    public static Factory getFactory() {
        if (factory == null) factory = new Factory();
        return factory;
    }
}

// Synchronized (thread-safe):
public static Factory getFactory() {
    synchronized (classLock) {
        if (factory == null) factory = new Factory();
        return factory;
    }
}

// Enum (thread-safe + serializable automáticamente):
public enum Factory {
    factory;
    private Factory() { }
}
```

**Serializable:** implementar `readResolve()` que devuelve `getFactory()`.

**Consideración moderna:** los singletons son puntos globales problemáticos. Usar contenedores de inyección de dependencias (Spring, Guice) en su lugar.

### Observer (Observador)

**Intención:** definir dependencia 1-a-N entre objetos de manera que cuando uno cambia de estado, todos los dependientes son notificados.

**Dos modelos:**

| Modelo Pull | Modelo Push |
|---|---|
| Observado no pasa info en la notificación | Observado pasa info al notificar |
| Observador interroga al observado | Observador ya tiene todo lo necesario |
| Menor acoplamiento | Mayor acoplamiento |

**Java API (deprecated desde Java 9):**
```java
public interface Observer { void update(Observable o, Object arg); }
public class Observable {
    public void addObserver(Observer o) { }
    public void notifyObservers(Object arg) { }
    protected void setChanged() { }
}
```

**Uso:**
```java
slider.addChangeListener(this); // registrar como observador
public void stateChanged(ChangeEvent e) { /* reaccionar */ }
```

**MVC (Modelo-Vista-Controlador):**
- Modelo: extiende `Observable`, contiene datos
- Vista/Controlador: implementan `Observer`, se registran en el modelo
- Cuando el modelo cambia: `setChanged(); notifyObservers();`

**Alternativas Java 9+:** `java.beans` event propagation, Flow API (reactive streams).

### Otros patrones de responsabilidad

- **Mediator:** centraliza en una clase la responsabilidad de coordinar cómo un conjunto de objetos interactúan
- **Proxy:** permite que un objeto actúe en nombre de otro
- **Chain of Responsibility:** petición se propaga por una serie de objetos hasta que hay uno que la trata
- **Flyweight:** centraliza responsabilidades en objetos compartidos de grano fino

---

## 4. Patrones de Construcción

> Ir más allá de la construcción normal con `new`.

| Si la intención es | Patrón |
|---|---|
| Obtener información gradualmente antes de pedir la construcción | **Builder** |
| Postergar la decisión de qué clase instanciar | **Factory Method** |
| Construir una familia de objetos relacionados | **Abstract Factory** |
| Especificar el objeto a crear dando un ejemplo | **Prototype** |
| Reconstruir un objeto desde una versión "durmiente" | **Memento** |

### Builder (Constructor)

**Intención:** trasladar la lógica de construcción a un objeto fuera de la clase a instanciar.

**Ventajas:**
- Clase a construir puede ser **inmutable** (todos los campos `final`)
- Múltiples builders para la misma clase (UnforgivingBuilder vs ForgivingBuilder)
- **Fluent API:** métodos del builder devuelven `this` → `StringBuilder`, `HttpRequest.Builder`

```java
// Uso típico: parsing
public void parse(String s) throws ParseException {
    String[] tokens = s.split(",\\s*");
    for (int i = 0; i < tokens.length; i += 2) {
        if ("headcount".equalsIgnoreCase(tokens[i]))
            builder.setHeadcount(Integer.parseInt(tokens[i+1]));
        // ...
    }
}
Reservation r = builder.build(); // construye el objeto inmutable
```

### Factory Method (Método de Fabricación)

**Intención:** permitir que el desarrollador de una clase defina una interfaz para crear un objeto manteniendo el control sobre qué clase se instancia.

```java
// Ejemplo clásico: Iterable.iterator() — nadie sabe qué clase concreta devuelve
Iterator<String> iter = list.iterator();

// Ejemplo CreditCheck:
public abstract class CreditCheckFactory {
    public static CreditCheck createCreditCheck() {
        return isAgencyUp() ? new CreditCheckOnline() : new CreditCheckOffline();
    }
}
```

**Jerarquías paralelas:** cuando dos jerarquías tienen clases correspondientes (Machine + Scheduler), Factory Method crea el Scheduler correcto para cada tipo de Machine.

### Abstract Factory (Factoría Abstracta)

**Intención:** facilitar la creación de **familias** de objetos relacionados.

```java
// GUI Kit: UI base + BetaUI subclase
public class UI {
    public JButton createButtonOk() { /* icono rocket */ }
    public JButton createButtonCancel() { /* icono rocket-down */ }
}
public class BetaUI extends UI {
    public JButton createButtonOk() { /* icono cherry */ }
    public JButton createButtonCancel() { /* icono cherry-down */ }
}
```

**Diferencia con Factory Method:** Abstract Factory crea una *familia* de objetos; Factory Method crea *un* tipo de objeto.

**Temas típicos:** look-and-feel (UI), localización geográfica (US vs Canada CheckFactory).

---

## 5. Patrones de Operación

> Operación (UML): especificación de un servicio. Método: implementación de una operación.

| Si la intención es | Patrón |
|---|---|
| Implementar un algoritmo, posponiendo algunos pasos a subclases | **Template Method** |
| Distribuir una operación según el estado | **State** |
| Encapsular alternativas intercambiables | **Strategy** |
| Encapsular la llamada a un método en un objeto | **Command** |
| Distribuir una operación por tipo de composición | **Interpreter** |

### Template Method (Método Plantilla)

**Intención:** implementar un algoritmo en un método, posponiendo la definición de algunos pasos para que las subclases puedan redefinirlos.

```java
// Superclase define el esquema:
public abstract class AsterStarPress {
    public void shutdown() {
        if (inProcess()) { stopProcessing(); markMoldIncomplete(currentMoldID); }
        usherInputMolds();
        dischargePaste();   // ← subclases pueden redefinir
        flush();
    }
    protected abstract void markMoldIncomplete(int id); // ← paso abstracto
}
// Subclase implementa los pasos específicos:
public class OzAsterStarPress extends AsterStarPress {
    public void dischargePaste() { super.dischargePaste(); getFactory().collectPaste(); }
    public void markMoldIncomplete(int id) { getManager().setMoldIncomplete(id); }
}
```

**Hook (gancho):** llamada opcional con implementación por defecto (puede estar vacía), que da oportunidad a subclases de insertar código en un punto del algoritmo.

**Consecuencias:**
- Inversión de Control (IoC) — **Principio de Hollywood:** "Don't call us, we'll call you"
- Reduce duplicación agrupando código común en superclases
- Suele usar Factory Method para crear objetos de subclases

### Strategy (Estrategia)

**Intención:** encapsular las diferentes alternativas a un problema en clases separadas que implementan una misma interfaz, separando el código de selección de las implementaciones.

**Antes (problemático):**
```java
public Firework getRecommended() {
    // mezcla selección + implementación en un método largo...
    if (promotedName != null) return Firework.lookup(promotedName);
    if (isRegistered()) return (Firework) Rel8.advise(this);
    if (spendingSince(cal.getTime()) > 1000) return (Firework) LikeMyStuff.suggest(this);
    return Firework.getRandom();
}
```

**Después (con Strategy):**
```java
public interface Advisor { Firework recommend(Customer c); }
// Implementaciones separadas: PromotionAdvisor, GroupAdvisor, ItemAdvisor, RandomAdvisor

public Firework getRecommended() { return getAdvisor().recommend(this); }
private Advisor getAdvisor() {
    if (promotionAdvisor.hasItem()) return promotionAdvisor;
    else if (isRegistered()) return groupAdvisor;
    else if (isBigSpender()) return itemAdvisor;
    else return randomAdvisor;
}
```

**Otro ejemplo:** `Comparator<T>` como estrategia de ordenación para `Collections.sort()`.

**Diferencia con Template Method:** Strategy usa composición (objeto externo); Template Method usa herencia (subclase).

---

## 6. Patrones de Extensión

| Si la intención es | Patrón |
|---|---|
| Acceso secuencial a elementos de una colección | **Iterator** |
| Nuevas operaciones para una jerarquía sin modificarla | **Visitor** |
| Componer dinámicamente comportamiento de un objeto | **Decorator** |

### Visitor (Visitante)

**Intención:** ampliar las operaciones de una jerarquía de clases sin necesidad de reprogramarla.

**Infraestructura necesaria:**
1. Añadir `accept(MachineVisitor v)` en cada clase de la jerarquía
2. Definir interfaz `MachineVisitor` con `visit()` por cada clase concreta

```java
public interface MachineVisitor {
    void visit(Machine m);
    void visit(MachineComposite mc);
}
// En cada clase de la jerarquía:
public void accept(MachineVisitor v) { v.visit(this); }
```

**Por qué `accept` no puede estar en la clase base:** aunque el código es idéntico, `this` tendría el tipo de la clase base, no el tipo concreto.

**Double dispatch:** la combinación de `accept` + `visit` asegura que se ejecuta la versión correcta basándose en el tipo runtime de *ambos* (el visitante y el elemento).

**Expression Problem (Wadler, 1998):** añadir subclasses sin actualizar la interfaz Visitor causa error de compilación — ventaja, ya que obliga a tratar los nuevos casos.

```java
public class FindVisitor implements MachineVisitor {
    public void visit(Machine m) { if (m.getId() == soughtId) found = m; }
    public void visit(MachineComposite mc) {
        if (mc.getId() == soughtId) { found = mc; return; }
        for (MachineComponent c : mc.getComponents()) c.accept(this);
    }
}
```

### Iterator (Iterador)

**Intención:** proporcionar acceso secuencial a los elementos de una colección.

**Mecanismos de iteración en Java:**
- `for/while/do-while` (básico)
- `java.util.Enumeration` (antiguo)
- `java.util.Iterator` (JDK 1.2)
- `foreach` (JDK 1.5)
- Streams (Java 8+)

**Iterador sobre Composite (complejo):**
```
ComponentIterator<E> (abstracto)
├── LeafIterator<E>       — devuelve el nodo una vez
└── CompositeIterator<E>  — gestiona children + subiterator
```

**Detección de ciclos:** `AcycliclyIterable<E>` pasa un `Set<E>` (basado en `IdentityHashMap`) para registrar nodos visitados.

```java
public interface AcycliclyIterable<E> extends Iterable<E> {
    default ComponentIterator<E> iterator() {
        return iterator(Collections.newSetFromMap(new IdentityHashMap<>()));
    }
    ComponentIterator<E> iterator(Set<E> visited);
}
```

**Tracking de profundidad:** `getDepth()` en CompositeIterator devuelve `subiterator.getDepth() + 1`.

**`ConcurrentModificationException`:** lanzado si se modifica la colección durante la iteración.

---

## 7. Resumen de los 23 Patrones GoF

| Patrón | Categoría (GoF) | Intención (Metsker) | Propósito |
|--------|-----------------|---------------------|-----------|
| Factory Method | Creación/Clase | Construcción | Postergar decisión de clase a instanciar |
| Abstract Factory | Creación/Objeto | Construcción | Crear familia de objetos relacionados |
| Builder | Creación/Objeto | Construcción | Construir objeto paso a paso |
| Prototype | Creación/Objeto | Construcción | Clonar un objeto existente |
| Singleton | Creación/Objeto | Responsabilidad | Única instancia con acceso global |
| Class Adapter | Estructural/Clase | Interfaces | Adaptar interfaz via herencia |
| Object Adapter | Estructural/Objeto | Interfaces | Adaptar interfaz via delegación |
| Bridge | Estructural/Objeto | Interfaces | Desacoplar abstracción de implementación |
| Composite | Estructural/Objeto | Interfaces | Tratar uniformemente objetos y composiciones |
| Decorator | Estructural/Objeto | Extensiones | Añadir responsabilidades dinámicamente |
| Façade | Estructural/Objeto | Interfaces | Interfaz simple para subsistema complejo |
| Flyweight | Estructural/Objeto | Responsabilidad | Compartir objetos de grano fino |
| Proxy | Estructural/Objeto | Responsabilidad | Representante de otro objeto |
| Chain of Responsibility | Comportamiento/Objeto | Responsabilidad | Propagar petición hasta que alguien la atienda |
| Command | Comportamiento/Objeto | Operaciones | Encapsular invocación como objeto |
| Interpreter | Comportamiento/Clase | Operaciones | Gramática para un lenguaje |
| Iterator | Comportamiento/Objeto | Extensiones | Acceso secuencial a colección |
| Mediator | Comportamiento/Objeto | Responsabilidad | Coordinar interacciones entre objetos |
| Memento | Comportamiento/Objeto | Construcción | Capturar y restaurar estado interno |
| Observer | Comportamiento/Objeto | Responsabilidad | Notificar cambios a dependientes |
| State | Comportamiento/Objeto | Operaciones | Comportamiento según estado |
| Strategy | Comportamiento/Objeto | Operaciones | Encapsular algoritmos intercambiables |
| Template Method | Comportamiento/Clase | Operaciones | Esquema de algoritmo con pasos redefinibles |
| Visitor | Comportamiento/Objeto | Extensiones | Nuevas operaciones sin modificar jerarquía |

---

## 8. Preguntas de repaso

1. ¿Cuál es la diferencia entre Class Adapter y Object Adapter? ¿Cuándo usar cada uno?
2. ¿Por qué el método `accept` en el patrón Visitor no puede estar en la clase base aunque el código sea idéntico?
3. ¿Qué problemas tiene el Singleton lazy en un entorno multihilo? ¿Cómo se resuelven?
4. ¿Cuál es la diferencia entre Template Method y Strategy para encapsular algoritmos?
5. ¿Por qué el patrón Composite puede tener problemas con grafos acíclicos y cómo se detectan los ciclos?
6. ¿En qué se diferencia Abstract Factory de Factory Method?
7. ¿Por qué Observer/Observable se marcaron como @Deprecated en Java 9?
8. ¿Qué es el "Expression Problem" y cómo lo aborda el patrón Visitor?
9. Explica el doble despacho (double dispatch) en el patrón Visitor.
10. ¿Por qué un Builder permite que la clase a construir sea inmutable?
