# Block 2: Web 2.0

**Asignatura:** Proyecto Web — Grado en Ingeniería Informática  
**Profesor:** Roberto García  
**Universidad:** Universitat de Lleida (UdL)

---

## Tema 1: Historia y Fundamentos Web

### Línea del tiempo

| Año | Evento |
|-----|--------|
| 1945 | Vannevar Bush propone el **Memex** (memory extender) en "As We May Think" — dispositivo conceptual para almacenar y acceder a información en microfilms |
| 1989 | Tim Berners-Lee presenta su propuesta al CERN para un sistema de información distribuido |
| 1990 | Primer sitio web en http://info.cern.ch |

### Los tres fundamentos de la Web (Berners-Lee)

1. **Identificadores (URIs)** — nombres (URNs) y localizadores (URLs)
2. **Protocolo de comunicación** — HTTP
3. **Lenguaje de representación** — HTML

> **URIs = URNs + URLs**
> - URN: nombre permanente (no indica dónde está)
> - URL: localización + protocolo (https://ejemplo.com/recurso)

### Web 1.0 vs Web 2.0

**Definición Web 2.0:** Segunda generación de servicios web que permite a los usuarios **colaborar y compartir información en línea** — la Web como plataforma que se beneficia del efecto de red.

| Web 1.0 | Web 2.0 |
|---------|---------|
| DoubleClick | AdSense |
| Ofoto | Flickr |
| Britannica Online | Wikipedia |
| Páginas personales | Blogs |
| Directorios / Taxonomía | Tagging / **Folksonomy** |
| Publicar (Publishing) | Participar (Participation) |

### Patrones de Web 2.0

| Patrón | Descripción |
|--------|-------------|
| **Long Tail** | Los nichos acumulados superan al mercado masivo (Amazon, Spotify) |
| **Web APIs / Mashups** | Combinar datos de terceros para crear nuevos servicios |
| **Cooperación** | Sindicación y reutilización de contenido |
| **Creative Commons** | Licencias que permiten uso y redistribución |
| **Usuarios añaden valor (explícito)** | Usuario consciente: escribir en Wikipedia, etiquetar en Flickr |
| **Efecto de red / valor implícito** | Datos agregados de uso: recomendaciones de Amazon |

> **Folksonomy** (folk + taxonomy): clasificación colaborativa y emergente mediante etiquetas asignadas por los propios usuarios.

---

## Tema 2: AJAX

### ¿Qué es AJAX?

**AJAX = Asynchronous JavaScript and XML**

Combinación de: HTML + JavaScript + CSS + Navegador + Servidor Web

**Para qué sirve:**
- Actualizar partes de una página HTML sin recargar completamente
- Enviar formularios en segundo plano
- Navegación interactiva sin interrupciones

### Modelo Síncrono (Web 1.0) vs Asíncrono (Web 2.0)

**Jesse James Garrett** describió la diferencia:

```
Web 1.0 (Síncrono):
Usuario → acción → petición → [ESPERA] → página recarga completa → ciclo nuevo

Web 2.0 (Asíncrono con AJAX):
Usuario → acción → motor AJAX → petición en 2º plano
         ↕ (sigue interactuando)
                              ← respuesta → actualiza solo el DOM afectado
```

### Los 4 pasos de una petición AJAX

#### Paso 1: Crear instancia XMLHttpRequest

```javascript
// IE7+ y navegadores modernos:
var xhr = new XMLHttpRequest();

// IE5/IE6 (ActiveXObject):
var xhr = new ActiveXObject("Microsoft.XMLHTTP");

// Compatibilidad completa:
var xhr;
if (window.XMLHttpRequest) {
    xhr = new XMLHttpRequest();
} else {
    xhr = new ActiveXObject("Microsoft.XMLHTTP");
}
```

#### Paso 2: Abrir la conexión

```javascript
xhr.open(method, url, async, user, password);
// Ejemplo:
xhr.open("GET", "products/34/", true);
xhr.open("POST", "/tasks/2024-06-09", true);
```

#### Paso 3: Definir callback `onreadystatechange`

```javascript
xhr.onreadystatechange = function() {
    if (xhr.readyState == 4 && xhr.status == 200) {
        // Procesar la respuesta
        document.getElementById("div").innerHTML = xhr.responseText;
    }
};
```

#### Paso 4: Enviar la petición

```javascript
xhr.send();              // GET — sin cuerpo
xhr.send(datos);         // POST/PUT/DELETE — con datos
```

### Valores de readyState

| Valor | Significado |
|-------|-------------|
| **0** | No inicializado — `open()` no llamado aún |
| **1** | Conexión establecida — `open()` llamado |
| **2** | Petición recibida — cabeceras disponibles |
| **3** | Procesando — descargando respuesta |
| **4** | **Finalizado/Listo** — respuesta completa disponible |

> La condición estándar es: `readyState == 4 && status == 200`

### Cabeceras HTTP en AJAX

```javascript
// Enviar datos de formulario:
xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");

// Enviar JSON:
xhr.setRequestHeader("Content-Type", "application/json");

// Indicar formato aceptado en respuesta:
xhr.setRequestHeader("Accept", "application/json");
xhr.setRequestHeader("Accept", "text/xml");
```

### Propiedades de respuesta

| Propiedad | Descripción |
|-----------|-------------|
| `xhr.responseText` | Respuesta como texto/string |
| `xhr.responseXML` | Respuesta como documento XML |
| `xhr.status` | Código HTTP (200, 404, 500...) |
| `xhr.statusText` | Texto del estado ('OK', 'Not Found') |
| `xhr.readyState` | Estado de la petición (0-4) |

### Ejemplo completo: POST con AJAX nativo

```javascript
var xhr = new XMLHttpRequest();
xhr.open("POST", "/tasks/2024-06-09", true);
xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");

xhr.onreadystatechange = function() {
    if (xhr.readyState == 4 && xhr.status == 201) {
        var location = xhr.getResponseHeader("Location");
        console.log("Tarea creada en: " + location);
    }
};

xhr.send("title=buy+some+fruit&state=pending");
```

### jQuery AJAX

```javascript
// Cargar contenido directamente en elemento:
$("#resultado").load("datos/fragmento.html");

// GET simple:
$.get("api/products", function(data) {
    console.log(data);
});

// POST con datos de formulario:
$.post("api/tasks", $("#miFormulario").serialize(), function(resp) {
    alert("Creado: " + resp.id);
});

// Codificar parámetros en URL:
var url = "search?q=" + encodeURIComponent("hello world & más");
```

---

## Tema 3: Resource Oriented Architecture (REST/ROA)

### Definiciones clave

- **REST** (Representational State Transfer): estilo arquitectónico para sistemas distribuidos — conjunto de **patrones** que restringen los roles y relaciones entre componentes.
- **RESTful**: sistema cuya arquitectura sigue **todos** los patrones REST.
- **ROA** (Resource-Oriented Architecture): implementación RESTful usando tecnologías Web (HTTP, URI, XML/JSON).

> **Criterios ROA:**
> - Toda la info de routing → en la **URI** (addressability)
> - Toda la info del método → en la **operación HTTP** (uniform interface)

### Los 4 componentes de ROA

| Componente | Descripción |
|------------|-------------|
| **Resources** (Recursos) | Cualquier cosa relevante para referenciar; puede ser física o abstracta |
| **URIs** | Nombre y dirección del recurso; deben ser descriptivas y predecibles |
| **Representations** | Bytes en formato específico; el servidor envía representaciones, no recursos |
| **Links** | Conectan los recursos entre sí (habilitan HATEOAS) |

### Los 4 principios de ROA

#### 1. Addressability (Direccionabilidad)

La aplicación expone sus datos como **recursos**, con una URI para cada "pieza" de información.

- Los recursos son **bookmarkable**, **cacheables**, **indexables** y **linkables**
- Sin addressability: no se puede enlazar a ni cachear la información

#### 2. Statelessness (Sin Estado)

Cada petición HTTP es **aislada** — contiene toda la información necesaria.

**Ventajas:**
- No hay timeouts de sesión
- Facilita el balanceo de carga
- Facilita el cacheo
- Mayor escalabilidad

**Emulación de sesiones:** session ID en cookie o en la URI

#### 3. Uniform Interface (Interfaz Uniforme)

Los métodos HTTP tienen semántica definida:

| Método | Seguro | Idempotente | Uso principal |
|--------|--------|-------------|---------------|
| **GET** | ✓ Sí | ✓ Sí | Obtener recurso |
| **HEAD** | ✓ Sí | ✓ Sí | Obtener solo cabeceras |
| **PUT** | ✗ No | ✓ Sí | Reemplazar recurso completo |
| **DELETE** | ✗ No | ✓ Sí | Eliminar recurso |
| **POST** | ✗ No | ✗ No | Crear recurso subordinado |
| **PATCH** | ✗ No | ✗ No | Actualización parcial |

> **Seguro** = no modifica el estado del servidor  
> **Idempotente** = el mismo efecto aplicado 1 o N veces

**Usos de POST:**
1. Crear recurso subordinado (el servidor asigna la URI → responde **201 Created + Location**)
2. Añadir datos a un recurso existente
3. Overloaded POST (anti-patrón RPC — viola la interfaz uniforme)

**Error más común:** exponer operaciones no seguras (modificadoras) mediante GET.

#### 4. Connectedness / HATEOAS

**HATEOAS** = Hypermedia As The Engine of Application State

- El servidor guía al cliente con **enlaces** y **formularios** en sus respuestas
- El cliente mantiene el estado de la aplicación **siguiendo enlaces**
- El cliente no necesita conocer las URLs de antemano
- Reduce el acoplamiento cliente-servidor

### Recursos y URIs

- Las URIs son el **nombre y dirección** del recurso
- Deben ser **descriptivas y predecibles**
- No debe haber recursos duplicados
- Un recurso puede tener **múltiples URIs** → usar **HTTP 303 "See Other"** para redirigir a la URI canónica
- Las representaciones se negocian con cabeceras `Accept` y `Accept-Language`

### REST vs REST-RPC

| REST (RESTful) | REST-RPC (Anti-patrón) |
|----------------|------------------------|
| GET /dogs | GET /getDog?id=5 |
| POST /dogs | POST /saveDog |
| DELETE /dogs/brian | POST /deleteDog?id=brian |
| PUT /dogs/brian | POST /updateDog |
| Sustantivos en URLs | **Verbos en URLs** |
| Operación en método HTTP | Operación en URL o cuerpo |
| Cumple interfaz uniforme | **Viola** interfaz uniforme |

---

## Tema 4: Diseño de APIs RESTful

### Metodología: La Tabla de Diseño

Para diseñar una API RESTful se define una **tabla**:
- **Filas** = tipos de recursos (con URL de ejemplo)
- **Columnas** = 4 métodos HTTP (GET, POST, PUT/PATCH, DELETE)

Cada celda define la operación o indica N/A.

### Ejemplo: API de Tareas

| URL ejemplo | GET | POST | PUT/PATCH | DELETE |
|-------------|-----|------|-----------|--------|
| `/tasks` | Listar todas las tareas | N/A | N/A | N/A |
| `/tasks/2024-06-09` | Listar tareas del día | Crear nueva tarea | N/A | N/A |
| `/tasks/2024-06-09/1` | Ver detalles de la tarea | N/A | Actualizar/cambiar estado | Eliminar tarea |

### Representación JSON de una tarea

```json
{
  "id": "/tasks/2024-06-09/1",
  "title": "buy some fruit",
  "state": "pending"
}
```

> El campo `id` es la **URI del propio recurso** — sigue el principio HATEOAS.

### Flujo de creación con POST

```
Cliente → POST /tasks/2024-06-09
          Body: {"title": "buy some fruit", "state": "pending"}

Servidor ← 201 Created
           Location: /tasks/2024-06-09/3
           Body: {"id": "/tasks/2024-06-09/3", "title": "buy some fruit", "state": "pending"}
```

### Estructura alternativa: `/dates/{date}/tasks`

```
GET /dates                      → listar fechas
GET /dates/2024-06-09           → info del día
GET /dates/2024-06-09/tasks     → tareas del día
POST /dates/2024-06-09/tasks    → crear tarea
```

**Cuándo usar cada estructura:**
- `/tasks/{date}` — cuando el foco son las tareas y la fecha es solo un filtro
- `/dates/{date}/tasks` — cuando las fechas son recursos de primer nivel con propiedades propias

### Reglas de diseño RESTful

1. **Sustantivos, no verbos** en las URLs
2. **Plural** para colecciones (`/dogs`, `/tasks`)
3. La **operación** siempre en el método HTTP, nunca en la URL
4. Jerarquía de recursos con `/` (recurso padre/hijo)
5. Responder con los **códigos HTTP correctos** (200, 201, 204, 303, 404...)
6. Incluir **enlaces** en las respuestas (HATEOAS)
7. URIs **descriptivas y predecibles**

### Códigos HTTP más importantes en REST

| Código | Nombre | Cuándo se usa |
|--------|--------|---------------|
| **200** | OK | Éxito general (GET, PUT, DELETE con cuerpo) |
| **201** | Created | POST exitoso — siempre con `Location` header |
| **204** | No Content | Éxito sin cuerpo (DELETE sin cuerpo) |
| **303** | See Other | Redirección a URI canónica (ROA) |
| **400** | Bad Request | Petición malformada |
| **404** | Not Found | Recurso no existe |
| **405** | Method Not Allowed | Método no permitido en ese recurso |
| **500** | Internal Server Error | Error del servidor |

---

## Resumen de Conceptos Clave

### Mapa mental de la asignatura

```
Web 2.0
├── Historia
│   ├── 1945: Memex (Vannevar Bush)
│   ├── 1989: Propuesta WWW (Berners-Lee)
│   └── 1990: Primer website (info.cern.ch)
├── Fundamentos Web
│   ├── URIs (URNs + URLs)
│   ├── HTTP
│   └── HTML
├── Web 2.0 Patrones
│   ├── Long Tail
│   ├── Mashups/Web APIs
│   ├── Folksonomy
│   └── Efectos de red
├── AJAX
│   ├── XMLHttpRequest (IE5+)
│   ├── 4 pasos: crear → open → onreadystatechange → send
│   ├── readyState: 0→1→2→3→4
│   └── jQuery: .load(), $.get(), $.post(), .serialize()
├── REST/ROA
│   ├── 4 componentes: Resources, URIs, Representations, Links
│   └── 4 principios: Addressability, Statelessness, Uniform Interface, HATEOAS
└── APIs RESTful
    ├── Tabla de diseño (filas=recursos, columnas=métodos)
    ├── Sustantivos en URLs (no verbos)
    ├── Métodos HTTP con semántica (GET/POST/PUT/DELETE)
    └── 201 Created + Location al crear recursos
```

### Comparativa Final: REST vs AJAX

| Aspecto | AJAX | REST/ROA |
|---------|------|----------|
| **Capa** | Cliente (navegador) | Arquitectura del servidor/API |
| **Qué define** | Cómo hacer peticiones asíncronas | Cómo estructurar recursos y URLs |
| **Tecnología** | XMLHttpRequest, JavaScript | HTTP, URIs, JSON/XML |
| **Problema que resuelve** | Recarga de página completa | Diseño inconsistente de APIs |
| **Principio clave** | Asincronía | Uniform Interface + Addressability |

AJAX y REST **se complementan**: AJAX es cómo el cliente hace peticiones; REST es cómo el servidor expone sus recursos.
