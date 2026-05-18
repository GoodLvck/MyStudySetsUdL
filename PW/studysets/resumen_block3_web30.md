# Bloque 3 — Web 3.0
**Proyecto Web — Grado en Ingeniería Informática | Universitat de Lleida (UdL)**  
**Profesor: Roberto García**

---

## Tabla de Contenidos
1. [Introducción a Web 3.0](#1-introducción-a-web-30)
2. [RDFa](#2-rdfa)
3. [Schema.org](#3-schemaorg)
4. [JSON-LD](#4-json-ld)
5. [Facebook Open Graph](#5-facebook-open-graph)
6. [Integración de Datos Semánticos](#6-integración-de-datos-semánticos)
7. [La Nueva Web 3.0 — Descentralización](#7-la-nueva-web-30--descentralización)
8. [Aplicaciones Descentralizadas (DApps)](#8-aplicaciones-descentralizadas-dapps)

---

## 1. Introducción a Web 3.0

### Evolución de la Web

| Versión | Modelo de Interacción | Característica Principal |
|---------|----------------------|--------------------------|
| Web 1.0 | Personas → Aplicaciones | Páginas estáticas, consumo pasivo |
| Web 2.0 | Personas → Personas | Redes sociales, contenido colaborativo |
| Web 3.0 | Aplicaciones → Aplicaciones | Semántica formal, datos enlazados |

### Dimensiones de Web 3.0 (Nova Spivack, 2007)

Web 3.0 = **Información** + **Conexiones** + **Semántica**

También expresado como: **Social** + **Conexiones** + **Semántica**

> Sin semántica explícita: solo estadísticas y NLP (Natural Language Processing)

### Semántica Implícita vs Explícita

**Semántica implícita:**
- Desambiguación mediante contexto estadístico y NLP
- Ejemplo: página de desambiguación de Wikipedia para "Ronaldo"
- Limitación: el ordenador infiere el significado, no lo conoce con certeza

**Semántica explícita:**
- Datos estructurados con propiedades formales y URIs únicos
- Ejemplo: `http://dbpedia.org/resource/Ronaldo` — tiene tipo (Footballer), nacimiento, clubes, etc.
- Permite: **búsqueda semántica real** (semantic search)

### Web of Documents vs Web of Data

**Web of Documents:**
```
Invoice + Web API → Mashup (Web of Applications)
```

**Web of Data:**
```
Proveedor → Producto → Pedido → Cliente
(datos estructurados enlazados con URIs y propiedades formales)
```

### Formatos de Anotación Semántica de HTML

Los cuatro formatos principales para anotar HTML con semántica:

1. **RDFa** — RDF in Attributes
2. **Microformats** — clases CSS con semántica predefinida
3. **JSON-LD** — JSON for Linked Data
4. **Microdata** — atributos HTML5 nativos (itemscope, itemprop)

> Estadísticas de adopción: [w3techs.com/technologies/history_overview/structured_data/all](https://w3techs.com/technologies/history_overview/structured_data/all)

---

## 2. RDFa

### ¿Qué es RDFa?

**RDFa = RDF in Attributes** — Permite incluir anotaciones RDF directamente como atributos en:
- HTML (uso principal)
- SVG
- MathML

RDFa reutiliza atributos HTML existentes (`href`, `src`) y añade cinco nuevos atributos semánticos.

### Los 5 Atributos Nuevos de RDFa

| Atributo | Función | Ejemplo |
|----------|---------|---------|
| `vocab` | Define el vocabulario base | `vocab="http://xmlns.com/foaf/0.1/"` |
| `typeof` | Declara el tipo de entidad | `typeof="Person"` |
| `property` | Declara una propiedad | `property="name"` |
| `resource` | Define el identificador URI | `resource="#manu"` |
| `prefix` | Define prefijos de vocabularios | `prefix="ov: http://open.vocab.org/terms/"` |

### Pasos para Añadir Anotaciones RDFa

```html
<!-- Paso 1: Añadir vocabulario -->
<p vocab="http://xmlns.com/foaf/0.1/"

   <!-- Paso 4: Definir identificador del recurso -->
   resource="#manu"

   <!-- Paso 2: Añadir tipo -->
   typeof="Person">

  My name is
  <!-- Paso 3: Describir propiedades reutilizando HTML existente -->
  <span property="name">Manu Sporny</span>
  and you can call me at
  <span property="phone">1-800-555-0527</span>.
</p>
```

```html
<!-- Paso 5: Combinar múltiples vocabularios con prefijos -->
<div prefix="ov: http://open.vocab.org/terms/"
     resource="#alice" typeof="foaf:Person">
  <span property="foaf:name">Alice</span>
  <span property="ov:codeRepository">https://github.com/alice</span>
</div>
```

### Vocabulario FOAF

**FOAF = Friend Of A Friend**  
URL: `http://xmlns.com/foaf/0.1/`  
Propósito: Describir personas y sus relaciones sociales.

Propiedades principales: `foaf:name`, `foaf:phone`, `foaf:email`, `foaf:knows`, `foaf:homepage`, `foaf:img`

### Herramientas

- **Visualización RDFa:** [rdfa.info/play/](https://rdfa.info/play/)
- **Validación Google:** [search.google.com/test/rich-results](https://search.google.com/test/rich-results)

---

## 3. Schema.org

### ¿Qué es Schema.org?

Vocabulario para la anotación semántica de HTML promovido por los principales motores de búsqueda:
- **Google**, **Yahoo**, **Bing**, **Yandex**

URL del catálogo: [schema.org/docs/schemas.html](http://schema.org/docs/schemas.html)

### Formatos Soportados

Schema.org soporta **tres formatos de serialización**:
1. **RDFa** — Atributos inline en HTML
2. **Microdata** — Atributos HTML5 nativos inline
3. **JSON-LD** — Bloque `<script>` separado (recomendado por Google actualmente)

### Rich Results (Rich Snippets)

Las anotaciones Schema.org habilitan **Rich Results** en los motores de búsqueda: resultados enriquecidos con:
- Estrellas de valoración
- Precios y disponibilidad
- Fechas de eventos
- Datos de recetas, personas, organizaciones...

**Ejemplo para un producto:**

| Propiedad Schema.org | Tipo de dato |
|---------------------|-------------|
| `dc:description` | Descripción del producto |
| `media:image` | Imagen del producto |
| `product:manufacturer` | Fabricante |
| `product:salePrice` | Precio de venta |
| `product:currency` | Moneda |
| `vcal:date` | Fecha |
| `review:text` | Texto de reseña |
| `review:rating` | Valoración |

---

## 4. JSON-LD

### ¿Qué es JSON-LD?

**JSON-LD = JSON for Linked Data**  
Serialización de RDF en formato JSON.

**Cómo se incluye en HTML:**
```html
<script type="application/ld+json">
{
  "@context": "http://schema.org",
  "@type": "WebPage",
  "name": "Mi Página"
}
</script>
```

### Propiedades Especiales (`@`)

| Propiedad | Función | Ejemplo |
|-----------|---------|---------|
| `@context` | Define el vocabulario base | `"http://schema.org"` |
| `@type` | Especifica el tipo de entidad | `"EventReservation"` |
| `@id` | Identificador URI único del recurso | `"http://example.com/reservation/1"` |

### Ejemplo Completo — Reserva de Evento

```json
{
  "@context": "http://schema.org",
  "@type": "EventReservation",
  "reservationNumber": "E123456789",
  "reservationStatus": "http://schema.org/Confirmed",
  "underName": {
    "@type": "Person",
    "name": "John Smith"
  },
  "reservationFor": {
    "@type": "Event",
    "name": "Foo Fighters Concert",
    "startDate": "2027-03-06T19:30:00-08:00",
    "location": {
      "@type": "Place",
      "name": "AT&T Park"
    }
  }
}
```

### JSON-LD en Gmail (Google Mail Markup)

Google Gmail usa JSON-LD para **anotar emails** y mejorar su visualización:
- Destacar emails en la bandeja de entrada con iconos
- Mostrar previsualizaciones de reservas de vuelos/hoteles
- Añadir botones de acción directos (Check in, View booking...)
- Mostrar información estructurada de pedidos

### Ventajas de JSON-LD vs RDFa/Microdata

| Ventaja | Descripción |
|---------|-------------|
| Separación | No modifica el HTML visible |
| Familiaridad | Sintaxis JSON conocida por todos los desarrolladores |
| Dinamismo | Fácil de generar desde el servidor |
| Soporte Google | Recomendado para Rich Results |
| Mantenimiento | Fácil de actualizar independientemente |

---

## 5. Facebook Open Graph

### ¿Qué es Open Graph?

**Protocolo Open Graph** — desarrollado por Facebook para metadatos de compartición social.  
URL: [ogp.me](http://ogp.me/)

Alternativa a Schema.org pero orientado a **redes sociales** en lugar de motores de búsqueda.

### Metaetiquetas Principales

```html
<head>
  <meta property="og:title"       content="Título del Artículo" />
  <meta property="og:type"        content="article" />
  <meta property="og:url"         content="https://example.com/articulo" />
  <meta property="og:image"       content="https://example.com/imagen.jpg" />
  <meta property="og:description" content="Descripción breve del contenido" />
  <meta property="og:site_name"   content="Mi Blog" />
</head>
```

> Nota: Open Graph usa `property` (no `name`) + `content`.

### Schema.org vs Open Graph

| Aspecto | Schema.org | Open Graph |
|---------|-----------|-----------|
| Objetivo | Motores de búsqueda | Redes sociales |
| Promotor | Google, Bing, Yahoo, Yandex | Facebook |
| Resultado | Rich Results en buscadores | Previsualizaciones al compartir |
| Formatos | RDFa, Microdata, JSON-LD | Metaetiquetas HTML |

---

## 6. Integración de Datos Semánticos

### OpenPHACTS

**OpenPHACTS = Open Pharmacological Data**
- Plataforma de integración de datos biomédicos y farmacológicos
- **3 billion (3.000 millones) de triples RDF**
- Integra: ChEMBL, UniProt, DrugBank y otros datasets
- Ejemplo de Knowledge Graph industrial a escala real

### FIBO

**FIBO = Financial Industry Business Ontology**
- Ontología semántica para el sector financiero y bancario
- Desarrollada por EDM Council y Object Management Group
- Permite interoperabilidad semántica entre instituciones financieras
- Define formalmente conceptos: préstamo, contrato, activo financiero...

### Knowledge Graphs Industriales

Las grandes empresas tecnológicas tienen grafos de conocimiento propios:
- **Google Knowledge Graph** — búsqueda semántica, asistente de voz
- **Amazon Product Graph** — recomendaciones, búsqueda de productos
- **Facebook Social Graph** — relaciones entre usuarios y entidades

---

## 7. La Nueva Web 3.0 — Descentralización

### Conceptos Clave

| Concepto | Definición |
|----------|-----------|
| **Centralización** | Control por una única entidad |
| **Descentralización** | Control compartido entre entidades independientes; nadie tiene el control absoluto |
| **Distribución** | Componentes en ubicaciones físicas separadas |

> **Bitcoin** es **descentralizado** (ninguna entidad lo controla) **Y distribuido** (nodos en múltiples ubicaciones del mundo).

### Blockchain

Una **blockchain** es una cadena de bloques donde:
- Cada bloque contiene: **datos** (transacciones) + **hash del bloque anterior**
- El hash del bloque anterior crea la "cadena" y garantiza la integridad
- Alterar cualquier bloque invalida todos los bloques posteriores
- La cadena es mantenida por múltiples nodos independientes

```
[Bloque 1: datos + hash(génesis)] → [Bloque 2: datos + hash(B1)] → [Bloque N: datos + hash(B(N-1))]
```

### Bitcoin Halving

El **halving** de Bitcoin es la reducción periódica a la mitad de la recompensa por bloque minado:
- Ocurre cada ~4 años (cada 210.000 bloques)
- Progresión: 50 → 25 → 12.5 → **6.25** → 3.125 BTC/bloque
- En el período del curso: **6.25 BTC por bloque**
- Máximo de Bitcoin: 21 millones (diseño deflacionario)

### Ethereum y Smart Contracts

**Ethereum** es una blockchain que soporta **smart contracts** (contratos inteligentes):
- Programas que se ejecutan automáticamente cuando se cumplen condiciones
- Inmutables una vez desplegados
- Ejecutados por todos los nodos de la red
- Sin intermediarios
- Base de las DApps

---

## 8. Aplicaciones Descentralizadas (DApps)

### ¿Qué es una DApp?

**DApp = Decentralized Application**

| Aspecto | App Web Tradicional | DApp |
|---------|--------------------|----|
| Backend | Servidores centralizados | Smart contracts en blockchain |
| Datos | Bases de datos privadas | Blockchain / IPFS |
| Control | Una empresa | Nadie (o la comunidad) |
| Censura | Posible | Muy difícil (código inmutable) |
| Transparencia | Opaca | Código y transacciones públicos |

Directorio de DApps: **DappRadar**

### CopyrightLY — Blockchain + Web Semántica

CopyrightLY es una DApp para gestión de derechos de autor que combina **blockchain** y **Web Semántica**:

**Componentes blockchain:**
1. Reclamar autoría de contenidos con evidencias (ej: vídeo de YouTube)
2. **Token-curated claims** usando staking de tokens **CLY** con **bonding curve**
3. Licenciar contenido como **NFTs**

**Componentes semánticos:**
- Metadatos semánticos en los NFTs de licencia
- Datos estructurados interoperables

**Marketplaces para los NFTs:** OpenSea, Rarible

#### Mecanismo Token-Curated Registry (TCR)

```
Creador → Staking CLY (reclamación)
                ↓
    Período de impugnación
                ↓
    Impugnador → Staking CLY (en contra)
                ↓
    Bonding curve evalúa soporte
                ↓
    TCR: Aceptar / Rechazar reclamación
                ↓
    Ganador recibe tokens del perdedor
```

#### Flujo de Licenciamiento

```
Contenido → Reclamación → Validación TCR → NFT de Licencia → OpenSea/Rarible
```

### Gitcoin

**Gitcoin** es una plataforma de financiación de desarrollo open source mediante blockchain:
- **Bounties:** recompensas cripto para resolver issues de GitHub
- **Grants:** donaciones colectivas con financiación cuadrática
- **Hackathons:** competiciones con premios en cripto
- Incentivos económicos alineados con el desarrollo de software libre

---

## Resumen de URLs Clave

| Recurso | URL |
|---------|-----|
| FOAF Vocabulary | http://xmlns.com/foaf/0.1/ |
| Schema.org Schemas | http://schema.org/docs/schemas.html |
| Open Graph Protocol | http://ogp.me/ |
| DBpedia (ejemplo Ronaldo) | http://dbpedia.org/resource/Ronaldo |
| RDFa Playground | https://rdfa.info/play/ |
| Google Rich Results Test | https://search.google.com/test/rich-results |
| Structured Data Adoption Stats | https://w3techs.com/technologies/history_overview/structured_data/all |
| DappRadar | https://dappradar.com/ |

---

## Preguntas de Repaso Rápido

1. ¿Cuáles son las 3 dimensiones de Web 3.0 según Nova Spivack?
2. ¿Qué diferencia hay entre semántica implícita y explícita? Da un ejemplo de cada una.
3. ¿Cuáles son los 5 atributos nuevos que introduce RDFa y para qué sirve cada uno?
4. ¿Qué tipo MIME se usa para incluir JSON-LD en HTML?
5. ¿Qué propiedades @ son las más importantes en JSON-LD?
6. ¿Qué motores de búsqueda promueven Schema.org?
7. ¿Qué diferencia hay entre descentralización y distribución? ¿Por qué Bitcoin es ambas?
8. ¿Qué hace que la blockchain sea prácticamente inalterable?
9. ¿Qué combina CopyrightLY y cómo funciona su sistema de token-curation?
10. ¿Qué es Open Graph y para qué se usa frente a Schema.org?
