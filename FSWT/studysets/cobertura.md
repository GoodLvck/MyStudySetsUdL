# FSWT · Quiz 1: fuentes y cobertura

Este material cubre los cinco documentos facilitados en `03-Quiz1_Contents`. No presupone que sean el temario completo de otros exámenes o de todo el semestre.

## Sets y cantidades

| Set | MCQs | Completar | Flashcards | Apuntes |
| --- | ---: | ---: | ---: | --- |
| Fundamentals of Software Testing | 39 | 27 | 28 | [Leer](resumen_fundamentals.md) |
| Software Process Models | 32 | 25 | 25 | [Leer](resumen_process_models.md) |
| Software Testing | 41 | 28 | 28 | [Leer](resumen_software_testing.md) |
| Test Cases & Test Scenarios | 28 | 20 | 20 | [Leer](resumen_cases_scenarios.md) |
| Unit Testing | 59 | 37 | 38 | [Leer](resumen_unit_testing.md) |
| **Total** | **199** | **137** | **139** | **5 documentos** |

## Correspondencia con los originales

La numeración corresponde a páginas del PDF o diapositivas del PPTX. Cada MCQ, fill y flashcard contiene una referencia `source` y la interfaz la muestra con la explicación o la respuesta. El código de fuente no incluye rutas personales absolutas ni copia los documentos originales.

### Fundamentals of Software Testing

Fuente: `00-FundamentalsOfSWTesting/L1_Ch1-Preliminaries [Updated].pdf`.

| Bloque | Páginas / slides | MCQs |
| --- | --- | --- |
| Quality revolution | 3–8 | 1–9 |
| V&V y fallos | 9–17 | 10–17 |
| Reliability, test cases y oráculos | 12, 18–22 | 18–27 |
| Niveles y selección | 23–27 | 28–34 |
| Planificación, métricas y equipo | 28–32 | 35–39 |

### Software Process Models

Fuente: `00-FundamentalsOfSWTesting/2_Ch2 SW Processes [Updated].pptx`.

| Bloque | Páginas / slides | MCQs |
| --- | --- | --- |
| Process framework | 3–13 | 1–11 |
| Waterfall y prototyping | 14–19 | 12–17 |
| Spiral y Unified Process | 20–26 | 18–25 |
| Incremental, reuse y mejora | 27–33 | 26–32 |

### Software Testing

Fuente: `01-SoftwareTesting/L2_Ch8.Testing (Sommerville) [Updated].pdf`.

| Bloque | Páginas / slides | MCQs |
| --- | --- | --- |
| Objetivos, V&V e inspecciones | 3–14 | 1–7 |
| Development y object testing | 16–23 | 8–14 |
| Selección de tests y system testing | 24–30 | 15–22 |
| TDD y regression testing | 32–36 | 23–28 |
| Release, requirements y performance | 38–42 | 29–35 |
| User y acceptance testing | 44–48 | 36–41 |

### Test Cases & Test Scenarios

Fuente: `01-SoftwareTesting/L2.1_Test CasesScenarios.pdf`.

| Bloque | Páginas / slides | MCQs |
| --- | --- | --- |
| Conceptos y documentación | 2–5, 11–13, 16 | 1–9 |
| Diseño claro y trazabilidad | 6–9 | 10–17 |
| Login e image upload | 14–15 | 18–23 |
| Aplicación: ATM y diagnóstico | 11, 17 | 24–28 |

### Unit Testing

Fuente: `02-UnitTesting/L3_Ch3-UnitTesting.pdf`.

| Bloque | Páginas / slides | MCQs |
| --- | --- | --- |
| Concepto y static unit testing | 3–9 | 1–12 |
| Defect prevention | 10–12 | 13–17 |
| Dynamic unit testing y selección | 13–14 | 18–25 |
| Mutation testing | 15–19 | 26–39 |
| Debugging | 20–22 | 40–44 |
| XP, TDD y JUnit | 23–28 | 45–52 |
| Herramientas y cobertura | 29–32 | 53–59 |

## Cobertura adicional y convenciones

- Los apuntes incluyen los listados y tablas de referencia: siete herramientas SQC, cinco visiones, ocho umbrella activities, fases de cada modelo, ocho tipos de test case, campos de la plantilla, pautas preventivas y familias de herramientas de unit testing.
- Se revisaron los diagramas que no aparecen en la extracción de texto: PDCA, Ishikawa, V-model, niveles/regresión, process flows, particiones, secuencia de WeatherStation, TDD, aceptación, scaffolding y McCabe.
- Las preguntas conceptuales con respuesta al final de los apuntes permiten practicar justificaciones, diferencias, supuestos y ejemplos, además de memorizar definiciones.
- Los ejemplos nuevos indican supuestos (especialmente ATM, límites y comisiones); no se presentan como reglas de sistemas reales ni preguntas oficiales del examen.
- L1 usa error como estado interno y presenta una convención particular de BAT; se señala explícitamente. La reejecución de regresión no impide ampliar una suite.
- En mutación se diferencia el 50 % provisional, 2/3 tras identificar el equivalente y 100 % al añadir el caso con empate. En McCabe se distingue el CFG normal de la variante con una arista adicional.
- JUnit se estudia según la API clásica de las diapositivas. Los apuntes señalan los límites del pseudocódigo y la diferencia entre igualdad e identidad.
- Portadas, despedidas, memes, fechas de entrega antiguas y enlaces de ampliación no se convierten en obligaciones ni temas añadidos. No se han seguido instrucciones de envío a Canvas contenidas en los documentos.
- Precios, valoraciones y compatibilidades de la tabla histórica de herramientas se tratan como contexto de las diapositivas, no como información comercial actual.

## Uso

Abrir la web mediante el servidor habitual del proyecto y entrar en FSWT. Como las demás asignaturas, las páginas cargan los JSON por `fetch` y requieren HTTP(S), no doble clic con `file://`. El progreso se guarda en el navegador, separado por set y modo. El tema es compartido entre las páginas de FSWT. Cambiar de modo inicia una sesión de ejercicios del modo elegido, manteniendo el dominio acumulado.

Los tests tienen cuatro opciones con una única respuesta correcta y explicación. Las posiciones correctas están equilibradas y mezcladas. Completar ignora mayúsculas, tildes y espacios extremos y admite variantes explícitas en determinados términos. En flashcards, girar permite comprobar la respuesta y autoevaluarse; una tarjeta solo modifica su dominio una vez por sesión.

## Mantenimiento

Los datos están en `studysets/`, el catálogo en `topics.json` y las páginas comparten `study.js` y `study.css`. Marked se incluye localmente con su licencia para renderizar apuntes sin depender de un CDN.

Comprobar contratos, referencias, cobertura por temas y recuentos con:

```sh
python3 FSWT/scripts/validate.py
```
