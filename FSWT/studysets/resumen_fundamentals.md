# Fundamentals of Software Testing

## Alcance y fuente

Preparación del **Quiz 1: MCQs and conceptual** a partir de `L1_Ch1-Preliminaries [Updated].pdf` (34 páginas). Los ejemplos de práctica son nuevos y se basan en sus conceptos. Las referencias indican páginas del PDF, no numeración de un libro. El set cubre el contenido docente, incluidos los diagramas; las portadas, memes, despedida y enlaces de ampliación no constituyen temas adicionales.

## 1. Calidad y mejora continua (pp. 3–8)

El contexto histórico enlaza a **Deming** con statistical quality control (SQC) y la difusión de **PDCA**, a **Juran** con total quality control (TQC), a **Ishikawa** con causa y efecto y a **Taiichi Ohno** con lean. Lean busca eliminar desperdicio: recursos que no aportan valor y tiempo innecesario entre pedido y cobro.

**Siete herramientas SQC:** Pareto analysis, trend chart, flow chart, histogram, scatter diagram, control chart y cause-and-effect diagram. En la figura de Ishikawa del material las 4 M son **Materials, Machines, Methods, Measurements**.

**PDCA:** Plan establece objetivos y cómo medirlos; Do ejecuta y mide; Check compara mediciones con objetivos; Act decide mejoras para la siguiente vuelta. Una revisión de resultados pertenece a Check; aplicar cambios derivados de ella pertenece a Act.

**TQC:** calidad antes que beneficio inmediato, cliente antes que productor, decisiones con hechos y datos, participación respetuosa y coordinación entre funciones. La comparación docente de premios asocia Baldrige (EE. UU., introducido en 1987) a calidad definida por el cliente y Deming a conformidad con especificaciones del productor. El documental de NBC de 1980 aparece como contexto de difusión, no como técnica de testing.

| Visión | Pregunta que la identifica |
| --- | --- |
| Transcendental | ¿Se percibe excelencia aunque sea difícil medirla? |
| User's | ¿Satisface necesidades y resulta usable? |
| Manufacturing | ¿Cumple especificaciones con un proceso controlado? |
| Product | ¿Qué propiedades, funciones y prestaciones tiene? |
| Value-based | ¿Compensan sus beneficios el coste? |

Un **quality factor** expresa una característica (correctness, reliability, efficiency, testability). Un **quality criterion** conecta esa característica con atributos del desarrollo, por ejemplo modularity. L1 cita ISO 9126, CMM, TPI y TMM como modelos/referencias de calidad; estudiar esa lista como contenido del curso, sin interpretarla como una relación de estándares vigentes equivalentes.

## 2. Testing, verification y validation (pp. 9–17)

- **Static analysis:** examinar código o artefactos sin ejecutarlos: review, inspection, análisis de algoritmos.
- **Dynamic analysis:** ejecutar con entradas y observar comportamientos representativos.
- Son complementarios. Revisar todo el texto del código no equivale a demostrar formalmente todos sus comportamientos.
- **Verification:** building the product right; conformidad con lo especificado para la fase.
- **Validation:** building the right product; adecuación al uso y necesidades reales.

Ejemplo conceptual: un buscador cumple una especificación que solo permite buscar por ID, pero sus usuarios necesitan buscar por nombre. Puede cumplir verification y fallar validation.

Según la terminología de **L1**:

| Concepto | Significado | Ejemplo de saldo |
| --- | --- | --- |
| Fault / defect / bug | Causa del estado incorrecto | Se suma una retirada en vez de restarla |
| Error | Estado interno incorrecto | Variable de saldo con valor incorrecto |
| Failure | Desviación externa observable | Se muestra o entrega un importe incorrecto |

Un fault puede quedar latente; un error puede corregirse o no propagarse a una salida. **No todo fault produce un failure en cada ejecución.** En otros materiales “error” puede designar un error humano: en preguntas basadas en L1 se usa la definición de estado.

Testing aporta evidencia de conformidad, rendimiento y calidad, y descubre defectos. Sus objetivos incluyen mostrar funcionamiento esperado, encontrar funcionamiento incorrecto, reducir riesgo de fallos y controlar coste. Pasar los tests **no demuestra ausencia de todos los faults**. Desarrolladores e independientes aportan conocimientos y perspectivas diferentes; la caricatura del material no convierte esas diferencias en reglas sobre cada persona.

La figura de p. 17 menciona ejecución manual/automatizada, white/black/grey box, funcional/no funcional y niveles. Son dimensiones que pueden combinarse, no categorías excluyentes. También enumera integración incremental (top-down, bottom-up) y no incremental, y pruebas de performance, usability y compatibility; dentro de performance aparecen load, stress, scalability y stability. Para interpretar sus etiquetas: **grey-box** usa conocimiento parcial del interior; funcional comprueba qué hace el sistema y no funcional cómo se comporta en aspectos de calidad. **Load** observa comportamiento bajo carga definida, **stress** excede condiciones normales, **scalability** considera cómo cambia la capacidad al crecer demanda/recursos y **stability** la continuidad del comportamiento a lo largo del tiempo. **Usability** trata de facilidad de uso y **compatibility** de funcionamiento en entornos o con otros sistemas. En integración, **top-down** avanza de módulos superiores a inferiores, **bottom-up** en sentido inverso y la no incremental integra el conjunto sin esa progresión por etapas. Estas glosas aclaran las etiquetas de la figura; el documento no desarrolla procedimientos detallados de cada técnica.

## 3. Reliability y selección representativa (p. 12)

**Reliability = probabilidad de operación sin fallos durante un tiempo determinado y en un entorno determinado.** Para estimarla mediante pruebas aleatorias, los datos deben reflejar la distribución de uso futuro: el **operational profile**. Aleatorio no significa necesariamente uniforme. Si el 90 % del uso real consiste en consultar y el 10 % en modificar, probar ambas operaciones al 50 % no representa por sí solo ese perfil.

Una suite orientada a descubrir defectos puede concentrarse en casos raros y límites. Una estimación de fiabilidad operacional necesita además representatividad del uso. El porcentaje de tests aprobados no es automáticamente una probabilidad de fiabilidad.

## 4. Test case, estado y oráculo (pp. 18–19)

Un caso básico es **〈input, expected outcome〉**. Para sistemas stateless basta con la entrada actual y las condiciones pertinentes. Para sistemas state-oriented importan el estado inicial y la secuencia de interacciones.

ATM del material: saldo inicial 500; retirar 200; comprobar efectivo 200 y saldo 300. El expected outcome puede ser un valor, un cambio de estado o una secuencia de valores que se interpreten juntos. “No hubo excepción” es una observación insuficiente si no se comprueba la lógica.

Un **test oracle** proporciona o verifica la respuesta correcta y permite contrastar expected con actual. Copiar la salida de la implementación como expectativa de esa misma ejecución crea una comprobación circular.

## 5. Límites y actividades de testing (pp. 20–22)

Exhaustive/complete testing aspiraría a que no queden faults ocultos. Normalmente es inviable por entradas válidas e inválidas, combinaciones, estados, diseño y entornos. Seleccionar D1 dentro de D puede ejercitar solo P1 del programa. La tarea central consiste en elegir un subconjunto útil bajo recursos limitados.

Secuencia: **objetivo → entradas → resultado esperado → entorno → ejecución → análisis**. Separar preparar un test de ejecutarlo y de interpretar su resultado.

## 6. Niveles, V-model y regresión (pp. 23–25)

| Nivel | Objeto principal | Ejemplo |
| --- | --- | --- |
| Unit | Unidad aislada | Cálculo de saldo |
| Integration | Interfaces entre unidades | Servicio envía correctamente el importe a persistencia |
| System | Sistema integrado, funcionalidad y propiedades globales | Retirada completa bajo carga |
| Acceptance | Criterios y expectativas del cliente | Cliente decide si acepta el producto |

El **V-model dibujado en L1** relaciona requirements con acceptance, high-level design con system, detailed design con integration y coding con unit. Usar estas correspondencias si se pregunta por esa figura.

**UAT** se presenta como comprobación de criterios contractuales. **BAT**, en la convención específica de estas diapositivas, es la evaluación de la organización del proveedor orientada a superar UAT. No extrapolar esta descripción a todos los usos de las siglas fuera del curso.

**Regression testing:** comprobar después de cambios que lo que funcionaba sigue funcionando. Reutiliza, selecciona y prioriza pruebas y puede darse en todos los niveles. La frase de p. 25 “no se diseñan nuevos casos” describe la reejecución de la suite; no prohíbe ampliar la suite para cubrir nuevas funciones o un bug corregido.

## 7. De dónde salen los tests (pp. 26–27)

Fuentes: requisitos y especificaciones, código, dominios de entrada/salida, operational profile y **fault models**. Estos últimos incluyen error guessing (experiencia sobre errores probables), fault seeding (introducir defectos deliberadamente) y mutation analysis (pequeñas variantes del programa para evaluar tests).

**White-box / structural:** estructura interna, control flow (orden de ejecución) y data flow (propagación de valores). **Black-box / functional:** entradas y resultados observables a través de interfaces. Puede aplicarse a una función o a todo el sistema. La descripción de equipos en L1 refleja una distribución habitual de responsabilidades; no convierte black-box en una técnica exclusiva de QA.

## 8. Planificación, métricas, automatización y personas (pp. 28–32)

Un test plan fija marco, alcance, recursos, esfuerzo, calendario y presupuesto. Los casos se construyen con test steps modulares. Distinguir métricas de ejecución, seguimiento de defectos, eficacia de casos (defect revealing ability) y eficacia del esfuerzo (incluye defectos escapados que descubre el cliente).

Automatizar puede aumentar productividad, cobertura de regresión y eficacia, acortar fases y reducir mantenimiento. Requiere casos bien definidos, infraestructura y herramientas, experiencia y presupuesto. Los scripts y oráculos también requieren mantenimiento.

La tabla de p. 31 cita **Katalon, Selenium, Appium, TestComplete y Cypress**. Presenta ámbitos web/API/móvil/escritorio y diferencias de configuración, scripting e informes. Es una tabla del material docente: precios, valoraciones y compatibilidades no se tratan como datos actuales ni como prioridades conceptuales del examen.

La organización puede separar grupos de desarrollo, integración y sistema, con especialidades de performance, scalability, automation y sustaining. Contratar y retener testers requiere entrevistas cuidadas y reconocimiento de su trabajo al nivel del desarrollo.

## Preguntas conceptuales para explicar en voz alta

1. ¿Cómo puede existir un fault sin failure? **Porque puede no ejecutarse, no activar un error o no propagarlo a una salida.**
2. ¿Por qué black-box no equivale a system testing? **Una técnica indica cómo seleccionas tests; un nivel indica qué alcance pruebas.**
3. ¿Qué falta a una suite que solo comprueba “no crash”? **Oráculos para valores, estado y requisitos.**
4. ¿Qué diferencia probar para romper de probar para estimar reliability? **La selección sesgada a defectos frente a una distribución representativa del uso.**
5. ¿Qué significa 100 % de tests pasados? **Todos los casos de esa suite cumplieron su oráculo, no ausencia universal de defectos.**
