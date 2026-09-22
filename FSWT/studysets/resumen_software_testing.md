# Software Testing

## Fuente

`L2_Ch8.Testing (Sommerville) [Updated].pdf`, pp. 1–51. El esquema del tema es **development testing → TDD → release testing → user testing**. Las preguntas combinan definiciones con situaciones nuevas basadas en los ejemplos del documento.

## 1. Objetivos y confianza (pp. 3–7)

Testing ejecuta con datos elegidos y observa resultados, anomalías y propiedades no funcionales. Puede mostrar presencia de defectos, pero no ausencia universal.

| Objetivo | Qué cuenta como éxito del esfuerzo |
| --- | --- |
| Validation testing | Evidenciar que opera como se requiere |
| Defect testing | Descubrir un defecto provocando comportamiento incorrecto |

No confundir **éxito al descubrir un bug** con **veredicto pass del caso**. Un test que falla puede ser muy valioso para defect testing.

Verification pregunta si se construye conforme a especificación. Validation pregunta si se construye lo que el usuario necesita. V&V busca confianza de **fit for purpose**, condicionada por criticidad/propósito, expectativas y entorno comercial. Una presión de entrega no demuestra que el riesgo técnico haya desaparecido.

## 2. Inspecciones y ejecución (pp. 8–13)

Inspecciones analizan representaciones estáticas: requisitos, arquitectura, modelos UML, esquemas de datos, código, configuración y test data. Pueden aplicarse antes de disponer de un ejecutable o sobre versiones incompletas. Ayudan a encontrar varios defectos sin que el primero en ejecutarse oculte otros y permiten considerar estándares, portabilidad y mantenibilidad.

Las pruebas dinámicas ejecutan y observan. Inspeccionar puede identificar riesgos, pero no sustituye medir rendimiento, experimentar usabilidad o comprobar funcionamiento en condiciones reales. El capítulo presenta ambas técnicas como complementarias.

El proceso de pruebas de la figura p. 13 es: **diseñar casos → preparar test data → ejecutar → comparar con casos → reportar**. Sus productos incluyen test cases, test data, test results y test reports.

## 3. Etapas y development testing (pp. 14–23)

**Development:** equipo que desarrolla encuentra defectos durante construcción. **Release:** equipo separado evalúa una versión completa antes de uso externo. **User:** usuarios prueban y aportan contexto operacional.

Dentro de development:

| Nivel | Foco |
| --- | --- |
| Unit | Funciones, métodos, clases o unidades con interfaz definida, en aislamiento |
| Component | Interfaces de unidades integradas en componentes |
| System | Interacciones, compatibilidad, datos correctos en el momento correcto y propiedades emergentes |

Para una **object class** se consideran todas las operaciones, asignación/consulta de atributos y estados. La herencia distribuye la información entre clases y complica el diseño. Esta aspiración de cobertura no elimina las limitaciones prácticas de exhaustividad.

**WeatherStation:** hay que probar operaciones y secuencias de estados. Ejemplos del texto: Shutdown → Running → Shutdown; Configuring → Running → Testing → Transmitting → Running; Running → Collecting → Running → Summarizing → Transmitting → Running. La interfaz ilustrada y el texto usan listas de métodos distintas: lo importante es cubrir el contrato de operaciones y transiciones, no deducir una API única de ambas listas.

Un test automatizado tiene **setup** (estado, inputs y expectativas), **call** (ejecutar) y **assertion** (contrastar resultado). Automatizar permite repetir y verificar sin intervención manual. Se necesitan casos normales y anormales: entrada inválida debe rechazarse o manejarse según contrato, sin fallo inesperado.

## 4. Selección de casos: particiones y guías (pp. 24–26)

**Equivalence partitioning:** agrupar valores que deben tratarse de forma similar y elegir representantes. Considerar particiones de entrada y, cuando proceda, salida, tanto válidas como inválidas. Es una reducción razonada, no una prueba matemática de equivalencia de todas las ejecuciones.

Ejemplos del gráfico:

| Dominio válido de enteros | Particiones | Fronteras y vecinos | Interior |
| --- | --- | --- | --- |
| 4–10 inclusive | <4; 4–10; >10 | 3, 4, 10, 11 | 7 |
| 10000–99999 inclusive | <10000; 10000–99999; >99999 | 9999, 10000, 99999, 100000 | 50000 |

Los valores fronterizos ayudan con errores como `<` frente a `<=`. Probar muchos interiores de una sola clase no compensa omitir clases inválidas. **Guideline-based testing** aprovecha experiencia de errores típicos para elegir casos y complementa las particiones.

## 5. Sistema, use cases y políticas (pp. 27–30)

Que todas las unidades pasen no prueba que colaboren correctamente. System testing comprueba comportamiento emergente e intercambios. Los **use cases** sirven para provocar recorridos reales a través de varios componentes; los **sequence diagrams** indican participantes, mensajes y orden.

El ejemplo de recoger datos meteorológicos conecta sistema de información, SatComms, WeatherStation, Commslink y WeatherData: solicitud, acuses, resumen y respuesta. Importan datos, destinatarios y secuencia, no solo que cada método retorne.

Como no se puede probar exhaustivamente, una **testing policy** define cobertura mínima: funciones accesibles por menús, combinaciones de funciones relacionadas, datos correctos e incorrectos donde hay input. “Cumple la política” es evidencia limitada a ese criterio.

## 6. TDD y regresión (pp. 32–36)

Ciclo: identificar pequeño incremento, escribir test automatizado, ejecutarlo junto con tests anteriores, comprobar que falla por la función ausente, implementar, reejecutar y refactorizar conservando comportamiento. Avanzar cuando toda la suite pasa.

Si el test pasa antes de implementar, investigar si la función ya existe o el test no comprueba lo que se cree. Si el nuevo test pasa pero uno anterior falla, corregir la regresión.

Beneficios citados: código acompañado de pruebas, suite de regresión que crece, debugging más localizado y tests como documentación ejecutable. No garantiza todos los escenarios ni oráculos correctos. TDD surgió en métodos ágiles como XP, pero también cabe en plan-driven.

**Regression testing** comprueba que cambios no rompen código previo. El flujo del capítulo reejecuta todos los tests y exige éxito antes de commit. Una suite verde solo acredita sus casos. Refactoring cambia estructura manteniendo comportamiento; los tests ayudan a comprobarlo.

## 7. Release y requirements-based testing (pp. 38–41)

Release testing evalúa si una versión es suficientemente buena para uso fuera del equipo. Verifica funcionalidad, performance y dependability especificadas y funcionamiento normal. Suele ser **black-box derivado de especificación**. Es una forma de system testing; difiere del system testing de desarrollo en equipo independiente y énfasis en validación, frente a descubrimiento de defectos.

**Requirements-based testing:** examinar cada requisito y diseñar uno o varios tests observables. El ejemplo de **Mentcare** contiene dos reglas: emitir alerta ante medicación alergénica y exigir motivo al ignorarla. Casos relevantes:

- Ninguna alergia relevante: no emitir alerta indebida.
- Una alergia: alertar ante el medicamento correspondiente.
- Varias alergias: comprobar cada medicamento y las alertas múltiples.
- Anular una alerta: exigir y registrar el motivo según el requisito.

No basta probar “aparece una alerta”: también hay que comprobar ausencia cuando procede y la regla de justificación.

## 8. Performance y stress (p. 42)

Performance testing observa propiedades emergentes con cargas que representen el **profile of use**. Una serie de cargas crecientes revela cuándo las prestaciones dejan de ser aceptables. **Stress testing** sobrecarga deliberadamente para estudiar cómo falla el sistema. Aumentar usuarios sin especificar operaciones, distribución y criterio de rendimiento deja la prueba incompleta.

## 9. User y acceptance testing (pp. 44–48)

El entorno real puede afectar fiabilidad, rendimiento, usabilidad y robustez de formas no reproducidas en laboratorio. Por eso user testing sigue siendo necesario tras release testing.

| Modalidad | Quién, dónde y para qué |
| --- | --- |
| Alpha | Usuarios trabajan con desarrollo en las instalaciones del desarrollador |
| Beta | Se facilita una versión para que usuarios experimenten y reporten problemas |
| Acceptance | El cliente decide si acepta y despliega, especialmente en sistemas a medida |

Proceso de aceptación: **criterios → plan → derivar casos → ejecutar → negociar resultados → aceptar/rechazar**. Se acuerdan condiciones antes de juzgar resultados; no se redefinen arbitrariamente para ocultar fallos.

En el enfoque agile presentado, el cliente integrado define tests de aceptación que se ejecutan automáticamente con cambios, sin una fase separada. El problema clave es si representa los intereses de todos los stakeholders. Esta es la descripción del modelo docente, no una afirmación de que toda organización ágil deba carecer de actividades formales de aceptación.

## Preguntas conceptuales con respuesta orientativa

1. ¿Por qué defect testing puede tener éxito cuando el caso falla? **Porque descubrió el defecto que se buscaba exponer.**
2. ¿Por qué system testing sigue siendo necesario? **Las propiedades emergentes y contratos entre componentes no se demuestran con aislamiento.**
3. ¿Qué prueba detecta mejor un error de frontera? **Una entrada en el límite y sus vecinos de clases distintas.**
4. ¿Release y acceptance son lo mismo? **Release informa al proveedor sobre uso externo; acceptance informa la decisión del cliente.**
5. ¿Qué separa stress de carga normal? **Sobrecarga deliberada y observación del modo de fallo.**
6. ¿Qué aporta ver rojo antes de verde en TDD? **Evidencia de que el test detecta la funcionalidad ausente.**
