# Unit Testing

## Fuente

`L3_Ch3-UnitTesting.pdf`, pp. 1–33. Incluye Naik & Tripathy, code review, prevención, pruebas dinámicas, mutation testing, debugging, XP y el ejemplo clásico de JUnit. Se conservan las convenciones docentes y se explican las limitaciones de sus simplificaciones.

## 1. Unidad y análisis estático/dinámico (pp. 3–5)

Una unidad puede ser función, procedimiento, método o clase. Tiene una interfaz por la que la invocan y puede invocar otras unidades. **Aislamiento** significa controlar su entorno y evaluar su comportamiento por separado, no exigir que nunca tenga dependencias.

**Static unit testing:** examinar código contra requisitos sin ejecutarlo. **Dynamic:** ejecutar y observar. Se recomienda revisión estática antes de dinámica, aunque en práctica puedan solaparse parcialmente. Se complementan.

**Inspection:** revisión formal por pares, pasos y criterios predeterminados. **Walkthrough:** autor guía informalmente una ejecución manual/simulada con escenarios. La revisión evalúa código, no a la persona. Dividir en partes pequeñas ayuda a no omitir detalles, pero revisar cada parte aisladamente no demuestra automáticamente que las interacciones sean correctas.

## 2. Code review completo (pp. 6–9)

| Etapa | Qué sucede |
| --- | --- |
| Readiness | Verificar completitud, funcionalidad mínima, legibilidad, complejidad y documentos de requisitos/diseño |
| Preparation | Preparar preguntas, potenciales CR y mejoras |
| Examination | Autor presenta, presenter lee, record keeper registra y moderator mantiene el foco |
| Re-work | Registrar decisiones y lista de CR/mejoras; el autor corrige |
| Validation | Comprobar de forma independiente los CR resueltos |
| Exit | Distribuir resumen/informe de la revisión |

Roles: **moderator, author, presenter, record keeper, reviewers, observer**. Una persona puede tener funciones organizativas concretas, pero conviene distinguir responsabilidades.

**Change Request (CR):** descripción, prioridad (major/minor en el material), persona responsable y fecha límite. Resolver y validar son pasos diferentes.

Métricas: LOC revisadas/hora, CR por KLOC, CR/hora, total de horas. Ejemplo nuevo: 12 CR sobre 2000 LOC = **6 CR/KLOC**. No interpretar automáticamente más CR como peor persona: complejidad, revisión y contexto importan.

Jerarquía documental: requirements (propuesta/necesidad), functional specification (respuesta funcional), high-level design (arquitectura), low-level design (detalle de módulos), code. También pueden revisarse guías de instalación, usuario y troubleshooting.

## 3. Defect prevention (pp. 10–12)

Los CR indican posibles problemas que consumen recursos y deben resolverse antes de integrar. Reducir introducción de defectos evita retrabajo. Pautas del material:

- Instrumentar para observar: el ejemplo incrementa `bb[0]` y `bb[1]` según rama.
- Controlar división por cero, índices y desbordamientos de contadores/buffers.
- Tratar todos los retornos pertinentes y validar argumentos.
- Centralizar mensajes de error y ayuda.
- Usar, conservar y documentar assertions que expresan condiciones que deberían cumplirse.
- Hacer comprobaciones inversas tras cálculos importantes y usar contadores de bucle cuando corresponda.

Una **assertion** es una condición booleana esperada en un punto; no reemplaza por sí sola el manejo previsto de inputs inválidos. Reverse-compute: calcular raíz y reconstruir el input elevando al cuadrado, con precisión/tolerancia pertinente. Es una comprobación de consistencia, no una prueba general de corrección.

**Instrumentation** observa; **mutation** altera deliberadamente para evaluar tests. No confundir propósitos.

## 4. Dynamic unit testing (pp. 13–14)

**Driver → UUT → stubs**. El driver llama a la unidad, proporciona datos y facilita reportar resultados. Los stubs emulan unidades llamadas por UUT. Driver + stubs = **scaffolding**. La prueba no demuestra integración real con la dependencia sustituida; para eso hacen falta otras pruebas.

El **low-level design** guía inputs. Familias de selección:

| Técnica | Base |
| --- | --- |
| Control flow | Orden de instrucciones, CFG, criterios y caminos |
| Data flow | Propagación de valores, definiciones y usos |
| Domain | Dominios y errores asociados, incluidas fronteras |
| Functional program testing | Relación entre inputs y outputs esperados |

Control flow: construir CFG, elegir criterio, seleccionar caminos, derivar **path predicate** y resolverlo para obtener inputs. Un camino gráfico puede ser inviable: `x > 10 AND x < 5` sin cambiar x no tiene solución. No inventar un input para una condición imposible.

## 5. Mutation testing (pp. 15–19)

Evalúa **calidad/adecuación de la suite** introduciendo pequeños cambios sintácticos legales. Cada programa modificado es un **mutant**.

| Estado | Interpretación |
| --- | --- |
| Killed / dead | Algún test detecta comportamiento incorrecto del mutante |
| Equivalent | Mismo comportamiento observable que el original para todos los inputs del modelo |
| Surviving | La suite actual no lo detectó; no demuestra equivalencia |
| Killable / stubborn en L3 | No equivalente que necesita un test que la suite actual no contiene |

Ejecutar una línea mutada no basta: el cambio debe producir una diferencia observable y el oráculo debe comprobarla.

**Mutation score = 100 × killed / (total mutants − equivalent mutants).** Se excluyen equivalentes. Ejemplo nuevo: 10 totales, 2 equivalentes, 6 matados → 6/8 = **75 %**. Si el denominador es cero, el cociente no está definido; no declarar un porcentaje informativo a partir de esa división.

Una suite **mutation-adequate** mata el 100 % de los no equivalentes considerados. La conclusión es relativa a ese conjunto de mutantes, no ausencia de todos los defectos del software.

### Ejemplo del índice del máximo, paso a paso (pp. 17–18)

El programa usa índices desde 1: `r=1`; recorre `i=2..3`; si `a[i] > a[r]`, pone `r=i`; devuelve r. Selecciona la **primera** posición del máximo.

Suite inicial:

| Input | Índice esperado |
| --- | --- |
| 1, 2, 3 | 3 |
| 1, 2, 1 | 2 |
| 3, 1, 2 | 1 |

| Mutante | Cambio | Resultado |
| --- | --- | --- |
| M1 | Bucle comienza en i=1 | Equivalente: primera comparación es a[1] > a[1], falsa |
| M2 | Compara i > a[r] | Lo mata 1,2,1: termina eligiendo índice 3 en vez de 2 |
| M3 | Cambia > por >= | Sobrevive sin empates; selecciona último máximo empatado |
| M4 | Compara a[r] > a[r] | Siempre falso; lo matan los dos primeros tests |

La slide 17 calcula **50 % provisional** al asumir cuatro no equivalentes. Tras demostrar M1 equivalente, la suite inicial tiene **2/3 ≈ 66.7 %**. Añadir **2,2,1** mata M3: original devuelve **1**, mutante devuelve **2**. Quedan **3/3 = 100 %**. El mutante equivalente no debe “matarse” mediante un oráculo incorrecto.

Hipótesis: **competent programmer** supone programas normalmente próximos a una solución correcta, no arbitrarios; **coupling effect** supone que tests sensibles a fallos simples detectarán muchos complejos relacionados. Son fundamentos del enfoque, no garantías universales.

## 6. Debugging (pp. 20–22)

Testing expone y evalúa fallos. **Debugging** parte de un síntoma, busca la causa y orienta su corrección. Puede ser costoso y propenso a errores.

| Enfoque | Movimiento del razonamiento |
| --- | --- |
| Brute force | Prints y trazas para observar ejecución |
| Cause elimination: induction | Recoger síntomas/datos, organizarlos, encontrar patrón y proponer hipótesis |
| Cause elimination: deduction | Enumerar causas por probabilidad y contrastarlas para descartar o confirmar |
| Backtracking | Retroceder desde el punto del fallo hasta su origen |

Backtracking resulta más difícil con muchos caminos posibles en programas grandes. Una correlación observada debe investigarse; un log no es automáticamente una explicación causal. Tras corregir, reejecutar el caso revelador y regresión relevante.

## 7. XP, test-first y pair programming (pp. 23–24)

Elegir story; escribir test pequeño fallido; implementar esa parte; ejecutar todos; corregir hasta pasar; repetir para completar la story.

Tres leyes docentes de TDD:

1. No escribir producción sin haber escrito primero un unit test fallido.
2. No escribir más test que lo suficiente para fallar.
3. No escribir más producción que lo suficiente para hacerlo pasar.

**Pair programming:** una persona desarrolla tácticamente y otra inspecciona con la perspectiva de la story. No son dos desarrollos independientes sin coordinación.

## 8. JUnit del material (pp. 25–28)

JUnit se presenta para Java. Otros ejemplos: NUnit (C#), CPPUnit (C++), fUnit (Fortran). Pasos: crear objeto, fijar inputs, calcular expected, llamar método, comparar actual y reportar.

Las slides usan **API clásica `junit.framework.TestCase`**, heredando de TestCase. Son ejemplos docentes/pseudocódigo, no una guía de configuración de JUnit actual. Los nombres `MyTest1`/`MyTest2` de la slide no deben copiarse como garantía de descubrimiento automático: en el estilo clásico los métodos de prueba se nombran `test...`.

| Assertion | Comprobación |
| --- | --- |
| assertTrue(condition) | Condición verdadera |
| assertFalse(condition) | Condición falsa |
| assertEquals(expected, actual) | Igualdad de valores según tipo/contrato |
| assertEquals(expectedDouble, actualDouble, tolerance) | Diferencia numérica dentro del margen |
| assertSame(expected, actual) | Misma referencia/identidad |
| assertNull(object) | Referencia nula |

En `assertEquals(365, x)`, **365 es expected y x actual**. En el ejemplo double, `0.0001` es tolerancia. Identidad no equivale a igualdad de valor: dos objetos distintos pueden ser iguales según equals.

El ejemplo de `assertTrue` lanza **AssertionFailedError** si la condición es falsa. La API clásica admite mensaje opcional inicial para diagnóstico. La frase docente sobre informar fallos no significa que ningún runner pueda mostrar tests correctos o un resumen.

## 9. Herramientas (pp. 29–32)

| Herramienta | Propósito |
| --- | --- |
| Code auditor | Comprobar estándares mínimos de código |
| Bound checker | Detectar escrituras/accesos fuera de áreas permitidas |
| Documenters | Extraer descripciones, árboles caller/callee y modelos |
| Interactive debuggers | Breakpoints e inspección de ejecución; el material cita también omniscient debuggers |
| In-circuit emulators | Conectar depuración con procesador objetivo |
| Memory leak detectors | Encontrar memoria solicitada que no se libera |
| Static code/path analyzers | Analizar estructura y caminos, incluida complejidad de McCabe |
| Inspection support | Organizar revisiones |
| Test coverage analyzers | Medir estructuras internas ejercitadas |
| Test data generators | Ayudar a obtener inputs para comportamientos deseados |
| Test harness | Soportar ejecución dinámica |
| Performance monitors | Medir características temporales |
| Network analyzers | Analizar tráfico y problemas |
| Simulators/emulators | Sustituir software/hardware no disponible |
| Traffic generators | Generar transacciones o paquetes |
| Version control | Conservar secuencia de revisiones y archivos relacionados |

**McCabe:** para un CFG conectado en la forma usual de entrada/salida, **V(G)=E−N+2**. La figura tiene E=9 y N=8, por tanto V=3. La variante del texto con arista adicional de salida a entrada usa E−N+1 sobre ese grafo modificado. No mezclar las dos fórmulas sobre el mismo número de aristas.

La complejidad cuenta caminos linealmente independientes; no todos los caminos de ejecución (los bucles pueden generar muchos) ni el número de bugs. Alta cobertura significa ejecución según un criterio; si no hay buenas assertions, pueden sobrevivir mutantes.

## Práctica conceptual con respuestas

1. ¿Por qué un mutante superviviente no tiene que ser equivalente? **Puede faltar un input revelador o una assertion que vea la diferencia.**
2. ¿Por qué los equivalentes salen del denominador? **Ningún test correcto puede distinguirlos del original en ese modelo.**
3. ¿Driver y stub son intercambiables? **Ocupan posiciones distintas: llamador de UUT y dependencia llamada.**
4. ¿Por qué independencia en validar CR? **Verifica la corrección en vez de confiar solo en la declaración del autor.**
5. ¿Induction y deduction? **Observaciones a hipótesis frente a hipótesis candidatas a pruebas de descarte.**
6. ¿Por qué cobertura alta puede coexistir con tests débiles? **Ejecutar una línea no garantiza comprobar correctamente sus efectos.**
