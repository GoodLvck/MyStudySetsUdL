# Test Cases & Test Scenarios

## Fuente y enfoque

Basado en `L2.1_Test CasesScenarios.pdf`, pp. 1–18. Incluye criterios de diseño, plantilla y ejemplos de login/image upload. La página 17 propone un ejercicio de ATM: aquí se utiliza solo como contexto de estudio. Sus instrucciones antiguas de entrega, fecha y Canvas no son acciones solicitadas por el usuario.

## 1. Case frente a scenario (pp. 2–4, 13, 16)

Un **test case** especifica cómo verificar un comportamiento: condiciones, pasos, inputs y expected results. Un **test scenario** describe una funcionalidad o historia a nivel alto y sirve de base para uno o varios casos. Relación **uno a muchos**.

| Aspecto | Test case | Test scenario |
| --- | --- | --- |
| Granularidad | Comprobación específica | Contexto amplio |
| Contenido | Pasos, datos y resultados | Funcionalidad y casos asociados |
| Identificación | ID único | Título descriptivo, a menudo con identificador |
| Ejemplo | Login con contraseña incorrecta debe rechazarse | Autenticación de usuarios |

Un escenario útil es creíble, relevante, suficientemente complejo y con resultado evaluable. Suele derivarse de use cases. “Test login” no es por sí solo un caso reproducible.

Objetivos de documentar casos: verificar requisitos, guiar testers, descubrir problemas de uso/diseño, registrar pasos para futuros bugs, facilitar reutilización y ayudar a personas nuevas en el equipo.

## 2. Manual, automatizado y tipos (pp. 3, 5)

Manual implica que una persona ejecuta y registra. Automatización permite ejecutar scripts, comprobar resultados y generar informes mediante herramientas. La presentación cita Selenium y Appium y lenguajes como Java, C#, JavaScript, Ruby, PHP y C++. Eso no significa que cualquier herramienta o lenguaje sea necesario en todos los proyectos ni que los tests se diseñen solos.

Tipos de casos del diagrama: **functionality, user interface, performance, integration, usability, database, security, user acceptance**. Son focos de evaluación; un recorrido puede aportar evidencia en más de uno, pero conviene conservar un objetivo claro por caso.

## 3. Cómo escribir un buen caso (pp. 6–9)

- Elegir técnica apropiada: **equivalence class partitioning** para clases de comportamiento, **boundary value analysis** para límites, **decision table testing** para combinaciones de condiciones y acciones. El documento cita estas técnicas sin desarrollar un algoritmo completo de cada una.
- Usar redacción clara, nomenclatura uniforme y pasos inequívocos.
- Mantener un objetivo **atomic**: una funcionalidad o condición concreta. No significa un solo clic.
- No inventar funciones ni estados previos; referenciar SRS y use cases, aclarar lo ambiguo.
- Evitar redundancia que no añada cobertura.
- Mantener **traceability matrix** de requisitos a casos para identificar huecos de alcance.
- Considerar performance, usability, robustness y negative testing, además de funcionalidad.
- Usar datos diversos y representativos, incluyendo inválidos relevantes.

**Trazabilidad no es exhaustividad.** Un caso enlazado a cada requisito no garantiza todas las condiciones, estados, combinaciones ni oráculos correctos.

Ejemplo de mejora: “La pantalla funciona bien” se convierte en “Se muestra rechazo de credenciales y no se crea una sesión autenticada”. Si el texto exacto del mensaje importa, se toma del requisito; si no, se evita hacer el test frágil por copiar textos no contractuales.

## 4. Campos de la plantilla (pp. 11–12)

| Grupo | Campos |
| --- | --- |
| Identidad y objetivo | Test Case ID, title/description, objective |
| Preparación | Preconditions, environment, test data preparation |
| Procedimiento | Test steps, input data |
| Oráculo y observación | Expected results, actual results |
| Ejecución y decisión | Execution date, status, conclusion, verdict |
| Defectos y evidencia | Bug details opcionales, attachments |
| Control del caso | Author, reviewer, version, notes/comments |

**Expected** se deriva de la especificación antes de evaluar. **Actual** se registra al ejecutar. **Preconditions** describen estado previo; **input data** valores utilizados; **test data preparation** cómo obtener/preparar esos valores. Entorno y versión permiten reproducir diferencias.

El ID no sustituye contenido. Un bug puede gestionarse en otra herramienta y enlazarse al caso. No cambiar expected para convertir una ejecución defectuosa en pass.

## 5. Escenario de login (p. 14)

Derivar casos separados para:

1. Email y password válidos.
2. Email inválido con password válida.
3. Email válido con password inválida.
4. Ambos inválidos.
5. Campos vacíos.
6. Recuperación de contraseña.
7. Funcionamiento de enlaces.

Aunque varios deban rechazarse, ejercitan condiciones distintas. Cada caso debe precisar cuenta, estado de sesión, valores y resultado observable. “Inválido” puede significar formato incorrecto o credencial inexistente: distinguir según requisitos.

## 6. Escenario de image upload (p. 15)

Cobertura que enumera el documento:

| Dimensión | Qué comprobar |
| --- | --- |
| Flujo básico | Ruta, subir y cambiar imagen |
| Formatos | Extensiones permitidas y rechazo de archivos no imagen |
| Nombres | Espacios, caracteres permitidos y nombres duplicados |
| Restricciones | Tamaño máximo, ancho y alto especificados |
| Durante transferencia | Progreso de cargas grandes y cancelación |
| Selección | Diálogo que muestra archivos soportados |
| Multiplicidad | Subida de varias imágenes |
| Resultado | Calidad conservada y posibilidad de visualizar |

Ejemplo de práctica **con requisito supuesto explícito**: máximo 5 MB inclusive. Probar justo por debajo, exactamente en el máximo y justo por encima; definir antes si MB son 1 000 000 bytes o si el requisito usa otra unidad. Rechazo, mensajes, duplicados y efectos de cancelar deben salir del contrato; el documento no fija cada una de esas reglas.

## 7. ATM: ejemplos completos para razonar (p. 17)

La fuente pide diseñar casos de PIN y retirada, pero no especifica reglas bancarias. Los siguientes son **ejemplos de estudio con supuestos**, no una solución atribuida a un banco real.

### Caso ATM-WITHDRAW-01

- **Objetivo:** retirada válida actualiza saldo y entrega efectivo.
- **Supuestos/requisitos del ejemplo:** sin comisión, importe permitido, saldo suficiente y disponibilidad de efectivo.
- **Preconditions:** cuenta de prueba autenticada, saldo inicial 500, ninguna operación pendiente.
- **Inputs:** importe 200.
- **Steps:** consultar saldo, solicitar retirar 200, confirmar, comprobar efectivo y consultar saldo otra vez.
- **Expected:** entrega 200, saldo persistido 300 y registro de operación según el contrato de ejemplo.
- **Actual/status:** pendientes de ejecución; no se inventa un pass.
- **Environment:** registrar versión, configuración y entorno simulado utilizado.

Variantes negativas: importe superior al saldo o no permitido, con rechazo y estado final definidos por requisitos. No presuponer comisiones, límites o política de intentos.

### Caso ATM-PIN-01

- **Objetivo:** cambiar un PIN válido y comprobar el efecto persistente.
- **Preconditions:** cuenta de prueba y PIN actual conocidos, autenticación válida, política de formato explícita.
- **Inputs:** PIN nuevo que satisfaga esa política y confirmación coincidente.
- **Steps:** acceder al cambio, introducir y confirmar nuevo PIN, cerrar sesión, comprobar autenticación nueva y comportamiento del PIN anterior según la política.
- **Expected:** cambio confirmado, nuevo PIN aceptado, anterior rechazado conforme a la especificación.
- **Actual/status:** pendientes de ejecución.

Variantes: confirmación distinta, formato inválido o autorización ausente. Para cada una, precisar rechazo y conservación del PIN previo según contrato.

## Práctica conceptual

1. ¿Qué falta en “probar upload”? **Casos concretos, datos, precondiciones y oráculo.**
2. ¿Por qué varios casos por escenario? **Una función tiene condiciones normales, negativas, límites y estados distintos.**
3. ¿Cómo distinguir un bug de un requisito ambiguo? **Contrastar observación con un contrato claro; si falta, aclarar antes de concluir.**
4. ¿Por qué no registrar pass sin ejecutar? **Expected es una expectativa; actual y verdict necesitan evidencia.**
5. ¿Qué revisar si dos testers discrepan? **Datos, precondiciones, pasos, versión, entorno y criterio de resultado.**
