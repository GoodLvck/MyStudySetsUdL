# Software Process Models

## Fuente y alcance

Basado en `2_Ch2 SW Processes [Updated].pptx`, diapositivas 1–34. Esta presentación está en la carpeta de fundamentos, pero se separa en un set para poder estudiar y practicar todos sus modelos. Las páginas citadas en ejercicios corresponden a diapositivas.

## 1. Framework, jerarquía y actividades transversales (slides 3–9)

Un **process framework** proporciona una estructura común adaptable al tamaño y complejidad del proyecto. El diagrama lo organiza en framework activities, engineering actions y task sets. Estos sets incluyen tareas, productos, puntos de aseguramiento de calidad e hitos.

| Framework activity | Propósito y ejemplo |
| --- | --- |
| Communication | Colaborar con cliente y stakeholders para objetivos y requisitos |
| Planning | Definir tareas, riesgos, recursos, productos y calendario |
| Modeling | Entender arquitectura, componentes y relaciones |
| Construction | Generar código y probarlo |
| Deployment | Entregar, evaluar con el cliente y recoger feedback |

Las actividades pueden repetirse; cada iteración puede generar un **increment** con parte de la funcionalidad. **Activity** expresa un objetivo amplio (comunicación), **action** produce un producto mayor (diseño arquitectónico) y **task** tiene un objetivo pequeño y tangible (ejecutar un unit test).

**Umbrella activities** actúan a lo largo del proceso:

| Actividad | Función |
| --- | --- |
| Project tracking and control | Comparar progreso con plan y corregir desviaciones |
| Risk management | Evaluar riesgos del proyecto y producto |
| Software quality assurance | Definir y realizar actividades de calidad |
| Technical reviews | Detectar errores en productos antes de propagarlos |
| Measurement | Recoger medidas de proceso, proyecto y producto |
| Configuration management | Gestionar los efectos de cambios |
| Reusability management | Establecer criterios y mecanismos de reutilización |
| Work product preparation/production | Elaborar modelos, documentos, registros y otros productos |

**Process flow:** linear ejecuta secuencialmente; iterative repite actividades; evolutionary produce incrementos mediante vueltas; parallel solapa actividades. “Iterativo” e “incremental” pueden coexistir: repetir para refinar y añadir funcionalidad son ideas relacionadas pero distintas.

## 2. Software process y formas de describirlo (slides 10–13)

Un proceso es un conjunto estructurado de actividades. Un **process model** es una representación abstracta desde una perspectiva. La descripción incluye actividades y orden, products (resultados), roles (responsabilidades) y pre/postconditions (condiciones antes/después).

La presentación también usa las cuatro actividades generales de Sommerville: **specification**, **design and implementation**, **validation**, **evolution**. Specification define qué hará el sistema; design/implementation lo organiza y construye; validation comprueba especificación y necesidades; evolution lo modifica para seguir siendo útil. No confundir este esquema con la lista de cinco framework activities: ambos son abstracciones docentes válidas.

**Plan-driven:** actividades planificadas anticipadamente, progreso contra ese plan. **Agile:** planificación incremental y adaptación más fácil a cambios. Agile sí planifica. En la práctica se combinan enfoques y no hay una receta universal.

## 3. Waterfall (slides 14–17)

Secuencia: **requirements analysis/definition → system/software design → implementation/unit testing → integration/system testing → operation/maintenance**.

Es lineal, con fases separadas que idealmente se cierran antes de avanzar. Resulta fácil de entender y planificar cuando los requisitos son estables y se conocen bien. El material lo menciona tanto en proyectos grandes como en pequeños bien entendidos: **no elegirlo solo por tamaño**.

Su debilidad central es el cambio tardío: obliga a revisar productos previos. El feedback del cliente sobre el producto completo y buena parte de las pruebas globales llegan tarde. Esto **no significa que no exista unit testing durante implementación**, pues la propia lista de fases lo incluye.

## 4. Prototyping y throw-away (slides 18–19)

El prototipo permite aclarar requisitos con feedback temprano y frecuente, reduce impacto de cambios y riesgo de rechazo. Dificultades: coordinación con usuarios, retrasos, planificación y gestión difíciles, trabajo perdido si se descarta y tentación de entregar la demo como sistema final.

Un **throw-away prototype** se construye para aprender y se descarta. Puede no admitir ajustes suficientes de rendimiento u otras propiedades no funcionales, estar mal documentado y tener una estructura deteriorada por cambios rápidos. Que una demo funcione no acredita calidad de producción.

## 5. Spiral (slides 20–23)

Modelo evolutivo que combina iteración/prototipado con disciplina de waterfall. Las vueltas producen especificaciones, modelos, prototipos y versiones cada vez más completas. **El riesgo se considera en cada vuelta**; prototyping es una herramienta para reducirlo. Feedback de entregas ajusta el plan, coste y calendario.

Puede aplicarse durante la vida del software, no solo hasta la primera entrega. Beneficia proyectos grandes, complejos y extensibles, con participación continua del cliente. Requiere experiencia y gestión; un análisis de riesgos deficiente puede comprometer el proyecto. Repetir ciclos sin evaluar riesgo no captura la característica central de spiral.

## 6. Unified Process (slides 24–26)

UP combina prácticas tradicionales y ágiles. Es iterativo e incremental, enfatiza comunicación, arquitectura, cambios futuros, reutilización y **use cases**. UML se relaciona con su modelado, pero es un lenguaje, no un proceso de desarrollo.

| Fase incluida en el curso | Resultado / foco |
| --- | --- |
| Inception | Necesidades de negocio, comunicación, planificación inicial y use cases preliminares |
| Elaboration | Refinar use cases y establecer architectural baseline |
| Construction | Implementar funciones del incremento, unit tests, tests de use cases e integración |
| Transition | Entrega y feedback, beta testing, convertir incremento en release usable |
| Production | Monitorizar y dar soporte, evaluar defectos y peticiones de cambio |

El trabajo del siguiente incremento puede empezar mientras el actual está en construction, transition o production. Ventajas: documentación, cliente continuo y adaptación a cambios; límites: use cases imprecisos, integración difícil, problemas por solapamiento y necesidad de experiencia.

## 7. Incremental development (slides 27–29)

La especificación, desarrollo y validación se entrelazan para producir versiones. Ventajas: menor coste de requisitos cambiantes, feedback concreto mediante demos y utilidad temprana. No se espera al producto completo para entregar valor.

Problemas: menor **visibilidad** si documentar cada versión rápida cuesta demasiado; **degradación estructural** al acumular modificaciones. El refactoring invierte en estructura para contener el coste de futuros cambios y debe preservar comportamiento externo.

## 8. Integration and configuration, evolución y mejora (slides 30–33)

Desarrollar mediante **reuse** significa combinar elementos existentes, configurarlos y adaptarlos a necesidades. Incluye componentes y aplicaciones **COTS = Commercial-Off-the-Shelf**. No equivale a integration testing: uno es un enfoque de construcción, el otro evalúa interacciones.

El software evoluciona para seguir siendo útil. Prototipos y entregas incrementales ayudan a gestionar cambios. Para mejorar procesos, el cierre distingue **agile approaches** (menos overhead) y **maturity-based approaches** (mejor gestión y buenas prácticas).

## Comparación para el examen

| Situación | Modelo / idea y justificación |
| --- | --- |
| Requisitos conocidos y estables, fases y plan explícitos | Waterfall facilita planificación; cambios tardíos cuestan |
| Usuario no puede concretar necesidades | Prototyping obtiene feedback sobre algo observable |
| Riesgo técnico alto en un sistema complejo | Spiral evalúa y reduce riesgo en cada vuelta |
| Use cases, arquitectura e incrementos con fases | Unified Process |
| Se necesita entregar valor útil cuanto antes | Incremental, cuidando estructura y evidencia de progreso |
| Hay productos existentes adecuados | Integration/configuration basada en reuse |

## Práctica conceptual con respuesta orientativa

1. ¿Agile es no planificar? **No; cambia la granularidad y frecuencia de planificación.**
2. ¿Por qué no entregar un throw-away prototype? **La velocidad de aprendizaje puede haber sacrificado estructura, documentación y requisitos no funcionales.**
3. ¿Cómo puede un proceso ser iterativo e incremental? **Refina mediante vueltas y cada entrega añade capacidad.**
4. ¿Por qué no basta elegir un modelo por tamaño? **También importan estabilidad, riesgo, experiencia y feedback.**
5. ¿Qué separa umbrella de framework? **Su carácter transversal, no una fase puntual del ciclo.**
