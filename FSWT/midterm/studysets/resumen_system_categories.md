# System Testing and Test Categories

Source: `L7_Ch8-SystemTestCategories.pdf`. Page references count PDF pages from the title slide. Examples and practice problems below are original applications unless explicitly identified as slide examples.

## System testing versus integration testing

SIT emphasizes interfaces and assembled interactions; system testing evaluates the integrated product against its requirements and operating expectations. A scenario can have several relevant qualities, so identify its **primary objective and measured evidence** instead of classifying only by the presence of a network, a UI, or a long duration (pp. 3–5).

The taxonomy contains twelve families. Security is discussed under functionality; high availability and degraded-node operation under robustness. These are the lecture's organizational choices, even though categories can overlap in practice.

| Category | Main question | Representative evidence |
| --- | --- | --- |
| Basic | Can it become operational? | Successful boot, configuration, upgrade/rollback |
| Functionality | Does it implement required capabilities? | Correct feature results, module behavior, GUI/security functions |
| Robustness | Does it handle errors and recover gracefully? | Recovery after invalid input, power glitches, module failure |
| Interoperability | Does it work with third-party products/platforms? | Correct exchange across supported products and revisions |
| Performance | How fast and resource-efficient is it? | Response time, throughput, CPU/memory utilization |
| Scalability | How does it behave as dimensions increase? | User/resource/geographic limits and growth trends |
| Stress | What happens at and beyond capacity? | Failure mode, bottlenecks, and recovery under overload |
| Load and stability | Does it stay stable under sustained full load? | Long-run errors, memory growth, slowdown, crashes |
| Reliability | How long does it keep operating without failure? | Failure-free operating times and failure measures |
| Regression | Did a change break previously working behavior? | Reused tests against affected and unchanged behavior |
| Documentation | Are instructions accurate, clear, and usable? | Review and execution of documented procedures |
| Regulatory | Does it meet applicable external compliance requirements? | Evidence required by the relevant certification process |

## Basic tests

**Boot** tests check supported startup options. **Upgrade/downgrade** tests check a graceful transition and rollback. **LED** tests check status indicators. **Diagnostic** tests exercise hardware/module health, with examples such as power-on self-test, loopback, and bit-error tests. **CLI** tests check configuration through command-line controls (pp. 6–7).

A product that computes correctly after startup may still fail basic tests if an upgrade loses its configuration. Do not confuse booting successfully with correct implementation of every feature.

## Functionality tests

Communication-system tests in the lecture include basic interconnection, capability, behavior, and system-resolution tests. Module tests at system level verify a module **inside the assembled system**, rather than necessarily in isolation. Logging/tracing tests verify configuration and operation, including crash logs. Element Management System (EMS) tests check management, monitoring, and upgrading; Management Information Base (MIB) tests check supported standard and enterprise management objects (pp. 9–10).

GUI tests cover visual controls and functionality behind them, plus usability. The lecture's characteristics are accessibility (enter/navigate/exit), responsiveness, efficiency (steps/time), and comprehensibility (understanding the structure). Online help and meaningful error messages also matter (p. 11).

Security uses the **CIA** objectives (pp. 12–13):

- **Confidentiality:** prevent unauthorized disclosure.
- **Integrity:** prevent unauthorized modification.
- **Availability:** preserve access for authorized users.

Security test scenarios include misuse/abuse cases in an authorized test environment. For exam classification, an unauthorized file read primarily concerns confidentiality, an unauthorized account-balance edit integrity, and denial of service availability. Correct encryption does not by itself establish all three objectives.

Feature tests cover requirements not already covered by the listed functional subcategories, for example data conversion and cross-functionality (p. 14).

## Robustness and recovery

Robustness asks how gracefully the product behaves under erroneous inputs or changed operational conditions. Subcategories include boundary/default values, power cycling, online insertion/removal (OIR), high availability, and degraded-node behavior (pp. 15–17).

A **power-cycling** test checks recovery after interruption/restoration. **OIR** checks insertion/removal of modules while running, including under heavy load. **High availability** checks redundancy/failover and short recovery without harmful service disruption. **Degraded node/failure containment** checks useful operation after part of the system becomes nonoperational. Availability and reliability are related but distinct: a system can recover quickly from frequent failures yet have poor failure-free reliability.

## Interoperability and compatibility

Interoperability checks interaction with third-party products. Configuration testing varies configurations in those checks. Compatibility covers supported platforms, operating systems, or DBMS combinations. **Backward compatibility** asks whether a current build still works with older supported platform versions (p. 18).

A simulated third-party service is useful during development, but real interoperability claims need evidence involving the supported actual counterpart or a justified conformance environment. A passing isolated stub test does not establish real product compatibility.

## Performance, scalability, stress, and sustained load

Use the objective to distinguish them (pp. 19–22):

- Measure request completion time under specified conditions: **performance**.
- Increase users and resources to find growth limits: **scalability**.
- Exceed designed capacity to characterize failure and recovery: **stress**.
- Run expected/full load for a long period to detect degradation: **load and stability**.

A test can measure performance during a stress run; the primary classification still follows its purpose. Response time is time per request; throughput is completed work per unit time. They are not interchangeable. Performance instrumentation may separate end-to-end time, CPU processing, connection delays, database access, and waiting. Avoid adding overlapping timing measurements as if they were always independent.

Scalability can be limited by storage, bandwidth, or processing speed. Increasing hardware does not prove linear growth. The slides mention extrapolation as a prediction technique, not as a substitute for validating every important supported limit.

Stress may expose memory leaks, allocation failures, or buffer-management issues. A sustained-load test may also expose a slow leak after many hours. The defect type alone does not uniquely determine the category.

## Reliability calculations

The slide on p. 23 uses these relationships:

```text
MTTF = mean failure-free operating time
MTTR = mean repair time
MTBF = MTTF + MTTR
```

Use this stated convention for these exercises. The slide also describes MTTF using “intervals between successive failures,” which can be ambiguous about repair time. To avoid double-counting, explicitly separate operating intervals from repair intervals before using its equation. Terminology differs across sources and repairability contexts.

Worked example: operating intervals 80,100,120 hours have mean MTTF=100 h. Repair times 1,2,3 hours have MTTR=2 h. The lecture equation gives MTBF=102 h. Keep units consistent; a repair time of 30 minutes is 0.5 hours.

Long duration alone does not determine the category: a reliability campaign estimates failure behavior, while a load/stability run emphasizes degradation under sustained load and high-availability tests emphasize failover and recovery.

## Regression and documentation

Regression selects tests from an existing pool to reveal side effects of a change. Retesting a defect confirms its fix; regression checks that previous behavior still works. Full reruns may be costly, so choose a justified subset based on impacted behavior and risk; do not infer that unchanged code cannot break (pp. 24–25).

The lecture gives three documentation-test levels (p. 26):

1. **Read test:** review clarity, organization, flow, and technical accuracy without executing instructions.
2. **Hands-on test:** exercise online help and error messages for accuracy and usefulness.
3. **Functional test:** follow documented procedures and verify the real system behaves as described.

A typo review does not prove an installation command works. Following the command against the supported product version provides different evidence.

## Regulatory testing and safety

Regulatory tests seek the relevant compliance evidence or marks for intended markets. The lecture mentions safety and electromagnetic compatibility/interference certification (p. 27). This is a description of a test category, not a claim that a particular product meets any current legal requirement.

A **hazard** is a system/environment condition with the potential to contribute to an accident. An **accident/mishap** is the harmful event, not just the possibility. Safety must be considered in the context of the complete system and environment; a software fault can contribute to physical harm through the equipment it controls (pp. 28–29).

Safety assurance identifies, tracks, evaluates, and eliminates or controls hazards and embeds safety into design/implementation. Passing ordinary functional tests does not by itself demonstrate safety under every hazardous circumstance (p. 30).

## Quick scenario decisions

- Upgrade then roll back while preserving configuration: **basic**.
- Fail a redundant power module and check seamless continued service: **robustness/high availability**.
- Run 2,000 supported users for a week and track memory drift: **load and stability**.
- Push past the rated 2,000-user limit and observe failure recovery: **stress**.
- Repeat previous export tests after an authentication change: **regression**.
- Verify a new build works with an older supported DBMS: **backward compatibility**.
