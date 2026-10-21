# System Integration Testing

Source: `L6_Ch7-SystemIntegrationTesting [Updated].pdf`. Page references count PDF pages from the title slide. Examples and practice problems below are original applications unless explicitly identified as slide examples.

## Purpose and level of testing

Unit tests check modules in isolation; **system integration testing (SIT)** checks that assembled components interact correctly. Individually correct modules can disagree on units, formats, timing, initialization, error handling, or API meaning. SIT aims to build a working integrated system without breaking already-integrated behavior. The final integrated build becomes a candidate for system testing (pp. 3–5, 11).

The lecture identifies three interface mechanisms: **procedure calls**, **shared memory**, and **message passing**. Interface faults may concern construction, missing or misplaced functionality, changed data structures, misuse or misunderstanding, initialization, data constraints, coordination, error processing, timing, or hardware/software interaction (pp. 5–6).

| Granularity | Focus | Example |
| --- | --- | --- |
| Intra-system | Integrate modules within one system | Cart, inventory, and order modules agree on reserved quantities |
| Inter-system | Connect separately tested systems | Shop and external payment system exchange transactions |
| Pairwise integration | Check two connected systems at a time | Verify one shop–gateway link while assuming other services behave normally |

Pairwise integration is about a pair of interconnected systems; it is not the same concept as pairwise combinatorial input testing. The p. 8 cart-function example can resemble unit testing; classify a test by what actual components interact, not its label alone (pp. 7–8).

## Stubs and drivers: follow the call direction

A **stub replaces a called dependency**. A **driver calls the unit or cluster under test**, replacing a missing caller. If A calls B, then while testing A without B, B's replacement is a stub. While testing real B without A, a small program that calls B is a driver (pp. 15–24).

Some payment examples on pp. 22–24 name a called payment-gateway replacement `PaymentGatewayDriver`. By its role in that code it behaves as a **stub/test double for a callee**. The outer test function invoking the real service performs the driver role. Do not memorize names in the examples at the expense of call direction. Simulated integration checks assumptions but does not prove compatibility with the real omitted dependency.

## Integration strategies

| Strategy | Assembly order | Typical temporary code | Strength / limitation |
| --- | --- | --- | --- |
| Top-down | Begin with high-level control modules, replace dependencies progressively | Stubs | Early control-flow visibility; unavailable low-level behavior may limit realism |
| Bottom-up | Begin with low-level clusters, add higher layers | Drivers | Early utility/service testing; high-level design faults can emerge late |
| Big-bang | Combine individually tested modules at once | Depends on missing externals | Many interfaces become active together, making fault localization harder |
| Sandwich | Mix upper and lower integration, then join the middle | Both | Lecture uses top-down above, bottom-up below, big-bang in the middle |

Top-down and bottom-up can both be incremental. “Incremental” describes gradual assembly and testing, rather than a unique direction. After adding a real module, test its new interfaces **and rerun relevant old tests** to detect interference with existing modules (pp. 9, 12, 15–20).

Example hierarchy: A calls B and C; C calls E and F. A top-down first stage can use real A and B with a C stub. Later replace C with the real module and add E/F stubs. Bottom-up can test E and F through a driver, replace that driver with real C, and then introduce an A-level driver before integrating A.

## Builds and regression

A **software image** is a compiled binary; a **build** is an interim image for internal testing. Construction gathers authorized, unit-tested module versions, compiles/links them, verifies subassemblies, and applies version control. A module's check-in request documents readiness before integration (pp. 9–14).

A build release note should record changes, fixed and outstanding defects, added/modified/deleted features, and any uncertain change areas. A per-build strategy selects new integration tests, reruns previously failing tests to check fixes, and chooses regression tests to find side effects. Re-testing a reported defect and regression testing unchanged functionality serve different objectives.

Daily-build practice supports small increments, automated reusable tests, quick feedback, and rollback through retained earlier builds. The lecture describes retaining 7–10 builds and trying to fix daily-build defects within 24 hours; these are described practices, not universal obligations (p. 14).

## Hardware/software integration

Iteratively load a minimal image onto prototype hardware, check that required modules are present, test essential behavior, and add functionality in later cycles. Hardware design verification tests (DVT) are prepared/executed by the hardware team before software integration. Examples include electrical, thermal, electrostatic discharge, emission, diagnostic, environmental, packaging/handling, acoustic, safety, and reliability tests (pp. 25–27).

A **compatibility matrix** records approved hardware/software revision combinations. An **Engineering Change Order (ECO)** formally documents a change and includes compatibility information. A passing test on hardware revision 1 does not establish compatibility with revision 2 (p. 28).

## SIT plan, entry, and exit criteria

The plan includes scope, phases and modules, schedule, environment/resources, per-phase entry/exit criteria and integration technique, test configurations, test IDs, inputs, initial conditions, expected results, execution/result-capture procedures, actual results, and references (p. 29, Table 7.3).

**Entry framework** (p. 30, Table 7.4): approved functional/design specifications and code; written/reviewed/executed unit plans with all unit tests passed; approved module check-in forms; approved hardware specifications, completed DVT, and successful hardware/software integration tests where applicable.

**Exit framework** (p. 31, Table 7.5): all code integrated and frozen; all SIT tests passed; no outstanding major defects; moderate defects fixed and retested; results documented. Its example numeric gates allow **at most 25 minor defects** and require **two weeks without anomalies/crashes** in the SIT environment. Treat these as the lecture's sample framework, not a universal software-release rule. Entry means ready to begin; exit means the phase has met its defined completion criteria.

The lecture's six integration-test categories (p. 32):

- **Interface integrity:** internal/external interfaces after adding components.
- **Functional validity:** an integrated module's functions.
- **End-to-end validity:** the complete workflow across components.
- **Pairwise validity:** two connected systems.
- **Interface stress:** behavior under interface load.
- **System endurance:** continued operation over weeks.

## Off-the-shelf components and built-in testing

For OTS/COTS integration, a **wrapper** isolates or simplifies an underlying component's interface; **glue** connects components; **tailoring** enriches behavior without modifying the vendor's source. They may coexist in one solution (p. 33).

Black-box component tests assess the purchased component's quality. **System-level fault injection** deliberately makes a component fail to observe system tolerance, error handling, and recovery. **Operational system testing** in this lecture observes the assembled system with the COTS component functioning correctly (p. 34).

Built-in testing embeds testability in a component. In **normal mode**, the capabilities are transparent to users. In **maintenance mode**, the user invokes internal tests, which execute, evaluate results, and return a summary (p. 35).

## Scenario practice

1. A real order service calls a fake shipping API returning a fixed cost. **The fake API is a stub.**
2. A small harness calls real low-level parsers before their controller exists. **The harness is a driver; this supports bottom-up testing.**
3. A gateway is added and an old cart test starts failing. **Test the new interface and run regression to detect interference.**
4. Every module passed unit testing, but one sends cents while another expects dollars. **An interface contract fault remains for SIT.**
