# FSWT Midterm Study Guide

## Confirmed scope and logistics

The professor's announcement says the midterm is **Wednesday, October 28, 2026**, and covers:

1. Control Flow Testing
2. Domain Testing
3. System Integration Testing
4. System Testing / System Test Categories

The format will be the same as the quiz. Bring your **laptop**; **LockDown Browser is required**. The announcement supplied here does **not** state a question count, duration, access code, permitted resources, or exact breakdown of question types. Do not carry over the rules or 65-minute duration of the separate Computational Geometry exam. Check the course's current quiz conventions and any later instructor directions for those details.

This preparation pack contains four topic notes, **130 quiz questions**, **84 fill-in-the-blank exercises**, and a **30-question mixed practice quiz** with a worked key. Thirty is a study-pack choice, not a claim about the real exam. All new study content is in English.

## How to study

Start each topic in the **Study notes** tab. Work the examples on paper before reading the solution. Then use the multiple-choice questions for application and the fill-in-the-blank questions for recall. Explain why each distractor is wrong. A correct guess needs review too.

The topic pages support shuffle, local mastery tracking, answer explanations, source-page references, and saved session state. Their FSWT-midterm browser-storage keys are separate from Quiz 1 and CG. The mixed practice quiz is a separate paper-style exercise with a hidden answer key, not an automatically scored or timed exam.

## Suggested preparation sequence

| Session | Focus | What to produce without notes |
| --- | --- | --- |
| 1 | CFGs and coverage | Draw a small graph; give a statement-covering suite that misses a branch |
| 2 | Symbolic execution and complexity | Solve path constraints; compute E-N+2P; explain infeasibility |
| 3 | Domains and boundary defects | Derive input domains and design ON–OFF–ON points |
| 4 | Integration | Trace a hierarchy top-down/bottom-up and identify every stub and driver |
| 5 | System categories | Classify scenarios by objective; compute reliability measures |
| 6 | Mixed practice and review | Attempt all 30 questions, then correct and explain each mistake |

Adapt these sessions to your schedule. The announcement gives the same date as the CG exam, so use shorter alternating review sessions if preparing for both.

## High-priority distinctions

| Distinction | What to remember |
| --- | --- |
| Statement vs branch | Visiting every statement can still miss an empty false branch |
| Branch vs all-path | Both outcomes at every decision do not enumerate every path combination |
| Predicate coverage | These slides require truth-combination coverage affecting selected paths |
| Structural path vs feasible path | A graph route can have contradictory input constraints |
| V(G) vs total paths | Cyclomatic complexity describes an independent path basis, not exhaustive loop paths |
| Computation vs domain error | Correct path/wrong calculation versus wrong input classification/path |
| Closure vs shift vs tilt | Equality inclusion versus parallel displacement versus changed orientation |
| ON/OFF membership | Closed: ON inside, OFF outside; open: ON excluded, OFF inside |
| Stub vs driver | A stub is called; a driver calls the real unit/cluster |
| Intra vs inter-system | Modules within one system versus separate systems |
| SIT vs system testing | Assembled interactions versus integrated-product requirements/qualities |
| Performance vs scalability | Metrics under conditions versus behavior as scale grows |
| Stress vs load/stability | Beyond capacity versus sustained anticipated/full load |
| Reliability vs high availability | Failure-free operation versus continued service and rapid recovery |
| Retest vs regression | Confirm a fix versus detect side effects on previous behavior |
| Documentation levels | Read, hands-on help/error-message use, functional execution of instructions |

## Formula and boundary sheet

```text
coverage percentage = covered counted obligations / total counted obligations * 100
V(G) = E - N + 2P
connected single-routine CFG: V(G) = E - N + 2
structured connected CFG with d binary decisions: V(G) = d + 1
n independent Booleans: 2^n truth combinations
OR true combinations: 2^n - 1
AND true combinations: 1
lecture reliability convention: MTBF = MTTF + MTTR
```

Use consistent CFG detail and units. For coverage, state whether the denominator includes unreachable structural obligations. For Boolean combinations, check feasibility and short-circuit evaluation. For reliability, separate operational intervals from repair intervals.

For each path, propagate assignments before interpreting later predicates. Solve the **conjunction** of all required outcomes. An inconsistent conjunction makes the path infeasible.

For a linear boundary, choose two distinct ON points satisfying the equality and a nearby OFF point on the domain-relative side specified by the lecture. Maintain other constraints. Integer inputs require representable integer choices; floating-point approximations must be checked by evaluating the actual predicate.

## Slide clarifications to review

These notes retain the course's terms while explaining conflicting examples:

- **Control flow pp. 23–25:** “predicate coverage” here means truth combinations, beyond merely toggling an overall decision.
- **Control flow p. 39:** the “all paths” shorthand around cyclomatic complexity should not be read as all complete execution paths; V describes a basis, and loops/infeasibility need separate treatment.
- **Domain pp. 13–16:** substituting the displayed code gives x>=4 for P2 on the P1-false path. The x>4 text omits equality, and the figure swaps the P2 path labels beside vertical lines. Input (4,0) is a useful code-level check.
- **Domain p. 20:** the slides call a domain open when some boundary is open; this is broader than the mathematical topology definition. Always state each boundary's membership.
- **Domain p. 27:** (0,0.858) satisfies x+7y>=6, so it is inside. The final outside wording conflicts with the arithmetic. Also, 6/7 is exact rationally even though a finite decimal approximation is not exact.
- **Integration pp. 22–24:** some fake gateway classes are named Driver while serving as called dependencies. By call direction, those are callee stubs/test doubles; the outer harness is the driver.
- **System categories p. 23:** the MTTF wording is ambiguous about between-failure intervals. These exercises explicitly separate failure-free operating time and repair time, then use the slide equation MTBF=MTTF+MTTR.

These are not instructions to ignore the instructor's question wording. If an exam explicitly supplies a convention, use it and state any assumptions in a written explanation.

## Sources and coverage

References in each question point to the supplied PDF file and PDF page range, including the title page. Numeric examples and scenario questions are original applications of the cited concepts. The source PDFs are not redistributed.

| Topic / source file | Main coverage | Quiz | Fill |
| --- | --- | ---: | ---: |
| `L4_Ch4-ControlFlowTesting [Updated].pdf` | CFGs, coverage, predicates, inputs, symbolic execution, feasibility, cyclomatic complexity; pp. 3–40 | 30 | 20 |
| `L5_Ch6-DomainTesting [Updated].pdf` | Domains, error classes, boundary inequalities/defects, ON/OFF points, selection; pp. 3–33 | 30 | 20 |
| `L6_Ch7-SystemIntegrationTesting [Updated].pdf` | Interfaces/granularity, builds, strategies, hardware, plans/criteria, OTS/COTS, built-in testing; pp. 3–35 | 30 | 20 |
| `L7_Ch8-SystemTestCategories.pdf` | All twelve families, detailed subcategories, reliability, documentation, regulatory/safety; pp. 3–30 | 40 | 24 |
| **Total** | **Four supplied lecture decks** | **130** | **84** |

The domain deck refers to Appendix L5Ch6, and the system-category deck points to textbook examples such as SNMP and 1xEV-DO. Those additional resources were not supplied in this folder. This pack covers the provided slides and original applications; it does not claim to reproduce the missing appendix or textbook examples.

## Final recall checklist

Can you construct a small CFG; calculate a coverage percentage; exhibit a statement-versus-branch counterexample; solve a path constraint; distinguish infeasibility from a missing test; compute V; derive domains through assignments; classify and expose a boundary fault; choose domain-relative ON/OFF points; select an integration strategy; identify stubs/drivers; explain SIT entry/exit criteria; classify system-test scenarios; and calculate the lecture's reliability measures?

For the exam, follow the FSWT instructor's permitted-resource directions and LockDown Browser requirements. This site supports preparation before the test.
