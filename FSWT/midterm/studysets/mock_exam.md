# FSWT Midterm: 30-Question Mixed Practice Quiz

Attempt every question before opening the answer key. This is original preparation based on the four supplied decks, not the professor's test. The practice length and question mix are study choices: the announcement only says the actual format matches the quiz and does not give duration or count. No timer is imposed here.

Questions 1–16 are multiple choice, 17–24 are fill-in-the-blank, and 25–30 are problems/scenarios. Use one point per correct item for self-assessment, then study the worked explanations. Assume independent Boolean conditions only where explicitly stated. Use the lecture's named conventions and exact arithmetic unless a question says otherwise.

## Multiple choice — 1–16

**1.** A test executes every statement in an `if` without an `else`, taking only the true outcome. What can remain uncovered?

A) The function entry · B) The false branch · C) Every assignment necessarily · D) Every return necessarily

**2.** The lecture's predicate coverage definition primarily requires:

A) Only one true result · B) Every method name listed · C) Feasible truth combinations affecting a selected path · D) Only a GUI screenshot

**3.** A path requires an unchanged x to satisfy x>5 and x<=5. It is:

A) Guaranteed executable · B) A complete branch suite · C) Executable for x=5 · D) Infeasible

**4.** Cyclomatic complexity is best interpreted as:

A) An independent path-basis measure · B) The number of all loop execution paths · C) The exact defect count · D) The number of input values

**5.** A correct path computes the wrong final sum. This is a:

A) Domain error necessarily · B) Computation error · C) Driver · D) Shifted boundary necessarily

**6.** Intended x<=8 is implemented as x<8. The defect is:

A) A parallel shift · B) A changed coefficient · C) A closure error · D) An interface driver

**7.** For a closed domain boundary, the lecture chooses an OFF point:

A) Far inside · B) Exactly on the included boundary · C) At any unrelated corner · D) Just outside the domain

**8.** For the displayed codedomain code, x=4,y=0 gives c=4,d=6. P1:c>5 and P2:d>=c+2 therefore yield:

A) FT · B) FF · C) TT · D) TF

**9.** A real order module calls a fake shipping dependency returning a fixed cost. That fake dependency is a:

A) Driver by definition · B) Stub · C) Build · D) Compatibility matrix

**10.** Bottom-up integration typically uses:

A) Only real top-level controllers from the start · B) Only manuals · C) Drivers for missing callers · D) No executable tests

**11.** Which statement best describes sandwich integration in these slides?

A) Only big-bang at every layer · B) Only unit testing · C) Only top-down · D) Top-down above, bottom-up below, joining the middle with big-bang

**12.** Which is a sample SIT exit criterion rather than initial entry readiness?

A) All modules integrated and results documented · B) First draft of module source · C) Initial check-in request being written · D) Initial hardware selection only

**13.** Exceeding designed capacity to observe failure and recovery primarily tests:

A) Documentation readability · B) Stress behavior · C) Only normal boot · D) Only compatibility

**14.** A new software build still operating on an older supported platform concerns:

A) Only a statement criterion · B) Only a new boundary · C) Backward compatibility · D) Only a GUI font

**15.** Unauthorized modification primarily threatens:

A) Confidentiality only · B) Throughput only · C) MTTF only · D) Integrity

**16.** Following manual instructions against the product to verify their behavior is a documentation:

A) Functional test · B) Read test only · C) Stress test · D) Regulatory test necessarily

## Fill in the blank — 17–24

**17.** A CFG has E edges, N nodes, and P connected components. V(G)=E-N+______.

**18.** A && B skips B when A is ______ under short-circuit semantics.

**19.** Wrong input classification into an unintended path is a ______ error.

**20.** The boundary test-selection sequence is ON–______–ON.

**21.** Integration of modules inside one system is ______-system testing.

**22.** Supported hardware/software revision combinations are recorded in a compatibility ______.

**23.** Completed work per unit time is ______.

**24.** Using the lecture convention, MTTF=120 h and MTTR=3 h give MTBF=______ h.

## Problems and scenarios — 25–30

**25. Coverage and V.** A connected CFG has nodes S, D1, A, D2, B, C, X and edges S→D1, D1(T)→A, D1(F)→D2, A→D2, D2(T)→B, D2(F)→C, B→X, C→X. D1 and D2 have independent conditions. Compute N,E,V. Do the two paths TT and FF cover all statements and branches? Do they cover all complete paths?

**26. Symbolic execution.** Consider y=x+3; if y>7, set y=y-2; then test y<10. Derive the integer x values forcing the true outcome at both tests. Explain why the second condition must use the updated y.

**27. Boundary tests.** For real-valued domain D:x+y>=12, choose two distinct exact ON points and a nearby OFF point satisfying the lecture's closed-boundary rule. Which point reveals a closure mutation to x+y>12? Explain what changes if the domain of interest is x+y<12.

**28. Fault classification.** Compare intended x+2y>=10 with faulty 2x+y>=10. Classify the main boundary fault and give an input whose classification differs. Then contrast with changing only >= to > in the intended predicate.

**29. Integration plan.** A calls B and C; C calls E and F. Describe a valid top-down first stage and a valid bottom-up first stage, naming the roles of temporary code. What should be rerun when a real dependency replaces a stub? Does the stub stage prove compatibility with the real dependency?

**30. Categories and reliability.** A system runs for weeks at its anticipated full user load while testers watch for slowdown and leaks. Name the primary category. A separate reliability campaign records operating intervals 40,60,80 h and repair intervals 1,2,3 h. Compute MTTF, MTTR, and MTBF under the lecture convention. What category best describes forcing a redundant module failure to check quick failover?

## Answer key

<details>
<summary>Show answers and worked solutions</summary>

### Multiple-choice key

1. **B.** Statement execution can miss the empty false branch. (L4 pp. 19–22)
2. **C.** The lecture uses a combination-oriented definition, not overall decision toggling alone. (L4 pp. 23–25)
3. **D.** The conditions contradict for unchanged x. (L4 pp. 31–32)
4. **A.** A basis is different from all complete paths, especially with loops. (L4 pp. 35–39)
5. **B.** Correct path with incorrect arithmetic is a computation error. (L5 p. 5)
6. **C.** Equality membership changes at the same boundary. (L5 p. 23)
7. **D.** The OFF point crosses to the nearby excluded side for a closed boundary. (L5 pp. 26–29)
8. **A.** 4>5 is false, and 6>=6 is true. This checks the code instead of repeating the x>4 slide typo. (L5 pp. 10–16)
9. **B.** It replaces a called dependency, regardless of its class name. (L6 pp. 15–24)
10. **C.** Drivers invoke low-level real modules before their callers exist. (L6 pp. 18–19)
11. **D.** This is the specific layered mix given in the slides. (L6 p. 20)
12. **A.** Entry establishes readiness; exit demonstrates completion. (L6 pp. 30–31)
13. **B.** The purpose is overload failure and recovery. (L7 p. 21)
14. **C.** The new build must continue working on supported older platforms. (L7 p. 18)
15. **D.** Integrity protects against unauthorized modification. (L7 p. 12)
16. **A.** This level executes documented procedures. (L7 p. 26)

### Fill-in-the-blank key

17. **2P.** For one component P=1, so the extra term is 2. (L4 p. 35)
18. **False.** The conjunction is already false and the right operand is skipped. (L4 pp. 9–12)
19. **Domain.** The input is routed to the wrong path. (L5 pp. 5, 8)
20. **OFF.** Two distinct ON points plus one nearby OFF point per boundary. (L5 p. 31)
21. **Intra.** Inter-system instead concerns separate systems. (L6 p. 7)
22. **Matrix.** It records supported revision combinations. (L6 p. 28)
23. **Throughput.** Response time instead measures time per request. (L7 p. 19)
24. **123.** Add 120+3 using the stated equation and consistent units. (L7 p. 23)

### Worked solutions

25. **N=7,E=8,V=8-7+2=3.** TT visits A and B; FF skips A and visits C. Together they visit all seven nodes and take both outcomes of D1 and D2, giving full statement and branch coverage in this graph. They do **not** cover all complete paths: TF and FT are missing. Four feasible outcome sequences can coexist with V=3 because V is a basis dimension, not the total path count. (L4 pp. 19–25, 35–39)

26. **x=5,6,7,8.** First condition: x+3>7 implies x>4. The true branch changes y to x+1, so the second condition becomes x+1<10, implying x<9. Combine: 4<x<9. Using the pre-branch y would solve the wrong path state. (L4 pp. 29–33)

27. **One valid choice:** ON A=(2,10), ON B=(10,2), OFF C=(6,5.99). A and B satisfy equality and are included in D; C has sum 11.99 and is just outside. Either exact ON point distinguishes >=12 from >12; C is rejected by both and does not distinguish that closure mutation. For D':x+y<12, A and B remain geometric ON points but are excluded; C is the OFF point inside D'. Integer-only tests would need a representable nearby integer rather than 5.99. (L5 pp. 23, 26–31)

28. **Tilted boundary.** The coefficients change the normal/orientation. Input (0,5) gives intended residual 10, accepted, and faulty residual 5, rejected. Replacing only >= with > is a closure error at the unchanged boundary; an exact equality point such as (0,5) also distinguishes those two operators. (L5 pp. 22–24)

29. **Top-down example:** real A and B with a C stub; later introduce real C with E/F stubs. **Bottom-up example:** a driver calls real E/F and their interactions as needed, before real C is introduced. A stub replaces a callee; a driver supplies a caller. When replacing a stub, test the new real interface, recheck the relevant existing behavior through regression, and retest any fixes. The stub stage validates only the simulated contract, not compatibility with the omitted real dependency. (L6 pp. 13, 15–24)

30. **Load and stability.** The objective is sustained operation at anticipated full load with degradation checks. Reliability campaign: MTTF=(40+60+80)/3=60 h; MTTR=(1+2+3)/3=2 h; MTBF=62 h. Forced redundant-module failure with quick failover primarily tests **robustness/high availability**. These related categories have different primary objectives. (L7 pp. 17, 22–23)

L4–L7 refer to the four full lecture filenames in the study guide. All scenarios and problems are original applications, not copied exam questions.

</details>
