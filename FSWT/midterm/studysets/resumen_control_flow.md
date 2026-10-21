# Control Flow Testing

Source: `L4_Ch4-ControlFlowTesting [Updated].pdf`. Page references count PDF pages from the title slide. Examples and practice problems below are original applications unless explicitly identified as slide examples.

## What the technique targets

Control flow is the order in which statements execute. Assignments update state; conditionals and loops change the next statement to execute. A **program path** is an entry-to-exit sequence. Control flow testing derives tests from program structure: code → control flow graph (CFG) → selected paths → path predicates → concrete input and expected output (pp. 3–7).

A CFG node represents a statement or basic block; a directed edge represents a possible transfer of control. Conditional nodes have outgoing edges for their outcomes. A block groups consecutive statements with no internal branching. State which CFG granularity you use: a compound Boolean expression may be one decision at a high level or several predicate nodes in a detailed short-circuit graph (pp. 8–14).

## Coverage is a criterion, not a correctness proof

| Criterion | What must be exercised | Main limitation |
| --- | --- | --- |
| Statement | Every executable statement at least once | Can miss a decision outcome |
| Branch | Every branch/outcome in the chosen CFG | Does not enumerate all truth combinations or complete paths |
| Predicate, as defined in these slides | All feasible truth combinations of conditions affecting a selected path | May need many combinations; some may be infeasible |
| All-path | Every feasible entry-to-exit path | Loops may create unboundedly many paths |

The lecture's **predicate coverage** means combination coverage, not merely making one overall decision true and false (pp. 23–25). Terminology varies elsewhere; use the definition given in a question. For three independent Booleans there are 8 truth combinations. OR is true in 7 and false in 1; AND is true in 1 and false in 7.

For ordinary reachable code, branch coverage subsumes statement coverage in a consistently modeled CFG. Statement coverage does not imply branch coverage. Example:

```text
x = 0
if a > 0:
    x = 1
return x
```

One test a=1 executes every statement, but never takes the false outcome. Tests a=1 and a=0 cover both outcomes. Both still need expected results; executing a line does not verify its calculation (pp. 19–22).

Two sequential binary decisions have four outcome sequences TT, TF, FT, FF. Tests TT and FF can cover both outcomes of each decision without covering TF or FT. Thus branch coverage does not imply all-path coverage. Correlated decisions can make some sequences infeasible.

Coverage percentages use executed obligations divided by total obligations under the declared model. If 9 of 12 branches execute, branch coverage is 75%. Do not silently remove unreachable obligations to improve a percentage; explain whether a question counts all structural branches or only feasible ones.

## Short-circuit evaluation and side effects

In C/Java-style short-circuit logic, `A && B` skips B when A is false; `A || B` skips B when A is true. If a skipped expression contains an assignment or increment, the side effect does not occur. The `openfiles()` example illustrates why a high-level single-condition CFG can hide important paths (pp. 9–12).

For post-increment `i++`, the expression's value is the old i; the increment then updates i. Starting from i=0, an evaluated `i++` contributes false in a C Boolean context, but i becomes 1. These are C semantics; Java does not permit an integer as a Boolean condition. A truth assignment to an unevaluated operand is not the same as having observed its computation.

## Inputs and symbolic execution

An **input vector** includes all relevant pre-entry state: parameters, globals, files, timers, or external connections. It is not limited to a function's parameter list (p. 27).

A predicate is a Boolean condition at a decision. A path predicate records the required outcome at every decision occurrence along a selected path. Symbolic execution propagates input symbols through assignments to eliminate local variables. The resulting **path predicate expression** is a set or conjunction of constraints over inputs and constants (pp. 26–33).

Example:

```text
y = x + 2
if y > 5:          # desired true
    y = y - 1
if y < 8:          # desired true
    return 1
return 0
```

For the TT path, substitute y=x+2 into the first predicate: x>3. On this path y then becomes x+1, so the second condition is x<7. The path constraints are `3<x<7`. Integer inputs 4,5,6 force this path; x=4 returns 1. A different path to the second decision gives a different interpretation of y.

An infeasible path has contradictory constraints. If x is unchanged, requiring x>0 followed by x<=0 is impossible. No amount of random testing can execute that path. In the lecture's `ReturnAverage` example, a path may require `tv>0` before any qualifying element was counted, reducing to the impossible `0>0` (pp. 31–32).

To test a one-iteration `ReturnAverage` path, one slide input is AS=1, MIN=25, MAX=35, value[0]=30. The selected value is in range and the returned average is 30. Also test zero accepted values, sentinel termination, bounds, and multiple iterations; one path is not the whole routine (pp. 13, 33).

## Cyclomatic complexity

The lecture gives `V(G)=E-N+2P`, where E is edges, N is nodes, and P is connected components. For a single connected routine, P=1. With E=10 and N=8, V=4. For a structured connected single-entry/single-exit graph with d binary decision nodes, `V=d+1`; count decisions consistently with how compound predicates are represented (pp. 35–39).

V counts the dimension of an independent path basis under the CFG convention. It is **not the total number of complete paths**, especially with loops, and it does not guarantee that V feasible tests achieve every coverage criterion. The slide's p. 39 shorthand about “all paths” should be read in the basis-path sense. A graph may have an infeasible basis path.

The lecture's interpretation bands are 1–5 low, 6–10 medium, and >10 high complexity. These are course heuristics, not a proof of defect counts.

## Paper practice and solutions

1. A graph has E=14,N=12,P=1. **V=4.**
2. A suite covers both outcomes of two sequential independent decisions using only TT and FF. Does it cover all paths? **No: TF and FT are missing.**
3. Solve `x+2>5` and `x+1<8` for integer x. **x=4,5,6.**
4. Can full coverage prove the specification is correct? **No. Missing requirements and wrong expected outputs can survive structural coverage.**
