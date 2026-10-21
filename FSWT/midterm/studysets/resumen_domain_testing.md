# Domain Testing

Source: `L5_Ch6-DomainTesting [Updated].pdf`. Page references count PDF pages from the title slide. Examples and practice problems below are original applications unless explicitly identified as slide examples.

## Input classification and faults

A **path domain** is a set of inputs that execute the same program path and hence the same sequence of computations. Numeric results can differ within the domain; “same computation” does not mean “same output value.” Predicates classify inputs into these domains (pp. 3, 5–9, 33).

- **Computation error:** the intended path executes, but its calculated result is wrong.
- **Domain error:** input is assigned to the wrong path/domain, often because a predicate or an assignment feeding a predicate is wrong.

If a discount applies at age>=65 but the implementation uses age>65, age 65 is misclassified: a domain error. If the correct eligible branch executes but subtracts 5% instead of 10%, that is a computation error.

Domain testing explicitly targets boundary-related faults. Control flow testing starts from structural coverage obligations without necessarily hypothesizing a particular boundary defect. The slides use both functional and white-box wording: the important distinction here is the **fault model and input classification**, not a universal label for every boundary test (pp. 3, 9).

## Derive domains by substituting assignments

For the lecture's `codedomain(x,y)` example (pp. 10–16):

```text
c = x+y
if c>5: d = c-x/2
else:   d = c+x/2
P2: d >= c+2
```

P1 has interpretation `x+y>5`. If P1 is true, substitute d=c-x/2 into P2; over real arithmetic this gives x<=-4. If P1 is false, substitute d=c+x/2; P2 becomes x>=4. The same thresholds hold for the displayed integer division semantics that truncate toward zero.

| Path | Input-domain constraints | Example integer input |
| --- | --- | --- |
| TT | x+y>5 and x<=-4 | (-4,10) |
| TF | x+y>5 and x>-4 | (2,6) |
| FT | x+y<=5 and x>=4 | (4,0) |
| FF | x+y<=5 and x<4 | (0,0) |

**Slide clarification:** pp. 13–15 state x>4 for the false-first-path interpretation, but direct substitution into the displayed `>=` predicate includes x=4. The figure on p. 16 also swaps the textual P2 True/False labels beside the vertical boundaries. Use the actual path assignments and relational operators when solving code questions; if asked explicitly to reproduce the slide statement, identify its convention or typo rather than silently changing the code.

Worked check at (4,0): c=4, so P1 is false; integer x/2=2; d=6; P2 compares 6>=6 and is true. Therefore the path is FT, not FF. Exact numeric execution is the best way to audit a symbolic boundary.

## Boundary terminology

The equality associated with a predicate defines the geometric boundary. For x+y>5, the boundary is x+y=5, while the selected side excludes the line.

| Term | Course meaning |
| --- | --- |
| Closed boundary | Equality is included, e.g. <= or >= |
| Open boundary | Equality is excluded, e.g. < or > |
| Closed domain | Every boundary is closed |
| Open domain in the slides | At least one boundary is open |
| Extreme point | An intersection of two or more boundaries |
| Adjacent domains | Domains sharing a boundary constraint |

The lecture's use of “open domain” on p. 20 is course terminology; in mathematical topology a region with some open and some closed boundaries is generally neither open nor closed. For course recall questions use the stated slide definition, while keeping each boundary's inclusion explicit (pp. 17–21).

## Defect classes

**Closure error:** `<` implemented as `<=`, or the reverse. The boundary stays in the same place but membership on equality changes. Test exactly at the boundary.

**Shifted boundary:** a wrong constant displaces a boundary parallel to its intended position. Intended x+y>4, implemented x+y>5: input (2,3) satisfies the intended predicate and fails the implementation. The accepted domain shrinks.

**Tilted boundary:** a coefficient changes, altering orientation. Intended x+0.5y>5, implemented x+y>5: at (3,3), the intended residual is 4.5 and the implemented one is 6, so the decisions differ.

A predicate error can combine effects. Changing a coefficient and the constant may both tilt and shift. The illustration on p. 24 additionally shows **missing** and **extra** boundaries; these remove or introduce constraints and can enlarge or reduce a domain (pp. 22–24).

## ON and OFF points

Use the lecture's domain-relative definitions (pp. 25–31):

| Domain boundary | ON point | OFF point |
| --- | --- | --- |
| Closed | On the boundary, included in this domain | Just outside this domain |
| Open | On the boundary, excluded from this domain | Just inside this domain |

An OFF point should be close, not arbitrarily far away. For real-valued inputs use a small epsilon; for integer-only inputs use the nearest representable values on the needed side. Do not invent 9.99 when the input is restricted to integers.

For D: x+y>=10, choose ON A=(2,8), ON B=(8,2), and OFF C=(5,4.99) for real inputs. C has sum 9.99 and lies just outside D. For integer inputs C=(5,4) is a near-boundary exterior choice. For D': x+y<10, the same exact boundary points are ON but excluded, and C=(5,4.99) is OFF and inside D'. ON/OFF names alone do not tell you membership without the boundary's openness.

The selection rule is **two ON points and one OFF point for each boundary of each domain**, conventionally written A,C,B in ON–OFF–ON order. The two distinct ON points help expose tilt; the near opposite-side point helps reveal shift and closure faults. Keep other domain constraints satisfied so that the test isolates the boundary of interest. Do not claim three points detect every possible nonlinear or combined defect (p. 31).

## Numeric caution in the slide example

For x+7y>=6, (-1,1) is exactly on the boundary. With x=0, y=6/7 is exact as a rational number, though a finite decimal representation is approximate. The rounded point (0,0.858) has 7y=6.006 and lies **inside** the closed domain. Page 27's final “outside” wording contradicts this arithmetic; (0,0.857) instead has 7y=5.999 and lies outside. Always evaluate the predicate after rounding.

## Worked fault-revealing tests

1. Intended integer x<=10, implemented x<10: **x=10** exposes a closure error; x=9 and 11 do not distinguish these predicates.
2. Intended x+y>=10, implemented x+y>=12: **(5,5)** is accepted by the specification and rejected by the faulty boundary.
3. Intended x+2y>=10, implemented 2x+y>=10: **(0,5)** is accepted only by the intended predicate. The coefficient change tilts the boundary.
4. Why not test only middle-of-domain values? **They may remain on the same side of both the correct and faulty boundaries.**
