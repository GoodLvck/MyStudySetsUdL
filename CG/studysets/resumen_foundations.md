# Algorithmic Foundations

Sources (PDF page numbers include title slides):
- `00) COT4521, 5520 slides - what is computational geometry.pdf`
- `01) COT4521, 5520 slides - complexity.pdf`

## What you should be able to do

Define computational geometry, distinguish computability from complexity, classify short algorithms, and justify a running-time bound. Computational geometry studies algorithms and data structures for geometric problems. Applications include graphics, GIS, robotics, computer vision, CAD, and circuit design (00, pp. 3–4).

## The cost model

An algorithm is a finite collection of mechanical steps. Computability asks whether a problem has an algorithm; complexity asks how many resources an algorithm consumes. The course uses the **Real RAM**: a memory cell can contain a primitive value, including a real number, and a primitive operation costs one unit. Arithmetic, comparisons, assignments, branching, and an individual array access have constant cost in this model. A loop or function call is not automatically constant time: count how many operations its execution causes (01, pp. 3–6).

For input size `n`, worst-case running time is the maximum number of executed instructions over all inputs of that size. An algorithm may have a best case faster than its worst case. State which case you analyze. Time and auxiliary space are different measures; a quadratic-time algorithm can use constant additional space.

## Asymptotic notation

For nonnegative functions and sufficiently large `n`:

- `f(n) = O(g(n))`: an upper bound. There are constants `c > 0`, `n0` with `f(n) <= c g(n)` for `n >= n0`.
- `f(n) = Ω(g(n))`: a lower bound. There are constants `c > 0`, `n0` with `f(n) >= c g(n)` for `n >= n0`.
- `f(n) = Θ(g(n))`: a tight bound, both upper and lower.
- `f(n) = o(g(n))`: a strict asymptotic upper bound; the ratio tends to zero in the usual positive-function setting.

An `O(n²)` bound does not prove a tight quadratic bound: a linear function is also `O(n²)`. Likewise, `Ω` does not mean “best case”; it describes a lower bound on the function being considered. The class slides introduce asymptotic notation on pp. 7–10; these explanations make the definitions explicit.

Order from slower to faster growth: `1`, `log n`, `n`, `n log n`, `n²`, `n³`, `2^n`, `n!`. Fixed logarithm bases differ only by a constant factor. Drop constant factors and lower-order terms: `7n² + 12n + 9 = Θ(n²)`.

## Count the work

**One full scan:** `n` constant-time operations gives `Θ(n)`.

**All unordered pairs:** loops `i = 0..n-1`, `j = i+1..n-1` execute `n(n-1)/2` times, so `Θ(n²)`. Do not count `(i,j)` and `(j,i)` twice.

**Repeated halving:** binary search reduces a search interval by a factor of two; it takes `Θ(log n)` steps in the worst case.

**Nested does not always mean quadratic:** an outer `n`-step loop and an inner loop that doubles its counter produce `Θ(n log n)` work. Sequential loops add; nested independent loops multiply.

**Divide and conquer:** `T(n) = 2T(n/2) + Θ(n)` has `Θ(n)` work at each of `Θ(log n)` levels, hence `Θ(n log n)`. If the merge instead costs `Θ(n²)`, the recurrence becomes quadratic. These recurrence exercises apply the foundations to the proximity lecture.

## Geometric algorithm design

Separate input, output, correctness, and resource costs. For closest pair, the input has `n` points but the output has only two point references: constant-size output does not imply constant running time. For range reporting with `k` returned points, writing the output alone costs `Ω(k)`.

Preprocessing pays work once to accelerate repeated queries. Sorting an array costs `O(n log n)`; a later interval query can use binary search and reporting in `O(log n + k)`. If there is only one query, a full scan might be the simpler choice. Always include preprocessing when comparing total workload costs.

## Worked practice

1. A loop checks all unordered pairs for `n = 12`. It performs `12·11/2 = 66` distance comparisons.
2. An algorithm sorts and then scans once: `O(n log n) + O(n) = O(n log n)`.
3. Doubling `n` roughly quadruples the dominant term of a quadratic algorithm. It adds one iteration to a pure binary-halving loop.
4. `T(n) = T(n/2) + Θ(1)` is `Θ(log n)`, while two half-size subproblems with linear merging give `Θ(n log n)`.

## Explain without looking

- Why does a faster implementation of every distance calculation not change the asymptotic order of all-pairs closest pair? **It changes the constant per pair, while the number of pairs remains quadratic.**
- Why must a range query include `k` in its cost? **Every reported point must be written or visited.**
- Is an algorithm optimal just because it is `O(n log n)`? **Only if a matching lower bound applies to the same problem and computational model.**
