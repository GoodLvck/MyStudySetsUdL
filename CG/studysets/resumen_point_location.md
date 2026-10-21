# Point Location

Sources (PDF page numbers include title slides):
- `07) COT4521, 5520 slides - point location.pdf`

## Define the query precisely

Given polygon Q and query point p, point location decides where p lies: interior, boundary, or exterior. A Boolean containment function needs a contract: does “in” include the boundary? The lecture's general algorithm includes boundary points (pp. 3, 14), while the exercise wording on p. 15 asks for the interior. These notes use **boundary-inclusive containment**, and explain how to implement the strict-interior alternative.

## Axis-aligned rectangle containment

For normalized bounds:

```text
minX <= px <= maxX and minY <= py <= maxY
```

This includes edges and corners and takes `O(1)` time (07, p. 4). For strict interior, use strict inequalities for all four comparisons. A point can satisfy the x interval but still be outside because its y is too large.

For rectangle `[1,5] x [2,6]`, `(1,4)` is on its left edge, `(3,4)` is interior, `(6,4)` is exterior. Boundary-inclusive containment accepts the first two. Normalize opposite corners before applying the test.

## Ray casting: parity

For a simple polygon, imagine a horizontal ray from p toward positive x (east). Each proper boundary crossing toggles inside/outside. After handling boundary points, an **odd** crossing count means inside, and an **even** count means outside. The ray extends to infinity, so do not count intersections west of p. A full ray test scans all edges in `O(n)` time (07, pp. 8–14).

The intuitive reason: sufficiently far to the right you are outside a bounded polygon. Traversing the ray toward p alternates membership whenever it crosses the boundary. Tangential contact at a vertex must not be mistaken for one proper crossing.

## Test boundary first

For edge AB and point p, check collinearity with an orientation determinant, then ensure p is within the inclusive x and y bounds of A and B. For a zero-length edge, membership reduces to equality with that endpoint. If p is on an edge, return true for boundary-inclusive containment, or false for strict interior. This prevents a boundary case from accidentally being decided by parity.

## Avoid double-counting vertices

The lecture's rule is: **lower endpoint included, upper endpoint excluded; horizontal edges skipped in the crossing count** (07, pp. 10–14).

For a nonhorizontal edge, let l be its lower-y endpoint and u its upper-y endpoint. It is eligible only when:

```text
ly <= py < uy
xHit = lx + (py-ly)*(ux-lx)/(uy-ly)
count the crossing only if px < xHit
```

“Lower” refers to smaller numerical y in this formula. If reasoning about mathematical geometry, use the Cartesian convention; Java's screen display does not change the arithmetic predicate. The half-open interval ensures an ordinary vertex crossing is counted once, not twice. At a local y extremum the incident edges contribute either zero or two crossings, preserving even parity for tangency.

Horizontal edges are excluded **only from the parity count**. They remain part of the polygon boundary and must be checked for containment first. The nonhorizontal condition also prevents division by zero in the interpolation formula.

## Worked example: rectangle viewed as a polygon

Let Q have vertices `(0,0),(4,0),(4,3),(0,3)`.

- p=(1,1): right edge intersects the eastward ray at (4,1); the left-edge hit is west of p. One crossing: interior.
- p=(5,1): every boundary intersection is west of p. Zero crossings: exterior.
- p=(4,1): on the boundary. Return the contract's boundary result before counting.
- p=(2,0): on the bottom horizontal edge. Skipping horizontal edges without a boundary check would mishandle this case.
- p=(0,0): a vertex is also on the boundary; handle it directly.

For a nonhorizontal edge `(2,0)` to `(6,4)` and query y=1, `xHit=2+(1/4)*4=3`. A query `(1,1)` counts this crossing; `(4,1)` does not, because its ray begins east of the hit.

## Compact pseudocode

```text
contains(p, vertices):
    inside = false
    for each cyclic edge (a,b):
        if onSegment(a,b,p): return true
        if a.y == b.y: continue
        let l,u be a,b ordered by y
        if l.y <= p.y and p.y < u.y:
            xHit = l.x + (p.y-l.y)*(u.x-l.x)/(u.y-l.y)
            if p.x < xHit: inside = not inside
    return inside
```

The scan takes `Θ(n)` time and `O(1)` extra storage. Testing a point against many polygons by independent scans costs linear time in their total edge count in the worst case. Searching a preprocessed subdivision is a different algorithmic problem.

## What to explain on an exam

Why odd parity? Why skip horizontal edges? Why is the upper endpoint excluded? Which contract accepts a point on an edge? How does `xHit` follow from linear interpolation? Practice these explanations in addition to memorizing the formula.
