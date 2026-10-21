# Orientation and Polygon Operations

Sources (PDF page numbers include title slides):
- `06) COT4521, 5520 slides - polygon operations.pdf`

## Orientation is a signed area test

For ordered points p0,p1,p2, compute the determinant:

```text
D = (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0)
```

In mathematical coordinates (positive y upward), D>0 means a **left** turn, D<0 a **right** turn, and D=0 collinearity. The lecture's `turnDirection` returns `1`, `-1`, or `0` respectively. The determinant is twice the triangle's signed area (06, pp. 3–6, 13).

For p0=(1,1), p1=(4,1), p2=(2,3), D=3*2-0*1=6: left turn, signed triangle area 3. Swapping p1 and p2 negates D and reverses the orientation. Collinearity describes the supporting line; it does not establish that p2 lies between p0 and p1. Repeated points make D zero as well.

In raw screen coordinates with y down, the same numerical determinant has the opposite **visual** turn interpretation. State your coordinate convention before interpreting a sign.

## Polygon representation

Store vertices in cyclic boundary order, clockwise or counterclockwise. For n vertices stored without a duplicated first vertex, edge i joins vertex i to vertex `(i+1)%n`. The modulo closes the last-to-first edge. Do not arbitrarily sort polygon vertices by x or angle and assume the same polygon remains (06, pp. 9–11).

A simple polygon has no self-crossing boundary. The usual shoelace area interpretation below assumes the input describes a simple polygon in boundary order. Degenerate inputs with fewer than three distinct, noncollinear vertices do not enclose positive area.

## Shoelace formula

```text
S = sum over i=0..n-1 of (xi*yNext - xNext*yi)
A_signed = S/2
A = abs(S)/2
```

Counterclockwise boundary order gives positive signed area; clockwise gives negative signed area. The lecture discusses signed area on p. 13, but its `area()` code on p. 15 returns the **absolute**, nonnegative area. These are compatible choices for different outputs.

The loop is `Θ(n)` time with `O(1)` auxiliary space. Reversing vertex order negates signed area and preserves unsigned area. A cyclic shift of the starting vertex changes neither. Translation and rotation preserve unsigned area; multiplying all coordinates by a factor s multiplies area by s² (the latter follows directly from the determinant).

## Worked shoelace example

Vertices `(0,0),(4,0),(4,3),(0,3)` in that order:

| Edge | xi*yNext | xNext*yi |
| --- | ---: | ---: |
| (0,0) to (4,0) | 0 | 0 |
| (4,0) to (4,3) | 12 | 0 |
| (4,3) to (0,3) | 12 | 0 |
| (0,3) to (0,0) | 0 | 0 |

S=24, signed area=12, unsigned area=12. Reverse the order to get signed area -12. For triangle `(0,0),(4,0),(0,3)`, S=12, so area=6. Forgetting the factor 1/2 doubles the answer.

The closing edge can contribute a nonzero amount. For triangle `(1,1),(4,1),(1,3)`, edge contributions are -3, 11, and -2. S=6 and area=3. Omitting the final edge incorrectly gives 4.

## Closest point on a polygon boundary

For each edge, project the query onto that finite segment, clamp the parameter to [0,1], and compare squared distances. Keep the point with the smallest distance. Include the closing edge and handle any zero-length edges. This is `O(n)` time for n edges (06, p. 17, using proximity lecture pp. 46–51).

A query at the center of a square has positive distance to the boundary despite being inside the filled square. If several edges give equally close points, any minimizer is valid unless a tie rule is specified. A nearest boundary point can lie in an edge interior, so comparing only vertices is insufficient.

## Rotation of a polygon

Rotate every vertex using the point rotation formulas. An origin rotation applies `(x*cos(t)-y*sin(t), x*sin(t)+y*cos(t))` to each vertex, using original x and y for both results. Preserve boundary order. This takes `Θ(n)` time; it preserves lengths, angles, orientation sign, and area for a nondegenerate polygon (06, p. 18).

## Common mistakes and explanations

- **Wrong sign:** reversing vector order in the cross product flips the turn.
- **Missing last edge:** loop through all n indices with wraparound.
- **Wrong polygon:** use boundary order, not a coordinate sort.
- **Unsigned versus signed:** read the requested output or method implementation.
- **Only checking vertices for proximity:** project onto every segment.

Try deriving the triangle orientation formula from shoelace: expand the three edge terms and regroup into `(p1-p0) cross (p2-p0)`. This connects two exam topics through one determinant.
