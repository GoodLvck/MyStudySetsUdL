# Geometric Objects and Transformations

Sources (PDF page numbers include title slides):
- `03) COT4521, 5520 slides - a library of geom objects (basic objects).pdf`
- `04) COT4521, 5520 slides - a library of geom objects (more objects).pdf`

## Object hierarchy and reusable geometry

The lecture uses an abstract `GeometricObject` base class for shared boundary/interior colors and an abstract `draw(Graphics g)` method. Concrete points, lines, segments, and rectangles supply drawing behavior. An array of `GeometricObject` references can hold different concrete subclasses; calling `draw` selects the implementation of the actual object. A `Vector` represents displacement and is not made a drawable subclass in these examples. A separate GeomLibrary project packages reusable geometry classes (03, pp. 5–10, 20–32).

## Points and vectors

A point represents a location; a vector represents displacement, magnitude, and direction. The displacement from point `p` to point `q` is `q-p`. Different directed segments can represent the same vector if they have equal magnitude and direction (03, pp. 33–44).

For `u=(ux,uy)`, `v=(vx,vy)`:

| Operation | Formula | Interpretation |
| --- | --- | --- |
| Addition | `(ux+vx, uy+vy)` | Combine displacements |
| Negation | `(-ux,-uy)` | Reverse direction |
| Magnitude | `sqrt(ux²+uy²)` | Length |
| Dot product | `ux*vx+uy*vy` | Projection and angle relation |
| 2D cross product | `ux*vy-uy*vx` | Signed scalar determinant |

For nonzero vectors, dot product zero means perpendicular, and cross product zero means parallel. The zero vector needs special care: its direction is not geometrically defined, and it must not be normalized. The identity `u·v = magnitude(u)*magnitude(v)*cos(theta)` connects dot products with projection (03, pp. 62–66).

Use `Math.atan2(y,x)` for direction: it accounts for quadrants and avoids division by x. It returns radians. Convert with `Math.toDegrees` when a degree angle is required; if the result is negative, add 360 degrees (or add `2*pi` before converting). For vector `(0,-2)`, direction is -90 degrees, normalized to 270 degrees.

Distance between points is `sqrt((qx-px)²+(qy-py)²)`. Compare squared distances when only ordering is needed, avoiding square roots without changing which distance is smaller.

## Translation and rotation

Translation by vector `(dx,dy)` maps `(x,y)` to `(x+dx,y+dy)`.

For a mathematical counterclockwise rotation about the origin by angle `theta`:

```text
x' = x*cos(theta) - y*sin(theta)
y' = x*sin(theta) + y*cos(theta)
```

Java trigonometric functions take radians. Preserve both original coordinates before computing both new ones. Reusing an overwritten x in the y formula is a common bug. Rotating `(3,1)` counterclockwise by 90 degrees gives `(-1,3)`; rotation preserves distances and vector lengths (03, pp. 67–76).

To rotate point p about center q: translate by `-q`, rotate about the origin, then translate back by `q`. A point at the rotation center stays fixed. Raw Java screen coordinates invert the visual sense of this mathematical rotation.

## Lines and intersections

The implicit line form is `a*x + b*y + c = 0`, with `(a,b) != (0,0)`. Vector `(a,b)` is a normal; `(-b,a)` is a direction along the line. A vertical line has `b=0`; a horizontal line has `a=0`. The implicit form handles both without an infinite slope (04, pp. 3–12).

Given distinct p and q, one choice is `a=py-qy`, `b=qx-px`, `c=px*qy-qx*py`. Scaling all three coefficients by a nonzero factor represents the same line. Identical points do not determine a unique line.

For two lines define `D=a1*b2-a2*b1`. If D is nonzero:

```text
x = (b1*c2 - b2*c1) / D
y = (c1*a2 - c2*a1) / D
```

If D is zero, the lines are parallel or coincident; distinguish them by checking whether a point on one also lies on the other. Do not divide by zero or claim all parallel normals imply distinct lines (04, pp. 21–22).

A point is on segment AB only if it is on the supporting line **and** in both coordinate bounds of the endpoints. A line intersection can lie outside one or both segments. For floating-point implementations, equality and orientation need a consistent numeric policy; the paper exercises here use exact arithmetic.

## More line operations

Distance from p to `a*x+b*y+c=0` is `abs(a*px+b*py+c)/sqrt(a²+b²)`. For `3x+4y-10=0` and p=(0,0), the distance is 10/5=2. Translating the line by v=(dx,dy) preserves a,b and changes c to `c-a*dx-b*dy` (04, pp. 16–20).

Do not infer “above” from the raw sign of `a*px+b*py+c` without considering b: multiplying all line coefficients by -1 leaves the line unchanged but reverses this sign. For nonvertical lines compare py with `(-a*px-c)/b`; for vertical lines compare px with `-c/a`. The lecture's relative-position method returns -1 below/left, 0 on, and 1 above/right (04, p. 17).

## Rectangles, chains, and polygons

Axis-aligned rectangles use minimum and maximum x and y. Width is `maxX-minX`; height is `maxY-minY`; area is their product. A rectangle represented by two arbitrary opposite corners must normalize bounds before containment checks (04, pp. 41–54).

A polygonal chain is a sequence of connected segments. A simple chain has no self-intersections beyond shared endpoints of adjacent segments. A rectilinear chain uses horizontal and vertical edges. An x-monotone chain has connected intersections with vertical lines; its vertices occur in nondecreasing or nonincreasing x order, permitting ties (04, pp. 55–59).

A simple polygon includes a non-self-intersecting closed boundary and its bounded interior. Vertices must be listed in boundary order, not arbitrarily sorted by x. A convex polygon contains the entire segment between every two of its points. A monotone polygon can have a reflex vertex, so monotonicity alone does not imply convexity. An x-monotone polygon splits into two x-monotone boundary chains (04, pp. 60–65).

## Bounding boxes and Minkowski sums

The smallest axis-aligned bounding rectangle of two rectangles uses the minimum of their min bounds and maximum of their max bounds. It can include points in neither input rectangle; it is not generally their exact union (04, p. 47).

The **Minkowski sum** `A ⊕ B` is the set of all vector sums `a+b` with a in A and b in B. For axis-aligned rectangles, add the interval endpoints independently. If A=[0,2] x [0,1] and B=[3,5] x [-1,2], then A⊕B=[3,7] x [-1,3]. This is not a union or intersection (04, pp. 49–52).

The reflection -A negates every point. If A has x interval [a,b], its reflection has x interval [-b,-a]. A translation vector t makes `(A+t)` intersect B exactly when `t` belongs to `(-A)⊕B`: intersection means `a+t=b`, hence `t=b-a`. This is the configuration-space collision interpretation of the rectangle theorem (04, pp. 53–54).

## Worked checks

For `u=(3,4)`, magnitude is 5. With `v=(-4,3)`, `u·v=-12+12=0`, so the nonzero vectors are perpendicular; `u×v=9+16=25`.

For lines `x+y-5=0` and `x-y-1=0`, solving gives `(3,2)`; substitute into both equations to verify. For points `(2,1)` and `(2,6)`, a supporting line is `x-2=0`, so `a=1,b=0,c=-2`.
