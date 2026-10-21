# Computational Geometry Midterm Study Guide

## Start here

The exam is **Wednesday, October 28, 2026, during class**, in the classroom. According to the announcement supplied by the student, it has **30 questions in 65 minutes**, including conceptual and problem questions. That is about **2 minutes 10 seconds per question** on average. These are original preparation exercises, not actual exam questions or a prediction of the exact question distribution.

Read a topic's notes first, work its quiz without looking at the explanation, and then use the fill-in-the-blank exercises for recall. Rework every calculation on paper. The site opens each topic in study-notes mode; quiz and fill modes retain the existing local progress and shuffle controls.

The professor says coverage includes, but is not limited to, the major concepts from the first half of the semester. The nine supplied lecture PDFs are the basis of this pack. Check any additional class material or announcements too.

## Seven-session preparation plan

| Session | Main work | Paper exercise |
| --- | --- | --- |
| 1 | Algorithmic foundations and drawing tools | Count pairs and loop iterations; derive circle bounding boxes |
| 2 | Points, vectors, transformations | Dot/cross products; rotate around an arbitrary center |
| 3 | Lines, rectangles, polygon types | Solve line intersections; compute a rectangle Minkowski sum |
| 4 | Closest pair and segment projections | Trace the strip argument; compute and clamp projection t |
| 5 | Polygon operations and point location | Shoelace with the closing edge; trace vertex and horizontal-edge cases |
| 6 | Geometric search | Find lower/upper bounds with duplicates; trace 2-D tree pruning |
| 7 | Mixed practice exam | Work 30 questions in 65 minutes, then review the answer key |

Use these as sessions to fit your schedule. Review mistaken answers again after a break; a correct guess is still worth revisiting.

## Formula and complexity reference

All orientation and rotation signs below assume Cartesian y upward.

| Task | Formula or key bound |
| --- | --- |
| Squared distance | `(qx-px)^2 + (qy-py)^2` |
| Dot product | `ux*vx + uy*vy` |
| 2D scalar cross product | `ux*vy - uy*vx` |
| Orientation | `(p1-p0) cross (p2-p0)`; positive left, negative right |
| Counterclockwise rotation | `(x*cos(t)-y*sin(t), x*sin(t)+y*cos(t))` |
| Line normal | `(a,b)` for `a*x+b*y+c=0` |
| Point-to-line distance | `abs(a*px+b*py+c)/sqrt(a*a+b*b)` |
| Translated line | Preserve a,b; set `c'=c-a*dx-b*dy` |
| Two-line determinant | `D=a1*b2-a2*b1`; unique intersection if D is nonzero |
| Intersection x,y | `(b1*c2-b2*c1)/D`, `(c1*a2-c2*a1)/D` |
| Segment projection | `v=e-b`, `w=p-b`, `t=(w dot v)/(v dot v)`; clamp to [0,1] |
| Zero-length segment | Return its endpoint before projection division |
| Signed polygon area | `0.5*sum(xi*yNext-xNext*yi)` |
| Polygon area and boundary proximity | Linear scan: O(n) |
| Ray crossings | Nonhorizontal edges; `lowY <= py < highY` and `px < xHit` |
| Boundary decision | Check segment membership before parity |
| Brute-force closest pair | `n(n-1)/2` pairs; Θ(n²) |
| Efficient closest pair | O(n log n), linear combine using y order and packing |
| 1D reporting | O(log n+k) after preprocessing |
| Balanced planar 2-D reporting | O(sqrt(n)+k), under the course's balanced spatial-tree assumptions |
| Rectangle Minkowski sum | Add x interval endpoints and y interval endpoints |

## Conventions you must keep straight

- Java screen y increases down; mathematical y increases up.
- `drawOval` takes a bounding box, not a center and radius.
- The lecture uses `paint`; the note about Swing's `paintComponent` is supplemental.
- Java trig functions take radians; the Vector direction method converts to degrees.
- The signed shoelace sum records orientation, while the slide's `area()` code takes an absolute value.
- The point-location algorithm includes the boundary; the exercise's “interior” wording requires a different boundary return value.
- The lecture's strip packing picture uses six opposite-side candidates. The asymptotic argument needs a constant candidate count, not a universal constant independent of the formulation.
- The closest-pair optimality lower bound depends on the computation model supporting the element-uniqueness reduction.
- A 2-D tree does not become balanced merely by alternating axes.

## Source coverage

PDF page numbers count from each file's title page. Every interactive question includes its source file and page range; question scenarios and numeric examples are newly written applications of the cited material. Supplemental implementation clarifications are labeled in the notes.

| Set | Lecture files and principal pages | Quiz | Fill |
| --- | --- | ---: | ---: |
| Foundations | 00 pp. 3–4; 01 pp. 3–11 | 18 | 12 |
| Drawing | 02 pp. 3–48; 05 pp. 21–36; 08 pp. 34–38 | 18 | 12 |
| Objects | 03 pp. 5–10, 20–76; 04 pp. 3–65 | 24 | 16 |
| Proximity | 05 pp. 3–20, 40–51 | 18 | 12 |
| Polygon operations | 06 pp. 3–18 | 18 | 12 |
| Point location | 07 pp. 3–15 | 18 | 12 |
| Search | 08 pp. 3–31 | 18 | 12 |
| **Total** | **All nine supplied PDFs** | **132** | **88** |

Lecture file inventory:

1. `00) COT4521, 5520 slides - what is computational geometry.pdf`
2. `01) COT4521, 5520 slides - complexity.pdf`
3. `02) COT4521, 5520 slides - drawing tools in Java.pdf`
4. `03) COT4521, 5520 slides - a library of geom objects (basic objects).pdf`
5. `04) COT4521, 5520 slides - a library of geom objects (more objects).pdf`
6. `05) COT4521, 5520 slides - proximity.pdf`
7. `06) COT4521, 5520 slides - polygon operations.pdf`
8. `07) COT4521, 5520 slides - point location.pdf`
9. `08) COT4521, 5520 slides - geometric search.pdf`

The original lecture PDFs are not redistributed in this repository. References identify the locally supplied course files. Exercises are paraphrased or original applications, rather than copied question banks.

## Exam-day checklist from the announcement

Bring a fully charged laptop and its charger. The exam uses Respondus LockDown Browser on a laptop; phone test-taking is not allowed. The access code will be given in class. Canvas is the only permitted software application, and scratch paper is the only listed resource; it must be destroyed at the end. Turn phones off and keep phones and other electronic devices in your bag or backpack.

Follow the professor's classroom directions: keep at least three feet between walls and chairs; begin seating near the center/front without leaving empty middle tables; follow the stated maximum of 4–6 students per table and placement directions; use the main door for entry and exit. Students at the back may be asked to move forward as others finish. Follow the instructor's instructions on the day for exact seating.

This study website and its notes are for preparation **before** the test, not a permitted exam resource.
