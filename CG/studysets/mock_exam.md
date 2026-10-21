# 30-Question Mixed Practice Exam

**Time:** 65 minutes. Use scratch paper. Attempt all questions before opening the answer key. These are original study questions, not the professor's exam. This practice mix has 12 multiple-choice questions, 8 fill-in-the-blank questions, and 10 short-answer/problem questions; the actual exam's type distribution was not specified.

Suggested pacing: 15 minutes for questions 1–12, 10 minutes for 13–20, 35 minutes for 21–30, then 5 minutes to review. For self-scoring, award one point for each correct question; use the worked key to diagnose partial understanding on problems.

Assume exact arithmetic, Cartesian y upward for geometry, and boundary-inclusive containment unless stated otherwise. Java screen-coordinate questions explicitly use y downward.

## Multiple choice — questions 1–12

**1.** A problem asks whether an algorithm exists. Which concept is being investigated?

A) Complexity · B) Computability · C) Convexity · D) Range reporting

**2.** What is the tight bound for `4n² + 100n + 7`?

A) Θ(n) · B) Θ(log n) · C) Θ(n²) · D) Θ(n³)

**3.** What arguments draw a circle centered at screen (50,70), radius 10?

A) `drawOval(50,70,10,10)` · B) `drawOval(40,60,10,10)` · C) `drawOval(50,70,20,20)` · D) `drawOval(40,60,20,20)`

**4.** What should a paint method normally do in the lecture framework?

A) Render stored state · B) Request a new console input every time · C) Sort all course files · D) Shut down after its first call

**5.** For two valid lines, `a1*b2-a2*b1=0` means:

A) Exactly one intersection · B) Parallel or coincident · C) Necessarily perpendicular · D) Both are invalid

**6.** A closest-pair divide-and-conquer algorithm needs a crossing-pair step because:

A) The left pair is always worse · B) The right pair is always better · C) The optimum may have one endpoint in each half · D) Every pair lies on the split

**7.** What keeps the efficient strip-combine step linear?

A) The strip always has at most two points · B) Comparing every strip pair · C) Ignoring y coordinates · D) Y order and a constant candidate bound from packing

**8.** A positive orientation determinant in Cartesian coordinates indicates:

A) Left turn · B) Right turn · C) Collinearity · D) A self-intersection necessarily

**9.** Reversing polygon boundary order:

A) Doubles unsigned area · B) Negates signed area and preserves unsigned area · C) Always gives zero · D) Rotates the polygon

**10.** In the lecture's ray-crossing count, which rule is correct?

A) Include both endpoints · B) Exclude every vertex from all processing · C) Include lower y, exclude upper y, and skip horizontal edges in parity · D) Count every horizontal edge once

**11.** A sorted 1D array range-reporting query returns k elements in:

A) O(1) · B) O(k²) · C) O(log n) independent of k · D) O(log n+k)

**12.** At an x-split node x=4, a rectangle query with x range [6,8] should explore:

A) Right only · B) Left only · C) Both necessarily · D) Neither necessarily

## Fill in the blank — questions 13–20

**13.** Java screen y increases ______.

**14.** The magnitude of vector (5,12) is ______.

**15.** `Math.atan2` returns an angle in ______.

**16.** The scalar cross product of u=(2,1) and v=(1,3) is ______.

**17.** The shoelace sum must be divided by ______ to get signed area.

**18.** A point off the boundary with an odd ray-crossing count is ______ the polygon.

**19.** Even levels in the lecture's 2-D data tree compare coordinate ______.

**20.** In a point-to-segment projection, clamp the parameter to [0,______].

## Problems and explanations — questions 21–30

**21. Pair counting.** An all-pairs closest-pair method processes 20 points. How many unordered distinct-index pairs are evaluated? Give the asymptotic time.

**22. Loop analysis.** An outer loop executes n times; on each iteration an independent counter repeatedly doubles from 1 until it reaches n. Give a tight bound and justify it.

**23. Rotation.** Rotate p=(4,2) counterclockwise by 90 degrees around q=(1,1). Show translation, rotation, and translation back.

**24. Lines.** Find the intersection of `2x+y-7=0` and `x-y-2=0`. Check the result in both equations.

**25. Closest pair.** For A=(0,0), B=(3,4), C=(4,4), D=(10,0), find the closest pair and its distance. List enough squared distances to justify the answer.

**26. Projection.** For segment b=(1,1), e=(5,1), compute the closest point to p=(7,4). Show raw t, clamped t, closest point, and distance.

**27. Polygon area.** Compute the signed and unsigned area of vertices `(1,1),(5,1),(5,4),(1,4)` in the given order. Include the closing edge.

**28. Point location.** For rectangle polygon `(0,0),(6,0),(6,4),(0,4)`, classify p=(2,2), q=(7,2), and r=(3,0). Explain why horizontal edges must still receive a boundary check.

**29. Range query.** In `[1,3,3,6,8,10]`, report [3,8]. Give the lower-bound index L, the first-greater-than-right index U, k, and query complexity. Use zero-based indexing.

**30. Minkowski sum and collision.** For A=[0,2] x [0,1] and B=[4,6] x [3,5], find `(-A)⊕B`. Does t=(3,3) translate A into contact or intersection with B? Justify using both the translation-vector region and the translated rectangle.

## Answer key — open after your attempt

<details>
<summary>Show answers and worked solutions</summary>

### Multiple-choice key

1. **B.** Computability concerns existence; complexity concerns resources. (01, p. 4)
2. **C.** The quadratic term dominates. (01, pp. 7–11)
3. **D.** Top-left is center minus radius; dimensions are diameter. (02, pp. 9–14)
4. **A.** The framework may redraw repeatedly; render state already acquired. (02, pp. 15–32)
5. **B.** A zero normal determinant does not distinguish coincident from distinct parallel lines. (04, pp. 21–22)
6. **C.** Neither recursive half considers a cross-half pair. (05, pp. 8–18)
7. **D.** There can be many strip points; separation inside each half bounds local candidates. (05, pp. 12–17)
8. **A.** Positive determinant means a left turn for y upward. (06, pp. 3–6)
9. **B.** Every directed edge reverses its signed contribution. (06, pp. 12–15)
10. **C.** Handle boundary membership separately before parity. (07, pp. 13–14)
11. **D.** Binary boundary searches plus reporting k results. (08, pp. 10–15)
12. **A.** The query lies strictly to the right of the split. (08, pp. 25–27)

### Fill-in-the-blank key

13. **Down / downward.** (02, p. 3)
14. **13.** sqrt(25+144)=13. (03, p. 43)
15. **Radians.** Convert explicitly if degrees are required. (03, p. 38)
16. **5.** 2*3-1*1=5. (03, pp. 63–66)
17. **2.** The determinant sum is twice signed area. (06, pp. 12–15)
18. **Inside / in the interior of.** Boundary cases were excluded first. (07, pp. 8–14)
19. **x.** Odd levels compare y. (08, p. 22)
20. **1.** A segment corresponds to t between zero and one. (05, pp. 48–50)

### Worked problem key

21. **190 pairs, Θ(n²).** `20*19/2=190`; each pair requires constant-time arithmetic. (05, pp. 6–7)
22. **Θ(n log n).** The inner loop has Θ(log n) doublings, paid on each of n outer iterations. This applies the instruction-count model. (01, pp. 5–11)
23. **(0,4).** Subtract q to obtain (3,1); rotate to (-1,3); add q to obtain (0,4). Rotation preserves distance to q. (03, pp. 67–76)
24. **(3,1).** From x-y=2 obtain y=x-2. Then 2x+x-2=7, so x=3 and y=1. Checks: 6+1-7=0; 3-1-2=0. (04, pp. 21–22)
25. **B,C at distance 1.** Squared distances AB=25, AC=32, AD=100, BC=1, BD=65, CD=52. The minimum is 1. (05, pp. 6–7)
26. **c=(5,1), distance sqrt(13).** v=(4,0), w=(6,3), t=24/16=1.5. Clamp to 1, giving the end point; squared distance is 2²+3²=13. (05, pp. 48–50)
27. **Signed area +12; unsigned area 12.** Edge terms: -4, 15, 16, -3. Sum=24; half is 12. Given order is counterclockwise. (06, pp. 12–15)
28. **p interior; q exterior; r boundary and accepted under the inclusive contract.** The eastward ray from p crosses only the right edge; q has zero eastward crossings. r lies on a horizontal edge, so skipping all processing for horizontal edges would lose a valid boundary hit. Strict-interior containment would reject r. (07, pp. 4, 8–15)
29. **3,3,6,8; L=1, U=5, k=4; O(log n+k).** The output is the half-open index interval [1,5). An arbitrary equality search could miss a duplicate 3. (08, pp. 10–15)
30. **[2,6] x [2,5]; yes.** -A=[-2,0] x [-1,0]; add B's bounds to get [2,6] x [2,5]. t=(3,3) lies inside this region. A+t=[3,5] x [3,4] intersects B in [4,5] x [3,4]. (04, pp. 49–54)

The shorthand 00–08 refers to the full lecture inventory in the study guide. All problem scenarios above are original applications of those lectures.

</details>
