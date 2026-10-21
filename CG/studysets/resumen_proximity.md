# Closest Pair and Proximity

Sources (PDF page numbers include title slides):
- `05) COT4521, 5520 slides - proximity.pdf`

## Distinguish the problems

**Closest pair:** given a set of n points, find two distinct input points minimizing their mutual Euclidean distance. **Nearest neighbor:** given a query q and a set S, find an element of S closest to q. **Closest point on a segment:** the answer can be an interior point not present as an endpoint. Input and output definitions matter (05, pp. 3–6, 46–51).

## Brute force

Enumerate unordered index pairs `i<j`. Maintain the smallest squared distance and the indices producing it. There are `n(n-1)/2` pairs, each with constant-time distance comparison, so the running time is `Θ(n²)`. Additional storage can be `O(1)` beyond the input and returned pair. If n<2, no pair exists; define how the method reports that situation. Two distinct input records can have identical coordinates, giving distance zero (05, pp. 6–7, 40–44).

For A=(0,0), B=(5,0), C=(2,1), D=(2,2), squared distances are AB=25, AC=5, AD=8, BC=10, BD=13, CD=1. The closest pair is C,D with distance 1. Do not pair a point record with itself.

## Divide and conquer

The efficient planar algorithm (05, pp. 8–18):

1. Sort by x, and maintain a y-sorted sequence. Partition the points into halves of nearly equal cardinality using x order.
2. Recurse to find distances `deltaLeft` and `deltaRight`.
3. Let `delta = min(deltaLeft,deltaRight)`.
4. Search for a closer pair crossing the dividing line.
5. Return the best pair from the left, right, and crossing candidates.

A crossing pair with distance below delta must have both endpoints within delta of the dividing line. Their y coordinates must also differ by less than delta. This restricts candidates to a vertical strip plus a short vertical window.

**A strip alone is not enough:** there can still be Θ(n) strip points. Comparing every strip point to every other strip point would cost quadratic time. Instead exploit y order and the fact that each half has no two points closer than delta. A packing argument bounds the number of candidates in the relevant small rectangle by a constant. The lecture's separate-halves construction illustrates at most six opposite-side candidates in a delta-by-2delta rectangle (p. 14). Other presentations use a different constant; the key fact for complexity is a constant number per point.

Maintain y order across recursion by linear partitioning/merging, rather than sorting every subproblem afresh. Then the combine step is `O(n)`, and `T(n)=2T(n/2)+O(n)=O(n log n)`. Re-sorting by y at each recursion level can produce `O(n log² n)`. Repeated x coordinates should be partitioned by membership in the x-sorted halves, with a stable tie rule, rather than placing all equal-x points on one side.

## Optimality and its model

The lecture gives a matching `Ω(n log n)` lower bound through element uniqueness (05, pp. 19–20). Map each real number ai to `(ai,0)`. A zero closest-pair distance detects duplicate numbers. A hypothetical faster closest-pair algorithm would give a faster solution to element uniqueness after linear conversion. The optimality statement is under the comparison/algebraic computation assumptions supporting that lower bound. Do not interpret it as a claim about every machine model, integer restriction, or randomized algorithm.

## Closest point on segment

For query p and segment endpoints b,e, define `v=e-b`, `w=p-b`:

```text
denominator = v dot v
if denominator == 0: return b
t = (w dot v) / denominator
t = max(0, min(1, t))
c = b + t*v
```

Before clamping, t is the parameter of the orthogonal projection onto the infinite supporting line. If t<=0, return b; if t>=1, return e; otherwise return the interior projection. A zero-length segment needs the explicit denominator check; the example code on p. 50 omits it. These notes add that degenerate-case guard (05, pp. 46–50).

For b=(0,0), e=(4,0), p=(1,3), v=(4,0), w=(1,3), t=4/16=1/4 and c=(1,0), at distance 3. If p=(7,2), t=28/16=1.75 clamps to 1, giving e=(4,0), at distance sqrt(13).

## Polygon boundary proximity

For a set of m segments, find each closest point and keep the nearest candidate: `O(m)`. For a polygon boundary, include the closing edge from the last vertex to the first. A point inside the polygon has distance zero to the filled polygon, but it generally has positive distance to its **boundary**. The question must specify the target set.

## Reasoning checks

- Why can the closest pair cross the split? **The recursive calls see only pairs within a half.**
- Why are only close y values relevant? **Euclidean distance is at least the absolute y difference.**
- Why clamp a projection? **The supporting line extends beyond the finite segment.**
