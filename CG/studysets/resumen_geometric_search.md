# Geometric Search

Sources (PDF page numbers include title slides):
- `08) COT4521, 5520 slides - geometric search.pdf`

## Search modes and costs

Geometric search includes point location, range reporting, and nearest-neighbor queries. Distinguish **one-time** from **repetitive** queries, and **static** input from **dynamic** input that receives insertions or deletions. Measure query time, preprocessing time, storage, and update time separately (08, pp. 3–6).

Orthogonal range reporting asks for all input points inside an axis-aligned product of intervals. In 1D, the range is `[l,r]`. In 2D it is `[minX,maxX] x [minY,maxY]`. The lecture uses inclusive endpoints. Let k be the number of reported points; writing them takes `Ω(k)` time (08, pp. 7–10).

## 1D sorted array queries

Sort n values once in `O(n log n)` time using `O(n)` storage for the stored values. For each query:

1. Binary search for **lower bound**: the first index L with `value >= l`.
2. Binary search for **upper bound**: the first index U with `value > r`.
3. Report indices L through U-1. If L>=U, report nothing.

The lecture alternatively defines the right index R as the last element <=r, giving R=U-1. Do not confuse these two endpoint conventions. A lower-bound search can return n when all values are smaller than l; the “last <=r” search can return -1 when all are larger than r (08, pp. 10–15).

For `[10,12,16,20,21,22,23,26,29,30]` and `[19,23]`, L=3, U=7, R=6, k=4; report 20,21,22,23. If duplicates are present, lower bound must find the first qualifying element, not just an arbitrary equal value. For `[2,2,5,8]` and `[2,5]`, report both copies of 2 and the 5.

Query cost is `O(log n+k)`. A simple scan costs `O(n)` per query without sorting; preprocessing may not be worthwhile for one query. Count-only queries can return U-L without listing k values. A sorted array can require `O(n)` shifting for insertion, so binary search does not make array updates logarithmic.

## Dynamic 1D data: balanced BST

A BST stores smaller keys in the left subtree and larger keys in the right, following a consistent duplicate policy. A balanced tree such as an AVL tree maintains height `O(log n)`. Range reporting visits relevant search paths and returned nodes, costing `O(log n+k)`; updates cost `O(log n)` with balancing (08, pp. 17–21).

An ordinary unbalanced BST can become a chain under sorted insertions. Its height and worst-case search/update time can become linear. “BST” alone does not guarantee logarithmic performance.

## 2-D trees

The lecture's 2-D tree alternates comparisons by level: the actual data root at level 0 compares x, level 1 compares y, level 2 compares x, and so on. Its sample implementation has a dummy root, so the public method begins at `root.getLeftChild()` with the level labeled “even” (08, pp. 22–29).

Insertion compares the relevant coordinate, goes left for strictly smaller values and right otherwise, then alternates axis. Equal split coordinates follow the right branch in the slide code. They may still have different values in the other coordinate, so equality does not imply identical points.

For rectangle range reporting, first test the node's point against **both** inclusive coordinate bounds. Then decide which subtrees can contain answers:

| Split axis | Visit left when | Visit right when |
| --- | --- | --- |
| x | minX <= node.x | maxX >= node.x |
| y | minY <= node.y | maxY >= node.y |

These are conservative tests that retain equality cases. A node outside the query rectangle can still have descendants inside it. Never prune an entire subtree merely because its root point is outside. Prune using the split axis or the subtree's spatial region.

At an x-split node with x=5 and query x range [6,9], only the right side can contain answers. With x range [2,8], both sides can contain answers. At a y-split node with y=4 and query y range [7,9], only the right side can contain answers.

## Performance assumptions

The lecture gives the following bounds (08, pp. 15, 21, 30–31):

| Structure | Preprocess | Query | Storage | Update |
| --- | --- | --- | --- | --- |
| Sorted 1D array | O(n log n) | O(log n+k) | O(n) | O(n) shifting in a simple array |
| Balanced 1D BST | O(n log n) | O(log n+k) | O(n) | O(log n) |
| Balanced planar 2-D tree | O(n log n) | O(sqrt(n)+k) | O(n) | Slide states O(log n) average-case |

The planar range-query bound assumes an appropriately balanced spatial partition with alternating axes, as in the course analysis. Ordinary incremental insertion does **not** guarantee this structure. Adversarial insertion can produce linear height and linear query work. Also do not turn the slide's average-case update claim into a worst-case guarantee for every insertion/deletion implementation.

## Practice and interpretation

For n=1024 and k=100, the 1D reporting cost includes both a logarithmic boundary search and 100 outputs. For a 2-D query that reports all n points, output alone takes linear time even if the search overhead is `O(sqrt(n))`.

Explain why preprocessing helps repeated queries, why a range may require visiting both subtrees, and why an output-sensitive cost must mention k. These are more useful than memorizing a bound without its assumptions.
