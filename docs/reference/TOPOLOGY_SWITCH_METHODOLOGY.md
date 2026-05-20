# Topology Switch Methodology: Type 1 and Type 2 Migrations

## Notation

Throughout this document we use the following conventions:

- **Vertices of the triangulation** are labeled by a letter indicating cell membership followed by a subscript for the vertex index. For example:
  - $v_1$ = vertex 1, belonging to cell $v$
  - $w_2$ = vertex 2, belonging to cell $w$
  - $x_3$ = vertex 3, belonging to cell $x$

- **Variable Points (VPs)** are contour points that sit on mesh edges. Each VP is parametrized by a scalar $\lambda_{(i,j)} \in [0,1]$ giving its position along the edge between vertex $i$ and vertex $j$:
  $$x_k = \lambda_{(i,j)} \cdot v_i + (1 - \lambda_{(i,j)}) \cdot v_j$$
  When $\lambda_{(i,j)} = 0$, the VP is at vertex $v_j$; when $\lambda_{(i,j)} = 1$, it is at vertex $v_i$.

- **Edge-based labeling**: VPs are stored in a dictionary keyed by sorted vertex-index pairs, e.g., `{(1,2): 0.47, (2,5): 0.53, ...}`. Each edge has at most one VP.

- **Segments**: A contour segment connects two VPs on two different edges of the same mesh triangle. In a standard (non-triple-point) triangle, there is at most one segment.

- **Triple-point triangles**: A mesh triangle whose three vertices belong to three different cells. It has one VP on each of its three edges, forming a **void triangle**. The three sides of the void triangle are direct connections between the three VPs. A **Steiner/Fermat point** is placed inside the void to create a Y-junction with three Steiner arms meeting at 120°. The Steiner arms replace the void triangle sides for perimeter calculation purposes.

- **Void triangle sides vs. Steiner arms**: The void triangle has three sides (VP↔VP direct connections). The Steiner tree has three arms (S→VP). Both structures exist: the void sides define the void geometry, while the Steiner arms define the actual partition boundaries inside the triple-point triangle.

- **1-ring vertex numbering convention**: When analyzing a migration at vertex 1 (the center), the neighbors are numbered starting from 2 in **counterclockwise cyclic order** around the 1-ring. In a Type 2 context, vertex 2 is chosen as the other vertex on the **collapsed edge** (the edge where the Steiner point has collapsed onto a VP). The triangles in the 1-ring are then $T_i = (1, \text{neighbor}_i, \text{neighbor}_{i+1})$, with indices wrapping cyclically. This convention ensures that the triple-point triangle $T$ and the target triangle $T'$ are always adjacent in the fan, sharing the collapsed edge (1,2).

---

## Context: When Are Topology Switches Needed?

After the constrained perimeter optimization (minimizing total boundary length subject to fixed cell areas), some VPs may have moved to positions that indicate the current topological structure is suboptimal:

1. **Type 1 trigger**: A VP's $\lambda$ value approaches 0 or 1, meaning the VP is collapsing toward a mesh vertex.
2. **Type 2 trigger**: The Steiner/Fermat point of a triple-point triangle collapses onto one of the three VPs forming the void triangle (angle at that VP ≥ 120°).

Both types of triggers indicate that the contour topology should change to allow a better local minimum.

---

## Part 1: Type 1 Migration

### 1.1 Detection

After each perimeter optimization run, scan all VPs. For each VP $\lambda_{(i,j)}$:

- If $\lambda_{(i,j)} < \delta$ (approaching vertex $j$): vertex $j$ is a Type 1 candidate.
- If $\lambda_{(i,j)} > 1 - \delta$ (approaching vertex $i$): vertex $i$ is a Type 1 candidate.

where $\delta$ is a threshold (e.g., $\delta \approx 0.01 - 0.05$).

When **multiple VPs** on edges emanating from the same vertex are all collapsing toward that vertex, this is a strong signal that the vertex should flip.

### 1.2 Determining the Target Cell

Suppose vertex 1 is currently in cell $v$, written $v_1$. Multiple VPs on edges from vertex 1 are collapsing toward it. Each such edge connects $v_1$ to a neighbor in a different cell. The vertex should flip to the cell that is "invading" — typically determined by examining which cell dominates among the neighbors, or by the optimization dynamics.

### 1.3 Concrete Example

**Setup — 1-ring of vertex 1:**

| Vertex | Cell |
|--------|------|
| $v_1$  | $v$ (the vertex to flip) |
| $w_2$  | $w$ |
| $w_3$  | $w$ |
| $w_4$  | $w$ |
| $v_5$  | $v$ |
| $v_6$  | $v$ |
| $v_7$  | $v$ |

Six triangles in the 1-ring (cyclically):

| Triangle | Vertices | Cell labels |
|----------|----------|-------------|
| $T_1$ | (1, 2, 3) | ($v$, $w$, $w$) |
| $T_2$ | (1, 3, 4) | ($v$, $w$, $w$) |
| $T_3$ | (1, 4, 5) | ($v$, $w$, $v$) |
| $T_4$ | (1, 5, 6) | ($v$, $v$, $v$) |
| $T_5$ | (1, 6, 7) | ($v$, $v$, $v$) |
| $T_6$ | (1, 7, 2) | ($v$, $v$, $w$) |

### 1.4 Before the Flip: Triangle Classification and Existing VPs

For each triangle, an edge carries a VP if and only if its two endpoints belong to different cells.

**$T_1$ = ($v_1$, $w_2$, $w_3$):** 1 vertex in $v$, 2 in $w$
- Edge (1,2): $v$–$w$ → **has** $\lambda_{(1,2)}$
- Edge (1,3): $v$–$w$ → **has** $\lambda_{(1,3)}$
- Edge (2,3): $w$–$w$ → no VP
- Contour segment: $\lambda_{(1,2)} \longleftrightarrow \lambda_{(1,3)}$

**$T_2$ = ($v_1$, $w_3$, $w_4$):** 1 vertex in $v$, 2 in $w$
- Edge (1,3): $v$–$w$ → **has** $\lambda_{(1,3)}$
- Edge (1,4): $v$–$w$ → **has** $\lambda_{(1,4)}$
- Edge (3,4): $w$–$w$ → no VP
- Contour segment: $\lambda_{(1,3)} \longleftrightarrow \lambda_{(1,4)}$

**$T_3$ = ($v_1$, $w_4$, $v_5$):** 2 vertices in $v$, 1 in $w$
- Edge (1,4): $v$–$w$ → **has** $\lambda_{(1,4)}$
- Edge (4,5): $w$–$v$ → **has** $\lambda_{(4,5)}$
- Edge (1,5): $v$–$v$ → no VP
- Contour segment: $\lambda_{(1,4)} \longleftrightarrow \lambda_{(4,5)}$

**$T_4$ = ($v_1$, $v_5$, $v_6$):** all in $v$ → no contour.

**$T_5$ = ($v_1$, $v_6$, $v_7$):** all in $v$ → no contour.

**$T_6$ = ($v_1$, $v_7$, $w_2$):** 2 vertices in $v$, 1 in $w$
- Edge (7,2): $v$–$w$ → **has** $\lambda_{(7,2)}$
- Edge (1,2): $v$–$w$ → **has** $\lambda_{(1,2)}$
- Edge (1,7): $v$–$v$ → no VP
- Contour segment: $\lambda_{(7,2)} \longleftrightarrow \lambda_{(1,2)}$

**Existing VPs in the 1-ring:**

| VP | Edge | Triangles |
|----|------|-----------|
| $\lambda_{(1,2)}$ | (1,2) | $T_1$ and $T_6$ |
| $\lambda_{(1,3)}$ | (1,3) | $T_1$ and $T_2$ |
| $\lambda_{(1,4)}$ | (1,4) | $T_2$ and $T_3$ |
| $\lambda_{(4,5)}$ | (4,5) | $T_3$ |
| $\lambda_{(7,2)}$ | (7,2) | $T_6$ |

**Contour path through the 1-ring:**
$$\lambda_{(7,2)} \to \lambda_{(1,2)} \to \lambda_{(1,3)} \to \lambda_{(1,4)} \to \lambda_{(4,5)}$$

The contour hugs vertex 1 on the $w$-side, since VPs on edges (1,2), (1,3), (1,4) are close to vertex 1.

### 1.5 Detection Signals

After optimization:
- $\lambda_{(1,2)} \to 0.02$ (near vertex 1)
- $\lambda_{(1,3)} \to 0.03$ (near vertex 1)
- $\lambda_{(1,4)} \to 0.01$ (near vertex 1)

All three VPs on edges from vertex 1 to $w$-neighbors are collapsing toward vertex 1. The optimizer is saying: vertex 1 should be in cell $w$.

### 1.6 Step 1 — Flip the Vertex Label

Change $v_1 \to w_1$. Vertex 1 is now in cell $w$.

New labels: $w_1$, $w_2$, $w_3$, $w_4$, $v_5$, $v_6$, $v_7$.

### 1.7 Step 2 — Reclassify Every Triangle in the 1-Ring

| Triangle | Vertices | New cell labels | Classification |
|----------|----------|-----------------|----------------|
| $T_1$ | (1, 2, 3) | ($w$, $w$, $w$) | all $w$ → no contour |
| $T_2$ | (1, 3, 4) | ($w$, $w$, $w$) | all $w$ → no contour |
| $T_3$ | (1, 4, 5) | ($w$, $w$, $v$) | 2 in $w$, 1 in $v$ → 1 segment |
| $T_4$ | (1, 5, 6) | ($w$, $v$, $v$) | 1 in $w$, 2 in $v$ → 1 segment |
| $T_5$ | (1, 6, 7) | ($w$, $v$, $v$) | 1 in $w$, 2 in $v$ → 1 segment |
| $T_6$ | (1, 7, 2) | ($w$, $v$, $w$) | 2 in $w$, 1 in $v$ → 1 segment |

New VP assignments:

**$T_3$ = ($w_1$, $w_4$, $v_5$):**
- Edge (4,5): $w$–$v$ → **has** $\lambda_{(4,5)}$ (kept)
- Edge (1,5): $w$–$v$ → **has** $\lambda_{(1,5)}$ ← **NEW**
- Edge (1,4): $w$–$w$ → no VP
- Segment: $\lambda_{(4,5)} \longleftrightarrow \lambda_{(1,5)}$

**$T_4$ = ($w_1$, $v_5$, $v_6$):**
- Edge (1,5): $w$–$v$ → **has** $\lambda_{(1,5)}$ ← **NEW**
- Edge (1,6): $w$–$v$ → **has** $\lambda_{(1,6)}$ ← **NEW**
- Edge (5,6): $v$–$v$ → no VP
- Segment: $\lambda_{(1,5)} \longleftrightarrow \lambda_{(1,6)}$

**$T_5$ = ($w_1$, $v_6$, $v_7$):**
- Edge (1,6): $w$–$v$ → **has** $\lambda_{(1,6)}$ ← **NEW**
- Edge (1,7): $w$–$v$ → **has** $\lambda_{(1,7)}$ ← **NEW**
- Edge (6,7): $v$–$v$ → no VP
- Segment: $\lambda_{(1,6)} \longleftrightarrow \lambda_{(1,7)}$

**$T_6$ = ($w_1$, $v_7$, $w_2$):**
- Edge (1,7): $w$–$v$ → **has** $\lambda_{(1,7)}$ ← **NEW**
- Edge (7,2): $v$–$w$ → **has** $\lambda_{(7,2)}$ (kept)
- Edge (1,2): $w$–$w$ → no VP
- Segment: $\lambda_{(1,7)} \longleftrightarrow \lambda_{(7,2)}$

### 1.8 Step 3 — Compare Old vs. New VPs

| Edge | Had VP before? | Needs VP after? | Action |
|------|:--------------:|:---------------:|--------|
| (1,2) | Yes ($\lambda = 0.02$) | No | **Delete** |
| (1,3) | Yes ($\lambda = 0.03$) | No | **Delete** |
| (1,4) | Yes ($\lambda = 0.01$) | No | **Delete** |
| (4,5) | Yes | Yes | **Keep** (preserve $\lambda$ value) |
| (7,2) | Yes | Yes | **Keep** (preserve $\lambda$ value) |
| (1,5) | No | Yes | **Create** (initialize $\lambda = 0.5$) |
| (1,6) | No | Yes | **Create** (initialize $\lambda = 0.5$) |
| (1,7) | No | Yes | **Create** (initialize $\lambda = 0.5$) |

**Key pattern:** VPs on edges from vertex 1 that become same-cell are **destroyed**. VPs on edges from vertex 1 that become different-cell are **created**. The destroyed VPs are effectively "reborn" on new edges — the boundary has jumped to the other side of vertex 1. VPs on edges not involving vertex 1 are **kept** with their optimized $\lambda$ values.

### 1.9 Step 4 — Handling $\lambda$ Values After Optimization

This is a critical detail for convergence:

- **Deleted VPs**: Their $\lambda$ values are discarded. These VPs no longer exist on their old edges.
- **Kept VPs** ($\lambda_{(4,5)}$, $\lambda_{(7,2)}$): Retain their optimized $\lambda$ values from the previous optimization. This is essential — these VPs are far from the switch site and their positions are already near-optimal.
- **Newly created VPs** ($\lambda_{(1,5)}$, $\lambda_{(1,6)}$, $\lambda_{(1,7)}$): Initialize at $\lambda = 0.5$ (midpoint of their edge). This is a neutral starting position that the next optimization round will adjust.

This "warm start" strategy means the optimizer only needs to move the new VPs significantly; the rest of the configuration is already near-optimal.

### 1.10 Step 5 — Rebuild the Ordered Contour

**Before:**
$$\lambda_{(7,2)} \to \lambda_{(1,2)} \to \lambda_{(1,3)} \to \lambda_{(1,4)} \to \lambda_{(4,5)}$$

**After:**
$$\lambda_{(4,5)} \to \lambda_{(1,5)} \to \lambda_{(1,6)} \to \lambda_{(1,7)} \to \lambda_{(7,2)}$$

The contour enters and exits the 1-ring at the same two points ($\lambda_{(4,5)}$ and $\lambda_{(7,2)}$) but now passes on the $v$-side of vertex 1 instead of the $w$-side. The boundary has "jumped over" vertex 1.

### 1.11 Step 6 — Validate: No New Triple-Point Triangles

In a **pure Type 1 migration**, the vertex flips between two cells that are already present in its neighborhood (e.g., from cell $v$ to cell $w$, where $w$-neighbors triggered the flip). After the flip, every triangle in the 1-ring should have vertices in **at most two cells**. If any triangle in the 1-ring ends up with three different cell labels after a pure Type 1 flip, this indicates an error — the wrong target cell was chosen, or the migration was applied incorrectly.

**Important:** This validation applies only to pure Type 1 migrations. In a **Type 2 migration** (Section 2), the vertex flips to the *third* cell (not one of the two on the collapsed boundary). In that case, new triple-point triangles in the 1-ring are **expected** — specifically, the target triangle $T'$ becomes the new triple-point triangle. The construction of the void triangle and Steiner tree for $T'$ is described in Section 2.8.

### 1.12 Step 7 — Rebuild Segments

For each triangle in the 1-ring, reconstruct its contour segments based on the new classification:

- **All-same-cell triangles**: No segments. Remove any old segments.
- **Two-cell triangles**: One segment connecting the two VPs on the boundary edges.
- **Triple-point triangles**: Three Steiner arms from the Fermat point to the three VPs. The three void triangle sides (VP↔VP direct connections) define the void geometry.

Segments in triangles **outside** the 1-ring are unchanged.

### 1.13 Step 8 — Re-Optimize

Run the constrained perimeter optimization again with the updated topology. The new VPs start at $\lambda = 0.5$; the kept VPs start from their previous values. Convergence should be fast since most of the configuration is unchanged.

### 1.14 Step 9 — Iterate

After re-optimization, check for new Type 1 or Type 2 triggers. Repeat the cycle:

$$\text{Optimize} \to \text{Detect triggers} \to \text{Switch topology} \to \text{Optimize} \to \cdots$$

until no more triggers are detected. The configuration has stabilized at a local minimum.

---

## Part 2: Type 2 Migration

Type 2 migration handles the case where a Steiner/Fermat point in a triple-point triangle degenerates. Through careful analysis, **Type 2 reduces to a specific Type 1 migration**, triggered by the Steiner collapse rather than by $\lambda$ thresholds. The Steiner collapse provides two pieces of information that a generic Type 1 does not: *which vertex* to flip and *which cell* to flip it to.

### 2.1 Setup: The Triple-Point Triangle and Its Neighbor

Consider a triple-point triangle:

**$T$ = ($v_1$, $w_2$, $x_3$)**

Three VPs form the void triangle:
- $A = \lambda_{(1,2)}$ on edge (1,2) — boundary between cells $v$ and $w$
- $B = \lambda_{(2,3)}$ on edge (2,3) — boundary between cells $w$ and $x$
- $C = \lambda_{(1,3)}$ on edge (1,3) — boundary between cells $v$ and $x$

The void triangle has three sides (direct VP↔VP connections):
- Side $A \leftrightarrow B$
- Side $B \leftrightarrow C$
- Side $A \leftrightarrow C$

The Fermat point $S$ sits inside the void triangle, and three Steiner arms radiate from it:
- $S \to A$: separates Zone\_$v$ from Zone\_$w$
- $S \to B$: separates Zone\_$w$ from Zone\_$x$
- $S \to C$: separates Zone\_$v$ from Zone\_$x$

Adjacent triangle across edge (1,2):

**$T'$ = ($v_1$, $w_2$, $v_4$)**

$T'$ is a simple two-cell triangle with a contour segment connecting VP $A = \lambda_{(1,2)}$ (on the shared edge) to VP $D = \lambda_{(2,4)}$ on edge (2,4). $D$ is a pre-existing VP on the $w$–$v$ boundary.

### 2.2 The Degeneration

After optimization, the angle at VP $A$ in the void triangle $ABC$ exceeds 120°. The Fermat point $S$ **collapses onto $A$**.

Consequences:
- Steiner arm $S \to A$ has **zero length** ($S = A$)
- The remaining two Steiner arms become $A \to B$ and $A \to C$
- The void triangle side $B \leftrightarrow C$ still exists, but sides $A \leftrightarrow B$ and $A \leftrightarrow C$ are now identical to the Steiner arms
- Three contour lines now meet at point $A$ on edge (1,2):
  1. $A \to C$: the $v$–$x$ boundary (inside $T$)
  2. $A \to B$: the $w$–$x$ boundary (inside $T$)
  3. $A \to D$: the $v$–$w$ boundary (inside $T'$)
- The triple junction is **pinned to edge (1,2)** with only 1 degree of freedom (sliding along the edge) instead of the 2 degrees of freedom it needs

### 2.3 Detection

For each triple-point triangle, after optimization:

1. Compute the Fermat point $S$ of the void triangle $(A, B, C)$
2. Check if $S$ coincides with any of the three VPs (within tolerance $\delta$)
3. Equivalently: check if any angle of the void triangle $\geq 120°$

If the angle at VP $A$ is $\geq 120°$, $S$ has collapsed onto $A$ on edge (1,2).

### 2.4 Identifying Which Vertex to Flip

The VP that $S$ collapsed onto ($A$) sits on edge (1,2). The mesh vertex **closest to $A$** on this edge is the one that should flip. Determine this from the $\lambda$ value:

- If $\lambda_{(1,2)}$ is closer to 1: $A$ is near vertex 1 → flip **vertex 1**
- If $\lambda_{(1,2)}$ is closer to 0: $A$ is near vertex 2 → flip **vertex 2**

In our example, $A$ is near $v_1$, so $v_1$ is the vertex to flip.

**Confirmation:** Other VPs on edges emanating from $v_1$ (including VP $C$ on edge (1,3)) should also be approaching $v_1$ (Type 1 signals), confirming the choice.

### 2.5 Determining the Target Cell

This is where Type 2 differs from a generic Type 1 flip. The target cell is determined by the Steiner collapse:

> **Rule:** When the Steiner point collapses onto the VP on the boundary between cells $a$ and $b$, the vertex flips to cell $c$ — the **third cell**, the one NOT involved in the collapsed boundary.

In our example:
- $S$ collapsed onto $A$ on the $v$–$w$ boundary (edge (1,2))
- The third cell is $x$
- Therefore: $v_1 \to x_1$

**Geometric intuition:** The $v$–$w$ Steiner arm has shrunk to zero, meaning the boundary between cells $v$ and $w$ has degenerated at this point. Cell $x$ is the one "pushing through" — it is expanding and absorbing vertex 1.

### 2.6 Step-by-Step Execution

#### Step 1: Flip the vertex

$v_1 \to x_1$

#### Step 2: Apply the standard Type 1 rebuild in the 1-ring of vertex 1

This is exactly the procedure from Part 1, but applied to a 1-ring where vertex 1 flips to the **third cell** ($x$) rather than to one of the two cells on the collapsed boundary. The 1-ring must be described in full to track all VP and segment changes.

**Full 1-ring of vertex 1 (cyclic order):**

| Vertex | Cell | Position in ring |
|--------|------|------------------|
| $v_1$ | $v$ (center, to be flipped) | center |
| $w_2$ | $w$ | 1st neighbor (shares triple-point triangle $T$) |
| $x_3$ | $x$ | 2nd neighbor |
| $v_4$ | $v$ | 3rd neighbor (shares target triangle $T'$ with $w_2$) |
| $v_5$ | $v$ | 4th neighbor |
| $v_6$ | $v$ | 5th neighbor |
| $x_7$ | $x$ | 6th neighbor (shares edge with $w_2$ in $T_6$) |

**Convention:** Neighbors are numbered 2 through $k$ in counterclockwise cyclic order around the 1-ring, starting from the neighbor that shares the collapsed edge with vertex 1 (i.e., vertex 2 is on the collapsed edge). The triangles $T_1, T_2, \ldots, T_6$ are defined cyclically: $T_i = (1, i+1, i+2)$ where indices wrap cyclically among the neighbors.

**Triangles in the 1-ring:**

| Triangle | Vertices | Cell labels before | Cell labels after ($v_1 \to x_1$) |
|----------|----------|-------------------|-----------------------------------|
| $T_1 = T$ | (1, 2, 3) | ($v$, $w$, $x$) — triple-point | ($x$, $w$, $x$) — two-cell |
| $T_2$ | (1, 3, 7) | ($v$, $x$, $x$) — all $v$-$x$ mix | ($x$, $x$, $x$) — all same |
| $T_3$ | (1, 7, 6) | ($v$, $x$, $v$) — two-cell | ($x$, $x$, $v$) — two-cell |
| $T_4$ | (1, 6, 5) | ($v$, $v$, $v$) — all same | ($x$, $v$, $v$) — two-cell |
| $T_5$ | (1, 5, 4) | ($v$, $v$, $v$) — all same | ($x$, $v$, $v$) — two-cell |
| $T' = T_6$ | (1, 4, 2) | ($v$, $v$, $w$) — two-cell | ($x$, $v$, $w$) — **triple-point** |

**Note on the example 1-ring:** This is a representative configuration. The actual number and arrangement of same-cell vs. different-cell neighbors will vary. The key invariant is: the triple-point triangle $T$ shares the collapsed edge (1,2) with the target triangle $T'$; both triangles include vertex 1 and vertex 2.

**Edges from vertex 1 and VP fate:**

For each edge from $x_1$ (after flip), classify based on the new cell labels:

| Edge | Before (v1) | After (x1) | VP before | VP after | Action |
|------|-------------|------------|-----------|----------|--------|
| (1,2) to $w_2$ | $v$–$w$ → VP A | $x$–$w$ → needs VP | VP A ($v$–$w$) | **Steiner-converted VP** ($x$–$w$) | A is **destroyed** by Type 1; S fills the slot (see Step 2b) |
| (1,3) to $x_3$ | $v$–$x$ → VP C | $x$–$x$ → no VP | VP C ($v$–$x$) | none | **Destroyed** |
| (1,7) to $x_7$ | $v$–$x$ → VP E | $x$–$x$ → no VP | VP E ($v$–$x$) | none | **Destroyed** |
| (1,6) to $v_6$ | $v$–$v$ → no VP | $x$–$v$ → needs VP | none | VP E-reborn ($x$–$v$) | **Created** ($\lambda = 0.5$) |
| (1,5) to $v_5$ | $v$–$v$ → no VP | $x$–$v$ → needs VP | none | VP C-reborn ($x$–$v$) | **Created** ($\lambda = 0.5$) |
| (1,4) to $v_4$ | $v$–$v$ → no VP | $x$–$v$ → needs VP | none | VP A-reborn ($x$–$v$) | **Created** ($\lambda = 0.5$) |

Type 1 balance: 3 destroyed (A, C, E) + 3 created (A-reborn, C-reborn, E-reborn) = **net 0 VPs**.

**VPs not on vertex-1 edges (kept unchanged):**
- VP B on edge (2,3): $w$–$x$ boundary → unchanged
- VP D on edge (2,4): $w$–$v$ boundary → unchanged
- VP G on edge (7,6): $x$–$v$ boundary → unchanged

#### Step 2b: Convert Steiner point S to VP on the collapsed edge

After the Type 1 rebuild, edge (1,2) has different-cell endpoints ($x_1$ and $w_2$) but **no VP** — VP A was destroyed. The old Steiner point $S$, which had collapsed onto $A$'s position on this edge, now becomes a **new VP** on edge (1,2):

- Create a new VP on edge (1,2) representing the $x$–$w$ boundary
- Set $\lambda$ to $A$'s old value (since $S$ was at $A$'s position)
- This VP did not exist as a VP before the migration — it is a **net +1 VP**
- Add to `variable_points` with `active = True`

#### Step 3: Handle $\lambda$ values

| VP category | Example | $\lambda$ handling |
|---|---|---|
| **Steiner-converted VP** (new VP from S on collapsed edge) | New VP on edge (1,2): $x$–$w$ | **Preserve** $\lambda$ from old VP $A$ (since $S = A$) |
| **Destroyed VPs** (edge becomes same-cell) | VP C on edge (1,3): was $v$–$x$, now $x$–$x$ | **Discard** $\lambda$ value |
| **Created VPs** (edge becomes different-cell) | A-reborn on edge (1,4): was $v$–$v$, now $x$–$v$ | **Initialize** at $\lambda = 0.5$ |
| **Kept VPs** (edges not involving vertex 1) | $B$ on (2,3), $D$ on (2,4), $G$ on (7,6) | **Preserve** $\lambda$ value |

### 2.7 What Happens to the Old Triple-Point Triangle $T$

Before: $T$ = ($v_1$, $w_2$, $x_3$) — triple-point triangle with degenerate Steiner tree.

After: $T$ = ($x_1$, $w_2$, $x_3$) — **two cells** ($x$ and $w$).

- Edge (1,2): $x$–$w$ → VP (the **Steiner-converted VP**, preserving $A$'s $\lambda$ value)
- Edge (2,3): $w$–$x$ → VP ($B$, the green VP, **unchanged**)
- Edge (1,3): $x$–$x$ → no VP ($C$ is **destroyed**)

$T$ now has **two VPs** forming a single contour segment:

$$\text{Steiner-converted VP} \longleftrightarrow B$$

The old Steiner tree is gone. The old void triangle sides ($A \leftrightarrow B$, $A \leftrightarrow C$, $B \leftrightarrow C$) are all gone — the void no longer exists. In their place is one simple segment connecting the Steiner-converted VP to $B$.

### 2.8 What Happens to the Target Triangle $T'$

Before: $T'$ = ($v_1$, $w_2$, $v_4$) — two cells ($v$ and $w$), with one contour segment:

$$A \longleftrightarrow D$$

where $A = \lambda_{(1,2)}$ and $D = \lambda_{(2,4)}$.

After: $T'$ = ($x_1$, $w_2$, $v_4$) — **three cells** ($x$, $w$, $v$). This is now a **triple-point triangle**.

Its three VPs are:

| VP | Edge | Boundary | Origin |
|----|------|----------|--------|
| Steiner-converted VP | (1,2) = ($x_1$, $w_2$) | $x$–$w$ | The old Steiner point became this VP. $\lambda$ preserved from old $A$. |
| Migrated VP (Red VP reborn) | (1,4) = ($x_1$, $v_4$) | $x$–$v$ | **Created** by Type 1 rebuild. This edge was $v$–$v$ before, now $x$–$v$. Initialized at $\lambda = 0.5$. |
| VP $D$ (Dark Red VP) | (2,4) = ($w_2$, $v_4$) | $w$–$v$ | **Pre-existing**, unchanged. $\lambda$ preserved. |

These three VPs form the **new void triangle** with three direct sides:
- Steiner-converted VP $\leftrightarrow$ Migrated VP
- Steiner-converted VP $\leftrightarrow$ $D$
- Migrated VP $\leftrightarrow$ $D$

The side **Steiner-converted VP $\leftrightarrow$ $D$** is the old contour segment from $T'$ (which was $A \leftrightarrow D$). The VP on edge (1,2) has changed identity (from $A$ representing $v$–$w$ to the Steiner-converted VP representing $x$–$w$), but the physical position and the segment to $D$ are preserved. This old segment becomes one side of the new void triangle.

The other two sides are **new** — they connect the newly created Migrated VP to the Steiner-converted VP and to $D$.

**Steiner tree construction for $T'$:**
1. Compute the Fermat point $S'$ of the new void triangle
2. Create three Steiner arms: $S' \to$ Steiner-converted VP, $S' \to$ Migrated VP, $S' \to D$
3. The arms meet at 120° angles

### 2.9 What Happens to Destroyed VPs ($C$ and $E$)

VP $C$ was on edge (1,3), the $v$–$x$ boundary. It is **destroyed** because edge (1,3) becomes $x$–$x$ after the flip. VP $E$ was on edge (1,7), the $v$–$x$ boundary. It is **destroyed** because edge (1,7) becomes $x$–$x$ after the flip.

However, the Type 1 rebuild creates **new VPs** on other edges from $x_1$ to $v$-neighbors:
- $C$-reborn is created on edge (1,5) ($x$–$v$ boundary)
- $E$-reborn is created on edge (1,6) ($x$–$v$ boundary)

Critically, these recreated VPs are **NOT** part of the new triple-point triangle $T'$. They end up in **different triangles** in the 1-ring of vertex 1, participating in the contour elsewhere.

### 2.10 Complete Summary of VP Fate

| VP | Edge | Before | After | Fate |
|----|------|--------|-------|------|
| $A$ (Red VP) | (1,2) | $v$–$w$ boundary, Steiner collapsed on top | Position taken by Steiner-converted VP ($x$–$w$) | **Destroyed** from (1,2); boundary migrated to edge (1,4) as A-reborn |
| $B$ (Green VP) | (2,3) | $w$–$x$ boundary, part of void in $T$ | $w$–$x$ boundary, part of simple segment in $T$ | **Kept**, now in simplified $T$ |
| $C$ (Blue VP) | (1,3) | $v$–$x$ boundary, part of void in $T$ | Edge becomes $x$–$x$ | **Destroyed**; recreated as C-reborn on edge (1,5) |
| $D$ (Dark Red VP) | (2,4) | $w$–$v$ boundary in $T'$ | $w$–$v$ boundary in $T'$ | **Kept**, now part of new void in $T'$ |
| $E$ | (1,7) | $v$–$x$ boundary | Edge becomes $x$–$x$ | **Destroyed**; recreated as E-reborn on edge (1,6) |
| $G$ | (7,6) | $x$–$v$ boundary | $x$–$v$ boundary | **Kept**, unchanged |
| Steiner point $S$ | (inside $T$) | Collapsed onto $A$ on edge (1,2) | On edge (1,2) as Steiner-converted VP ($x$–$w$) | **Converted to VP** — this is a **net +1 VP** |
| A-reborn (Migrated VP) | (1,4) | Did not exist (edge was $v$–$v$) | $x$–$v$ boundary | **Created** by Type 1 rebuild, becomes VP of new void in $T'$ |
| C-reborn | (1,5) | Did not exist (edge was $v$–$v$) | $x$–$v$ boundary | **Created** by Type 1 rebuild |
| E-reborn | (1,6) | Did not exist (edge was $v$–$v$) | $x$–$v$ boundary | **Created** by Type 1 rebuild |

**Net VP accounting:**
- Type 1 rebuild: 3 destroyed (A, C, E) + 3 created (A-reborn, C-reborn, E-reborn) = **net 0**
- Steiner-to-VP conversion: +1 (S becomes Steiner-converted VP)
- **Total: +1 VP per Type 2 forward migration**

### 2.11 Complete Summary of Segment Changes

**Segments destroyed in the old triple-point triangle $T$:**
1. Steiner arm $S \to A$ (zero length, degenerate): gone
2. Steiner arm $S \to B$ (= $A \to B$ since $S = A$): gone because $T$ is no longer a triple-point triangle
3. Steiner arm $S \to C$ (= $A \to C$ since $S = A$): gone because $C$ is destroyed and $T$ is simplified
4. Void triangle side $A \leftrightarrow B$: gone
5. Void triangle side $A \leftrightarrow C$: gone
6. Void triangle side $B \leftrightarrow C$: gone

**Segments destroyed elsewhere in the 1-ring:**
7. Any segments involving destroyed VPs (A, C, E) in their neighboring triangles are removed

**Segments created in the simplified $T$:**
1. Simple segment (Steiner-converted VP $\leftrightarrow B$): the new contour through $T$

**Segments created in the new triple-point triangle $T'$:**
2. Void triangle side (Steiner-converted VP $\leftrightarrow$ A-reborn): new
3. Void triangle side (A-reborn $\leftrightarrow D$): new
4. Steiner arm $S' \to$ Steiner-converted VP
5. Steiner arm $S' \to$ A-reborn
6. Steiner arm $S' \to D$

**Segments created elsewhere in the 1-ring:**
7. Segments in triangles that gained new VPs (C-reborn, E-reborn) from the Type 1 rebuild

**Segment preserved (with changed identity):**
The old segment $A \leftrightarrow D$ in $T'$ becomes the void triangle side (Steiner-converted VP $\leftrightarrow D$). The physical segment is the same (same edge endpoints), but the VP on edge (1,2) has changed identity from $A$ to the Steiner-converted VP.

**Net segment accounting:**

Before migration: 9 segments total (6 in $T$ (3 arms + 3 void sides) + 1 in $T'$ + 1 in $T_3$ or $T_6$ + 1 external)
After migration: 10 segments total (1 in $T$ + 6 in $T'$ (3 arms + 3 void sides) + segments in rebuilt triangles from the 1-ring)
**Net: +1 segment per Type 2 forward migration**

The +1 segment is a direct consequence of the +1 VP: the Steiner-converted VP participates in segments that did not exist before.

### 2.12 The Unified Trigger Hierarchy

In practice, Type 1 and Type 2 triggers can coexist. The recommended processing order:

1. **Check all VPs for Type 1 triggers** ($\lambda$ near 0 or 1). Process the most extreme ones first.
2. **Check all triple-point triangles for Type 2 triggers** (Steiner collapse onto a VP). For each:
   - Identify the closest vertex to the collapsed VP using the $\lambda$ value
   - Confirm Type 1 signals are present at that vertex (other VPs approaching it)
   - Determine the target cell using the Type 2 rule: the **third cell**, not on the collapsed boundary
   - Execute as a standard Type 1 flip with the determined target cell
3. **After all switches in a round, re-optimize.**
4. **Repeat until no triggers remain.**

### 2.13 Re-Optimization After Migration

After completing all topology switches in a round:

1. Assemble the new $\lambda$ vector for the optimizer:
   - Kept VPs: previous optimized $\lambda$ values
   - Steiner-converted VPs: previous $\lambda$ values (position preserved)
   - Newly created / migrated VPs: $\lambda = 0.5$
   - Deleted VPs: removed from the vector
2. Rebuild the void triangles and Steiner trees for any new triple-point triangles
3. Rebuild the contour ordering (ordered sequence of VPs and segments per cell)
4. Rebuild area/perimeter data structures for affected triangles
5. Run constrained perimeter optimization
6. Check for new triggers → repeat if necessary

The process terminates when no $\lambda$ values are near 0 or 1 and no Steiner points are degenerate. The configuration has reached a stable local minimum.

---

## Part 3: Type 2 Reversal and the Undo Stack

### 3.1 The Reversal Problem

A Type 2 forward migration creates **+1 VP** and **+1 segment** (the Steiner-converted VP). If, after re-optimization, the triple junction tries to migrate back to the original triangle, we **cannot** simply apply the forward migration mechanics again in reverse. Doing so would create *another* +1 VP and +1 segment, accumulating artifacts with each back-and-forth oscillation.

**The correct approach is snapshot-based rollback:** before each forward migration, we capture a complete local state snapshot. To reverse, we restore the snapshot exactly, which implicitly destroys the extra VP and segment.

### 3.2 What Must Be Captured (LocalStateSnapshot)

For each Type 2 forward migration, before any changes:

1. **Vertex labels** for all vertices in the 1-ring of the flipped vertex
2. **VP data** for all VPs in the affected region (edge → lambda, active flag, cell membership)
3. **Triangle segments** for all triangles in the 1-ring (deep copy)
4. **VP adjacency** for all affected VPs
5. **Steiner infrastructure** (SteinerSnapshot): Steiner point position, the 3 VP indices forming the void, the 3 cell indices, void side pairs, arm endpoints

Additionally, after the forward migration completes, the snapshot records:

6. **`steiner_converted_vp_idx`**: the index of the newly created Steiner-converted VP (the +1 VP). This is essential for rollback — this VP must be deactivated.

### 3.3 The TriplePointHistory Structure

Each triple junction (identified by its three cell indices) maintains a history:

- `triple_id`: frozenset of {cell_a, cell_b, cell_c}
- `visited_triangles`: [T, T', T'', ...] — ordered list of triangles the junction has occupied
- `snapshots`: [snapshot_0, snapshot_1, ...] — one per forward migration
- `flipped_vertices`: [v1, v2, ...] — the vertex flipped at each step

### 3.4 Reversal Detection

A reversal is detected when a Type 2 trigger would move the triple junction to a triangle that appears **earlier** in its `visited_triangles` list:

- $T \to T' \to T$: reversal to position 0 (full reversal, 1 step back)
- $T \to T' \to T'' \to T$: reversal to position 0 (2 steps back)
- $T \to T' \to T'' \to T'$: reversal to position 1 (1 step back)

### 3.5 Reversal Execution

To roll back from the current triangle to a target triangle at position $k$ in the history:

1. Let $n$ = current position (length of `visited_triangles` - 1)
2. For each step $i$ from $n-1$ down to $k$:
   a. **Tear down** the current triple-point infrastructure (Steiner point, arms, void sides) in the current triangle
   b. **Deactivate the Steiner-converted VP** stored in `snapshots[i].steiner_converted_vp_idx` — set `active = False`, remove from `edge_to_varpoint` and `_vp_adjacency`. This is the **-1 VP** that reverses the +1.
   c. **Restore** `snapshots[i]`: write back vertex labels, VP data (including reactivating destroyed VPs like A, C, E), triangle segments, VP adjacency
   d. **Reconstruct** the old Steiner point and arms from `snapshots[i].steiner_data`
3. **Truncate** `visited_triangles`, `snapshots`, and `flipped_vertices` to position $k$
4. **Rebuild** `_active_vp_indices` to reflect the restored VP state
5. **Validate:**
   - The target triangle at position $k$ is a proper triple-point triangle with a valid Steiner tree
   - VP count matches the pre-migration state recorded in the snapshot
   - Segment count matches the pre-migration state

### 3.6 Multi-Step Reversal Example

Consider the migration path: $T \to T' \to T'' \to T$

- Forward $T \to T'$: creates Steiner-conv-VP-1, snapshot_0 records state before this step
- Forward $T' \to T''$: creates Steiner-conv-VP-2, snapshot_1 records state before this step

Reversal to $T$ (2 steps back):
1. Tear down T'', deactivate Steiner-conv-VP-2, restore snapshot_1 → triple point is at $T'$
2. Tear down T', deactivate Steiner-conv-VP-1, restore snapshot_0 → triple point is at $T$
3. Truncate history to [T]

After rollback: both Steiner-converted VPs are inactive, all original VPs are restored, VP and segment counts match the state before any migration.

### 3.7 Why Forward Migration Cannot Be Used for Reversal

If we tried to "migrate back" by applying forward migration mechanics (Type 2 detection → vertex flip → Type 1 rebuild):

- The Steiner-converted VP from the first migration would be treated as a regular VP
- The vertex flip would create/destroy VPs following the standard rules
- A **new** Steiner-converted VP would be created on a different edge
- Net result: +2 VPs instead of 0 (two round-trip migrations = two extra VPs)
- The contour topology would progressively accumulate artifacts

This is why snapshot-based rollback is mandatory for Type 2 reversals.

### 3.8 Type 1 Reversal (Oscillation Prevention)

Type 1 migrations are VP-neutral (3 destroyed + 3 created = net 0). Therefore, applying forward migration mechanics in reverse is safe from a VP-counting perspective. However, to prevent infinite oscillation, a simple history of (vertex, old_cell, new_cell) tuples is maintained. If a migration would exactly reverse a previous one, it is blocked.
