# Steiner tree solver
This is an exact solver for Steiner tree problem which was submitted to the Parameterized Algorithms and Computational Experiments Challenge 2018 (https://pacechallenge.wordpress.com/pace-2018/).
Preliminary results against the public data set: [result1.tsv](result1.tsv), [result2.tsv](result2.tsv), [result3.tsv](result3.tsv)

## Required
- Rust (https://www.rust-lang.org/)
- g++

## How to build
The following command creates two binaries `track1` and `track2` at `./target/release/`.
~~~
$ cargo build --release
~~~
`track1` is for inputs with a small number of terminals and `track2` is for inputs with a small tree-width.

## Algorithm

### Track1 (small number of terminals)
Based on the standard O(3^k n + 2^k m log n)-time FPT algorithm by Erickson, Monma, and Veinott [1], where k is the number of terminals, we implemented the following separator-based pruned DP algorithm.
Let A be the set of terminals. For a vertex v and a set I \subseteq A, we compute the minimum weight d[v][I] of a Steiner tree for I+v as follows.
 - Sending step: Pick the smallest unprocessed set I and update d[v][I] for every v by the Dijkstra's algorithm.
 - Splitting step: For every processed set J with I \cap J = \emptyset and for every v, update d[v][I \cup J] \gets min(d[v][I \cup J], d[v][I] + d[v][J]).

Instead of computing d[v][I] for every (v, I) by the above dynamic programming, we skip the computation of d[v][I] which cannot be a part of any optimal solutions by using the following key lemma.

*Let a be the maximum value such that in A-I is connected in the graph G[{v | d[v][I] >= a}]. Then any Steiner tree for A-I must contain some vertex v with d[v][I] <= a. Therefore (v, I) with d[v][I] > a cannot be a part of any optimal solutions.*

After the sending step, we safely delete every (v, I) with d[v][I] > a by the above lemma. We then we apply the splitting step only for the remaining pairs. This significantly reduces the number of pairs we need to consider. We also use several other pruning and speeding-up techniques. We plan to publish the detail later in a paper.

#### References
1. Erickson, R.E., Monma, C.L., Veinott Jr., A.F.: Send-and-split method for minimum-concave-cost network flows. Math. Oper. Res. 12(4), 634–664 (1987)


### Track2 (small tree-width)
We implemented the standard O^*(tw^tw)-time dynamic programming algorithm on a tree-decomposition. This solves all the public instances of tw <= 8 and several instances of tw <= 10.
The track1 solver also works very well for inputs with small tree-width because it can prune many pairs thanks to the existence of small separators.
So in our track2 solver, we use the tree-decomposition DP algorithm only when tw <= 8 or (tw <= 10 and #terminals >= 300), and for the other cases, we use the track1 solver.

### Track3 (heuristic)
We just submitted the track1 solver.


## Authors
- Yoichi Iwata
- Takuto Shigemura
