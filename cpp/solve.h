#pragma once
extern "C" {
	/*
		input:

			N: 
			M: 
			edges: x1 y1 c1 x2 y2 c2 ... xM yM cM
			T: num of terminals
			terminals: t1 t2 .. tT

			V: the number of vertices in tree-decomposition
			td: 
				k1 v1 v2 .. v_{k1}
				k2 v1 v2 .. v_{k2}
				:
				kV v1 v2 .. v_{kV}
			
			tedges:
				x1 y1
				x2 y2
				:
				x_{K-1} y_{K-1}

		output:
			return K: the number of edges used
			ans:
				x1 y1 x2 y2 .. x_K y_K
	*/
	int solve(int N, int M, int* edges, int T, int* terminals, int V, int* td, int* tedges, int* ans);
}