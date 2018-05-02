#![allow(non_snake_case)]

use std::ops::*;
use std::collections::{BinaryHeap, BTreeMap, BTreeSet};
use std::cmp::Ordering;
use std::cell::Cell;

#[macro_use]
pub mod util {
	
	pub struct OnDrop {
		f: Box<Fn()>
	}

	impl OnDrop {
		#[inline]
		pub fn new<F: 'static + Fn()>(f: F) -> OnDrop {
			OnDrop { f: Box::new(f) }
		}
	}

	impl Drop for OnDrop {
		#[inline]
		fn drop(&mut self) {
			(*self.f)();
		}
	}
	
	pub static mut PROFILER: *mut Vec<(&str, &(f64, usize))> = 0 as *mut Vec<_>;
	
	#[macro_export]
	macro_rules! profile {
		($id:ident) => {}
	}
	
	// #[macro_export]
	// macro_rules! profile {
	// 	($id:ident) => {
	// 		static mut __PROF: (f64, usize) = (0.0, 0);
	// 		unsafe {
	// 			if __PROF.1 == 0 {
	// 				if $crate::util::PROFILER.is_null() {
	// 					$crate::util::PROFILER = Box::into_raw(Box::new(Vec::new()));
	// 				}
	// 				(*$crate::util::PROFILER).push((stringify!($id), &__PROF));
	// 			}
	// 			__PROF.1 += 1;
	// 		}
	// 		let t = ::std::time::SystemTime::now();
	// 		let _b = $crate::util::OnDrop::new(move || {
	// 			let d = t.elapsed().unwrap();
	// 			let s = d.as_secs() as f64 + d.subsec_nanos() as f64 * 1e-9;
	// 			unsafe {
	// 				__PROF.0 += s;
	// 			}
	// 		});
	// 	}
	// }

	pub fn write_profile<W>(w: &mut W) where W: ::std::io::Write {
		if unsafe { PROFILER.is_null() } {
			return;
		}
		let mut ps: Vec<_> = unsafe { (*PROFILER).clone() };
		ps.sort_by(|&(_, a), &(_, b)| b.partial_cmp(&a).unwrap());
		let _ = writeln!(w, "########## Profile ##########");
		for (id, &(t, c)) in ps {
			let _ = writeln!(w, "{}:\t{:.3}\t{}", id, t, c);
		}
		let _ = writeln!(w, "#############################");
	}
	
}

pub type W = i32;
pub type G = Vec<Vec<(usize, W)>>;
pub type E = (usize, usize);
pub type TD = (Vec<Vec<usize>>, Vec<(usize, usize)>);

pub const INF: W = std::i32::MAX;

pub trait SetMin {
	fn setmin(&mut self, v: Self) -> bool;
}
impl<T> SetMin for T where T: PartialOrd {
	fn setmin(&mut self, v: T) -> bool {
		*self > v && { *self = v; true }
	}
}
pub trait SetMax {
	fn setmax(&mut self, v: Self) -> bool;
}
impl<T> SetMax for T where T: PartialOrd {
	fn setmax(&mut self, v: T) -> bool {
		*self < v && { *self = v; true }
	}
}

#[derive(Clone, Debug)]
pub struct UnionFind {
	/// size / parent
	ps: Vec<Cell<usize>>,
	pub is_root: Vec<bool>
}

impl UnionFind {
	pub fn new(n: usize) -> UnionFind {
		UnionFind { ps: vec![Cell::new(1); n], is_root: vec![true; n] }
	}
	pub fn find(&self, x: usize) -> usize {
		if self.is_root[x] { x }
		else {
			let p = self.find(self.ps[x].get());
			self.ps[x].set(p);
			p
		}
	}
	pub fn unite(&mut self, x: usize, y: usize) {
		let mut x = self.find(x);
		let mut y = self.find(y);
		if x == y { return }
		if self.ps[x].get() < self.ps[y].get() {
			::std::mem::swap(&mut x, &mut y);
		}
		*self.ps[x].get_mut() += self.ps[y].get();
		self.ps[y].set(x);
		self.is_root[y] = false;
	}
	pub fn same(&self, x: usize, y: usize) -> bool {
		self.find(x) == self.find(y)
	}
	pub fn size(&self, x: usize) -> usize {
		self.ps[self.find(x)].get()
	}
}

#[derive(Clone, Debug)]
pub struct InitVec<T: Clone> {
	version: u32,
	init: T,
	data: Vec<(T, u32)>
}

impl<T: Clone> InitVec<T> {
	pub fn new(v: T, n: usize) -> InitVec<T> {
		InitVec { version: 0, init: v.clone(), data: vec![(v, 0); n] }
	}
	pub fn init(&mut self) {
		if self.version == u32::max_value() {
			for v in &mut self.data {
				v.1 = 0;
			}
			self.version = 1;
		} else {
			self.version += 1;
		}
	}
	#[inline]
	pub fn len(&self) -> usize {
		self.data.len()
	}
}

impl<T: Clone> Index<usize> for InitVec<T> {
	type Output = T;
	#[inline]
	fn index(&self, i: usize) -> &T {
		if self.data[i].1 == self.version {
			&self.data[i].0
		} else {
			&self.init
		}
	}
}

impl<T: Clone> IndexMut<usize> for InitVec<T> {
	#[inline]
	fn index_mut(&mut self, i: usize) -> &mut T {
		if self.data[i].1 != self.version {
			self.data[i].1 = self.version;
			self.data[i].0 = self.init.clone();
		}
		&mut self.data[i].0
	}
}

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub struct BitSet {
	data: Vec<u64>
}

impl BitSet {
	
	pub fn new(n: usize) -> Self {
		BitSet { data: vec![0; (n + 63) / 64] }
	}
	
	#[inline]
	pub fn set(&mut self, i: usize, b: bool) {
		if b {
			self.data[i / 64] |= 1 << (i & 63)
		} else {
			self.data[i / 64] &= !(1 << (i & 63))
		}
	}
	
	pub fn intersect(&self, other: &Self) -> bool {
		for i in 0..self.data.len() {
			if self.data[i] & other.data[i] != 0 {
				return true;
			}
		}
		false
	}
	
	pub fn count(&self) -> usize {
		let mut c = 0;
		for &a in &self.data {
			c += u64::count_ones(a);
		}
		c as usize
	}
	
}

impl Index<usize> for BitSet {
	type Output = bool;
	#[inline]
	fn index(&self, i: usize) -> &bool {
		if self.data[i / 64] >> (i & 63) & 1 == 0 {
			&false
		} else {
			&true
		}
	}
}

impl<'a> BitAnd for &'a BitSet {
	type Output = BitSet;
	fn bitand(self, other: &BitSet) -> BitSet {
		BitSet { data: self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a & b).collect() }
	}
}

impl<'a> BitOr for &'a BitSet {
	type Output = BitSet;
	fn bitor(self, other: &BitSet) -> BitSet {
		BitSet { data: self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a | b).collect() }
	}
}

impl<'a> BitXor for &'a BitSet {
	type Output = BitSet;
	fn bitxor(self, other: &BitSet) -> BitSet {
		BitSet { data: self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a ^ b).collect() }
	}
}


#[derive(Debug)]
enum Modify {
	/// (u, v, w): v with N(v)={u,w} is contracted to u.
	Contract(usize, usize, usize),
	/// (u, v, Nv): uv is included and v is contracted to u. Nv is the set of neighbors w of v such that c(vw)<c(uw).
	Include(usize, usize, Vec<usize>)
}

pub struct Reduction {
	orig_n: usize,
	t0: usize,
	pub g: G,
	pub ts: Vec<usize>,
	pub td: Option<TD>,
	ids: Vec<usize>,
	w0: W,
	modify: Vec<Modify>
}

impl Reduction {
	
	fn size(g: &Vec<BTreeMap<usize, W>>, is_t: &Vec<bool>) -> (usize, usize, usize) {
		let mut n = 0;
		let mut m = 0;
		let mut t = 0;
		for (i, a) in g.iter().enumerate() {
			if a.len() > 0 {
				n += 1;
				m += a.len();
				if is_t[i] {
					t += 1;
				}
			}
		}
		(n, m / 2, t)
	}
	
	fn get_ts(g: &Vec<BTreeMap<usize, W>>, is_t: &Vec<bool>) -> (Vec<usize>, Vec<usize>) {
		let n = g.len();
		let mut ts = vec![];
		let mut id = vec![!0; n];
		for u in 0..n {
			if is_t[u] && g[u].len() > 0 {
				id[u] = ts.len();
				ts.push(u);
			}
		}
		(ts, id)
	}
	
	// Include all the edges of weight zero.
	fn weight0(g: &mut Vec<BTreeMap<usize, W>>, is_t: &mut Vec<bool>, modify: &mut Vec<Modify>) -> bool {
		let n = g.len();
		let mut modified = false;
		for u in 0..n {
			if let Some((&v, _)) = g[u].iter().find(|&(_, &w)| w == 0) {
				is_t[v] |= is_t[u];
				is_t[u] = false;
				let mut xs = vec![];
				for (x, w) in std::mem::replace(&mut g[u], BTreeMap::new()) {
					g[x].remove(&u);
					if x != v && g[v].entry(x).or_insert(INF).setmin(w) {
						g[x].insert(v, w);
						xs.push(x);
					}
				}
				modify.push(Modify::Include(v, u, xs));
				modified = true;
			}
		}
		modified
	}
	
	// Lightweight reduction which runs in linear time.
	fn light(g: &mut Vec<BTreeMap<usize, W>>, is_t: &mut Vec<bool>, modify: &mut Vec<Modify>, w0: &mut W) -> bool {
		profile!(light);
		let n = g.len();
		let mut modified = false;
		let mut t_count = is_t.iter().filter(|b| **b).count();
		for u in 0..n {
			if t_count <= 1 {
				break;
			}
			if g[u].len() == 0 {
			} else if g[u].len() == 1 {
				// When terminal u has degree one, we can include the incident edge.
				// When non-terminal u has degree one, we can remove it.
				let fs = std::mem::replace(&mut g[u], BTreeMap::new());
				for (v, w) in fs {
					if is_t[u] {
						if is_t[v] {
							t_count -= 1;
						}
						is_t[u] = false;
						is_t[v] = true;
						modify.push(Modify::Include(v, u, vec![]));
						*w0 += w;
					}
					g[v].remove(&u);
					modified = true;
				}
			} else if g[u].len() == 2 && !is_t[u] {
				// When non-terminal u has degree two, we can contract it.
				let fs: Vec<_> = std::mem::replace(&mut g[u], BTreeMap::new()).into_iter().collect();
				let (v1, w1) = fs[0];
				let (v2, w2) = fs[1];
				g[v1].remove(&u);
				g[v2].remove(&u);
				modified = true;
				if g[v1].entry(v2).or_insert(INF).setmin(w1 + w2) {
					g[v2].insert(v1, w1 + w2);
					modify.push(Modify::Contract(v1, u, v2));
				}
			} else if is_t[u] {
				// When the edge uv connecting two terminal u and v has the smallest weight among all the edges incident to u, we can include it.
				let min_w = g[u].iter().map(|(_, &w)| w).min().unwrap();
				let mut v = !0;
				for (&a, &w) in &g[u] {
					if w == min_w && is_t[a] {
						v = a;
						break;
					}
				}
				if v != !0 {
					t_count -= 1;
					is_t[u] = false;
					*w0 += min_w;
					let mut xs = vec![];
					for (x, w) in std::mem::replace(&mut g[u], BTreeMap::new()) {
						g[x].remove(&u);
						if x != v && g[v].entry(x).or_insert(INF).setmin(w) {
							g[x].insert(v, w);
							xs.push(x);
						}
					}
					modify.push(Modify::Include(v, u, xs));
					modified = true;
				}
			}
		}
		modified
	}
	
	// s(u,v) := bottleneck distance from u to v in the distance graph on A+u+v.
	// When s(u,v)<c(uv), we can delete uv.
	fn sd(g: &mut Vec<BTreeMap<usize, W>>, is_t: &Vec<bool>) -> bool {
		profile!(sd);
		let mut del = vec![];
		{
			let n = g.len();
			let (ts, _) = Reduction::get_ts(g, is_t);
			let mut tid = vec![!0; n];
			for t in 0..ts.len() {
				tid[ts[t]] = t;
			}
			// Perturbation for tie-breaking.
			let g: Vec<Vec<_>> = (0..n).map(|u| g[u].iter().map(|(&v, &w)| (v, ((w as i64) << 32) - if is_t[u] { 2 } else { 1 } - if is_t[v] { 2 } else { 1 })).collect()).collect();
			let mut max_w = 0;
			for u in 0..n {
				for &(_, w) in &g[u] {
					max_w.setmax(w);
				}
			}
			let mut dist_t = vec![vec![std::i64::MAX; n]; ts.len()];
			let mut gt = vec![vec![]; ts.len()];
			let mut que = BinaryHeap::new();
			{
				profile!(dist_t);
				for i in 0..ts.len() {
					let dist = &mut dist_t[i];
					dist[ts[i]] = 0;
					que.push((0, ts[i]));
					while let Some((d, u)) = que.pop() {
						let d = -d;
						if dist[u] != d {
							continue;
						}
						for &(v, w) in &g[u] {
							let d2 = d + w;
							if d2 < max_w && dist[v].setmin(d2) {
								que.push((-d2, v));
							}
						}
					}
					for &t in &ts {
						if t == ts[i] {
							continue;
						}
						if dist[t] < max_w {
							gt[i].push((dist[t], t));
						}
						gt[i].sort();
					}
				}
			}
			let mut dist = InitVec::new(std::i64::MAX, n);
			for s in 0..n {
				if g[s].len() == 0 {
					continue;
				}
				let mut d_max = std::i64::MIN;
				let mut adj = vec![];
				for &(v, w) in &g[s] {
					if s < v {
						d_max.setmax(w);
						adj.push(v);
					}
				}
				if adj.len() == 0 {
					continue;
				}
				dist.init();
				dist[s] = 0;
				que.push((0, s));
				while let Some((d, u)) = que.pop() {
					let d = -d;
					if d != dist[u] {
						continue;
					}
					let i = tid[u];
					if i != !0 {
						for &v in &adj {
							dist[v].setmin(d.max(dist_t[i][v]));
						}
						for &(w, t) in &gt[i] {
							if w >= d_max {
								break;
							}
							let d2 = d.max(w);
							if dist[t].setmin(d2) {
								que.push((-d2, t));
							}
						}
					} else {
						for &(v, w) in &g[u] {
							let d2 = d + w;
							if d2 < d_max && dist[v].setmin(d2) {
								que.push((-d2, v));
							}
						}
					}
				}
				for &(v, w) in &g[s] {
					if s < v && w > dist[v] {
						del.push((s, v));
					}
				}
			}
		}
		for &(u, v) in &del {
			g[u].remove(&v);
			g[v].remove(&u);
		}
		del.len() > 0
	}
	
	// Let T be a minimum spanning tree.
	// By removing an edge uv\in T, we split T into T_u and T_v, and terminals A into A_u and A_v.
	// If d(u, A_u+x) + c(uv) + d(v, A_v+y) <= c(xy) holds for every chord xy between T_u and T_v, we can include uv.
	fn nsc(g: &mut Vec<BTreeMap<usize, W>>, is_t: &mut Vec<bool>, modify: &mut Vec<Modify>, w0: &mut W) -> bool {
		profile!(nsc);
		let n = g.len();
		let mut fs = vec![];
		for u in 0..n {
			for (&v, &w) in &g[u] {
				if u < v {
					fs.push((w, if is_t[u] { 0 } else { 1 } + if is_t[v] { 0 } else { 1 }, u, v));
				}
			}
		}
		fs.sort();
		let mut uf = UnionFind::new(n);
		let mut mst = vec![vec![]; n];
		for (w, _, u, v) in fs {
			if !uf.same(u, v) {
				uf.unite(u, v);
				mst[u].push((v, w));
				mst[v].push((u, w));
			}
		}
		let mut parent = vec![!0; n];
		let mut depth = vec![!0; n];
		let mut r = !0;
		for u in 0..n {
			if mst[u].len() > 0 {
				r = u;
				break;
			}
		}
		if r == !0 {
			return false;
		}
		let mut root_to_leaf = vec![];
		let mut p = 0;
		root_to_leaf.push(r);
		depth[r] = 0;
		while p < root_to_leaf.len() {
			let u = root_to_leaf[p];
			p += 1;
			for &(v, w) in &mst[u] {
				if depth[v] == !0 {
					depth[v] = depth[u] + w;
					parent[v] = u;
					root_to_leaf.push(v);
				}
			}
		}
		let (ts, id) = Reduction::get_ts(g, is_t);
		let mut dist_t = vec![vec![INF; n]; ts.len()];
		for i in 0..ts.len() {
			let mut que = BinaryHeap::new();
			let dist = &mut dist_t[i];
			dist[ts[i]] = 0;
			que.push((0, ts[i]));
			while let Some((d, u)) = que.pop() {
				let d = -d;
				if dist[u] != d {
					continue;
				}
				for (&v, &w) in &g[u] {
					let d2 = d + w;
					if dist[v].setmin(d2) {
						que.push((-d2, v));
					}
				}
			}
		}
		let mut ts_below = vec![vec![]; n];
		for &u in root_to_leaf.iter().rev() {
			if is_t[u] {
				ts_below[u].push(id[u]);
			}
			let v = parent[u];
			if v != !0 {
				let a = ts_below[u].clone();
				ts_below[v].extend(a);
			}
		}
		let mut min_below = vec![INF; n];
		let mut min_above = vec![INF; n];
		let mut is_below = InitVec::new(false, ts.len());
		for &u in &root_to_leaf {
			is_below.init();
			for &b in &ts_below[u] {
				is_below[b] = true;
			}
			for i in 0..ts.len() {
				if is_below[i] {
					min_below[u].setmin(dist_t[i][u]);
				} else {
					min_above[u].setmin(dist_t[i][parent[u]]);
				}
			}
		}
		let mut ok: Vec<_> = (0..n).map(|u| parent[u] != !0 && min_below[u] < INF && min_above[u] < INF).collect();
		for u in 0..n {
			for (&v, &w) in &g[u] {
				if v == parent[u] || u == parent[v] || u < v {
					continue;
				}
				let mut x = u;
				let mut y = v;
				while x != y {
					if depth[x] < depth[y] {
						y = parent[y];
					} else {
						x = parent[x];
					}
				}
				let lca = x;
				let mut x = u;
				let mut y = v;
				while x != y {
					if depth[x] < depth[y] {
						if ok[y] {
							let c = depth[y] - depth[parent[y]];
							let d_above = depth[u] - depth[lca] + depth[parent[y]] - depth[lca];
							let d_below = depth[v] - depth[y];
							if d_above.min(min_above[y]) + d_below.min(min_below[y]) + c > w {
								ok[y] = false;
							}
						}
						y = parent[y];
					} else {
						if ok[x] {
							let c = depth[x] - depth[parent[x]];
							let d_above = depth[v] - depth[lca] + depth[parent[x]] - depth[lca];
							let d_below = depth[u] - depth[x];
							if d_above.min(min_above[x]) + d_below.min(min_below[x]) + c > w {
								ok[x] = false;
							}
						}
						x = parent[x];
					}
				}
			}
		}
		let mut modified = false;
		for &u in root_to_leaf.iter().rev() {
			if ok[u] {
				modified = true;
				let v = parent[u];
				let w = depth[u] - depth[v];
				is_t[v] |= is_t[u];
				is_t[u] = false;
				let mut xs = vec![];
				for (x, w) in std::mem::replace(&mut g[u], BTreeMap::new()) {
					g[x].remove(&u);
					if x != v && g[v].entry(x).or_insert(INF).setmin(w) {
						g[x].insert(v, w);
						xs.push(x);
					}
				}
				modify.push(Modify::Include(v, u, xs));
				*w0 += w;
			}
		}
		modified
	}
	
	fn deg3(g: &mut Vec<BTreeMap<usize, W>>, is_t: &Vec<bool>) -> bool {
		profile!(deg3);
		let n = g.len();
		let mut modified = false;
		let mut dist = InitVec::new(INF, n);
		// Compute d(s,t) on G-deleted, or return ub if the distance is at least ub.
		let mut dist = |g: &Vec<BTreeMap<usize, W>>, deleted: &InitVec<bool>, s: &[usize], t: usize, ub: W| -> W {
			dist.init();
			let mut que = BinaryHeap::new();
			for &u in s {
				dist[u] = 0;
				que.push((0, u));
			}
			while let Some((d, u)) = que.pop() {
				let d = -d;
				if u == t {
					return d;
				}
				if dist[u] != d {
					continue;
				}
				for (&v, &w) in &g[u] {
					let d2 = d + w;
					if d2 < ub && !deleted[v] && dist[v].setmin(d2) {
						que.push((-d2, v));
					}
				}
			}
			ub
		};
		let mut deleted = InitVec::new(false, n);
		for u in 0..n {
			if g[u].len() == 3 && !is_t[u] {
				let gu: Vec<_> = g[u].iter().map(|(&u, &w)| (u, w)).collect();
				for i in 0..3 {
					let (mut p, mut up) = gu[i];
					let (x, ux) = gu[(i + 1) % 3];
					let (y, uy) = gu[(i + 2) % 3];
					deleted.init();
					deleted[u] = true;
					let xp = dist(g, &deleted, &[x], p, ux + up + 1);
					if xp > ux + up {
						continue;
					}
					let yp = dist(g, &deleted, &[y], p, uy + up + 1);
					if yp > uy + up {
						continue;
					}
					// u has three neighbors {p, x, y} and there are shortest paths from p to x and p to y both avoiding u.
					// Let OPT be the set of edges contained in every optimal solution.
					// Suppose that up is in OPT. Then ux and uy are also in OPT.
					let xy = dist(g, &deleted, &[x], y, ux + uy);
					if xp + yp + xy - xp.max(yp).max(xy) <= ux + uy + up {
						// Instead of using {up, ux, uy}, we can use MST of {p, x, y}.
						modified = true;
						g[u].remove(&p);
						g[p].remove(&u);
						break;
					} else {
						let mut del = false;
						loop {
							if dist(g, &deleted, &[x, y], p, up + 1) <= up {
								// If there exists a path from {x,y} to p of length at most up, we can replace up by the path.
								del = true;
								break;
							}
							if is_t[p] {
								break;
							}
							// If p is not a terminal, p must have degree at least two in OPT.
							deleted[p] = true;
							let mut ps = vec![];
							for (&q, &pq) in &g[p] {
								// If there exists a path from {x, y} to q of length at most max(up, pq), we can replace tp or pq by the path.
								if ps.len() < 2 && !deleted[q] && dist(g, &deleted, &[x, y], q, up.max(pq) + 1) > up.max(pq) {
									ps.push((q, pq));
								}
							}
							if ps.len() > 1 {
								break;
							} else if ps.len() == 1 {
								// If there is only one candidate q, p has degree two in OPT.
								// We repeat the process by setting p<-q.
								p = ps[0].0;
								up += ps[0].1;
							} else {
								del = true;
								break;
							}
						}
						if del {
							modified = true;
							let p = gu[i].0;
							g[u].remove(&p);
							g[p].remove(&u);
							break;
						}
					}
				}
			}
		}
		modified
	}
	
	pub fn reduce(g: &G, ts: &Vec<usize>, td: &Option<TD>) -> Self {
		let n = g.len();
		let t0 = ts[0];
		let mut g2 = vec![BTreeMap::new(); n];
		let mut is_t = vec![false; n];
		for &t in ts {
			is_t[t] = true;
		}
		for u in 0..n {
			for &(v, w) in &g[u] {
				g2[u].entry(v).or_insert(INF).setmin(w);
			}
		}
		let mut g = g2;
		let mut modify = vec![];
		let mut w0 = 0;
		{
			let (n, m, t) = Reduction::size(&g, &is_t);
			if Reduction::weight0(&mut g, &mut is_t, &mut modify) {
				let (n2, m2, t2) = Reduction::size(&g, &is_t);
				eprintln!("weight0: ({}, {}, {}) -> ({}, {}, {})", n, m, t, n2, m2, t2);
			}
		}
		loop {
			let (n, m, t) = Reduction::size(&g, &is_t);
			// print_input(&g.iter().map(|a| a.iter().map(|(&u, &w)| (u, w)).collect()).collect(), &(0..is_t.len()).filter(|&i| is_t[i]).collect());
			if t <= 1 {
				eprintln!("reduce: ({}, {}, {}) -> ({}, {}, {})", n, m, t, 0, 0, 0);
				break;
			}
			if Reduction::light(&mut g, &mut is_t, &mut modify, &mut w0) {
				let (n2, m2, t2) = Reduction::size(&g, &is_t);
				eprintln!("light: ({}, {}, {}) -> ({}, {}, {})", n, m, t, n2, m2, t2);
				continue;
			}
			if Reduction::sd(&mut g, &is_t) {
				let (n2, m2, t2) = Reduction::size(&g, &is_t);
				eprintln!("sd: ({}, {}, {}) -> ({}, {}, {})", n, m, t, n2, m2, t2);
				continue;
			}
			if Reduction::nsc(&mut g, &mut is_t, &mut modify, &mut w0) {
				let (n2, m2, t2) = Reduction::size(&g, &is_t);
				eprintln!("nsc: ({}, {}, {}) -> ({}, {}, {})", n, m, t, n2, m2, t2);
				continue;
			}
			if Reduction::deg3(&mut g, &is_t) {
				let (n2, m2, t2) = Reduction::size(&g, &is_t);
				eprintln!("deg3: ({}, {}, {}) -> ({}, {}, {})", n, m, t, n2, m2, t2);
				continue;
			}
			break;
		}
		let mut ids = vec![];
		let mut name = vec![!0; n];
		for u in 0..n {
			if g[u].len() > 0 {
				name[u] = ids.len();
				ids.push(u);
			}
		}
		let mut ts = vec![];
		for t in 0..n {
			if is_t[t] && name[t] != !0 {
				ts.push(name[t]);
			}
		}
		let mut g2 = vec![vec![]; ids.len()];
		for u in 0..ids.len() {
			for (&v, &w) in &g[ids[u]] {
				g2[u].push((name[v], w));
			}
		}
		for u in 0..n {
			for (&v, &w) in &g[u] {
				assert!(g[v][&u] == w);
			}
		}
		let td = if let Some(ref td) = *td {
			for m in modify.iter().rev() {
				match *m {
					Modify::Contract(u, v, _) | Modify::Include(u, v, _) => {
						name[v] = name[u];
					},
				}
			}
			let bs: Vec<_> = td.0.iter().map(|bag| {
				let mut vs = vec![];
				for &v in bag {
					if name[v] != !0 {
						vs.push(name[v]);
					}
				}
				vs.sort();
				vs.dedup();
				vs
			}).collect();
			Some((bs, td.1.clone()))
		} else {
			None
		};
		util::write_profile(&mut std::io::stderr());
		Self { orig_n: n, t0, g: g2, ts, ids, modify, w0, td }
	}
	
	pub fn restore(&self, w: W, es: Vec<E>) -> (W, Vec<E>) {
		let mut g = vec![BTreeSet::new(); self.orig_n];
		for (u, v) in es {
			let u = self.ids[u];
			let v = self.ids[v];
			g[u].insert(v);
			g[v].insert(u);
		}
		for modify in self.modify.iter().rev() {
			match *modify {
				Modify::Contract(u, v, w) => {
					if g[u].contains(&w) {
						g[u].remove(&w);
						g[w].remove(&u);
						g[u].insert(v);
						g[v].insert(u);
						g[w].insert(v);
						g[v].insert(w);
					}
				},
				Modify::Include(u, v, ref ws) => {
					for &w in ws {
						if g[u].contains(&w) {
							g[u].remove(&w);
							g[w].remove(&u);
							g[v].insert(w);
							g[w].insert(v);
						}
					}
					g[u].insert(v);
					g[v].insert(u);
				}
			}
		}
		let mut visited = vec![false; self.orig_n];
		let mut stack = vec![];
		visited[self.t0] = true;
		stack.push(self.t0);
		while let Some(u) = stack.pop() {
			for &v in &g[u] {
				if visited[v].setmax(true) {
					stack.push(v);
				}
			}
		}
		let mut es = vec![];
		for u in 0..self.orig_n {
			for &v in &g[u] {
				if visited[u] && visited[v] && u < v {
					es.push((u, v));
				}
			}
		}
		(w + self.w0, es)
	}
	
}

#[derive(Copy, Clone, Debug)]
struct Data {
	v: usize,
	cost: i64,
	prev: (usize, usize) // leaf: (!0, !0), move: (v, !0), merge: (x, y)
}

struct IntersectIter<'a> {
	d1: &'a[Data],
	d2: &'a[Data],
	p: usize,
	q: usize,
}

/// returns all (x, y)'s with d1[x].v == d2[y].v
fn intersect<'a>(d1: &'a[Data], d2: &'a[Data]) -> IntersectIter<'a> {
	IntersectIter { d1, d2, p: 0, q: 0 }
}

impl<'a> Iterator for IntersectIter<'a> {
	type Item = (usize, usize);
	#[inline]
	fn next(&mut self) -> Option<(usize, usize)> {
		while self.p < self.d1.len() && self.q < self.d2.len() {
			match self.d1[self.p].v.cmp(&self.d2[self.q].v) {
				Ordering::Equal => {
					self.p += 1;
					self.q += 1;
					return Some((self.p - 1, self.q - 1));
				},
				Ordering::Less => {
					self.p += 1;
				},
				Ordering::Greater => {
					self.q += 1;
				}
			}
		}
		None
	}
}

struct BitTree {
	ts: Vec<SubTree>,
	root: usize,
}

struct SubTree {
	i_and: BitSet,
	v_or: BitSet,
	p: usize,
	i_pos: usize,
	cs: [usize; 2]
}

impl BitTree {
	fn new() -> Self {
		BitTree { ts: vec![], root: !0 }
	}
	fn insert(&mut self, i: BitSet, v: BitSet, p: usize) {
		if self.root == !0 {
			self.new_leaf(i, v, p);
			self.root = 0;
		} else {
			let root = self.root;
			self.root = self._insert(i, v, p, root, 0);
		}
	}
	fn new_leaf(&mut self, i: BitSet, v: BitSet, p: usize) -> usize {
		let i_pos = i.data.len() * 64;
		self.ts.push(SubTree {
			i_and: i,
			v_or: v,
			p,
			i_pos,
			cs: [!0, !0]
		});
		self.ts.len() - 1
	}
	fn _insert(&mut self, i: BitSet, v: BitSet, p: usize, x: usize, i_pos: usize) -> usize {
		if self.ts[x].i_pos == i_pos {
			let j = if i[i_pos] {
				1
			} else {
				0
			};
			let c = self.ts[x].cs[j];
			self.ts[x].i_and = &self.ts[x].i_and & &i;
			self.ts[x].v_or = &self.ts[x].v_or | &v;
			self.ts[x].cs[j] = self._insert(i, v, p, c, i_pos + 1);
			x
		} else if self.ts[x].i_and[i_pos] != i[i_pos] {
			let leaf = self.new_leaf(i.clone(), v.clone(), p);
			let tmp = SubTree {
				i_and: &i & &self.ts[x].i_and,
				v_or: &v | &self.ts[x].v_or,
				p: !0,
				i_pos,
				cs: if i[i_pos] {
					[x, leaf]
				} else {
					[leaf, x]
				}
			};
			self.ts.push(tmp);
			self.ts.len() - 1
		} else {
			self._insert(i, v, p, x, i_pos + 1)
		}
	}
	fn find(&self, i: &BitSet, v: &BitSet) -> Vec<usize> {
		if self.root == !0 {
			return vec![];
		}
		let mut ps = vec![];
		self._find(i, v, self.root, &mut ps);
		ps
	}
	fn _find(&self, i: &BitSet, v: &BitSet, x: usize, ps: &mut Vec<usize>) {
		if self.ts[x].i_and.intersect(&i) || !self.ts[x].v_or.intersect(&v) {
			return;
		} else if self.ts[x].i_pos == i.data.len() * 64 {
			ps.push(self.ts[x].p);
		} else {
			self._find(i, v, self.ts[x].cs[0], ps);
			if !i[self.ts[x].i_pos] {
				self._find(i, v, self.ts[x].cs[1], ps);
			}
		}
	}
}

pub mod pruned {
	
	use ::*;
	
	pub fn solve(g: &G, ts: &Vec<usize>) -> (W, Vec<E>) {
		if ts.len() <= 1 {
			return (0, vec![]);
		}
		let n = g.len();
		let div = (1i64 << 32) / n as i64;
		// Perturbation for tie-breaking.
		let g: Vec<Vec<_>> = (0..n).map(|u| g[u].iter().map(|&(v, w)| (v, (w as i64) << 32 | (u + v) as i64 % div)).collect()).collect();
		const INF: i64 = std::i64::MAX;
		let T = ts.len();
		let mut tid = vec![!0; n];
		for t in 0..T {
			tid[ts[t]] = t;
		}
		let mut all = BitSet::new(T);
		for i in 0..T {
			all.set(i, true);
		}
		let mut i_set = BTreeSet::new();
		let mut ongoing = BTreeMap::new();
		for i in 0..T {
			let mut bit = BitSet::new(T);
			bit.set(i, true);
			i_set.insert((1, bit.clone()));
			ongoing.insert(bit, vec![Data { v: ts[i], cost: 0, prev: (!0, !0) }]);
		}
		let mut min = INF;
		let mut min_x = !0;
		let mut min_prev = (!0, !0);
		let mut bit_tree = BitTree::new();
		let mut bits = vec![];
		let mut bit_counts = vec![];
		let mut offsets = vec![0];
		let mut data = vec![];
		let mut total_size = 0;
		let mut process = 0;
		while i_set.len() > 0 {
			let (i_count, i) = i_set.iter().next().unwrap().clone();
			i_set.remove(&(i_count, i.clone()));
			let crt = ongoing.remove(&i).unwrap();
			let i_count = i.count();
			if process.setmax(i_count) {
				eprintln!("Finished = {}/{}", process - 1, T / 2);
				eprintln!("total = {}", total_size);
				eprintln!("subsets = {}", bits.len());
			}
			// eprintln!("{:?}: {:?}", (0..T).filter(|&t| i[t]).map(|t| ts[t]).collect::<Vec<_>>(), crt.iter().map(|c| (c.v, c.cost >> 32)).collect::<Vec<_>>());
			let mut dist = vec![INF; n]; // dist[r] := minimum steiner tree for i rooted at r.
			let mut ub = vec![INF; n]; // No optimal subtrees for T-i contain vertices at distance < ub[u] from u.
			let mut prev = vec![(!0, !0); n];
			let mut valid = vec![false; n];
			{
				profile!(trace);
				// Construct steiner trees for i by tracing prev.
				// If u is contained in every tree, set ub[u] <- min_tree maximum length of an induced degree-2 path from u to the root.
				let mut cs: Vec<_> = (0..crt.len()).map(|i| (crt[i].cost, i)).collect();
				cs.sort();
				let mut count = vec![0; n];
				let mut stack = vec![];
				let mut valid_count = 0;
				for (_, i) in cs {
					let c = &crt[i];
					let d = c.cost;
					if dist[c.v] < d {
						continue;
					}
					dist[c.v] = d;
					valid[c.v] = true;
					prev[c.v] = c.prev;
					valid_count += 1;
					count[c.v] += 1;
					ub[c.v] = 0;
					if c.prev.0 != !0 {
						let (x, y) = c.prev;
						stack.push((x, 0, 0));
						stack.push((y, 0, 0));
						while let Some((x, c, w)) = stack.pop() {
							let Data { cost, prev: (x, y), .. } = data[x];
							if x != !0 {
								if y != !0 {
									stack.push((x, 0, w));
									stack.push((y, 0, w));
								} else {
									let v2 = data[x].v;
									let c2 = c + cost - data[x].cost;
									let w2 = w.max(c2);
									dist[v2].setmin(d);
									count[v2] += 1;
									ub[v2].setmin(w2);
									stack.push((x, c2, w2));
								}
							}
						}
					}
				}
				for u in 0..n {
					if count[u] < valid_count {
						ub[u] = 0;
					}
				}
			}
			let mut s = !0;
			{
				profile!(dist);
				// Compute dist by Dijkstra's algorithm.
				// For speeding up, we stop when it visits a terminal or all the neighbors of a terminal.
				let mut adj_count = vec![0; T];
				let mut que = BinaryHeap::new();
				for u in 0..n {
					if dist[u] < INF {
						que.push((-dist[u], u));
					}
				}
				'dij: while let Some((d, u)) = que.pop() {
					let d = -d;
					if d != dist[u] {
						continue;
					}
					if tid[u] != !0 && !i[tid[u]] {
						s = u;
						break;
					}
					let ok = valid[u];
					for &(v, w) in &g[u] {
						let d2 = d + w;
						if dist[v].setmin(d2) {
							que.push((-d2, v));
							valid[v] = ok;
							prev[v] = (u, !0);
						} else if dist[v] == d2 && !ok {
							valid[v] = false;
						}
						if tid[v] != !0 && !i[tid[v]] && T - i_count > 1 {
							adj_count[tid[v]] += 1;
							if adj_count[tid[v]] == g[v].len() {
								s = v;
								break 'dij;
							}
						}
					}
				}
			}
			if T - i_count > 1 {
				profile!(prune);
				// Propagate ub by Dijkstra's algorithm.
				let mut que = BinaryHeap::new();
				for u in 0..n {
					if ub[u] > 0 {
						que.push((ub[u], u));
					}
				}
				while let Some((d, u)) = que.pop() {
					if d != ub[u] {
						continue;
					}
					for &(v, w) in &g[u] {
						let d2 = d - w;
						if ub[v].setmax(d2) {
							que.push((d2, v));
						}
					}
				}
				// Let V_d := {v | dist[v] >= d && ub[v] <= 0}.
				// Let atmost := max d s.t. T-i is connected on G[V_d].
				// Then every optimal steiner tree for T-i contains a vertex v with dist[v] <= atmost.
				// Therefore the current steiner tree rooted at v is valid only when dist[v] <= atmost.
				let mut atmost = -1;
				if (0..T).find(|&t| !i[t] && ub[ts[t]] > 0).is_none() {
					let mut count_t = 0;
					let mut min_d = vec![-1; n];
					min_d[s] = dist[s];
					que.push((dist[s], s));
					let mut stack = vec![];
					'cut: while let Some((d, u)) = que.pop() {
						if d != min_d[u] {
							continue;
						}
						stack.push(u);
						while let Some(u) = stack.pop() {
							if tid[u] != !0 && !i[tid[u]] {
								count_t += 1;
								if count_t == T - i_count {
									atmost = d;
									break 'cut;
								}
							}
							for &(v, _) in &g[u] {
								if ub[v] > 0 {
									continue;
								}
								let d2 = d.min(dist[v]);
								if min_d[v].setmax(d2) {
									if d == d2 {
										stack.push(v);
									} else {
										que.push((d2, v));
									}
								}
							}
						}
					}
				}
				for u in 0..n {
					if dist[u] > atmost { // || ub[u] > 0 {
						valid[u] = false;
					}
				}
			}
			if valid.iter().find(|&&v| v).is_none() {
				continue;
			}
			let mut di: Vec<_> = (0..n).filter(|&v| valid[v]).map(|v| Data { v, cost: dist[v], prev: prev[v] }).collect();
			let mut id = vec![!0; n];
			for x in 0..di.len() {
				id[di[x].v] = total_size + x;
			}
			for a in &mut di {
				if a.prev.0 != !0 && a.prev.1 == !0 {
					a.prev.0 = id[a.prev.0];
				}
			}
			profile!(merge);
			let mut valid_bit = BitSet::new(n);
			for u in 0..n {
				if valid[u] {
					valid_bit.set(u, true);
				}
			}
			for q in bit_tree.find(&i, &valid_bit) { // Find all j s.t. i & j == 0 and valids[i] & valids[j] != 0.
				if i_count * 2 + bit_counts[q] > T {
					continue;
				}
				let j = &bits[q];
				let dj = &data[offsets[q]..offsets[q + 1]];
				let tmp: Vec<Data> = intersect(&di, dj).map(|(x, y)| Data { v: di[x].v, cost: di[x].cost + dj[y].cost, prev: (total_size + x, offsets[q] + y)}).collect();
				let k = &i | j;
				let mut merged = false;
				if let Some(dk) = ongoing.get_mut(&k) {
					let mut dk2 = vec![];
					let mut x = 0;
					let mut y = 0;
					while x < dk.len() || y < tmp.len() {
						let v1 = if x < dk.len() {
							dk[x].v
						} else {
							n
						};
						let v2 = if y < tmp.len() {
							tmp[y].v
						} else {
							n
						};
						match v1.cmp(&v2) {
							Ordering::Equal => {
								if dk[x].cost <= tmp[y].cost {
									dk2.push(dk[x]);
								} else {
									dk2.push(tmp[y]);
								}
								x += 1;
								y += 1;
							},
							Ordering::Less => {
								dk2.push(dk[x]);
								x += 1;
							},
							Ordering::Greater => {
								dk2.push(tmp[y]);
								y += 1;
							}
						}
					}
					*dk = dk2;
					merged = true;
				}
				if !merged {
					if k.count() * 2 <= T {
						i_set.insert((k.count(), k.clone()));
					}
					ongoing.insert(k, tmp);
				}
			}
			let j = &all ^ &i;
			if let Some(ref dj) = ongoing.get(&j) {
				for (x, y) in intersect(&di, dj) {
					if min.setmin(di[x].cost + dj[y].cost) {
						min_x = total_size + x;
						min_prev = dj[y].prev;
					}
				}
			}
			let size = di.len();
			if i_count * 3 <= T {
				bit_tree.insert(i.clone(), valid_bit, bits.len());
			}
			bits.push(i);
			bit_counts.push(i_count);
			data.extend(di);
			total_size += size;
			offsets.push(total_size);
		}
		
		if min == INF {
			return (0, vec![]);
		}
		assert!(min != INF);
		util::write_profile(&mut std::io::stderr());
		eprintln!("total = {}", total_size);
		eprintln!("subsets = {}", bits.len());
		let mut es = vec![];
		let mut stack = vec![min_x];
		if min_prev.0 != !0 {
			stack.push(min_prev.0);
		}
		if min_prev.1 != !0 {
			stack.push(min_prev.1);
		}
		while let Some(x) = stack.pop() {
			if data[x].prev.0 != !0 {
				stack.push(data[x].prev.0);
				if data[x].prev.1 != !0 {
					stack.push(data[x].prev.1);
				} else {
					es.push((data[x].v, data[data[x].prev.0].v));
				}
			}
		}
		((min >> 32) as W, es)
	}
		
}

pub fn validate(g: &G, ts: &Vec<usize>, w: W, es: &Vec<E>) -> bool {
	let n = g.len();
	let mut w2 = 0;
	let mut g2 = vec![BTreeMap::new(); n];
	for u in 0..n {
		for &(v, w) in &g[u] {
			g2[u].entry(v).or_insert(INF).setmin(w);
		}
	}
	let mut st = vec![vec![]; n];
	for &(u, v) in es {
		st[u].push(v);
		st[v].push(u);
		if let Some(&w) = g2[u].get(&v) {
			w2 += w as i64;
		} else {
			eprintln!("illegal edge");
			return false;
		}
	}
	if w2 != w as i64 {
		eprintln!("wrong value: {} vs {}", w, w2);
		return false;
	}
	let mut visited = vec![false; n];
	let mut stack = vec![];
	visited[ts[0]] = true;
	stack.push(ts[0]);
	while let Some(u) = stack.pop() {
		for &v in &st[u] {
			if !visited[v] {
				visited[v] = true;
				stack.push(v);
			}
		}
	}
	for &t in ts {
		if !visited[t] {
			eprintln!("not connected");
			return false;
		}
	}
	return true;
}
