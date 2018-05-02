#![allow(non_snake_case)]

extern crate steiner;
use std::collections::{BTreeSet, BTreeMap, BinaryHeap};
use steiner::*;

use std::io::BufRead;

// Each edge must have positive weight (weight zero is not allowed).
fn read_input() -> (G, Vec<usize>) {
	let mut es: Vec<(usize, usize, W)> = vec![];
	let mut ts: Vec<usize> = vec![];
	let stdin = std::io::stdin();
	for line in stdin.lock().lines() {
		let line = line.unwrap();
		let ss: Vec<&str> = line.split_whitespace().collect();
		if ss.len() > 0 {
			if ss[0] == "E" {
				es.push((ss[1].parse().unwrap(), ss[2].parse().unwrap(), ss[3].parse().unwrap()));
			} else if ss[0] == "T" {
				ts.push(ss[1].parse::<usize>().unwrap() - 1);
			} else if ss[0] == "SECTION" && ss[1] == "Tree" {
				break;
			}
		}
	}
	let mut n = 0;
	for &(u, v, _) in &es {
		n.setmax(u);
		n.setmax(v);
	}
	let mut g = vec![vec![]; n];
	for &(u, v, w) in &es {
		g[u - 1].push((v - 1, w));
		g[v - 1].push((u - 1, w));
	}
	(g, ts)
}

fn read_tree_decomposition() -> Option<TD> {
	let stdin = std::io::stdin();
	let mut lines = stdin.lock().lines();
	let n;
	loop {
		if let Some(line) = lines.next() {
			let line = line.unwrap();
			let ss: Vec<&str> = line.split_whitespace().collect();
			if ss.len() > 0 && ss[0] == "s" && ss[1] == "td" {
				n = ss[2].parse().unwrap();
				break;
			}
		} else {
			return None;
		}
	}
	let mut bs = vec![vec![]; n];
	for _ in 0..n {
		let line = lines.next().unwrap().unwrap();
		let ss: Vec<&str> = line.split_whitespace().collect();
		assert_eq!(ss[0], "b");
		let id = ss[1].parse::<usize>().unwrap() - 1;
		for j in 2..ss.len() {
			bs[id].push(ss[j].parse::<usize>().unwrap() - 1);
		}
		bs[id].sort();
	}
	let mut es = vec![];
	for _ in 0..n - 1 {
		let line = lines.next().unwrap().unwrap();
		let ss: Vec<&str> = line.split_whitespace().collect();
		es.push((ss[0].parse::<usize>().unwrap() - 1, ss[1].parse::<usize>().unwrap() - 1));
	}
	Some((bs, es))
}

pub fn compute_tree_decomposition(g: &G) -> TD {
	let n = g.len();
	let mut g: Vec<BTreeSet<usize>> = g.iter().map(|a| {
		a.iter().map(|&(u, _)| u).collect()
	}).collect();
	let mut bags = vec![vec![]; n];
	let mut depth = vec![!0; n];
	let mut que = BinaryHeap::new();
	for u in 0..n {
		que.push((!g[u].len(), u));
	}
	let mut k = 0;
	while let Some((d, u)) = que.pop() {
		let d = !d;
		if depth[u] != !0 || g[u].len() != d {
			continue;
		}
		depth[u] = k;
		k += 1;
		g[u].insert(u);
		bags[u] = std::mem::replace(&mut g[u], BTreeSet::new()).into_iter().collect();
		for &i in &bags[u] {
			if i == u {
				continue;
			}
			let di = g[i].len();
			g[i].remove(&u);
			for &j in &bags[u] {
				if i != j && j != u {
					g[i].insert(j);
				}
			}
			if g[i].len() != di {
				que.push((!g[i].len(), i));
			}
		}
	}
	let mut es = vec![];
	for u in 0..n {
		if bags[u].len() > 1 {
			let mut p = !0;
			for &v in &bags[u] {
				if v != u && (p == !0 || depth[p] > depth[v]) {
					p = v;
				}
			}
			if p != !0 {
				es.push((u, p));
			}
		}
	}
	(bags, es)
}

pub fn validate_td(g: &G, td: &TD) -> bool {
	let n = g.len();
	let mut es = BTreeSet::new();
	for u in 0..n {
		for &(v, _) in &g[u] {
			if u < v {
				es.insert((u, v));
			}
		}
	}
	for bag in &td.0 {
		for i in 0..bag.len() {
			for j in i + 1..bag.len() {
				es.remove(&(bag[i], bag[j]));
			}
		}
	}
	if es.len() > 0 {
		let &(u, v) = es.iter().next().unwrap();
		eprintln!("The tree-decomposition does not contain edge {}-{}.", u, v);
		return false;
	}
	let mut count_v = vec![0; n];
	let mut count_e = vec![0; n];
	for bag in &td.0 {
		for &v in bag {
			count_v[v] += 1;
		}
	}
	for &(i, j) in &td.1 {
		let mut p = 0;
		let mut q = 0;
		while p < td.0[i].len() && q < td.0[j].len() {
			match td.0[i][p].cmp(&td.0[j][q]) {
				std::cmp::Ordering::Equal => {
					count_e[td.0[i][p]] += 1;
					p += 1;
					q += 1;
				},
				std::cmp::Ordering::Less => {
					p += 1;
				},
				std::cmp::Ordering::Greater => {
					q += 1;
				},
			}
		}
	}
	for v in 0..n {
		if count_v[v] > 0 && count_v[v] != count_e[v] + 1 {
			eprintln!("The bags containing {} are not connected.", v);
			return false;
		}
	}
	true
}

pub mod td {
	
	use ::*;
	
	mod cpp {
		extern {
			pub fn solve(N: i32, M: i32, edges: *const i32, T: i32, terminals: *const i32, V: i32, td: *const i32, tedges: *const i32, ans: *mut i32) -> i32;
		}
	}
	
	pub fn solve(g: &G, ts: &Vec<usize>, td: &TD) -> (W, Vec<E>) {
		if ts.len() <= 1 {
			return (0, vec![]);
		}
		let N = g.len();
		let mut edges = vec![];
		for u in 0..N {
			for &(v, w) in &g[u] {
				if u < v {
					edges.push(u as i32);
					edges.push(v as i32);
					edges.push(w as i32);
				}
			}
		}
		let M = edges.len() / 3;
		let mut terminals = vec![];
		for &t in ts {
			terminals.push(t as i32);
		}
		let V = td.0.len();
		let mut bags = vec![];
		for bag in &td.0 {
			bags.push(bag.len() as i32);
			for &v in bag {
				bags.push(v as i32);
			}
		}
		let mut tedges = vec![];
		for &(u, v) in &td.1 {
			tedges.push(u as i32);
			tedges.push(v as i32);
		}
		let mut ans = vec![0; M * 2];
		let K = unsafe { cpp::solve(N as i32, M as i32, edges.as_ptr(), ts.len() as i32, terminals.as_ptr(), V as i32, bags.as_ptr(), tedges.as_ptr(), ans.as_mut_ptr()) } as usize;
		let mut es = vec![];
		for i in 0..K {
			es.push((ans[i * 2] as usize, ans[i * 2 + 1] as usize));
		}
		let mut w = 0;
		let mut g2 = vec![BTreeMap::new(); N];
		for u in 0..N {
			for &(v, w) in &g[u] {
				g2[u].entry(v).or_insert(INF).setmin(w);
			}
		}
		for &(u, v) in &es {
			if let Some(w1) = g2[u].get(&v) {
				w += w1;
			} else {
				panic!("illegal edge");
			}
		}
		(w, es)
	}
	
}

fn main() {
	let stime = std::time::SystemTime::now();
	let (g, ts) = read_input();
	eprintln!("n = {}\nm = {}\nT = {}", g.len(), g.iter().map(|a| a.len()).sum::<usize>() / 2, ts.len());
	let td = read_tree_decomposition();
	if let Some(ref td) = td {
		let tw = td.0.iter().map(|b| b.len()).max().unwrap_or(0);
		eprintln!("tw = {}", tw);
	}
	let mut reduced = Reduction::reduce(&g, &ts, &td);
	eprintln!("n' = {}\nm' = {}\nT' = {}", reduced.g.len(), reduced.g.iter().map(|a| a.len()).sum::<usize>() / 2, reduced.ts.len());
	let time = std::time::SystemTime::now().duration_since(stime).unwrap();
	eprintln!("reduce = {:.3}", time.as_secs() as f64 + time.subsec_nanos() as f64 * 1e-9);
	let mut td = compute_tree_decomposition(&reduced.g);
	let time = std::time::SystemTime::now().duration_since(stime).unwrap();
	eprintln!("decompose = {:.3}", time.as_secs() as f64 + time.subsec_nanos() as f64 * 1e-9);
	let mut tw = td.0.iter().map(|b| b.len()).max().unwrap_or(0);
	if let Some(ref td1) = reduced.td {
		let tw1 = td1.0.iter().map(|b| b.len()).max().unwrap_or(0);
		if tw >= tw1 {
			td = td1.clone();
			tw = tw1;
		}
	}
	reduced.td = Some(td);
	eprintln!("tw' = {}", tw);
	if !validate_td(&reduced.g, reduced.td.as_ref().unwrap()) {
		panic!("orz");
	}
	let (w, es) = if tw <= 9 || tw <= 11 && reduced.ts.len() >= 300 {
		td::solve(&reduced.g, &reduced.ts, reduced.td.as_ref().unwrap())
	} else {
		pruned::solve(&reduced.g, &reduced.ts)
	};
	let (w, es) = reduced.restore(w, es);
	let time = std::time::SystemTime::now().duration_since(stime).unwrap();
	eprintln!("value = {}", w);
	eprintln!("time = {:.3}", time.as_secs() as f64 + time.subsec_nanos() as f64 * 1e-9);
	if !validate(&g, &ts, w, &es) {
		eprintln!("value = WA");
		eprintln!("time = WA");
		panic!("orz");
	}
	println!("VALUE {}", w);
	for (u, v) in es {
		println!("{} {}", u + 1, v + 1);
	}
}
