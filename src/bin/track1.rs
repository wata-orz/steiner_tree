#![allow(non_snake_case)]

extern crate steiner;
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

fn main() {
	let stime = std::time::SystemTime::now();
	let (g, ts) = read_input();
	eprintln!("n = {}\nm = {}\nT = {}", g.len(), g.iter().map(|a| a.len()).sum::<usize>() / 2, ts.len());
	let reduced = Reduction::reduce(&g, &ts, &None);
	eprintln!("n' = {}\nm' = {}\nT' = {}", reduced.g.len(), reduced.g.iter().map(|a| a.len()).sum::<usize>() / 2, reduced.ts.len());
	let time = std::time::SystemTime::now().duration_since(stime).unwrap();
	eprintln!("reduce = {:.3}", time.as_secs() as f64 + time.subsec_nanos() as f64 * 1e-9);
	let (w, es) = pruned::solve(&reduced.g, &reduced.ts);
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
