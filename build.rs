extern crate cc;

fn main() {
	cc::Build::new()
		.cpp(true)
		.file("cpp/solve.cpp")
		.flag("-std=c++11")
		.compile("solve")
}
