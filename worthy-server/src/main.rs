extern crate rand;

use std::net::TcpListener;
use std::io::{BufReader, BufRead, Write};
use std::thread::spawn;
use std::mem::swap;
use rand::random;

fn main() {
    let listener = TcpListener::bind("127.0.0.1:64335").unwrap();

    let mut connections = listener.incoming();

    let mut player1 = connections.next().unwrap().unwrap();
    let mut player2 = connections.next().unwrap().unwrap();

    if random() { swap(&mut player1, &mut player2) }

    let p1_in = BufReader::new(player1.try_clone().unwrap());
    let p2_in = BufReader::new(player2.try_clone().unwrap());

    player1.write(b"gen\n").unwrap();
    player1.flush().unwrap();

    spawn(move || relay(p1_in, &mut player2));
    relay(p2_in, &mut player1);
}

fn relay<R: BufRead, W: Write>(r: R, w: &mut W) {
    for line in r.lines().flat_map(|x| x) {
        if line.starts_with("!") {
            w.write(line[1..].as_bytes()).unwrap();
            w.write(b"\ngen\n").unwrap();
            w.flush().unwrap();
        }
    }
}
