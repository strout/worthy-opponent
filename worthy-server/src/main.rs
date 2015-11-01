use std::net::TcpListener;
use std::io::{BufReader, BufRead, Write};

fn main() {
    let listener = TcpListener::bind("127.0.0.1:64335").unwrap();

    let mut connections = listener.incoming();

    let mut player1 = connections.next().unwrap().unwrap();
    let mut player2 = connections.next().unwrap().unwrap();

    let mut p1_in = BufReader::new(player1.try_clone().unwrap());
    let mut p2_in = BufReader::new(player2.try_clone().unwrap());

    player1.write(b"gen\n").unwrap();
    player1.flush().unwrap();

    loop {
        let mut buf = String::new();
        p1_in.read_line(&mut buf).unwrap();
        if buf.starts_with("!") {
            let mv = &buf[1..];
            player2.write(mv.as_bytes()).unwrap();
            player2.write(b"gen\n").unwrap();
            player2.flush().unwrap();
            std::mem::swap(&mut p1_in, &mut p2_in);
            std::mem::swap(&mut player1, &mut player2);
        }
        print!("{}", buf);
    }

    println!("Hello, world!");
}
