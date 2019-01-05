extern crate worthy_opponent;
extern crate rand;

#[cfg(test)]
extern crate bencher;

use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Duration;
use rand::{Rng, FromEntropy, rngs::SmallRng, prelude::SliceRandom};
use std::io::{BufReader, BufRead, Write};
use std::collections::HashMap;
use std::fmt::Display;
use std::str::from_utf8;

use worthy_opponent::game::Game;
use worthy_opponent::tictactoe::TicTacToe;
use worthy_opponent::connectfour::ConnectFour;
use worthy_opponent::ninemensmorris::NineMensMorris;
use worthy_opponent::go::Go;

pub struct MCTree<M> {
    wins: f64,
    plays: usize,
    urgency: u32,
    replies: Option<Vec<(M, MCTree<M>)>>
}

impl<M: PartialEq> MCTree<M> {
    fn new(urgency: u32) -> MCTree<M> { MCTree { wins: 0.0, plays: 0, replies: None, urgency: urgency } }
    fn next(self: MCTree<M>, mv: &M) -> MCTree<M> {
        self.replies.map(|mut replies| {
            let idx = replies.iter().position(|&(ref m, _)| m == mv).unwrap();
            replies.swap_remove(idx).1
        }).unwrap_or(MCTree::new(0))
    }
}

fn mc_score<M>(mc: &MCTree<M>, lnt: f64, explore: f64) -> f64 {
    let default = std::f64::INFINITY;
    match *mc {
        MCTree { plays: 0, .. } => default,
        MCTree { wins, plays, .. } => wins / plays as f64 + explore * (lnt / (plays as f64)).sqrt()
    }
}

fn print_mc<M>(mc: &MCTree<M>, chosen: Option<&M>) where M: Display + PartialEq {
    if let Some(ref rs) = mc.replies {
        let max_plays : usize = rs.iter().fold(0, |max, &(_, MCTree { plays, .. })| std::cmp::max(max, plays));
        for &(ref m, ref r) in rs.iter() {
            println!("{} => {:.5} / {} {}{}", m, r.wins / r.plays as f64, r.plays, if r.plays == max_plays { "*" } else { "" }, if chosen.map(|c| m == c).unwrap_or(false) { "+" } else { "" })
        }
    }
    println!("");
}

fn mc_expand<G: Game>(mc: &mut MCTree<G::Move>, g: &G) {
    if mc.replies.is_none() {
        mc.replies = Some({
            let mut reps = Vec::new();
            let mvs = g.legal_moves();
            for (item, weight) in mvs {
                reps.push((item, MCTree::new(weight)));
            }
            reps
        })
    }
}

fn mc_move<'a, M, T: Rng>(rng: &mut T, mc: &'a mut MCTree<M>, explore: f64) -> (&'a M, &'a mut MCTree<M>) {
    let lnt = (mc.plays as f64).ln();
    // TODO should this be true? Possibly. debug_assert_eq!(mc.plays, mc.replies.as_ref().unwrap().iter().fold(0, |acc, &(_, ref r)| acc + r.plays) + 1);
    let mut best_score = std::f64::NEG_INFINITY;
    let mut best_urgency = 0;
    let mut best = None;
    let mut count = 0;
    for (p, &(_, ref rep)) in mc.replies.as_ref().unwrap().iter().enumerate() {
        let score = mc_score(rep, lnt, explore);
        let urgency = rep.urgency;
        if score > best_score || (score == best_score && urgency > best_urgency) {
            best = Some(p);
            best_score = score;
            best_urgency = urgency;
            count = 1;
        } else if score == best_score {
            count += 1;
            if rng.gen_bool(1.0 / count as f64) {
                best = Some(p);
            }
        }
    }
    let x = &mut mc.replies.as_mut().unwrap()[best.unwrap()];
    (&x.0, &mut x.1)
}

fn random_move<R: Rng, G: Game>(rng: &mut R, g: &G) -> G::Move {
    let moves = g.playout_moves();
    moves.choose_weighted(rng, |m| m.1).unwrap().0.clone() // TODO can we move it out instead of cloning?
}

fn play_out<T: Rng, G: Game>(rng: &mut T, g: &mut G) -> f64 {
    debug_assert!(g.payoff().is_none());
    let mut flip = false;
    let mut discount = 1.00;
    loop
    {
        let mv = random_move(rng, g);
        g.play(&mv);
        flip = !flip;
        discount = 0.99 * discount;
        match g.payoff() {
            Some(p) => return if flip { discount * (1.0 - p) + (1.0 - discount) * 0.5 } else { discount * p + (1.0 - discount) * 0.5 },
            None => {}
        }
    }
}

pub fn mc_iteration<T: Rng, G: Game>(rng: &mut T, g: &mut G, mc: &mut MCTree<G::Move>) -> f64 {
    let p = 0.99 * (1.0 - match g.payoff() {
        Some(p) => p,
        None => if mc.plays == 0 {
            play_out(rng, g)
        } else {
            mc_expand(mc, g);
            let (mv, rep) = mc_move(rng, mc, 2.0);
            g.play(mv);
            mc_iteration(rng, g, rep)
        }
    }) + 0.01 * 0.05;
    mc.wins += p;
    mc.plays += 1;
    p
}

#[derive(PartialEq, Clone, Copy)]
enum Cmd<M> { Move(M), Gen, Reset }

fn think<G: Game>(cmds: Receiver<Cmd<G::Move>>, mvs: Sender<G::Move>) {
    let mut rng = SmallRng::from_entropy();
    let mut g = G::init();
    let mut mc = MCTree::new(0);
    loop {
        match cmds.try_recv() {
            Err(TryRecvError::Empty) => {},
            Err(TryRecvError::Disconnected) => break,
            Ok(Cmd::Reset) => {
                mc = MCTree::new(0);
                g = G::init()
            }
            Ok(Cmd::Move(mv)) => {
                print_mc(&mc, Some(&mv));
                mc = mc.next(&mv);
                g.play(&mv);
                g.print();
            }
            Ok(Cmd::Gen) => {
                let mv = if mc.replies.is_some() {
                    mc_move(&mut rng, &mut mc, 0.0).0.clone()
                } else {
                    random_move(&mut rng, &g)
                };
                mvs.send(mv).unwrap();
            }
        }
        mc_iteration(&mut rng, &mut g.clone(), &mut mc);
    }
}

fn run<'a, G: Game, R: BufRead, W: Write>(think_ms: u64, input: &mut R, output: &mut W) where G::Move: 'static {
    let (sendcmd, recvcmd) = channel();
    let (sendmv, recvmv) = channel();
    thread::spawn(move || think::<G>(recvcmd, sendmv));
    let mut buf = vec![];
    loop {
        buf.clear();
        input.read_until(0, &mut buf).unwrap();
        let buf_str = from_utf8(&buf).unwrap();
        println!("Got: {}", &buf_str[..buf_str.len()-1]);
        let cscmd = parse_client_server_command(buf_str).unwrap();
        let cmd = match cscmd.name {
            "update-game-state" => cscmd.args.get("state").and_then(|x| G::parse_move(x)).map(Cmd::Move).unwrap_or(Cmd::Reset),
            "make-move" => Cmd::Gen,
            "announce-winner" => return,
            x => panic!("Protocol error! I don't know the command '{}'.", x)
        };
        let is_gen = if let Cmd::Gen = cmd { true } else { false };
        if is_gen { thread::sleep(Duration::from_millis(think_ms)) }
        sendcmd.send(cmd).unwrap();
        if is_gen {
            let mv = recvmv.recv().unwrap();
            println!("Sending {}.", mv);
            write!(output, "move {}\0", mv).unwrap();
        } else {
            output.write(b"ready\0").unwrap();
        }
    }
}

#[derive(Debug)]
struct ClientServerCmd<'a> {
    name: &'a str,
    args: HashMap<&'a str, &'a str>
}

fn parse_client_server_command(string: &str) -> Option<ClientServerCmd> {
    let string = &string[..string.len() - 1]; // avoid the \0 at the end
    let mut name_and_args = string.split(' ');
    name_and_args.next().and_then(|name| {
        let args = name_and_args.map(|arg| {
            let mut arg_parts = arg.splitn(2, '=');
            arg_parts.next().and_then(|arg_name| arg_parts.next().map(|arg_val| (arg_name, arg_val)))
        });
        let mut acc = HashMap::new();
        for mkv in args { match mkv {
            None => return None,
            Some((k, v)) => { acc.insert(k, v); }
        }}
        Some(ClientServerCmd { name: name, args: acc })
    })
}

fn main() {
    use std::net::TcpStream;
    let args = std::env::args().take(3).collect::<Vec<_>>();
    let supplied_server = args.get(2);
    let server_addr = supplied_server.map(|x| x as &str).unwrap_or("127.0.0.1:11873");
    let stream = TcpStream::connect(server_addr).expect(&format!("Couldn't connect to {}", server_addr));
    let mut input = BufReader::new(stream.try_clone().unwrap());
    let mut output = stream;
    let mut buf = vec![];
    write!(&mut output, "bot:{}\0", args.get(1).expect(&format!("Usage: {} name [server:port]", args[0]))).unwrap();
    input.read_until(0, &mut buf).unwrap();
    loop {
        buf.clear();
        input.read_until(0, &mut buf).unwrap();
        let cmd = parse_client_server_command(from_utf8(&buf).unwrap()).unwrap();
        match cmd.name {
            "prepare-new-game" => {
                let think_ms = cmd.args["milliseconds-per-move"].parse().unwrap();
                output.write(b"ready\0").unwrap();
                match cmd.args["game"] {
                    "tictactoe" => run::<TicTacToe,_,_>(think_ms, &mut input, &mut output),
                    "connectfour" => run::<ConnectFour,_,_>(think_ms, &mut input, &mut output),
                    "ninemensmorris" => run::<NineMensMorris,_,_>(think_ms, &mut input, &mut output),
                    "go" => run::<Go,_,_>(think_ms, &mut input, &mut output),
                    x => panic!("I don't know how to play {}.", x)
                };
            },
            x => panic!("Protocol error! I don't know the command '{}'.", x)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use worthy_opponent::game::Game;
    use bencher::Bencher;
    use rand::{FromEntropy, rngs::SmallRng};
    use worthy_opponent::go::Go;
    use worthy_opponent::ninemensmorris::NineMensMorris;
    use worthy_opponent::tictactoe::TicTacToe;
    use worthy_opponent::connectfour::ConnectFour;

    fn bench_mc_iteration<G: Game>(bench: &mut Bencher) {
        let mut mc = MCTree::new(0);
        let g = G::init();
        let mut rng = SmallRng::from_entropy();
        bench.iter(|| { let mut g2 = g.clone(); mc_iteration(&mut rng, &mut g2, &mut mc); })
    }

/*
    #[bench]
    fn go(bench: &mut Bencher) {
        bench_mc_iteration::<Go>(bench)
    }

    #[bench]
    fn ninemensmorris(bench: &mut Bencher) {
        bench_mc_iteration::<NineMensMorris>(bench)
    }

    #[bench]
    fn tictactoe(bench: &mut Bencher) {
        bench_mc_iteration::<TicTacToe>(bench)
    }

    #[bench]
    fn connectfour(bench: &mut Bencher) {
        bench_mc_iteration::<ConnectFour>(bench)
    }
*/
}
