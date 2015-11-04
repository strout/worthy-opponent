#![cfg_attr(test, feature(test))]

extern crate rand;
extern crate bit_set;

#[cfg(test)]
extern crate test;
#[cfg(test)]
extern crate quickcheck;

use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::thread;
use rand::Rng;
use rand::distributions::{Weighted, WeightedChoice, IndependentSample};
use std::io::{BufReader, BufRead, Write};
use std::collections::HashMap;
use std::fmt::Display;
use std::str::from_utf8;

mod game;
mod basics;
mod go;
mod tictactoe;
mod ninemensmorris;
mod ggp;

use game::Game;
use tictactoe::TicTacToe;
use ninemensmorris::NineMensMorris;
use go::Go;

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

fn print_mc<M>(mc: &MCTree<M>) where M: Display {
    let lnt = (mc.plays as f64).ln();
    let explore = 2.0;
    if let Some(ref rs) = mc.replies {
        for &(ref m, ref r) in rs.iter() {
            println!("{} => {:.5} / {:.5} / {}", m, r.wins / r.plays as f64, mc_score(r, lnt, explore), r.plays)
        }
    }
    println!("");
}

fn mc_expand<G: Game>(mc: &mut MCTree<G::Move>, g: &G) {
    if mc.replies.is_none() {
        mc.replies = Some({
            let mut reps = Vec::new();
            let mvs = g.legal_moves();
            for Weighted { item, weight } in mvs {
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
            if rng.gen_weighted_bool(count) {
                best = Some(p);
            }
        }
    }
    let x = &mut mc.replies.as_mut().unwrap()[best.unwrap()];
    (&x.0, &mut x.1)
}

fn random_move<R: Rng, G: Game>(rng: &mut R, g: &G) -> G::Move {
    let mut moves = g.playout_moves();
    let dist = WeightedChoice::new(&mut moves[..]);
    dist.ind_sample(rng)
}

fn play_out<T: Rng, G: Game>(rng: &mut T, g: &mut G) -> f64 {
    debug_assert!(g.payoff().is_none());
    let mut flip = false;
    loop
    {
        let mv = random_move(rng, g);
        g.play(&mv);
        flip = !flip;
        match g.payoff() {
            Some(p) => return if flip { 1.0 - p } else { p },
            None => {}
        }
    }
}

pub fn mc_iteration<T: Rng, G: Game>(rng: &mut T, g: &mut G, mc: &mut MCTree<G::Move>) -> f64 {
    let p = 1.0 - match g.payoff() {
        Some(p) => p,
        None => if mc.plays == 0 {
            play_out(rng, g)
        } else {
            mc_expand(mc, g);
            let (mv, rep) = mc_move(rng, mc, 2.0);
            g.play(mv);
            mc_iteration(rng, g, rep)
        }
    };
    mc.wins += p;
    mc.plays += 1;
    p
}

#[derive(PartialEq, Clone, Copy)]
enum Cmd<M> { Move(M), Gen, Reset }

fn think<G: Game>(cmds: Receiver<Cmd<G::Move>>, mvs: Sender<G::Move>) {
    let mut rng = rand::weak_rng();
    let mut g = G::init();
    let mut mc = MCTree::new(0);
    loop {
        match cmds.try_recv() {
            Err(TryRecvError::Empty) => {},
            Err(TryRecvError::Disconnected) => return,
            Ok(Cmd::Reset) => {
                mc = MCTree::new(0);
                g = G::init()
            }
            Ok(Cmd::Move(mv)) => {
                print_mc(&mc);
                mc = mc.next(&mv);
                g.play(&mv);
                g.print();
                if g.payoff().is_some() { return }
            }
            Ok(Cmd::Gen) => {
                let mv = if mc.replies.is_some() {
                    mc_move(&mut rng, &mut mc, 0.0).0.clone()
                } else {
                    random_move(&mut rng, &g)
                };
                mvs.send(mv).unwrap();
                if g.payoff().is_some() { return }
            }
        }
        mc_iteration(&mut rng, &mut g.clone(), &mut mc);
    }
}

fn run<'a, G: Game, R: BufRead, W: Write>(think_ms: u32, input: &mut R, output: &mut W) where G::Move: 'static {
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
        if is_gen { thread::sleep_ms(think_ms) }
        sendcmd.send(cmd).unwrap();
        if is_gen {
            let mv = recvmv.recv().unwrap();
            output.write_fmt(format_args!("move {}\0", mv)).unwrap();
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
    let stream = TcpStream::connect(std::env::args().nth(2).as_ref().map(|x| x as &str).unwrap_or("127.0.0.1:11873")).unwrap();
    let mut input = BufReader::new(stream.try_clone().unwrap());
    let mut output = stream;
    let mut buf = vec![];
    output.write_fmt(format_args!("bot:{}\0", std::env::args().nth(1).unwrap())).unwrap();
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
    use game::Game;
    use test::Bencher;
    use rand::weak_rng;
    use go::Go;
    use ninemensmorris::NineMensMorris;
    use tictactoe::TicTacToe;

    fn bench_mc_iteration<G: Game>(bench: &mut Bencher) {
        let mut mc = MCTree::new(0);
        let g = G::init();
        let mut rng = weak_rng();
        bench.iter(|| { let mut g2 = g.clone(); mc_iteration(&mut rng, &mut g2, &mut mc); })
    }

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
}
