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
use std::io;

mod game;
mod basics;
mod go;
mod tictactoe;
mod ninemensmorris;

use game::Game;

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

fn print_mc<G: Game>(mc: &MCTree<G::Move>) {
    let lnt = (mc.plays as f64).ln();
    let explore = 2.0;
    if let Some(ref rs) = mc.replies {
        for &(ref i, ref r) in rs.iter() {
            G::print_move(i);
            println!(" => {:.5} / {:.5} / {}", r.wins / r.plays as f64, mc_score(r, lnt, explore), r.plays)
        }
    }
    println!("");
}

fn mc_expand<G: Game>(mc: &mut MCTree<G::Move>, g: &G) where G::Move : PartialEq {
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
    debug_assert_eq!(mc.plays, mc.replies.as_ref().unwrap().iter().fold(0, |acc, &(_, ref r)| acc + r.plays) + 1);
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

pub fn mc_iteration<T: Rng, G: Game>(rng: &mut T, g: &mut G, mc: &mut MCTree<G::Move>) -> f64 where G::Move : PartialEq {
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
enum Cmd<M> { Move(M), Gen }

fn think<G: Game>(cmds: Receiver<Cmd<G::Move>>, mvs: Sender<G::Move>, payoffs: Sender<Option<f64>>) where G::Move : PartialEq {
    let mut rng = rand::weak_rng();
    let mut g = G::init();
    let mut mc = MCTree::new(0);
    let mut g2 = g.clone();
    loop {
        match cmds.try_recv() {
            Err(TryRecvError::Empty) => {},
            Err(TryRecvError::Disconnected) => return,
            Ok(Cmd::Move(mv)) => {
                mc = mc.next(&mv);
                g.play(&mv);
                let payoff = g.payoff();
                payoffs.send(payoff).unwrap();
                if payoff.is_some() { return }
            }
            Ok(Cmd::Gen) => {
                let mv = if mc.replies.is_some() {
                    mc_move(&mut rng, &mut mc, 0.0).0.clone()
                } else {
                    random_move(&mut rng, &g)
                };
                mc = mc.next(&mv);
                g.play(&mv);
                g.print();
                let payoff = g.payoff();
                mvs.send(mv).unwrap();
                payoffs.send(payoff.map(|p| 1.0 - p)).unwrap();
                if payoff.is_some() { return }
            }
        }
        g2.clone_from(&g);
        mc_iteration(&mut rng, &mut g2, &mut mc);
    }
}

fn parse_command<G: Game>(string: &str) -> Cmd<G::Move> {
    match string.trim() {
        "gen" => Cmd::Gen,
        x => Cmd::Move(G::parse_move(x))
    }
}

fn run<G: Game>(think_time: u32) where G::Move : Send + PartialEq + 'static {
    let (sendcmd, recvcmd) = channel();
    let (sendmv, recvmv) = channel();
    let (sendpayoff, recvpayoff) = channel();
    thread::spawn(move || think::<G>(recvcmd, sendmv, sendpayoff));
    loop {
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let cmd = parse_command::<G>(&input);
        let is_gen = if let Cmd::Gen = cmd { true } else { false };
        if is_gen { thread::sleep_ms(think_time) }
        sendcmd.send(cmd).unwrap(); if is_gen {
            let mv = recvmv.recv().unwrap();
            print!("!");
            G::print_move(&mv);
            println!("");
        }
        if let Some(payoff) = recvpayoff.recv().unwrap() {
            println!(";bye {}", payoff);
            return;
        }
    }
}

fn main() {
    let game = std::env::args().nth(1).expect("Please supply a game (t = Tic-Tac-Toe, n = Nine Men's Morris, g = Go)");
    let think_time = std::env::args().nth(2).and_then(|tt| tt.parse().ok()).expect("Please supply a thinking time (in milliseconds)");
    match game.as_ref() {
        "t" => run::<tictactoe::TicTacToe>(think_time),
        "n" => run::<ninemensmorris::NineMensMorris>(think_time),
        "g" => run::<go::Go>(think_time),
        x => panic!("I don't know how to play '{}'.", x)
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

    fn bench_mc_iteration<G: Game>(bench: &mut Bencher) where G::Move : PartialEq {
        let mut mc = MCTree::new(0);
        let mut g = G::init();
        let mut g2 = g.clone();
        let mut rng = weak_rng();
        bench.iter(|| { g2.clone_from(&g); mc_iteration(&mut rng, &mut g2, &mut mc); })
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
