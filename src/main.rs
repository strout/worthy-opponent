#![cfg_attr(test, feature(test))]

const THINK_TIME: u32 = 1000;

extern crate rand;
extern crate bit_set;
extern crate vec_map;

#[cfg(test)]
extern crate test;
#[cfg(test)]
extern crate quickcheck;

use vec_map::VecMap;
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

#[derive(Debug)]
struct MCTree {
    wins: f64,
    plays: usize,
    replies: Option<VecMap<MCTree>>
}

impl MCTree {
    fn new() -> MCTree { MCTree { wins: 0.0, plays: 0, replies: None } }
    fn next(self: MCTree, mv: usize) -> MCTree {
        self.replies.and_then(|mut replies| {
            replies.remove(&mv)
        }).unwrap_or(MCTree::new())
    }
    // this can panic; only do it for legal moves
    fn get_mut(self: &mut MCTree, mv: &usize) -> Option<&mut MCTree> {
        let replies = self.replies.as_mut().unwrap();
        replies.get_mut(mv)
    }
}

fn mc_score(mc: &MCTree, lnt: f64, explore: f64) -> f64 {
    let default = std::f64::INFINITY;
    match *mc {
        MCTree { plays: 0, .. } => default,
        MCTree { wins, plays, .. } => wins / plays as f64 + explore * (lnt / (plays as f64)).sqrt()
    }
}

fn print_mc<G: Game>(mc: &MCTree) {
    let lnt = (mc.plays as f64).ln();
    let explore = 2.0;
    if let Some(ref rs) = mc.replies {
        for (i, r) in rs.iter() {
            G::print_move(i);
            println!(" => {:.5} / {:.5} / {}", r.wins / r.plays as f64, mc_score(r, lnt, explore), r.plays)
        }
    }
    println!("");
}

fn mc_expand<G: Game>(mc: &mut MCTree, g: &G) {
    if mc.replies.is_none() {
        mc.replies = Some({
            let mut reps = VecMap::new();
            for Weighted { item: m, .. } in g.legal_moves() {
                reps.insert(m, MCTree::new());
            }
            reps
        })
    }
}

fn mc_move<T: Rng>(rng: &mut T, mc: &MCTree, explore: f64) -> usize {
    let lnt = (mc.plays as f64).ln();
    debug_assert_eq!(mc.plays, mc.replies.as_ref().unwrap().iter().fold(0, |acc, (_, r)| acc + r.plays) + 1);
    let mut best_score = std::f64::NEG_INFINITY;
    let mut best = None;
    let mut count = 0;
    for (p, rep) in mc.replies.as_ref().unwrap().iter() {
        let score = mc_score(rep, lnt, explore);
        if score > best_score {
            best = Some(p);
            best_score = score;
            count = 1;
        } else if score == best_score {
            count += 1;
            if rng.gen_weighted_bool(count) {
                best = Some(p);
            }
        }
    }
    best.unwrap() // TODO what if no legal moves?
}

fn random_move<R: Rng, G: Game>(rng: &mut R, g: &G) -> usize {
    let mut moves = g.legal_moves();
    let dist = WeightedChoice::new(&mut moves[..]);
    dist.ind_sample(rng)
}

fn play_out<T: Rng, G: Game>(rng: &mut T, g: &mut G) -> f64 {
    debug_assert!(g.payoff().is_none());
    let mut flip = false;
    loop
    {
        let mv = random_move(rng, g);
        g.play(mv);
        flip = !flip;
        match g.payoff() {
            Some(p) => return if flip { 1.0 - p } else { p },
            None => {}
        }
    }
}

fn mc_iteration<T: Rng, G: Game>(rng: &mut T, g: &mut G, mc: &mut MCTree) -> f64 {
    let p = 1.0 - match g.payoff() {
        Some(p) => p,
        None => if mc.plays == 0 {
            play_out(rng, g)
        } else {
            mc_expand(mc, g);
            let mv = mc_move(rng, mc, 2.0);
            g.play(mv);
            mc_iteration(rng, g, mc.get_mut(&mv).unwrap())
        }
    };
    mc.wins += p;
    mc.plays += 1;
    p
}

#[derive(PartialEq, Clone, Copy)]
enum Cmd { Move(usize), Gen }

fn think<G: Game>(cmds: Receiver<Cmd>, mvs: Sender<usize>) {
    let mut rng = rand::weak_rng();
    let mut g = G::init();
    let mut mc = MCTree::new();
    let mut g2 = g.clone();
    loop {
        match cmds.try_recv() {
            Err(TryRecvError::Empty) => {},
            Err(TryRecvError::Disconnected) => return,
            Ok(Cmd::Move(mv)) => {
                print_mc::<G>(&mc);
                mc = mc.next(mv);
                g.play(mv);
                g.print();
                if g.payoff().is_some() { return }
            }
            Ok(Cmd::Gen) => {
                if mc.replies.is_some() {
                    mvs.send(mc_move(&mut rng, &mc, 0.0)).unwrap()
                } else {
                    mvs.send(random_move(&mut rng, &g)).unwrap();
                }
            }
        }
        g2.clone_from(&g);
        mc_iteration(&mut rng, &mut g2, &mut mc);
    }
}

fn parse_command<G: Game>(string: &str) -> Cmd {
    match string.trim() {
        "gen" => Cmd::Gen,
        x => Cmd::Move(G::parse_move(x))
    }
}

fn run<G: Game>() {
    let (sendcmd, recvcmd) = channel();
    let (sendmv, recvmv) = channel();
    thread::spawn(move || think::<G>(recvcmd, sendmv));
    loop {
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let cmd = parse_command::<G>(&input);
        if cmd == Cmd::Gen { thread::sleep_ms(THINK_TIME) }
        match sendcmd.send(cmd) {
            Err(_) => return,
            _ => {}
        }
        if cmd == Cmd::Gen {
            let mv = match recvmv.recv() {
                Ok(mv) => mv,
                Err(_) => return
            };
            print!("!");
            G::print_move(mv);
            println!("");
            match sendcmd.send(Cmd::Move(mv)) {
               Err(_) => return,
               _ => {}
            }
        }
    }
}

fn main() {
    let game = std::env::args().nth(1).expect("Please supply a game (t = Tic-Tac-Toe, n = Nine Men's Morris, g = Go)");
    match game.as_ref() {
        "t" => run::<tictactoe::TicTacToe>(),
        "n" => run::<ninemensmorris::NineMensMorris>(),
        "g" => run::<go::GoState>(),
        x => panic!("I don't know how to play '{}'.", x)
    }
}
