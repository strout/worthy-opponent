#![cfg_attr(test, feature(test))]

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

mod game;
mod basics;
mod go;
mod tictactoe;
mod ninemensmorris;

use game::Game;

const THINK_MS : u32 = 1000;

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
    fn get_mut(self: &mut MCTree, mv: usize) -> &mut MCTree {
        let replies = self.replies.as_mut().unwrap();
        &mut replies[mv]
    }
}

fn mc_score(mc: &MCTree, lnt: f64, explore: f64) -> f64 {
    let default = std::f64::INFINITY;
    match *mc {
        MCTree { plays: 0, .. } => default,
        MCTree { wins, plays, .. } => wins / plays as f64 + explore * (lnt / (plays as f64)).sqrt()
    }
}

fn print_mc(mc: &MCTree) {
    let lnt = (mc.plays as f64).ln();
    let explore = 2.0;
    if let Some(ref rs) = mc.replies {
        for (i, r) in rs.iter() {
            println!("{} => {:.5} / {:.5} / {}", i, r.wins / r.plays as f64, mc_score(r, lnt, explore), r.plays)
        }
    }
    println!("");
}

fn mc_expand<G: Game>(mc: &mut MCTree, g: &G) {
    mc.replies = Some({
        let mut reps = VecMap::new();
        for m in g.legal_moves() {
            reps.insert(m, MCTree::new());
        }
        reps
    })
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
            if rng.gen_range(0, count) == 0 {
                best = Some(p);
            }
        }
    }
    best.unwrap() // TODO what if no legal moves?
}

fn play_out<T: Rng, G: Game>(rng: &mut T, g: &mut G) -> f64 {
    debug_assert!(g.payoff().is_none());
    let mut flip = false;
    loop
    {
        let mv = *rng.choose(&g.legal_moves()[..]).expect("Either 'payoff' or 'legal_moves' is lying.");
        g.play(mv);
        flip = !flip;
        match g.payoff() {
            Some(p) => return if flip { 1.0 - p } else { p },
            None => {}
        }
    }
}

fn mc_iteration<T: Rng, G: Game>(rng: &mut T, g: &mut G, mc: &mut MCTree) -> f64 {
    let p = match g.payoff() {
        Some(p) => p,
        None => if mc.replies.is_none() {
            mc_expand(mc, g);
            play_out(rng, g)
        } else {
            let mv = mc_move(rng, mc, 2.0);
            g.play(mv);
            1.0 - mc_iteration(rng, g, mc.get_mut(mv))
        }
    };
    mc.wins += p;
    mc.plays += 1;
    p
}

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
                print_mc(&mc);
                mc = mc.next(mv);
                g.play(mv);
                g.print();
                if g.payoff().is_some() { return }
            }
            Ok(Cmd::Gen) => {
                mvs.send(mc_move(&mut rng, &mc, 0.0)).unwrap()
            }
        }
        g2.clone_from(&g);
        mc_iteration(&mut rng, &mut g2, &mut mc);
    }
}

fn main() {
    let (sendcmd, recvcmd) = channel();
    let (sendmv, recvmv) = channel();
    let game = std::env::args().nth(1).expect("Please supply a game (t = Tic-Tac-Toe, n = Nine Men's Morris, g = Go)");
    thread::spawn(move || match game.as_ref() {
        "t" => think::<tictactoe::TicTacToe>(recvcmd, sendmv),
        "n" => think::<ninemensmorris::NineMensMorris>(recvcmd, sendmv),
        "g" => think::<go::GoState>(recvcmd, sendmv),
        x => panic!("I don't know how to play '{}'.", x)
    });
    loop {
        thread::sleep_ms(THINK_MS);
        match sendcmd.send(Cmd::Gen) {
            Err(_) => return,
            _ => {}
        }
        let mv = match recvmv.recv() {
            Ok(mv) => mv,
            Err(_) => return
        };
        println!("{}", mv);
        match sendcmd.send(Cmd::Move(mv)) {
           Err(_) => return,
           _ => {}
        }
    }
}
