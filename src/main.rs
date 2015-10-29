#![feature(test)]

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
use bit_set::BitSet;

mod game;
mod go;
mod tictactoe;
mod ninemensmorris;

use game::Game;

const THINK_MS : u32 = 1000;

#[cfg(test)]
use quickcheck::*;

#[derive(Debug)]
struct MCTree {
    wins: f64,
    plays: usize,
    reply_plays: usize,
    replies: Option<VecMap<MCTree>>
}

impl MCTree {
    fn new() -> MCTree { MCTree { wins: 0.0, plays: 0, reply_plays: 0, replies: None } }
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
        MCTree { plays: 0, .. } => default + explore * lnt.sqrt(),
        MCTree { wins, plays, .. } => wins / plays as f64 + explore * (lnt / (plays as f64)).sqrt()
    }
}

fn print_mc(mc: &MCTree) {
    let lnt = (mc.reply_plays as f64).ln();
    let explore = 2.0f64.sqrt();
    if let Some(ref rs) = mc.replies {
        for (i, r) in rs.iter() {
            println!("{} => {:.5} / {:.5} / {}", i, r.wins / r.plays as f64, mc_score(r, lnt, explore), r.plays)
        }
    }
    println!("");
}

fn mc_expand<G: Game>(mc: &mut MCTree, g: &G) {
    if mc.replies.is_none() {
        mc.replies = Some({
            let mut reps = VecMap::new();
            for m in g.legal_moves() {
                reps.insert(m, MCTree::new());
            }
            reps
        })
    }
}

fn mc_move<T: rand::Rng, G: Game>(rng: &mut T, g: &G, mc: &mut MCTree, explore: f64) -> usize {
    mc_expand(mc, g);
    let lnt = if mc.reply_plays == 0 { 0.0 } else { (mc.reply_plays as f64).ln() };
    debug_assert_eq!(mc.reply_plays, mc.replies.as_ref().unwrap().iter().fold(0, |acc, (_, r)| acc + r.plays));
    let mut best_score = -1.0;
    let mut best = Vec::new();
    for (p, rep) in mc.replies.as_ref().unwrap().iter() {
        let score = mc_score(rep, lnt, explore);
        if score > best_score {
            best.clear();
            best_score = score;
        }
        if score >= best_score {
            best.push(p);
        }
    }
    *rng.choose(&best[..]).unwrap() // TODO what if no legal moves?
}

fn play_out<T: rand::Rng, G: Game>(rng: &mut T, g: &mut G) -> f64 {
    let mut flip = false;
    loop
    {
        match g.payoff() {
            None => {
                let mv = *rng.choose(&g.legal_moves()[..]).expect("Either 'payoff' or 'legal_moves' is lying.");
                g.play(mv);
                flip = !flip;
            },
            Some(p) => return if flip { 1.0 - p } else { p }
        }
    }
}

fn mc_iteration<T: rand::Rng, G: Game>(rng: &mut T, g: &mut G, mc: &mut MCTree) -> f64 {
    let mv = mc_move(rng, g, mc, 2.0f64.sqrt());
    mc.reply_plays += 1;
    let mut reply = mc.get_mut(mv);
    let expanding = reply.plays == 0;
    g.play(mv);
    let wr = 1.0 - (match g.payoff() {
        Some(p) => p,
        None => if expanding {
            play_out(rng, g)
        } else {
            mc_iteration(rng, g, reply)
        }
    });
    reply.wins += wr;
    reply.plays += 1;
    wr
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
                mvs.send(mc_move(&mut rng, &mut g, &mut mc, 0.0)).unwrap()
            }
        }
        for _ in 0..50 {
            g2.clone_from(&g);
            mc_iteration(&mut rng, &mut g2, &mut mc);
        }
    }
}

fn main() {
    let (sendcmd, recvcmd) = channel();
    let (sendmv, recvmv) = channel();
    thread::spawn(|| think::<tictactoe::TicTacToe>(recvcmd, sendmv));
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
