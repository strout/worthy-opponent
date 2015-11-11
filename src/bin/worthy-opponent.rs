extern crate tiny_http;
extern crate rand;
extern crate worthy_opponent;

use tiny_http::{ServerBuilder, Response};
use rand::{Rng, weak_rng};
use std::collections::HashMap;
use worthy_opponent::ggp::{Expr, GGP, db_from_str};
use std::hash::Hash;
use std::fmt::Display;

#[derive(Debug)]
struct MCTree<A: Eq + Hash, B: Eq + Hash> {
    payoff: f64,
    plays: usize,
    children: HashMap<A, MCTree<B, A>>
}

impl<A: Eq + Hash, B: Eq + Hash> MCTree<A, B> {
    fn new() -> MCTree<A, B> {
        MCTree { payoff: 0.0, plays: 0, children: HashMap::new() }
    }
}

fn mc_score<A: Eq + Hash, B: Eq + Hash>(mc: &MCTree<A, B>, lnt: f64, explore: f64) -> f64 {
    let default = std::f64::INFINITY;
    match *mc {
        MCTree { plays: 0, .. } => default,
        MCTree { payoff, plays, .. } => payoff / plays as f64 + explore * (lnt / (plays as f64)).sqrt()
    }
}

fn print_mc<A: Eq + Hash + Display, B: Eq + Hash>(mc: &MCTree<A, B>, chosen: Option<&A>) {
    let max_plays : usize = mc.children.iter().fold(0, |max, (_, &MCTree { plays, .. })| std::cmp::max(max, plays));
    for (m, r) in mc.children.iter() {
        println!("{} => {:.5} / {} {}{}", m, r.payoff / r.plays as f64, r.plays, if r.plays == max_plays { "*" } else { "" }, if chosen.map(|c| m == c).unwrap_or(false) { "+" } else { "" })
    }
    println!("");
}

fn choose_move<'a, 'b, T: Rng>(rng: &mut T, ggp: &GGP<'b>, role: &'b str, mc: &'a mut MCTree<Expr<'b>, Vec<Expr<'b>>>, explore: f64) -> (Expr<'b>, &'a mut MCTree<Vec<Expr<'b>>, Expr<'b>>) {
    if mc.children.is_empty() {
        let mvs = ggp.legal_moves_for(role);
        mc.children.reserve(mvs.len());
        for mv in mvs { mc.children.insert(mv, MCTree::new()); }
    }

    let lnt = (mc.plays as f64).ln();

    let mut best_score = std::f64::NEG_INFINITY;
    let mut best = None;
    let mut count = 0;
    for (mv, child) in mc.children.iter() {
        let score = mc_score(child, lnt, explore);
        if score > best_score {
            best = Some(mv.clone());
            best_score = score;
            count = 1;
        } else if score == best_score {
            count += 1;
            if rng.gen_weighted_bool(count) { best = Some(mv.clone()) }
        }
    }
    let mv = best.unwrap();
    let child = mc.children.get_mut(&mv).unwrap();
    (mv, child)
}

fn tree_search<'a, T: Rng>(rng: &mut T, ggp: &mut GGP<'a>, mcs: &mut [(&'a str, &mut MCTree<Expr<'a>, Vec<Expr<'a>>>)]) -> HashMap<&'a str, f64> {
    let result = if ggp.is_done() {
        println!("I found a terminal state!");
        ggp.goals().into_iter().map(|(k, v)| (k, v as f64)).collect()
    } else {
        let mut result = if mcs.iter().any(|&(_, ref mc)| mc.plays > 0) {
            let mut next = mcs.iter_mut().map(|&mut (r, ref mut mc)| (r, choose_move(rng, ggp, r, mc, 20.0))).collect::<Vec<_>>();
            let mvs = next.iter().map(|&(r, (ref mv, _))| (r, mv.clone())).collect::<Vec<_>>();
            ggp.play(&mvs[..]);
            let mvs2 = mvs.iter().map(|x| x.1.clone()).collect::<Vec<_>>();
            let result = tree_search(rng, ggp, &mut next.iter_mut().map(|&mut (r, (_, ref mut mc))| (r, mc.children.entry(mvs2.clone()).or_insert(MCTree::new()))).collect::<Vec<_>>()[..]);
            for (r, (_, mc)) in next {
                mc.payoff += result[r];
                mc.plays += 1;
            }
            result
        } else {
            play_out(rng, ggp, &mcs.iter().map(|&(r, _)| r).collect::<Vec<_>>()[..])
        };
        for (_, res) in result.iter_mut() {
            *res *= 0.99;
        }
        result
    };
    for &mut (r, ref mut mc) in mcs.iter_mut() {
        mc.payoff += result[r];
        mc.plays += 1;
    }
    result
}

fn play_out<'a, T: Rng>(rng: &mut T, ggp: &mut GGP<'a>, roles: &[&'a str]) -> HashMap<&'a str, f64> {
    if ggp.is_done() {
        ggp.goals().into_iter().map(|(k, v)| (k, v as f64)).collect()
    } else {
        let moves = roles.iter().map(|&r| {
            (r, rng.choose(&ggp.legal_moves_for(r)[..]).unwrap().clone())
        }).collect::<Vec<_>>();
        ggp.play(&moves[..]);
        let mut result = play_out(rng, ggp, roles);
        for (_, res) in result.iter_mut() {
            *res = *res * 0.99;
        }
        result
    }
}

fn main() {
    let srv = ServerBuilder::new().with_port(8080).build().unwrap();
    let mut rng = weak_rng();
    for mut req in srv.incoming_requests() {
        let mut body = String::new();
        req.as_reader().read_to_string(&mut body).unwrap();
        let ggp = GGP::from_rules(db_from_str(&body).unwrap());
        let mut mcs = ggp.roles().into_iter().map(|r| (r, MCTree::new())).collect::<Vec<_>>();
        for _ in 0.. {
            tree_search(&mut rng, &mut ggp.clone(), &mut mcs.iter_mut().map(|&mut (r, ref mut mc)| (r, mc)).collect::<Vec<_>>()[..]);
            for &(ref r, ref mc) in mcs.iter() {
                println!("For {}", r);
                print_mc(mc, None);
            }
        }
        req.respond(Response::from_string(""));
    }
}
