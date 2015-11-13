extern crate tiny_http;
extern crate rand;
extern crate worthy_opponent;
extern crate thread_scoped;

use tiny_http::{ServerBuilder, Response, Header};
use rand::{Rng, weak_rng};
use std::collections::HashMap;
use worthy_opponent::ggp::{IExpr, Expr, GGP, SExpr, sexpr_to_db, sexpr_to_expr, parse_sexpr};
use std::hash::Hash;
use std::fmt::Display;
use std::sync::mpsc::{channel, Sender, Receiver, TryRecvError};
use thread_scoped::scoped;
use std::thread::{spawn, sleep_ms};

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

fn choose_move<'a, 'b, T: Rng>(rng: &mut T, ggp: &GGP<'b>, role: usize, mc: &'a mut MCTree<IExpr, Vec<IExpr>>, explore: f64) -> (IExpr, &'a mut MCTree<Vec<IExpr>, IExpr>) {
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

fn tree_search<T: Rng>(rng: &mut T, ggp: &mut GGP, mcs: &mut [(usize, &mut MCTree<IExpr, Vec<IExpr>>)]) -> HashMap<usize, f64> {
    let result = if ggp.is_done() {
        ggp.goals().into_iter().map(|(k, v)| (k, v as f64)).collect()
    } else {
        let mut result = if mcs.iter().any(|&(_, ref mc)| mc.plays > 0) {
            let mut next = mcs.iter_mut().map(|&mut (r, ref mut mc)| (r, choose_move(rng, ggp, r, mc, 20.0))).collect::<Vec<_>>();
            let mvs = next.iter().map(|&(r, (ref mv, _))| (r, mv.clone())).collect::<Vec<_>>();
            ggp.play(&mvs[..]);
            let mvs2 = mvs.iter().map(|x| x.1.clone()).collect::<Vec<_>>();
            let result = tree_search(rng, ggp, &mut next.iter_mut().map(|&mut (r, (_, ref mut mc))| (r, mc.children.entry(mvs2.clone()).or_insert(MCTree::new()))).collect::<Vec<_>>()[..]);
            for (r, (_, mc)) in next {
                mc.payoff += result[&r];
                mc.plays += 1;
            }
            result
        } else {
            play_out(rng, ggp, &mcs.iter().map(|&(r, _)| r).collect::<Vec<_>>()[..])
        };
        for (_, res) in result.iter_mut() {
            *res *= 0.99999;
        }
        result
    };
    for &mut (r, ref mut mc) in mcs.iter_mut() {
        mc.payoff += result[&r];
        mc.plays += 1;
    }
    result
}

fn play_out<T: Rng>(rng: &mut T, ggp: &mut GGP, roles: &[usize]) -> HashMap<usize, f64> {
    if ggp.is_done() {
        ggp.goals().into_iter().map(|(k, v)| (k, v as f64)).collect()
    } else {
        let moves = roles.iter().map(|&r| {
            (r, rng.choose(&ggp.legal_moves_for(r)[..]).unwrap().clone())
        }).collect::<Vec<_>>();
        ggp.play(&moves[..]);
        let mut result = play_out(rng, ggp, roles);
        for (_, res) in result.iter_mut() {
            *res = *res * 0.99999;
        }
        result
    }
}

enum Message<'a> {
    Info,
    Preview(SExpr<'a>, u32),
    Start(&'a str, &'a str, SExpr<'a>, u32, u32),
    Play(&'a str, Vec<Expr<'a>>),
    Stop(&'a str, Vec<Expr<'a>>),
    Abort(&'a str)
}

fn parse_message(s: &str) -> Option<Message> {
    parse_sexpr(s).ok().and_then(|(sexpr, _)| sexpr.as_list().and_then(|list| list[0].as_str().and_then(|name| match name {
        "info" => Some(Message::Info),
        "preview" => list[2].as_str().and_then(|s| s.parse().ok().map(|clock| Message::Preview(list[1].clone(), clock))),
        "start" => list[1].as_str().and_then(|id| list[2].as_str().and_then(|role| list[4].as_str().and_then(|s| s.parse().ok().and_then(|start_clock| list[5].as_str().and_then(|s| s.parse().ok().map(|play_clock| Message::Start(id, role, list[3].clone(), start_clock, play_clock))))))),
        "play" => list[1].as_str().and_then(|id| match list[2] {
            SExpr::Atom("nil") => Some(Message::Play(id, vec![])),
            SExpr::Atom(_) => None,
            SExpr::List(ref list) => Some(Message::Play(id, list.iter().flat_map(|sexpr| sexpr_to_expr(sexpr)).collect()))
        }),
        "stop" => list[1].as_str().and_then(|id| match list[2] {
            SExpr::Atom("nil") => Some(Message::Stop(id, vec![])),
            SExpr::Atom(_) => None,
            SExpr::List(ref list) => Some(Message::Stop(id, list.iter().flat_map(|sexpr| sexpr_to_expr(sexpr)).collect()))
        }),
        "abort" => list[1].as_str().map(|id| Message::Abort(id)),
        _ => None
    })))
}

fn think(role: usize, mut ggp: GGP, recvmvs: Receiver<Option<String>>, sendreplies: Sender<IExpr>) {
    let roles = ggp.roles();
    let mut mcs = roles.iter().map(|&r| (r, MCTree::new())).collect::<Vec<_>>();
    let mut rng = weak_rng();
    loop {
        match recvmvs.try_recv() {
            Err(TryRecvError::Empty) => {},
            Err(TryRecvError::Disconnected) => return,
            Ok(Some(mvs)) => {
                if let Some(Message::Play(_, mvs)) = parse_message(&mvs) {
                    for mv in mvs.iter() { println!("{}", mv) }
                    let mvs = roles.iter().cloned().zip(mvs.into_iter().map(|mv| mv.try_thru(&ggp.labeler).unwrap())).collect::<Vec<_>>();
                    let just_mvs = mvs.iter().map(|&(_, ref mv)| mv).cloned().collect::<Vec<_>>();
                    for (&mut (_, ref mut mc), &(_, ref mv)) in mcs.iter_mut().zip(mvs.iter()) {
                        // print_mc(mc, Some(mv));
                        *mc = mc.children.remove(mv).and_then(|mut mc| mc.children.remove(&just_mvs)).unwrap_or(MCTree::new());
                    }
                    if !mvs.is_empty() { ggp.play(&mvs[..]) }
                };
            },
            Ok(None) => {
                for &mut (r, ref mut mc) in mcs.iter_mut() {
                    if r == role {
                        let mv = choose_move(&mut rng, &ggp, role, mc, 0.0).0;
                        sendreplies.send(mv).unwrap()
                    }
                }
                // TODO send reply
            }
        }
        tree_search(&mut rng, &mut ggp.clone(), &mut mcs.iter_mut().map(|&mut (r, ref mut mc)| (r, mc)).collect::<Vec<_>>()[..]);
    }
}

fn run_match(desc: String, recvmvs: Receiver<String>, sendreply: Sender<String>) {
    if let Some(Message::Start(id, role, db, _, play_clock)) = parse_message(&desc) {
        let (db, labeler) = sexpr_to_db(&db).unwrap();
        let role = labeler.check(role).unwrap();
        let ggp = GGP::from_rules(db, labeler.clone()).unwrap();
        let (sendmovexprs, recvmovexprs) = channel();
        let (sendreplies, recvreplies) = channel();
        let handle = unsafe { scoped(move || think(role, ggp, recvmovexprs, sendreplies)) };
        println!("Match {} ready.", id);
        while let Ok(mvs) = recvmvs.recv() {
            sendmovexprs.send(Some(mvs)).unwrap();
            sleep_ms(play_clock * 1000 - 500);
            sendmovexprs.send(None).unwrap();
            sendreply.send(recvreplies.recv().unwrap().thru(&labeler).unwrap().to_string()).unwrap();
        }
        println!("Match {} shutting down.", id);
        drop(sendmovexprs);
        drop(recvreplies);
        handle.join();
    }
}

fn main() {
    let srv = ServerBuilder::new().with_port(if cfg!(debug_assertions) { 64335 } else { 9147 }).build().unwrap();
    let mut ongoing = HashMap::new();
    let mut body = String::new();
    for mut req in srv.incoming_requests() {
        body.clear();
        req.as_reader().read_to_string(&mut body).unwrap();
        let body = body.to_lowercase();
        println!("Got: [[\n{}\n]]", body);
        match parse_message(&body) {
            Some(Message::Info) => req.respond(Response::from_data("available").with_header(Header::from_bytes("Content-Type", "text/acl").unwrap())),
            Some(Message::Preview(_, _)) => req.respond(Response::from_data("ready").with_header(Header::from_bytes("Content-Type", "text/acl").unwrap())),
            Some(Message::Start(id, _, _, _, _)) => {
                let (sendmvs, recvmvs) = channel();
                let (sendreply, recvreply) = channel();
                ongoing.insert(id.to_string(), (sendmvs, recvreply));
                let desc = body.clone();
                spawn(move || run_match(desc, recvmvs, sendreply));
            },
            Some(Message::Play(id, _)) => if let Some(&mut (ref mut sendmvs, ref mut recvreply)) = ongoing.get_mut(id) {
                sendmvs.send(body.clone()).unwrap();
                req.respond(Response::from_data(recvreply.recv().unwrap()).with_header(Header::from_bytes("Content-Type", "text/acl").unwrap()))
            } else { println!("Asked to play in unknown match {}", id) },
            Some(Message::Stop(id, _)) | Some(Message::Abort(id)) => { ongoing.remove(id); },
            None => { println!("Bad request: {}", body); req.respond(Response::from_string("dunno-lol").with_status_code(400)) }
        }
    }
}
