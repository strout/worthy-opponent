extern crate tiny_http;
extern crate rand;
extern crate worthy_opponent;

use tiny_http::{Server, Request, Response};
use rand::{Rng, FromEntropy, rngs::SmallRng, seq::SliceRandom};
use std::collections::HashMap;
use worthy_opponent::ggp::{IExpr, Expr, GGP, SExpr, sexpr_to_db, sexpr_to_expr, parse_sexpr};
use std::hash::Hash;
use std::fmt::Display;
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::thread::{spawn, sleep};
use std::time::Duration;
use std::sync::{Mutex, Arc};

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

fn choose_move<'a, T: Rng>(rng: &mut T, ggp: &GGP, role: usize, mc: &'a mut MCTree<IExpr, Vec<IExpr>>, explore: f64) -> (IExpr, &'a mut MCTree<Vec<IExpr>, IExpr>) {
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
            if rng.gen_bool(1.0 / count as f64) { best = Some(mv.clone()) }
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
            let mut next = mcs.iter_mut().map(|&mut (r, ref mut mc)| (r, choose_move(rng, ggp, r, mc, 2.0_f64.sqrt()))).collect::<Vec<_>>();
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
            (r, ggp.legal_moves_for(r).choose(rng).unwrap().clone())
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
    Preview(SExpr<'a>, u64),
    Start(&'a str, &'a str, SExpr<'a>, u64, u64),
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

fn think(role: usize, mut ggp: GGP, recvmvs: Receiver<Vec<IExpr>>, reply: &Mutex<Option<IExpr>>) {
    let roles = ggp.roles();
    let mut mcs = roles.iter().map(|&r| (r, MCTree::new())).collect::<Vec<_>>();
    let mut rng = SmallRng::from_entropy();
    loop {
        match recvmvs.try_recv() {
            Err(TryRecvError::Empty) => {},
            Err(TryRecvError::Disconnected) => return,
            Ok(mvs) => {
                let mvs = roles.iter().cloned().zip(mvs).collect::<Vec<_>>();
                let just_mvs = mvs.iter().map(|&(_, ref mv)| mv).cloned().collect::<Vec<_>>();
                for (&mut (_, ref mut mc), &(_, ref mv)) in mcs.iter_mut().zip(mvs.iter()) {
                    print_mc(mc, Some(mv));
                    *mc = mc.children.remove(mv).and_then(|mut mc| mc.children.remove(&just_mvs)).unwrap_or(MCTree::new());
                }
                if !mvs.is_empty() { ggp.play(&mvs[..]) }
            }
        }
        tree_search(&mut rng, &mut ggp.clone(), &mut mcs.iter_mut().map(|&mut (r, ref mut mc)| (r, mc)).collect::<Vec<_>>()[..]);
        for &mut (r, ref mut mc) in mcs.iter_mut() {
            if r == role {
                let mv = choose_move(&mut rng, &ggp, role, mc, 0.0).0;
                *reply.lock().unwrap() = Some(mv);
            }
        }
    }
}

fn run_match(recvreqs: Receiver<(String, Request)>) {
    let (desc, req) = recvreqs.recv().unwrap();
    if let Some(Message::Start(id, role, db, start_clock, play_clock)) = parse_message(&desc) {
        let (db, labeler, lens) = sexpr_to_db(&db).unwrap();
        let role = labeler.check(role).unwrap();
        let ggp = GGP::from_rules(db, &labeler, &lens).unwrap();
        let (sendmovexprs, recvmovexprs) = channel();
        let reply = Arc::new(Mutex::new(None));
        {
            let reply = reply.clone();
            spawn(move || think(role, ggp, recvmovexprs, &*reply))
        };
        println!("Match {} preparing.", id);
        sleep(Duration::from_millis(1000 * start_clock - 200));
        req.respond(Response::from_data("ready")).unwrap();
        println!("Match {} started.", id);
        while let Ok((desc, req)) = recvreqs.recv() {
            if let Some(Message::Play(_, mvs)) = parse_message(&desc) {
                sendmovexprs.send(mvs.iter().map(|mv| mv.try_thru(&labeler, &lens)).collect()).unwrap();
                sleep(Duration::from_millis(play_clock * 1000 - 200));
                let mut reply = reply.lock().unwrap();
                req.respond(Response::from_data(match (&*reply).as_ref() {
                    None => "oops-i-timed-out".into(),
                    Some(mv) => mv.thru(&labeler, &lens).unwrap().to_string()
                })).unwrap();
                *reply = None;
            }
        }
        println!("Match {} shutting down.", id);
    }
}

fn main() {
    let srv = Server::http(if cfg!(debug_assertions) { "0.0.0.0:64335" } else { "0.0.0.0:9147" }).unwrap();
    let mut ongoing = HashMap::new();
    let mut body = String::new();
    for mut req in srv.incoming_requests() {
        println!("{} matches running.", ongoing.len());
        body.clear();
        req.as_reader().read_to_string(&mut body).unwrap();
        let body = body.to_lowercase();
        println!("Got: [[\n{}\n]]", body);
        match parse_message(&body) {
            Some(Message::Info) => req.respond(Response::from_data("available")).unwrap(),
            Some(Message::Preview(_, _)) => {},
            Some(Message::Start(id, _, _, _, _)) => {
                let (sendreqs, recvreqs) = channel();
                sendreqs.send((body.clone(), req)).unwrap();
                ongoing.insert(id.to_string(), sendreqs);
                spawn(move || run_match(recvreqs));
            },
            Some(Message::Play(id, _)) => if let Some(sendreqs) = ongoing.get_mut(id) {
                sendreqs.send((body.clone(), req)).unwrap();
            } else { println!("Asked to play in unknown match {}", id) },
            Some(Message::Stop(id, _)) | Some(Message::Abort(id)) => { ongoing.remove(id); },
            None => { println!("Bad request: {}", body); req.respond(Response::from_string("dunno-lol").with_status_code(400)).unwrap() }
        }
    }
}
