extern crate tiny_http;
extern crate rand;
extern crate worthy_opponent;
extern crate thread_scoped;

use tiny_http::{ServerBuilder, Response, Header};
use rand::{Rng, weak_rng};
use std::collections::HashMap;
use worthy_opponent::ggp::{Expr, Var, Atom, Pred, DB, GGP, parse_db, parse_expr, Parsed};
use std::hash::Hash;
use std::fmt::Display;
use std::result::Result;
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

enum Message<'a> {
    Info,
    Preview(DB<'a>, u32),
    Start(&'a str, &'a str, DB<'a>, u32, u32),
    Play(&'a str, Vec<Expr<'a>>),
    Stop(&'a str, Vec<Expr<'a>>),
    Abort(&'a str)
}

fn parse_exprs(mut s: &str) -> Parsed<Vec<Expr>> {
    let mut ret = vec![];
    s = &s.trim_left()[1..];
    loop {
        s = s.trim_left();
        if s.starts_with(')') { return Ok((ret, s)); }
        let (expr, rest) = try!(parse_expr(s).map_err(|_| println!("Bad expr with {}", s)));
        ret.push(expr);
        s = rest;
    }
}

fn parse_message(s: &str) -> Result<Message, ()> {
    let s = s.trim();
    Ok(
        if s == "(info)" { Message::Info }
        else if s.starts_with("(preview") {
            let (db, rest) = try!(parse_db(&s[8..]));
            let rest = rest.trim_left();
            let clock = try!(rest[..rest.len() - 1].parse().map_err(|_| ()));
            Message::Preview(db, clock)
        } else if s.starts_with("(start") {
            let rest = &s[6..].trim_left();
            let (id, rest) = rest.split_at(rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len()));
            let rest = rest.trim_left();
            let (role, rest) = rest.split_at(rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len()));
            let rest = rest.trim_left();
            let (db, rest) = try!(parse_db(&rest[1..]).map_err(|_| println!("\n\nDB parse failed for\n\n{}\n\n", rest)));
            let rest = rest[1..].trim_left();
            let (start_clock, rest) = rest.split_at(rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len()));
            let play_clock = rest[..rest.len() - 1].trim();
            Message::Start(id, role, db, try!(start_clock.parse().map_err(|err| println!("Bad start_clock: {} ({})", start_clock, err))), try!(play_clock.parse().map_err(|err| println!("Bad play_clock: {} ({})", play_clock, err))))
        } else if s.starts_with("(play") {
            let rest = &s[5..].trim_left();
            let (id, rest) = rest.split_at(rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len()));
            let rest = rest.trim_left();
            if rest.starts_with("nil") {
                Message::Play(id, vec![])
            } else {
                let (mvs, _) = try!(parse_exprs(rest));
                Message::Play(id, mvs)
            }
        } else if s.starts_with("(stop") {
            let rest = &s[5..].trim_left();
            let (id, rest) = rest.split_at(rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len()));
            let (mvs, _) = try!(parse_exprs(rest));
            Message::Stop(id, mvs)
        } else if s.starts_with("(abort") {
            Message::Abort(s[6..s.len() - 1].trim())
        } else { return Err(()) }
    )
}

fn same(l: &Expr, r: &Expr) -> bool {
    match (l, r) {
        (&Var(ref l), &Var(ref r)) => l == r,
        (&Atom(ref l), &Atom(ref r)) => l == r,
        (&Pred(ref ln, ref la), &Pred(ref rn, ref ra)) => ln == rn && la.len() == ra.len() && la.iter().zip(ra.iter()).all(|(l, r)| same(l, r)),
        _ => false
    }
}

fn think<'a>(role: &'a str, mut ggp: GGP<'a>, recvmvs: Receiver<Option<String>>, sendreplies: Sender<Expr<'a>>) {
    let roles = ggp.roles();
    let mut mcs = roles.iter().map(|&r| (r, MCTree::new())).collect::<Vec<_>>();
    let mut rng = weak_rng();
    loop {
        match recvmvs.try_recv() {
            Err(TryRecvError::Empty) => {},
            Err(TryRecvError::Disconnected) => return,
            Ok(Some(mvs)) => {
                if let Ok(Message::Play(_, mvs)) = parse_message(&mvs) {
                    for mv in mvs.iter() { println!("{}", mv) }
                    let mvs = roles.iter().cloned().zip(mvs.into_iter()) // TODO this is a hack because I need the right lifteime for the moves.
                        .map(|(r, mv)| {
                            let mv = ggp.legal_moves_for(r).into_iter().find(|x| same(x, &mv)).unwrap();
                            (r, mv)
                        }).collect::<Vec<_>>();
                    let just_mvs = mvs.iter().map(|&(_, ref mv)| mv).cloned().collect::<Vec<_>>();
                    for (&mut (_, ref mut mc), &(_, ref mv)) in mcs.iter_mut().zip(mvs.iter()) {
                        print_mc(mc, Some(mv));
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
    if let Ok(Message::Start(id, role, db, _, play_clock)) = parse_message(&desc) {
        let ggp = GGP::from_rules(db);
        let (sendmovexprs, recvmovexprs) = channel();
        let (sendreplies, recvreplies) = channel();
        let handle = unsafe { scoped(move || think(role, ggp, recvmovexprs, sendreplies)) };
        println!("Match {} ready.", id);
        while let Ok(mvs) = recvmvs.recv() {
            sendmovexprs.send(Some(mvs)).unwrap();
            sleep_ms(play_clock * 1000 - 500);
            sendmovexprs.send(None).unwrap();
            sendreply.send(recvreplies.recv().unwrap().to_string()).unwrap();
        }
        println!("Match {} shutting down.", id);
        drop(sendmovexprs);
        drop(recvreplies);
        handle.join();
    }
}

fn main() {
    let srv = ServerBuilder::new().with_port(64335).build().unwrap();
    let mut ongoing = HashMap::new();
    let mut body = String::new();
    for mut req in srv.incoming_requests() {
        body.clear();
        req.as_reader().read_to_string(&mut body).unwrap();
        println!("Got: [[\n{}\n]]", body);
        match parse_message(&body) {
            Ok(Message::Info) => req.respond(Response::from_data("available").with_header(Header::from_bytes("Content-Type", "text/acl").unwrap())),
            Ok(Message::Preview(_, _)) => req.respond(Response::from_data("ready").with_header(Header::from_bytes("Content-Type", "text/acl").unwrap())),
            Ok(Message::Start(id, _, _, _, _)) => {
                let (sendmvs, recvmvs) = channel();
                let (sendreply, recvreply) = channel();
                ongoing.insert(id.to_string(), (sendmvs, recvreply));
                let desc = body.clone();
                spawn(move || run_match(desc, recvmvs, sendreply));
            },
            Ok(Message::Play(id, _)) => if let Some(&mut (ref mut sendmvs, ref mut recvreply)) = ongoing.get_mut(id) {
                sendmvs.send(body.clone()).unwrap();
                req.respond(Response::from_data(recvreply.recv().unwrap()).with_header(Header::from_bytes("Content-Type", "text/acl").unwrap()))
            } else { println!("Asked to play in unknown match {}", id) },
            Ok(Message::Stop(id, _)) | Ok(Message::Abort(id)) => { ongoing.remove(id); },
            _ => { println!("Bad request: {}", body); req.respond(Response::from_string("dunno-lol").with_status_code(400)) }
        }
    }
}
