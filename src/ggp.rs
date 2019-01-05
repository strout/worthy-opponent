// A general game player

use std::iter::*; use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt::{Display, Formatter, Error};
use std::result::Result;
pub use self::Expr::*;
use self::{ValArg as V};
use std::borrow::Cow;
use std::usize;
use labeler::Labeler;
use std::cmp::max;

#[derive(Clone, Debug)]
pub struct Assignments {
    vals: Vec<ValArg>
}

#[derive(Clone, Debug)]
enum ValArg {
    Const(usize),
    Var(usize),
    Filler
}

#[derive(Clone, Debug)]
struct ValExpr {
    name: usize,
    args: Vec<ValArg>
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Expr<'a> {
    Var(Cow<'a, str>),
    Pred(&'a str, Vec<Expr<'a>>)
}

impl<'a> Display for Expr<'a> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        match self {
            &Var(ref s) => write!(fmt, "?{}", s),
            &Pred(ref name, ref args) => {
                if args.is_empty() { 
                    name.fmt(fmt)
                } else { 
                    try!(write!(fmt, "({}", name));
                    for arg in args.iter() {
                        try!(write!(fmt, " {}", arg));
                    }
                    write!(fmt, ")")
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IArg {
    Const(usize),
    Var(usize),
    Filler
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IExpr {
    name: usize,
    args: Vec<IArg>
}

impl Display for IExpr {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        let write_interned = |fmt: &mut _, x: usize| if x > usize::MAX - 256 { (usize::MAX - x).fmt(fmt) } else { write!(fmt, "${}", x) };
        if self.args.is_empty() {
            write_interned(fmt, self.name)
        } else {
            try!(write!(fmt, "("));
            try!(write_interned(fmt, self.name));
            for arg in self.args.iter() {
                try!(write!(fmt, " {}", arg));
            }
            write!(fmt, ")")
        }
    }
}

impl Display for IArg {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        match *self {
            IArg::Const(x) => x.fmt(fmt),
            IArg::Var(x) => write!(fmt, "?{}", x),
            IArg::Filler => "?".fmt(fmt)
        }
    }
}

impl<'a> Expr<'a> {
    fn record_lens(&self, lens: &mut HashMap<&'a str, usize>) -> usize { // TODO each one should have some kind of inner_lens as well!
        match *self {
            Var(_) => 1,
            Pred(n, ref args) => {
                let mut largs = 0;
                for arg in args {
                    largs += arg.record_lens(lens)
                }
                1 + match lens.entry(n) {
                    Entry::Vacant(v) => { v.insert(largs); largs },
                    Entry::Occupied(mut o) => {
                        let n = max(*o.get(), largs);
                        o.insert(n);
                        n
                    }
                }
            }
        }
    }
    pub fn try_thru(&self, labeler: &Labeler, lens: &HashMap<&str, usize>) -> IExpr {
        match *self {
            Var(_) => panic!("Can't have a var at the top level."),
            Pred(n, ref args) => {
                let mut iargs = repeat(IArg::Filler).take(lens[n]).collect::<Vec<_>>();
                let mut i = 0;
                for arg in args {
                    match *arg {
                        Var(Cow::Borrowed(x)) => {
                            iargs[i] = IArg::Var(labeler.check(x).unwrap());
                            i += 1
                        }
                        Var(Cow::Owned(_)) => panic!("Don't use owned vars"),
                        Pred(_, _) => {
                            let IExpr { name, args: xs } = arg.try_thru(labeler, lens);
                            iargs[i] = IArg::Const(name);
                            i += 1;
                            for x in xs {
                                iargs[i] = x;
                                i += 1;
                            }
                        }
                    }
                }
                IExpr { name: labeler.check(n).unwrap(), args: iargs }
            }
        }
    }
    pub fn arg_thru(&self, labeler: &mut Labeler<'a>) -> IArg {
        match *self {
            Var(Cow::Borrowed(n)) => IArg::Var(labeler.put(n)),
            Var(Cow::Owned(_)) => panic!("What are you doing with an owned string?"),
            Pred(n, _) => IArg::Const(labeler.put(n))
        }
    }
    pub fn thru(&self, labeler: &mut Labeler<'a>, lens: &HashMap<&str, usize>) -> IExpr {
        match *self {
            Var(_) => panic!("Can't have a var at the top level."),
            Pred(n, ref args) => {
                let mut iargs = repeat(IArg::Filler).take(lens[n]).collect::<Vec<_>>();
                let mut i = 0;
                for arg in args {
                    match *arg {
                        Var(Cow::Borrowed(x)) => {
                            iargs[i] = IArg::Var(labeler.put(x));
                            i += 1
                        }
                        Var(Cow::Owned(_)) => panic!("Don't use owned vars"),
                        Pred(_, _) => {
                            let IExpr { name, args: xs } = arg.thru(labeler, lens);
                            iargs[i] = IArg::Const(name);
                            i += 1;
                            for x in xs {
                                iargs[i] = x;
                                i += 1;
                            }
                        }
                    }
                }
                IExpr { name: labeler.put(n), args: iargs }
            }
        }
    }
}

impl IExpr {
    pub fn thru<'a>(&self, labeler: &Labeler<'a>, lens: &HashMap<&str, usize>) -> Option<Expr<'a>> {
        go_thru(self.name, &self.args[..], labeler, lens)
    }
}

fn go_thru<'a>(name: usize, args: &[IArg], labeler: &Labeler<'a>, lens: &HashMap<&str, usize>) -> Option<Expr<'a>> {
    labeler.get(name).and_then(|n| {
        let mut eargs = vec![];
        let mut i = 0;
        while i < args.len() {
            i += 1;
            let a = match args[i - 1] {
                IArg::Filler => continue,
                IArg::Var(n) => labeler.get(n).map(|x| Var(x.into())),
                IArg::Const(x) => labeler.get(x).and_then(|n| {
                    let l = lens[n];
                    i += l;
                    go_thru(x, &args[i-l..i], labeler, lens)
                })
            };
            match a {
                None => return None,
                Some(a) => eargs.push(a)
            }
        }
        Some(Pred(n, eargs))
    })
}

pub type Parsed<'a, T> = Result<(T, &'a str), ()>;

fn skip_comments(s: &str) -> &str {
    let mut s = s.trim_left();
    while s.starts_with(';') {
        s = &s[s.find('\n').unwrap_or(s.len())..].trim_left();
    }
    s
}

fn parse_word(s: &str) -> Parsed<&str> {
    let s = skip_comments(s);
    let end = s.find(|c: char| c == '(' || c == ')' || c == ';' || c.is_whitespace()).unwrap_or(s.len());
    if end == 0 { Err(()) } else { Ok(s.split_at(end)) }
}

#[derive(Clone, Debug)]
pub enum SExpr<'a> {
    Atom(&'a str),
    List(Vec<SExpr<'a>>)
}

impl<'a> SExpr<'a> {
    pub fn as_str(&self) -> Option<&'a str> {
        match self {
            &SExpr::Atom(s) => Some(s),
            _ => None
        }
    }
    pub fn as_list(&self) -> Option<&[SExpr<'a>]> {
        match self {
            &SExpr::List(ref list) => Some(&list[..]),
            _ => None
        }
    }
}

pub fn parse_sexpr(s: &str) -> Parsed<SExpr> {
    let s = skip_comments(s);
    if s.starts_with('(') {
        let mut rest = skip_comments(&s[1..]);
        let mut args = vec![];
        while !rest.starts_with(')') && !rest.is_empty() {
            let (arg, next) = try!(parse_sexpr(rest));
            args.push(arg);
            rest = skip_comments(next);
        }
        if rest.is_empty() { Err(()) } else { Ok((SExpr::List(args), &rest[1..])) }
    } else {
        parse_word(s).map(|(s, r)| (SExpr::Atom(s), r))
    }
}

pub fn sexpr_to_expr<'a>(sexpr: &SExpr<'a>) -> Option<Expr<'a>> {
    match sexpr {
        &SExpr::Atom(s) => Some(if s.starts_with('?') { Var(s[1..].into()) } else { Pred(s, vec![]) }),
        &SExpr::List(ref args) => if args.is_empty() {
            None
        } else {
            let name = match &args[0] {
                &SExpr::Atom(s) => s,
                _ => return None
            };
            let mut pred_args = Vec::with_capacity(args.len() - 1);
            for arg in args.iter().skip(1) {
                match sexpr_to_expr(arg) {
                    None => return None,
                    Some(x) => pred_args.push(x)
                }
            }
            Some(Pred(name, pred_args))
        }
    }
}

#[derive(Clone, Debug)]
pub enum Thing<'a> {
    True(Expr<'a>),
    False(Expr<'a>),
    Distinct(Expr<'a>, Expr<'a>)
}

impl<'a> Thing<'a> {
    fn record_lens(&self, lens: &mut HashMap<&'a str, usize>) {
        match *self {
            Thing::True(ref expr) => { expr.record_lens(lens); },
            Thing::False(ref expr) => { expr.record_lens(lens); },
            Thing::Distinct(ref left, ref right) => {
                left.record_lens(lens);
                right.record_lens(lens);
            }
        }
    }
    fn thru(&self, labeler: &mut Labeler<'a>, lens: &HashMap<&str, usize>) -> IThing {
        match *self {
            Thing::True(ref expr) => IThing::True(expr.thru(labeler, lens)),
            Thing::False(ref expr) => IThing::False(expr.thru(labeler, lens)),
            Thing::Distinct(ref left, ref right) => IThing::Distinct(left.arg_thru(labeler), right.arg_thru(labeler))
        }
    }
}

#[derive(Clone, Debug)]
pub struct Fact<'a> {
    head: Expr<'a>,
    body: Vec<Thing<'a>>
}

#[derive(Clone, Debug)]
pub enum IThing {
    True(IExpr),
    False(IExpr),
    Distinct(IArg, IArg)
}

#[derive(Clone, Debug)]
pub struct IFact {
    head: IExpr,
    body: Vec<IThing>,
}

impl<'a> Fact<'a> {
    fn record_lens(&self, lens: &mut HashMap<&'a str, usize>) {
        self.head.record_lens(lens);
        for x in self.body.iter() {
            x.record_lens(lens);
        }
    }
    fn thru(&self, labeler: &mut Labeler<'a>, lens: &HashMap<&str, usize>) -> IFact {
        IFact {
            head: self.head.thru(labeler, lens),
            body: self.body.iter().map(|x| x.thru(labeler, lens)).collect::<Vec<_>>(),
        }
    }
}

impl<'a> Display for Thing<'a> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        match *self {
            Thing::True(ref e) => e.fmt(fmt),
            Thing::False(ref e) => write!(fmt, "(not {})", e),
            Thing::Distinct(ref l, ref r) => write!(fmt, "(distinct {} {})", l, r)
        }
    }
}

impl<'a> Display for Fact<'a> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        if self.body.is_empty() {
            write!(fmt, "{}", self.head)
        } else {
            try!(write!(fmt, "(<= {}", self.head));
            for p in self.body.iter() {
                try!(write!(fmt, " {}", p));
            }
            write!(fmt, ")")
        }
    }
}

fn add_arg<'a, 'b, I: Iterator<Item=Vec<Thing<'b>>> + 'a>(sofar: I, arg: &'a Expr<'b>) -> Box<Iterator<Item=Vec<Thing<'b>>> + 'a> {
    match arg {
        &Pred(ref name, ref args) => match name as &str {
            "not" => Box::new(sofar.map(move |mut body| { body.push(Thing::False(args[0].clone())); body })),
            "and" => Box::new(args.iter().fold(Box::new(sofar) as Box<Iterator<Item=_>>, add_arg)),
            "or" => Box::new(sofar.flat_map(move |body| args.iter().flat_map(move |arg| add_arg(Box::new(once(body.clone())), arg)))),
            "distinct" => Box::new(sofar.map(move |mut body| { body.push(Thing::Distinct(args[0].clone(), args[1].clone())); body })),
            _ => Box::new(sofar.map(move |mut body| { body.push(Thing::True(arg.clone())); body }))
        },
        &Var(_) => panic!("That's not right")
    }
}

fn add_args<'a, 'b, I: Iterator<Item=&'a Expr<'b>>>(head: &'a Expr<'b>, from: I) -> Box<Iterator<Item=Fact<'b>> + 'a> {
    let base = Box::new(once(vec![])) as Box<Iterator<Item=_>>;
    let ret = from.fold(base, add_arg);
    Box::new(ret.map(move |body| Fact { head: head.clone(), body: body }))
}

fn expr_to_fact<'a>(expr: Expr<'a>) -> Option<Vec<Fact<'a>>> {
    // TODO make this implementation less hacky -- right now it relies on Exprs instead of SExprs
    match expr {
        Var(_) => None,
        Pred(name, args) => {
            match &name as &str {
                "<=" => {
                    let (head, body) = args.split_at(1);
                    Some(add_args(&head[0], body.iter()).collect())
                },
                _ => Some(vec![Fact { head: Pred(name, args), body: vec![] }])
            }
        }
    }
}

impl Assignments {
    pub fn new() -> Assignments { Assignments { vals: vec![] } }
    fn get_val(&self, base: &V) -> V {
        match base {
            &V::Var(mut i) => {
                loop {
                    match &self.vals[i] {
                        &V::Var(j) if i != j => i = j,
                        x => return x.clone()
                    };
                }
            },
            _ => base.clone()
        }
    }
    fn to_val(&mut self, &IExpr { name, ref args }: &IExpr, vars: &mut HashMap<usize, usize>) -> ValExpr {
        let mut vargs = Vec::with_capacity(args.len());
        for arg in args {
            vargs.push(match *arg {
                IArg::Const(x) => V::Const(x),
                IArg::Var(x) => V::Var(*vars.entry(x).or_insert_with(|| { let i = self.vals.len(); self.vals.push(V::Var(i)); i })),
                IArg::Filler => V::Filler
            })
        }
        ValExpr { name: name, args: vargs }
    }
    fn to_valarg_immut(&self, arg: &IArg, vars: &HashMap<usize, usize>) -> ValArg {
        match *arg {
                IArg::Const(x) => V::Const(x),
                IArg::Var(x) => V::Var(vars[&x]),
                IArg::Filler => V::Filler
        }
    }
    fn to_val_immut(&self, &IExpr { name, ref args }: &IExpr, vars: &HashMap<usize, usize>) -> ValExpr {
        let mut vargs = Vec::with_capacity(args.len());
        for arg in args {
            vargs.push(self.to_valarg_immut(arg, vars))
        }
        ValExpr { name: name, args: vargs }
    }
    fn from_val(&self, val: &ValExpr) -> IExpr {
        IExpr { name: val.name, args: val.args.iter().map(|arg| match self.get_val(arg) {
            V::Const(x) => IArg::Const(x),
            V::Var(x) => IArg::Var(x),
            V::Filler => IArg::Filler
        }).collect() }
    }
    fn check_val(&self, left: &ValArg, right: &ValArg) -> bool {
        match (self.get_val(left), self.get_val(right)) {
            (V::Const(x), V::Const(y)) => x == y,
            (V::Filler, V::Filler) => true,
            (V::Var(_), _) | (_, V::Var(_)) => panic!("check_val shouldn't be called with unresolved vars"),
            (V::Const(_), V::Filler) | (V::Filler, V::Const(_)) => false
        }
    }
    fn unify_val(mut self, left: &ValExpr, right: &ValExpr) -> Option<Assignments> {
        debug_assert_eq!(left.args.len(), right.args.len());
        let ok = left.name == right.name && left.args.iter().zip(right.args.iter()).all(|(l, r)| match (self.get_val(l), self.get_val(r)) {
            (V::Const(x), V::Const(y)) => x == y,
            (V::Filler, V::Filler) => true,
            (V::Var(i), x) | (x, V::Var(i)) => { self.bind(i, x); true }
            (V::Const(_), V::Filler) | (V::Filler, V::Const(_)) => false
        });
        if ok { Some(self) } else { None }
    }
    fn bind(&mut self, mut var: usize, mut val: ValArg) {
        if let V::Var(x) = val {
            if x == var {
                return
            } else if x > var {
                val = V::Var(var);
                var = x;
            }
        }
        self.vals[var] = val;
    }
}

#[derive(Clone, Debug)]
pub struct DB { facts: HashMap<usize, Vec<IFact>> }

pub fn sexpr_to_db<'a>(sexpr: &SExpr<'a>) -> Option<(DB, Labeler<'a>, HashMap<&'a str, usize>)> {
    let mut facts = vec![];
    match sexpr {
        &SExpr::Atom(_) => return None,
        &SExpr::List(ref list) => {
            for sexpr in list {
                match sexpr_to_expr(sexpr).and_then(expr_to_fact) {
                    None => return None,
                    Some(mut fs) => facts.append(&mut fs)
                }
            }
        }
    }
    let mut lens = HashMap::new();
    for f in facts.iter() { f.record_lens(&mut lens) }
    let mut db = DB::new();
    let mut labeler = Labeler::new();
    for f in facts { db.add(f.thru(&mut labeler, &lens)) }
    Some((db, labeler, lens))
}

impl DB {
    pub fn new() -> DB { DB { facts: HashMap::new() } }
    pub fn query<'b>(&'b self, expr: &'b IExpr) -> Box<Iterator<Item=IExpr> + 'b> {
        let mut asg = Assignments::new();
        let expr = asg.to_val(expr, &mut HashMap::new());
        Box::new(self.query_inner(expr.clone(), asg).map(move |asg| { asg.from_val(&expr) }))
    }
    pub fn check(&self, e: &IExpr) -> bool {
        self.query(e).next().is_some()
    }
    pub fn add(&mut self, f: IFact) {
        self.facts.entry(f.head.name).or_insert(vec![]).push(f)
    }
    fn query_inner<'b>(&'b self, expr: ValExpr, asg: Assignments) -> Box<Iterator<Item=Assignments> + 'b> {
        match self.facts.get(&expr.name) {
            Some(relevant) => {
                let depth = asg.vals.len();
                Box::new(relevant.iter().flat_map(move |&IFact { ref head, ref body }| {
                    let mut asg = asg.clone();
                    let mut vars = HashMap::new();
                    let head = asg.to_val(head, &mut vars);
                    body.iter().fold(Box::new(asg.unify_val(&expr, &head).map(|asg| (asg, vars)).into_iter()) as Box<Iterator<Item=(Assignments, HashMap<usize, usize>)> + 'b>, move |asgs, t| {
                        match *t {
                            IThing::True(ref p) => {
                                Box::new(asgs.flat_map(move |(mut asg, mut vars)| {
                                    let p = asg.to_val(p, &mut vars);
                                    self.query_inner(p, asg).map(move |asg| (asg, vars.clone()))
                                }))
                            },
                            IThing::False(ref n) => {
                                Box::new(asgs.filter(move |&(ref asg, ref vars)| {
                                    let n = asg.to_val_immut(n, vars);
                                    self.query_inner(n, asg.clone()).next().is_none()
                                }))
                            },
                            IThing::Distinct(ref l, ref r) => {
                                Box::new(asgs.filter(move |&(ref asg, ref vars)| {
                                    let l = asg.to_valarg_immut(l, &vars);
                                    let r = asg.to_valarg_immut(r, &vars);
                                    !asg.check_val(&l, &r)
                                }))
                            }
                        }
                    }).map(move |(asg, _)| asg)
                }.map(move |mut asg| { asg.vals.truncate(depth); asg })))
            },
            None => {
                Box::new(empty())
            }
        }
    }
}

#[derive(Clone)]
pub struct GGP {
    db: DB,
    tru: usize,
    role: usize,
    legal: usize,
    legal_len: usize,
    does: usize,
    next: usize,
    next_len: usize,
    terminal: usize,
    goal: usize
}

impl GGP {
    pub fn from_rules(mut db: DB, labeler: &Labeler, lens: &HashMap<&str, usize>) -> Option<GGP> {
        let tru = match labeler.check("true") {
            Some(x) => x,
            None => return None
        };
        let role = match labeler.check("role") {
            Some(x) => x,
            None => return None
        };
        let legal = match labeler.check("legal") {
            Some(x) => x,
            None => return None
        };
        let legal_len = lens["legal"];
        let does = match labeler.check("does") {
            Some(x) => x,
            None => return None
        };
        let next = match labeler.check("next") {
            Some(x) => x,
            None => return None
        };
        let next_len = lens["next"];
        let terminal = match labeler.check("terminal") {
            Some(x) => x,
            None => return None
        };
        let goal = match labeler.check("goal") {
            Some(x) => x,
            None => return None
        };
        if let Some(init) = labeler.check("init") {
            let init_query = IExpr { name: init, args: (0..lens["init"]).map(IArg::Var).collect() };
            let inits = db.query(&init_query).collect::<Vec<_>>();
            for init in inits {
                db.add(IFact { head: IExpr { name: tru, args: init.args.clone() }, body: vec![] })
            }
        }
        Some(GGP { db: db, tru: tru, role: role, legal: legal, legal_len: legal_len, does: does, next: next, next_len: next_len, terminal: terminal, goal: goal })
    }
    pub fn roles(&self) -> Vec<usize> {
        let query = IExpr { name: self.role, args: vec![IArg::Var(0)] };
        let ret = { self.db.query(&query).map(|x| match x.args[0] {
            IArg::Const(x) => x,
            _ => unreachable!()
        }).collect() };
        ret
    }
    pub fn legal_moves_for(&self, r: usize) -> Vec<IExpr> {
        let query = IExpr { name: self.legal, args: once(IArg::Const(r)).chain((0..self.legal_len-1).map(IArg::Var)).collect() };
        let ret = self.db.query(&query).map(|x| {
            let args = x.args[2..].iter().cloned().collect();
            IExpr { name: match x.args[1] { IArg::Const(n) => n, _ => panic!("Legal move has no name?") }, args: args }
        }).collect();
        ret
    }
    pub fn play(&mut self, moves: &[(usize, IExpr)]) {
        for &(r, ref m) in moves.iter() {
            self.db.add(IFact { head: IExpr { name: self.does, args: once(IArg::Const(r)).chain(once(IArg::Const(m.name))).chain(m.args.iter().cloned()).collect() }, body: vec![] });
        }
        let next_query = IExpr { name: self.next, args: (0..self.next_len).map(IArg::Var).collect() };
        let mut nexts = self.db.query(&next_query).collect::<Vec<_>>();
        nexts.sort();
        nexts.dedup(); // TODO shouldn't have to do this..
        self.db.facts.get_mut(&self.does).unwrap().clear();
        self.db.facts.get_mut(&self.tru).unwrap().clear();
        for next in nexts {
            self.db.add(IFact { head: IExpr { name: self.tru, args: next.args.clone() }, body: vec![] })
        }
    }
    pub fn is_done(&self) -> bool {
        self.db.check(&IExpr { name: self.terminal, args: vec![] })
    }
    pub fn goals(&self) -> HashMap<usize, u8> {
        let query = IExpr { name: self.goal, args: vec![IArg::Var(0), IArg::Var(1)] };
        let ret = self.db.query(&query).map(|g| match (&g.args[0], &g.args[1]) {
            (&IArg::Const(r), &IArg::Const(s)) => (r, (usize::MAX - s) as u8),
            _ => unreachable!()
        }).collect();
        ret
    }
    pub fn state(&self) -> Vec<IExpr> {
        let query = IExpr { name: self.legal, args: (0..self.legal_len).map(IArg::Var).collect() }; // next len == true len
        let ret = self.db.query(&query).collect();
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use labeler::Labeler;
    use std::collections::HashMap;

    fn fact<'a>(head: Expr<'a>, pos: Vec<Expr<'a>>, neg: Vec<Expr<'a>>, distinct: Vec<(Expr<'a>, Expr<'a>)>) -> Fact<'a> { Fact { head: head, body: pos.into_iter().map(|p| Thing::True(p)).chain(neg.into_iter().map(|n| Thing::False(n))).chain(distinct.into_iter().map(|(l, r)| Thing::Distinct(l, r))).collect() } }
    fn pred<'a>(name: &'a str, args: Vec<Expr<'a>>) -> Expr<'a> { Pred(name.into(), args) }
    fn atom<'a>(x: &'a str) -> Expr<'a> { Pred(x.into(), vec![]) }
    fn var<'a>(x: &'a str) -> Expr<'a> { Var(x.into()) }

    #[test]
    fn atoms() {
        let mut db = DB::new();
        let mut labeler = Labeler::new();
        let mut lens = HashMap::new();
        let truth = atom("truth");
        let f = fact(truth.clone(), vec![], vec![], vec![]);
        f.record_lens(&mut lens);
        db.add(f.thru(&mut labeler, &lens));
        assert!(db.check(&truth.thru(&mut labeler, &lens)));
    }

    #[test]
    fn preds() {
        let mut db = DB::new();
        let mut labeler = Labeler::new();
        let mut lens = HashMap::new();
        let man_atom = pred("man", vec![atom("socrates")]);
        let man_var_ = pred("man", vec![var("X")]);
        man_var_.record_lens(&mut lens);
        let man_var = man_var_.thru(&mut labeler, &lens);
        let thing_atom_ = pred("thing", vec![atom("socrates")]);
        thing_atom_.record_lens(&mut lens);
        let thing_atom = thing_atom_.thru(&mut labeler, &lens);
        let thing_var = pred("thing", vec![var("X")]);
        db.add(fact(man_atom.clone(), vec![], vec![], vec![]).thru(&mut labeler, &lens));
        db.add(fact(thing_var.clone(), vec![], vec![], vec![]).thru(&mut labeler, &lens));
        let man_atom = man_atom.thru(&mut labeler, &lens);
        let thing_var = thing_var.thru(&mut labeler, &lens);
        assert_eq!(1, db.query(&man_atom).count()); // man(socrates)?
        assert_eq!(1, db.query(&man_var).count()); // man(X)?
        assert_eq!(1, db.query(&thing_atom).count()); // thing(socrates)?
        assert_eq!(1, db.query(&thing_var).count()); // thing(X)?
    }

    #[test]
    fn tic_tac_toe() {
        let (mut db, mut labeler, lens) = set_up_tic_tac_toe();

        let role_query = pred("role", vec![var("X")]).thru(&mut labeler, &lens);
        assert_eq!(2, db.query(&role_query).count());

        let input_query = pred("input", vec![var("R"), var("X"), var("Y"), var("Z")]).thru(&mut labeler, &lens);
        assert_eq!(20, db.query(&input_query).count());

        let base_query = pred("base", vec![var("X"), var("Y"), var("Z"), var("A")]).thru(&mut labeler, &lens);
        assert_eq!(29, db.query(&base_query).count());

        let init_query = pred("init", vec![var("X"), var("Y"), var("Z"), var("A")]).thru(&mut labeler, &lens);
        let init = db.query(&init_query).collect::<Vec<_>>();
        assert_eq!(10, init.len());

        for expr in init.iter() {
            db.add(IFact { head: IExpr { name: labeler.put("true"), args: expr.args.clone() }, body: vec![] });
        }

        let goal_query = pred("goal", vec![var("R"), var("X")]).thru(&mut labeler, &lens);
        assert_eq!(2, db.query(&goal_query).count());

        let legal_query = pred("legal", vec![var("R"), var("X"), var("Y"), var("Z")]).thru(&mut labeler, &lens);
        println!("{:?}", db.query(&legal_query).map(|x| x.thru(&labeler, &lens).unwrap()).collect::<Vec<_>>());
        assert_eq!(10, db.query(&legal_query).count());

        db.add(fact(pred("does", vec![atom("x"), pred("mark", vec![atom("1"), atom("1")])]), vec![], vec![], vec![]).thru(&mut labeler, &lens));
        db.add(fact(pred("does", vec![atom("o"), atom("noop")]), vec![], vec![], vec![]).thru(&mut labeler, &lens));

        // let next_query = pred("next", vec![var("X")]);
        // assert_eq!(10, db.query(&next_query).count()); // This would pass if results had no duplicates. (Should they?)
    }

    fn set_up_tic_tac_toe() -> (DB, Labeler<'static>, HashMap<&'static str, usize>) {
        // based on the example in http://games.stanford.edu/index.php/intro-to-gdl
        let mut db = DB::new();
        let mut labeler = Labeler::new();
        let mut lens = HashMap::new();
        let mut facts = vec![];

        let roles = ["x", "o"];
        for r in roles.iter() { facts.push(fact(pred("role", vec![atom(r)]), vec![], vec![], vec![])) }

        facts.push(fact(pred("input", vec![var("R"), pred("mark", vec![var("M"), var("N")])]), vec![pred("role", vec![var("R")]), pred("index", vec![var("M")]), pred("index", vec![var("N")])], vec![], vec![]));
        facts.push(fact(pred("input", vec![var("R"), atom("noop")]), vec![pred("role", vec![var("R")])], vec![], vec![]));

        for &i in ["1", "2", "3"].iter() { facts.push(fact(pred("index", vec![atom(i)]), vec![], vec![], vec![])) }

        for m in ["x", "o", "b"].iter() { facts.push(fact(pred("base", vec![pred("cell", vec![var("M"), var("N"), atom(m)])]), vec![pred("index", vec![var("M")]), pred("index", vec![var("N")])], vec![], vec![])) }
        for r in roles.iter() { facts.push(fact(pred("base", vec![pred("control", vec![atom(r)])]), vec![], vec![], vec![])) }

        for &x in ["1", "2", "3"].iter() { for &y in ["1", "2", "3"].iter() { facts.push(fact(pred("init", vec![pred("cell", vec![atom(x), atom(y), atom("b")])]), vec![], vec![], vec![])) } }
        facts.push(fact(pred("init", vec![pred("control", vec![atom("x")])]), vec![], vec![], vec![]));

        facts.push(fact(pred("legal", vec![var("W"), pred("mark", vec![var("X"), var("Y")])]),
            vec![pred("true", vec![pred("cell", vec![var("X"), var("Y"), atom("b")])]), pred("true", vec![pred("control", vec![var("W")])])],
            vec![], vec![]));
        facts.push(fact(pred("legal", vec![atom("x"), atom("noop")]), vec![pred("true", vec![pred("control", vec![atom("o")])])], vec![], vec![]));
        facts.push(fact(pred("legal", vec![atom("o"), atom("noop")]), vec![pred("true", vec![pred("control", vec![atom("x")])])], vec![], vec![]));
        
        facts.push(fact(pred("next", vec![pred("cell", vec![var("M"), var("N"), var("R")])]),
            vec![pred("does", vec![var("R"), pred("mark", vec![var("M"), var("N")])]),
                pred("true", vec![pred("cell", vec![var("M"), var("N"), atom("b")])])],
            vec![], vec![]));
        facts.push(fact(pred("next", vec![pred("cell", vec![var("M"), var("N"), var("W")])]),
            vec![pred("true", vec![pred("cell", vec![var("M"), var("N"), var("W")])])],
            vec![],
            vec![(var("W"), atom("b"))]));
        facts.push(fact(pred("next", vec![pred("cell", vec![var("M"), var("N"), atom("b")])]),
            vec![pred("does", vec![var("W"), pred("mark", vec![var("J"), var("K")])]),
                pred("true", vec![pred("cell", vec![var("M"), var("N"), atom("b")])])],
            vec![],
            vec![(var("M"), var("J"))]));
        facts.push(fact(pred("next", vec![pred("cell", vec![var("M"), var("N"), atom("b")])]),
            vec![pred("does", vec![var("W"), pred("mark", vec![var("J"), var("K")])]),
                pred("true", vec![pred("cell", vec![var("M"), var("N"), atom("b")])])],
            vec![],
            vec![(var("N"), var("K"))]));
        facts.push(fact(pred("next", vec![pred("control", vec![atom("o")])]),
            vec![pred("true", vec![pred("control", vec![atom("x")])])],
            vec![], vec![]));
        facts.push(fact(pred("next", vec![pred("control", vec![atom("x")])]),
            vec![pred("true", vec![pred("control", vec![atom("o")])])],
            vec![], vec![]));

        facts.push(fact(pred("goal", vec![atom("x"), atom("100")]),
            vec![pred("line", vec![atom("x")])],
            vec![pred("line", vec![atom("o")])],
            vec![]));
        facts.push(fact(pred("goal", vec![atom("x"), atom("50")]),
            vec![],
            vec![pred("line", vec![atom("x")]), pred("line", vec![atom("o")])],
            vec![]));
        facts.push(fact(pred("goal", vec![atom("x"), atom("0")]),
            vec![pred("line", vec![atom("o")])],
            vec![pred("line", vec![atom("x")])],
            vec![]));
        facts.push(fact(pred("goal", vec![atom("o"), atom("100")]),
            vec![pred("line", vec![atom("o")])],
            vec![pred("line", vec![atom("x")])],
            vec![]));
        facts.push(fact(pred("goal", vec![atom("o"), atom("50")]),
            vec![],
            vec![pred("line", vec![atom("o")]), pred("line", vec![atom("x")])],
            vec![]));
        facts.push(fact(pred("goal", vec![atom("o"), atom("0")]),
            vec![pred("line", vec![atom("x")])],
            vec![pred("line", vec![atom("o")])],
            vec![]));

        facts.push(fact(pred("line", vec![var("X")]),
            vec![pred("row", vec![var("M"), var("X")])],
            vec![],
            vec![]));
        facts.push(fact(pred("line", vec![var("X")]),
            vec![pred("column", vec![var("M"), var("X")])],
            vec![],
            vec![]));
        facts.push(fact(pred("line", vec![var("X")]),
            vec![pred("diagonal", vec![var("X")])],
            vec![],
            vec![]));

        facts.push(fact(pred("row", vec![var("M"), var("X")]),
            ["1","2","3"].iter().map(|&i| pred("true", vec![pred("cell", vec![var("M"), atom(i), var("X")])])).collect(),
            vec![], vec![]));
        facts.push(fact(pred("column", vec![var("M"), var("X")]),
            ["1","2","3"].iter().map(|&i| pred("true", vec![pred("cell", vec![atom(i), var("M"), var("X")])])).collect(),
            vec![], vec![]));
        facts.push(fact(pred("diagonal", vec![var("X")]),
            ["1","2","3"].iter().map(|&i| pred("true", vec![pred("cell", vec![atom(i), atom(i), var("X")])])).collect(),
            vec![], vec![]));
        facts.push(fact(pred("diagonal", vec![var("X")]),
            ["1","2","3"].iter().zip(["3","2","1"].iter()).map(|(&i, &j)| pred("true", vec![pred("cell", vec![atom(i), atom(j), var("X")])])).collect(),
            vec![], vec![]));

        facts.push(fact(atom("terminal"), vec![pred("line", vec![var("W")]), pred("role", vec![var("W")])], vec![], vec![]));
        facts.push(fact(atom("terminal"), vec![], vec![atom("open")], vec![]));

        facts.push(fact(atom("open"), vec![pred("true", vec![pred("cell", vec![var("M"), var("N"), atom("b")])])], vec![], vec![]));

        for f in facts.iter() { f.record_lens(&mut lens) }
        for f in facts { db.add(f.thru(&mut labeler, &lens)) }

        (db, labeler, lens)
    }
}
