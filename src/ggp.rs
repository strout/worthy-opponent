// A general game player

use std::iter::*;
use std::collections::HashMap;
use std::fmt::{Display, Formatter, Error};
use std::result::Result;
pub use self::Expr::*;
use self::{ValExpr as V};
use std::borrow::Cow;
use std::usize;
use labeler::Labeler;

#[derive(Clone, Debug)]
pub struct Assignments {
    vals: Vec<ValExpr>,
    binds: Vec<usize>
}

#[derive(Clone, Debug)]
enum ValExpr {
    Var(usize),
    Pred(usize, Vec<ValExpr>)
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
pub enum IExpr {
    Var(usize),
    Pred(usize, Vec<IExpr>)
}

impl Display for IExpr {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        let write_interned = |fmt: &mut _, x: usize| if x > usize::MAX - 256 { (usize::MAX - x).fmt(fmt) } else { write!(fmt, "${}", x) };
        match self {
            &IExpr::Var(x) => write!(fmt, "?{}", x),
            &IExpr::Pred(x, ref args) => {
                if args.is_empty() {
                    write_interned(fmt, x)
                } else {
                    try!(write!(fmt, "("));
                    try!(write_interned(fmt, x));
                    for arg in args.iter() {
                        try!(write!(fmt, " {}", arg));
                    }
                    write!(fmt, ")")
                }
            }
        }
    }
}

impl<'a> Expr<'a> {
    pub fn thru(&self, labeler: &mut Labeler<'a>) -> IExpr {
        match self {
            &Var(ref x) => IExpr::Var(labeler.put(match x { &Cow::Borrowed(x) => x, _ => panic!("I don't know what to do here.") })),
            &Pred(ref n, ref args) => IExpr::Pred(labeler.put(n), args.iter().map(|x| x.thru(labeler)).collect())
        }
    }
    pub fn try_thru(&self, labeler: &Labeler) -> Option<IExpr> {
        match self {
            &Var(ref x) => labeler.check(&*x).map(|x| IExpr::Var(x)),
            &Pred(ref n, ref args) => labeler.check(n).and_then(|n| args.iter().fold(Some(vec![]), |acc, arg| acc.and_then(|mut acc| arg.try_thru(labeler).map(|x| { acc.push(x); acc }))).map(|args| IExpr::Pred(n, args)))
        }
    }
}

impl IExpr {
    pub fn thru<'a>(&self, labeler: &Labeler<'a>) -> Option<Expr<'a>> {
        match self {
            &IExpr::Var(x) => labeler.get(x).map(|x| Var(x.into())).or_else(|| Some(Var(x.to_string().into()))),
            &IExpr::Pred(n, ref args) => labeler.get(n).and_then(|n| args.iter().fold(Some(vec![]), |acc, arg| acc.and_then(|mut acc| arg.thru(labeler).map(|x| { acc.push(x); acc }))).map(|args| Pred(n, args)))
        }
    }
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

#[derive(Clone)]
pub enum Thing<'a> {
    True(Expr<'a>),
    False(Expr<'a>),
    Distinct(Expr<'a>, Expr<'a>)
}

impl<'a> Thing<'a> {
    fn thru(&self, labeler: &mut Labeler<'a>) -> IThing {
        match *self {
            Thing::True(ref expr) => IThing::True(expr.thru(labeler)),
            Thing::False(ref expr) => IThing::False(expr.thru(labeler)),
            Thing::Distinct(ref left, ref right) => IThing::Distinct(left.thru(labeler), right.thru(labeler))
        }
    }
}

#[derive(Clone)]
pub struct Fact<'a> {
    head: Expr<'a>,
    body: Vec<Thing<'a>>
}

#[derive(Clone)]
pub enum IThing {
    True(IExpr),
    False(IExpr),
    Distinct(IExpr, IExpr)
}

#[derive(Clone)]
pub struct IFact {
    head: IExpr,
    body: Vec<IThing>,
}

impl<'a> Fact<'a> {
    fn thru(&self, labeler: &mut Labeler<'a>) -> IFact {
        IFact {
            head: self.head.thru(labeler),
            body: self.body.iter().map(|x| x.thru(labeler)).collect::<Vec<_>>(),
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
    pub fn new() -> Assignments { Assignments { vals: vec![], binds: vec![] } }
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
    fn to_val(&mut self, expr: &IExpr, vars: &mut HashMap<usize, usize>) -> V {
        match expr {
            &IExpr::Var(x) => {
                V::Var(*vars.entry(x).or_insert_with(|| { let i = self.vals.len(); self.vals.push(V::Var(i)); i }))
            },
            &IExpr::Pred(name, ref args) => {
                let mut v_args = Vec::with_capacity(args.len());
                for arg in args.iter() {
                    v_args.push(self.to_val(arg, vars));
                }
                V::Pred(name, v_args)
            }
        }
    }
    fn to_val_immut(&self, expr: &IExpr, vars: &HashMap<usize, usize>) -> V {
        match expr {
            &IExpr::Var(x) => {
                V::Var(vars[&x])
            },
            &IExpr::Pred(name, ref args) => {
                let mut v_args = Vec::with_capacity(args.len());
                for arg in args.iter() {
                    v_args.push(self.to_val_immut(arg, vars));
                }
                V::Pred(name, v_args)
            }
        }
    }
    fn from_val(&self, val: &V) -> IExpr {
        match self.get_val(val) {
            V::Var(i) => IExpr::Var(i),
            V::Pred(name, args) => IExpr::Pred(name, args.iter().map(|arg| self.from_val(arg)).collect())
        }
    }
    fn check_val(&self, left: &V, right: &V) -> bool {
        match (self.get_val(left), self.get_val(right)) {
            (V::Pred(l_name, l_args), V::Pred(r_name, r_args)) => l_name == r_name && l_args.iter().zip(r_args.iter()).all(|(l, r)| self.check_val(l, r)),
            (V::Var(_), _) | (_, V::Var(_)) => panic!("check_val shouldn't be called with unresolved vars.")
        }
    }
    fn unify_val(mut self, left: &V, right: &V) -> Result<Assignments, Assignments> {
        match (self.get_val(left), self.get_val(right)) {
            (V::Pred(l_name, l_args), V::Pred(r_name, r_args)) => if l_name == r_name {
                for (l, r) in l_args.iter().zip(r_args.iter()) {
                    match self.unify_val(l, r) {
                        Ok(slf) => self = slf,
                        Err(slf) => return Err(slf)
                    }
                }
                Ok(self)
            } else {
                Err(self)
            },
            (V::Var(i), x) | (x, V::Var(i)) => { self.bind(i, x); Ok(self) },
        }
    }
    fn bind(&mut self, var: usize, val: ValExpr) {
        self.vals[var] = val;
        self.binds.push(var)
    }
    fn unwind(&mut self, depth: usize) {
        while depth < self.binds.len() {
            let i = self.binds.pop().unwrap();
            if let Some(v) = self.vals.get_mut(i)
            {
                *v = V::Var(i)
            }
        }
    }
}

#[derive(Clone)]
pub struct DB { facts: HashMap<usize, Vec<IFact>> }

pub fn sexpr_to_db<'a>(sexpr: &SExpr<'a>) -> Option<(DB, Labeler<'a>)> {
    match sexpr {
        &SExpr::Atom(_) => return None,
        &SExpr::List(ref list) => {
            let mut db = DB::new();
            let mut labeler = Labeler::new();
            for sexpr in list.iter() {
                match sexpr_to_expr(sexpr) {
                    None => return None,
                    Some(expr) => match expr_to_fact(expr) {
                        None => return None,
                        Some(fs) => for f in fs { db.add(f.thru(&mut labeler)) }
                    }
                }
            }
            Some((db, labeler))
        }
    }
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
        let e = match f.head {
            IExpr::Pred(n, _) => self.facts.entry(n),
            IExpr::Var(_) => panic!("Can't add a variable as a fact.")
        };
        e.or_insert(vec![]).push(f)
    }
    fn query_inner<'b>(&'b self, expr: ValExpr, asg: Assignments) -> Box<Iterator<Item=Assignments> + 'b> {
        let relevant = match expr {
            V::Var(_) => panic!("Can't query for a variable."),
            V::Pred(n, _) => self.facts.get(&n).expect("Can't query something that isn't in the database.")
        };
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
                            let l = asg.to_val_immut(l, &vars);
                            let r = asg.to_val_immut(r, &vars);
                            !asg.check_val(&l, &r)
                        }))
                    }
                }
            }).map(move |(asg, _)| asg)
        }))
    }
}

#[derive(Clone)]
pub struct GGP {
    db: DB,
    tru: usize,
    role: usize,
    legal: usize,
    does: usize,
    next: usize,
    terminal: usize,
    goal: usize
}

impl GGP {
    pub fn from_rules(mut db: DB, labeler: &Labeler) -> Option<GGP> {
        let init = match labeler.check("init") {
            Some(x) => x,
            None => return None
        };
        let init_query = IExpr::Pred(init, vec![IExpr::Var(0)]);
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
        let does = match labeler.check("does") {
            Some(x) => x,
            None => return None
        };
        let next = match labeler.check("next") {
            Some(x) => x,
            None => return None
        };
        let terminal = match labeler.check("terminal") {
            Some(x) => x,
            None => return None
        };
        let goal = match labeler.check("goal") {
            Some(x) => x,
            None => return None
        };
        let inits = db.query(&init_query).collect::<Vec<_>>();
        for init in inits {
            match init {
                IExpr::Pred(_, args) => db.add(IFact { head: IExpr::Pred(tru, args.clone()), body: vec![] }),
                _ => unreachable!()
            }
        }
        Some(GGP { db: db, tru: tru, role: role, legal: legal, does: does, next: next, terminal: terminal, goal: goal })
    }
    pub fn roles(&self) -> Vec<usize> {
        let query = IExpr::Pred(self.role, vec![IExpr::Var(0)]);
        let ret = { self.db.query(&query).map(|x| match x {
            IExpr::Pred(_, args) => match &args[0] {
                &IExpr::Pred(x, _) => x,
                _ => unreachable!()
            },
            _ => unreachable!()
        }).collect() };
        ret
    }
    pub fn legal_moves_for(&self, r: usize) -> Vec<IExpr> {
        let query = IExpr::Pred(self.legal, vec![IExpr::Pred(r, vec![]), IExpr::Var(0)]);
        let ret = self.db.query(&query).map(|x| match x { IExpr::Pred(_, args) => args[1].clone(), _ => unreachable!() }).collect();
        ret
    }
    pub fn play(&mut self, moves: &[(usize, IExpr)]) {
        for &(r, ref m) in moves.iter() {
            self.db.add(IFact { head: IExpr::Pred(self.does, vec![IExpr::Pred(r, vec![]), m.clone()]), body: vec![] });
        }
        let next_query = IExpr::Pred(self.next, vec![IExpr::Var(0)]);
        let mut nexts = self.db.query(&next_query).collect::<Vec<_>>();
        nexts.sort();
        nexts.dedup(); // TODO shouldn't have to do this..
        self.db.facts.get_mut(&self.does).unwrap().clear();
        self.db.facts.get_mut(&self.tru).unwrap().clear();
        for next in nexts {
            match next {
                IExpr::Pred(_, args) => self.db.add(IFact { head: IExpr::Pred(self.tru, args.clone()), body: vec![] }),
                _ => unreachable!()
            }
        }
    }
    pub fn is_done(&self) -> bool {
        self.db.check(&IExpr::Pred(self.terminal, vec![]))
    }
    pub fn goals(&self) -> HashMap<usize, u8> {
        let query = IExpr::Pred(self.goal, vec![IExpr::Var(0), IExpr::Var(1)]);
        let ret = self.db.query(&query).map(|g| match g {
            IExpr::Pred(_, args) => match (&args[0], &args[1]) {
                (&IExpr::Pred(r, _), &IExpr::Pred(s, _)) => (r, (usize::MAX - s) as u8),
                _ => unreachable!()
            },
            _ => unreachable!()
        }).collect();
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use labeler::Labeler;

    fn fact<'a>(head: Expr<'a>, pos: Vec<Expr<'a>>, neg: Vec<Expr<'a>>, distinct: Vec<(Expr<'a>, Expr<'a>)>) -> Fact<'a> { Fact { head: head, body: pos.into_iter().map(|p| Thing::True(p)).chain(neg.into_iter().map(|n| Thing::False(n))).chain(distinct.into_iter().map(|(l, r)| Thing::Distinct(l, r))).collect() } }
    fn pred<'a>(name: &'a str, args: Vec<Expr<'a>>) -> Expr<'a> { Pred(name.into(), args) }
    fn atom<'a>(x: &'a str) -> Expr<'a> { Pred(x.into(), vec![]) }
    fn var<'a>(x: &'a str) -> Expr<'a> { Var(x.into()) }

    #[test]
    fn atoms() {
        let mut db = DB::new();
        let mut labeler = Labeler::new();
        let truth = atom("truth");
        db.add(fact(truth.clone(), vec![], vec![], vec![]).thru(&mut labeler));
        assert!(db.check(&truth.thru(&mut labeler)));
    }

    #[test]
    fn preds() {
        let mut db = DB::new();
        let mut labeler = Labeler::new();
        let man_atom = pred("man", vec![atom("socrates")]);
        let man_var = pred("man", vec![var("X")]).thru(&mut labeler);
        let thing_atom = pred("thing", vec![atom("socrates")]).thru(&mut labeler);
        let thing_var = pred("thing", vec![var("X")]);
        db.add(fact(man_atom.clone(), vec![], vec![], vec![]).thru(&mut labeler));
        db.add(fact(thing_var.clone(), vec![], vec![], vec![]).thru(&mut labeler));
        let man_atom = man_atom.thru(&mut labeler);
        let thing_var = thing_var.thru(&mut labeler);
        assert_eq!(1, db.query(&man_atom).count()); // man(socrates)?
        assert_eq!(1, db.query(&man_var).count()); // man(X)?
        assert_eq!(1, db.query(&thing_atom).count()); // thing(socrates)?
        assert_eq!(1, db.query(&thing_var).count()); // thing(X)?
    }

    #[test]
    fn tic_tac_toe() {
        let (mut db, mut labeler) = set_up_tic_tac_toe();

        let role_query = pred("role", vec![var("X")]).thru(&mut labeler);
        assert_eq!(2, db.query(&role_query).count());

        let input_query = pred("input", vec![var("R"), var("I")]).thru(&mut labeler);
        assert_eq!(20, db.query(&input_query).count());

        let base_query = pred("base", vec![var("X")]).thru(&mut labeler);
        assert_eq!(29, db.query(&base_query).count());

        let init_query = pred("init", vec![var("X")]).thru(&mut labeler);
        let init = db.query(&init_query).collect::<Vec<_>>();
        assert_eq!(10, init.len());

        for expr in init.iter() {
            if let &IExpr::Pred(_, ref args) = expr {
                db.add(IFact { head: IExpr::Pred(labeler.put("true"), args.iter().cloned().collect()), body: vec![] });
            } else { unreachable!() }
        }

        let goal_query = pred("goal", vec![var("R"), var("X")]).thru(&mut labeler);
        assert_eq!(2, db.query(&goal_query).count());

        let legal_query = pred("legal", vec![var("R"), var("X")]).thru(&mut labeler);
        let legal = db.query(&legal_query).collect::<Vec<_>>();
        assert_eq!(10, legal.len());

        db.add(fact(pred("does", vec![atom("x"), pred("mark", vec![atom("1"), atom("1")])]), vec![], vec![], vec![]).thru(&mut labeler));
        db.add(fact(pred("does", vec![atom("o"), atom("noop")]), vec![], vec![], vec![]).thru(&mut labeler));

        // let next_query = pred("next", vec![var("X")]);
        // assert_eq!(10, db.query(&next_query).count()); // This would pass if results had no duplicates. (Should they?)
    }

    #[bench]
    fn tic_tac_toe_playthrough(bench: &mut Bencher) {
        use rand::{weak_rng, Rng};

        let mut rng = weak_rng();
        let (db, labeler) = set_up_tic_tac_toe();
        let ggp = GGP::from_rules(db, &labeler).unwrap();

        assert!(!ggp.is_done());
        bench.iter(|| {
            let mut ggp = ggp.clone();
            let roles = ggp.roles();
            while !ggp.is_done() {
                let moves = roles.iter().map(|&r| {
                    let all = ggp.legal_moves_for(r);
                    assert!(!all.is_empty());
                    (r, rng.choose(&all[..]).unwrap().clone())
                }).collect::<Vec<_>>();
                ggp.play(&moves[..]);
            }
        });
    }

    fn set_up_tic_tac_toe() -> (DB, Labeler<'static>) {
        // based on the example in http://games.stanford.edu/index.php/intro-to-gdl
        let mut db = DB::new();
        let mut labeler = Labeler::new();

        let roles = ["x", "o"];
        for r in roles.iter() { db.add(fact(pred("role", vec![atom(r)]), vec![], vec![], vec![]).thru(&mut labeler)) }
        
        db.add(fact(pred("input", vec![var("R"), pred("mark", vec![var("M"), var("N")])]), vec![pred("role", vec![var("R")]), pred("index", vec![var("M")]), pred("index", vec![var("N")])], vec![], vec![]).thru(&mut labeler));
        db.add(fact(pred("input", vec![var("R"), atom("noop")]), vec![pred("role", vec![var("R")])], vec![], vec![]).thru(&mut labeler));

        for &i in ["1", "2", "3"].iter() { db.add(fact(pred("index", vec![atom(i)]), vec![], vec![], vec![]).thru(&mut labeler)) }

        for m in ["x", "o", "b"].iter() { db.add(fact(pred("base", vec![pred("cell", vec![var("M"), var("N"), atom(m)])]), vec![pred("index", vec![var("M")]), pred("index", vec![var("N")])], vec![], vec![]).thru(&mut labeler)) }
        for r in roles.iter() { db.add(fact(pred("base", vec![pred("control", vec![atom(r)])]), vec![], vec![], vec![]).thru(&mut labeler)) }

        for &x in ["1", "2", "3"].iter() { for &y in ["1", "2", "3"].iter() { db.add(fact(pred("init", vec![pred("cell", vec![atom(x), atom(y), atom("b")])]), vec![], vec![], vec![]).thru(&mut labeler)) } }
        db.add(fact(pred("init", vec![pred("control", vec![atom("x")])]), vec![], vec![], vec![]).thru(&mut labeler));

        db.add(fact(pred("legal", vec![var("W"), pred("mark", vec![var("X"), var("Y")])]),
            vec![pred("true", vec![pred("cell", vec![var("X"), var("Y"), atom("b")])]), pred("true", vec![pred("control", vec![var("W")])])],
            vec![], vec![]).thru(&mut labeler));
        db.add(fact(pred("legal", vec![atom("x"), atom("noop")]), vec![pred("true", vec![pred("control", vec![atom("o")])])], vec![], vec![]).thru(&mut labeler));
        db.add(fact(pred("legal", vec![atom("o"), atom("noop")]), vec![pred("true", vec![pred("control", vec![atom("x")])])], vec![], vec![]).thru(&mut labeler));
        
        db.add(fact(pred("next", vec![pred("cell", vec![var("M"), var("N"), var("R")])]),
            vec![pred("does", vec![var("R"), pred("mark", vec![var("M"), var("N")])]),
                pred("true", vec![pred("cell", vec![var("M"), var("N"), atom("b")])])],
            vec![], vec![]).thru(&mut labeler));
        db.add(fact(pred("next", vec![pred("cell", vec![var("M"), var("N"), var("W")])]),
            vec![pred("true", vec![pred("cell", vec![var("M"), var("N"), var("W")])])],
            vec![],
            vec![(var("W"), atom("b"))]).thru(&mut labeler));
        db.add(fact(pred("next", vec![pred("cell", vec![var("M"), var("N"), atom("b")])]),
            vec![pred("does", vec![var("W"), pred("mark", vec![var("J"), var("K")])]),
                pred("true", vec![pred("cell", vec![var("M"), var("N"), atom("b")])])],
            vec![],
            vec![(var("M"), var("J"))]).thru(&mut labeler));
        db.add(fact(pred("next", vec![pred("cell", vec![var("M"), var("N"), atom("b")])]),
            vec![pred("does", vec![var("W"), pred("mark", vec![var("J"), var("K")])]),
                pred("true", vec![pred("cell", vec![var("M"), var("N"), atom("b")])])],
            vec![],
            vec![(var("N"), var("K"))]).thru(&mut labeler));
        db.add(fact(pred("next", vec![pred("control", vec![atom("o")])]),
            vec![pred("true", vec![pred("control", vec![atom("x")])])],
            vec![], vec![]).thru(&mut labeler));
        db.add(fact(pred("next", vec![pred("control", vec![atom("x")])]),
            vec![pred("true", vec![pred("control", vec![atom("o")])])],
            vec![], vec![]).thru(&mut labeler));

        db.add(fact(pred("goal", vec![atom("x"), atom("100")]),
            vec![pred("line", vec![atom("x")])],
            vec![pred("line", vec![atom("o")])],
            vec![]).thru(&mut labeler));
        db.add(fact(pred("goal", vec![atom("x"), atom("50")]),
            vec![],
            vec![pred("line", vec![atom("x")]), pred("line", vec![atom("o")])],
            vec![]).thru(&mut labeler));
        db.add(fact(pred("goal", vec![atom("x"), atom("0")]),
            vec![pred("line", vec![atom("o")])],
            vec![pred("line", vec![atom("x")])],
            vec![]).thru(&mut labeler));
        db.add(fact(pred("goal", vec![atom("o"), atom("100")]),
            vec![pred("line", vec![atom("o")])],
            vec![pred("line", vec![atom("x")])],
            vec![]).thru(&mut labeler));
        db.add(fact(pred("goal", vec![atom("o"), atom("50")]),
            vec![],
            vec![pred("line", vec![atom("o")]), pred("line", vec![atom("x")])],
            vec![]).thru(&mut labeler));
        db.add(fact(pred("goal", vec![atom("o"), atom("0")]),
            vec![pred("line", vec![atom("x")])],
            vec![pred("line", vec![atom("o")])],
            vec![]).thru(&mut labeler));

        db.add(fact(pred("line", vec![var("X")]),
            vec![pred("row", vec![var("M"), var("X")])],
            vec![],
            vec![]).thru(&mut labeler));
        db.add(fact(pred("line", vec![var("X")]),
            vec![pred("column", vec![var("M"), var("X")])],
            vec![],
            vec![]).thru(&mut labeler));
        db.add(fact(pred("line", vec![var("X")]),
            vec![pred("diagonal", vec![var("X")])],
            vec![],
            vec![]).thru(&mut labeler));

        db.add(fact(pred("row", vec![var("M"), var("X")]),
            ["1","2","3"].iter().map(|&i| pred("true", vec![pred("cell", vec![var("M"), atom(i), var("X")])])).collect(),
            vec![], vec![]).thru(&mut labeler));
        db.add(fact(pred("column", vec![var("M"), var("X")]),
            ["1","2","3"].iter().map(|&i| pred("true", vec![pred("cell", vec![atom(i), var("M"), var("X")])])).collect(),
            vec![], vec![]).thru(&mut labeler));
        db.add(fact(pred("diagonal", vec![var("X")]),
            ["1","2","3"].iter().map(|&i| pred("true", vec![pred("cell", vec![atom(i), atom(i), var("X")])])).collect(),
            vec![], vec![]).thru(&mut labeler));
        db.add(fact(pred("diagonal", vec![var("X")]),
            ["1","2","3"].iter().zip(["3","2","1"].iter()).map(|(&i, &j)| pred("true", vec![pred("cell", vec![atom(i), atom(j), var("X")])])).collect(),
            vec![], vec![]).thru(&mut labeler));

        db.add(fact(atom("terminal"), vec![pred("line", vec![var("W")]), pred("role", vec![var("W")])], vec![], vec![]).thru(&mut labeler));
        db.add(fact(atom("terminal"), vec![], vec![atom("open")], vec![]).thru(&mut labeler));

        db.add(fact(atom("open"), vec![pred("true", vec![pred("cell", vec![var("M"), var("N"), atom("b")])])], vec![], vec![]).thru(&mut labeler));

        (db, labeler)
    }
}
