// A general game player

use std::iter::*;
use std::collections::HashMap;
use std::fmt::{Display, Formatter, Error};
use std::result::Result;
pub use self::Expr::*;
use self::{ValExpr as V};
use std::mem::replace;
use std::borrow::Cow;

#[derive(Clone, Debug)]
struct Assignments<'a> {
    vars: HashMap<Cow<'a, str>, usize>,
    vals: Vec<ValExpr<'a>>
}

impl<'a> Display for Assignments<'a> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        let mut first = true;
        for (k, &v) in self.vars.iter() {
            if !first { try!(write!(fmt, ", ")) }
            try!(write!(fmt, "{} = {}", k, self.from_val(&V::Var(v))));
            first = false
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
enum ValExpr<'a> {
    Atom(&'a str),
    Var(usize),
    Pred(&'a str, Box<[ValExpr<'a>]>)
}

impl<'a> Display for ValExpr<'a> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        match self {
            &V::Atom(ref s) => s.fmt(fmt),
            &V::Var(ref i) => write!(fmt, "?{}", i),
            &V::Pred(ref name, ref args) => {
                try!(write!(fmt, "({}", name));
                for arg in args.iter() {
                    try!(write!(fmt, " {}", arg))
                }
                write!(fmt, ")")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Expr<'a> {
    Atom(&'a str),
    Var(Cow<'a, str>),
    Pred(&'a str, Box<[Expr<'a>]>)
}

impl<'a> Display for Expr<'a> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        match self {
            &Atom(ref s) => s.fmt(fmt),
            &Var(ref s) => write!(fmt, "?{}", s),
            &Pred(ref name, ref args) => {
                try!(write!(fmt, "({}", name));
                for arg in args.iter() {
                    try!(write!(fmt, " {}", arg));
                }
                write!(fmt, ")")
            }
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
        &SExpr::Atom(s) => Some(if s.starts_with('?') { Var(s[1..].into()) } else { Atom(s) }),
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
            Some(Pred(name, pred_args.into_boxed_slice()))
        }
    }
}

#[derive(Clone)]
pub struct Fact<'a> {
    cons: Expr<'a>,
    pos: Box<[Expr<'a>]>,
    neg: Box<[Expr<'a>]>,
    distinct: Box<[(Expr<'a>, Expr<'a>)]>
}

impl<'a> Display for Fact<'a> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        if self.pos.is_empty() && self.neg.is_empty() && self.distinct.is_empty() {
            return write!(fmt, "{}", self.cons);
        }
        try!(write!(fmt, "(<= {}", self.cons));
        for p in self.pos.iter() {
            try!(write!(fmt, " {}", p));
        }
        for n in self.neg.iter() {
            try!(write!(fmt, " (not {})", n));
        }
        for &(ref l, ref r) in self.distinct.iter() {
            try!(write!(fmt, " distinct({}, {})", l, r));
        }
        write!(fmt, ")")
    }
}

fn add_arg<'a, 'b, I: Iterator<Item=(Vec<Expr<'b>>, Vec<Expr<'b>>, Vec<(Expr<'b>, Expr<'b>)>)> + 'a>(sofar: I, arg: &'a Expr<'b>) -> Box<Iterator<Item=(Vec<Expr<'b>>, Vec<Expr<'b>>, Vec<(Expr<'b>, Expr<'b>)>)> + 'a> {
    match arg {
        &Pred(ref name, ref args) => match name as &str {
            "not" => Box::new(sofar.map(move |(pos, mut neg, distinct)| { neg.push(args[0].clone()); (pos, neg, distinct) })),
            "and" => Box::new(args.iter().fold(Box::new(sofar) as Box<Iterator<Item=_>>, add_arg)),
            "or" => Box::new(sofar.flat_map(move |(pos, neg, distinct)| args.iter().flat_map(move |arg| add_arg(Box::new(once((pos.clone(), neg.clone(), distinct.clone()))), arg)))),
            "distinct" => Box::new(sofar.map(move |(pos, neg, mut distinct)| { distinct.push((args[0].clone(), args[1].clone())); (pos, neg, distinct) })),
            _ => Box::new(sofar.map(move |(mut pos, neg, distinct)| { pos.push(arg.clone()); (pos, neg, distinct) }))
        },
        _ => Box::new(sofar.map(move |(mut pos, neg, distinct)| { pos.push(arg.clone()); (pos, neg, distinct) }))
    }
}

fn add_args<'a, 'b, I: Iterator<Item=&'a Expr<'b>>>(cons: &'a Expr<'b>, from: I) -> Box<Iterator<Item=Fact<'b>> + 'a> {
    let base = Box::new(once((vec![], vec![], vec![]))) as Box<Iterator<Item=_>>;
    let ret = from.fold(base, add_arg);
    Box::new(ret.map(move |(pos, neg, distinct)| Fact { cons: cons.clone(), pos: pos.into_boxed_slice(), neg: neg.into_boxed_slice(), distinct: distinct.into_boxed_slice() }))
}

fn expr_to_fact<'a>(expr: Expr<'a>) -> Option<Vec<Fact<'a>>> {
    // TODO make this implementation less hacky -- right now it relies on Exprs instead of SExprs
    match expr {
        Var(_) => None,
        Atom(_) => Some(vec![Fact { cons: expr, pos: Box::new([]), neg: Box::new([]), distinct: Box::new([]) }]),
        Pred(name, args) => {
            match &name as &str {
                "<=" => {
                    let (head, body) = args.split_at(1);
                    Some(add_args(&head[0], body.iter()).collect())
                },
                _ => Some(vec![Fact { cons: Pred(name, args), pos: Box::new([]), neg: Box::new([]), distinct: Box::new([]) }])
            }
        }
    }
}

impl<'a> Assignments<'a> {
    fn new() -> Assignments<'a> { Assignments { vars: HashMap::new(), vals: vec![] } }
    fn get_val(&self, base: &V<'a>) -> V<'a> {
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
    fn to_val(&mut self, expr: &Expr<'a>, suf: Option<usize>) -> V<'a> {
        match expr {
            &Atom(ref x) => V::Atom(x.clone()),
            &Var(ref x) => {
                let vals = &mut self.vals;
                let x = match suf { Some(n) => format!("{}${}", x, n).into(), None => x.clone() };
                V::Var(*self.vars.entry(x).or_insert_with(|| { let i = vals.len(); vals.push(V::Var(i)); i }))
            },
            &Pred(ref name, ref args) => {
                let mut v_args = Vec::with_capacity(args.len());
                for arg in args.iter() {
                    v_args.push(self.to_val(arg, suf));
                }
                V::Pred(name.clone(), v_args.into_boxed_slice())
            }
        }
    }
    fn from_val(&self, val: &V<'a>) -> Expr<'a> {
        match self.get_val(val) {
            V::Atom(x) => Atom(x),
            V::Var(i) => Var(i.to_string().into()),
            V::Pred(name, args) => Pred(name, args.iter().map(|arg| self.from_val(arg)).collect::<Vec<_>>().into_boxed_slice())
        }
    }
    fn unify(mut self, left: &Expr<'a>, l_depth: Option<usize>, right: &Expr<'a>, r_depth: Option<usize>) -> Option<Assignments<'a>> {
        let l_val = self.to_val(left, l_depth);
        let r_val = self.to_val(right, r_depth);
        if self.unify_val(&l_val, &r_val) {
            Some(self)
        } else {
            None
        }
    }
    fn unify_val(&mut self, left: &V<'a>, right: &V<'a>) -> bool {
        match (self.get_val(left), self.get_val(right)) {
           (V::Atom(x), V::Atom(y)) => x == y,
           (V::Pred(l_name, l_args), V::Pred(r_name, r_args)) => l_name == r_name && l_args.iter().zip(r_args.iter()).all(|(l, r)| self.unify_val(l, r)),
           (V::Var(i), V::Var(j)) => { if i != j { self.vals[i] = V::Var(j) }; true },
           (V::Var(i), x) | (x, V::Var(i)) => { self.vals[i] = x; true },
           (V::Atom(_), V::Pred(_, _)) | (V::Pred(_, _), V::Atom(_)) => false
        }
    }
}

#[derive(Clone)]
pub struct DB<'a> { facts: HashMap<&'a str, Vec<Fact<'a>>> }

impl<'a> Display for DB<'a> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        for f in self.facts.values().flat_map(|x| x.iter()) { try!(writeln!(fmt, "{}", f)) }
        Ok(())
    }
}

pub fn sexpr_to_db<'a>(sexpr: &SExpr<'a>) -> Option<DB<'a>> {
    match sexpr {
        &SExpr::Atom(_) => return None,
        &SExpr::List(ref list) => {
            let mut db = DB::new();
            for sexpr in list.iter() {
                match sexpr_to_expr(sexpr) {
                    None => return None,
                    Some(expr) => match expr_to_fact(expr) {
                        None => return None,
                        Some(fs) => for f in fs { db.add(f) }
                    }
                }
            }
            Some(db)
        }
    }
}

impl<'a> DB<'a> {
    pub fn new() -> DB<'a> { DB { facts: HashMap::new() } }
    pub fn query<'b>(&'b self, expr: &'b Expr<'a>) -> Box<Iterator<Item=Expr<'a>> + 'b> {
        Box::new(self.query_inner(expr, Assignments::new(), None).map(move |mut asg| { let val = asg.to_val(expr, None); asg.from_val(&val) }))
    }
    pub fn check(&self, e: &Expr) -> bool {
        self.query(e).next().is_some()
    }
    pub fn add(&mut self, f: Fact<'a>) {
        let e = match f.cons {
            Atom(ref n) | Pred(ref n, _) => self.facts.entry(n.clone()),
            Var(_) => panic!("Can't add a variable as a fact.")
        };
        e.or_insert(vec![]).push(f)
    }
    fn query_inner<'b>(&'b self, expr: &'b Expr<'a>, asg: Assignments<'a>, depth: Option<usize>) -> Box<Iterator<Item=Assignments<'a>> + 'b> {
        let relevant : Box<Iterator<Item=&'b Vec<Fact>> + 'b> = match expr {
            &Var(_) => Box::new(self.facts.values()),
            &Atom(ref n) | &Pred(ref n, _) => Box::new(self.facts.get(n).into_iter())
        };
        Box::new(relevant.flat_map(|x| x.iter()).flat_map(move |&Fact { ref cons, ref pos, ref neg, ref distinct }| {
            let r_depth = Some(asg.vars.len());
            let pos = pos.iter().fold(Box::new(asg.clone().unify(expr, depth, &cons, r_depth).into_iter()) as Box<Iterator<Item=Assignments> + 'b>, move |asgs, p| Box::new(asgs.flat_map(move |asg| self.query_inner(p, asg, r_depth))));
            let neg = neg.iter().fold(pos, move |asgs, n| Box::new(asgs.filter(move |asg| self.query_inner(n, asg.clone(), r_depth).next().is_none())));
            distinct.iter().fold(neg, move |asgs, &(ref l, ref r)| Box::new(asgs.filter(move |asg| asg.clone().unify(l, r_depth, r, r_depth).is_none())))
        }))
    }
}

#[derive(Clone)]
pub struct GGP<'a> {
    base: DB<'a>,
    cur: DB<'a>
}

impl<'a> GGP<'a> {
    pub fn from_rules(base: DB<'a>) -> GGP<'a> {
        let mut cur = base.clone();
        let init_query = Pred("init".into(), Box::new([Var("X".into())]));
        for init in base.query(&init_query) {
            match init {
                Pred(_, args) => cur.add(Fact { cons: Pred("true".into(), args.clone()), pos: Box::new([]), neg: Box::new([]), distinct: Box::new([]) }),
                _ => unreachable!()
            }
        }
        GGP { base: base, cur: cur }
    }
    pub fn roles(&self) -> Vec<&'a str> {
        let query: Expr<'a> = Pred("role".into(), Box::new([Var("X".into())]));
        let ret = { self.cur.query(&query).map(|x| match x {
            Pred(_, args) => match &args[0] {
                &Atom(ref s) => s.clone(),
                _ => unreachable!()
            },
            _ => unreachable!()
        }).collect() };
        ret
    }
    pub fn legal_moves_for(&self, r: &'a str) -> Vec<Expr<'a>> {
        let query = Pred("legal".into(), Box::new([Atom(r.clone()), Var("X".into())]));
        let ret = self.cur.query(&query).map(|x| match x { Pred(_, args) => args[1].clone(), _ => unreachable!() }).collect();
        ret
    }
    pub fn play<'b>(&mut self, moves: &[(&'a str, Expr<'a>)]) where 'a {
        let mut db = replace(&mut self.cur, self.base.clone());
        for &(r, ref m) in moves.iter() {
            db.add(Fact { cons: Pred("does".into(), Box::new([Atom(r.clone()), m.clone()])), pos: Box::new([]), neg: Box::new([]), distinct: Box::new([]) });
        }
        let next_query = Pred("next".into(), Box::new([Var("X".into())]));
        let mut nexts = db.query(&next_query).collect::<Vec<_>>();
        nexts.sort();
        nexts.dedup(); // TODO shouldn't have to do this..
        for next in nexts {
            match next {
                Pred(_, args) => self.cur.add(Fact { cons: Pred("true".into(), args.clone()), pos: Box::new([]), neg: Box::new([]), distinct: Box::new([]) }),
                _ => unreachable!()
            }
        }
    }
    pub fn is_done(&self) -> bool {
        self.cur.check(&Atom("terminal".into()))
    }
    pub fn goals(&self) -> HashMap<&'a str, u8> {
        let query = Pred("goal".into(), Box::new([Var("X".into()), Var("Y".into())]));
        let ret = self.cur.query(&query).map(|g| match g {
            Pred(_, args) => match (&args[0], &args[1]) {
                (&Atom(ref r), &Atom(ref s)) => (r.clone(), s.parse().unwrap()),
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
    use super::Expr::*;
    use test::Bencher;

    fn fact<'a>(cons: Expr<'a>, pos: Vec<Expr<'a>>, neg: Vec<Expr<'a>>, distinct: Vec<(Expr<'a>, Expr<'a>)>) -> Fact<'a> { Fact { cons: cons, pos: pos.into_boxed_slice(), neg: neg.into_boxed_slice(), distinct: distinct.into_boxed_slice() } }
    fn pred<'a>(name: &'a str, args: Vec<Expr<'a>>) -> Expr<'a> { Pred(name.into(), args.into_boxed_slice()) }
    fn atom<'a>(x: &'a str) -> Expr<'a> { Atom(x.into()) }
    fn var<'a>(x: &'a str) -> Expr<'a> { Var(x.into()) }

    #[test]
    fn atoms() {
        let mut db = DB::new();
        let truth = atom("truth");
        db.add(fact(truth.clone(), vec![], vec![], vec![]));
        assert!(db.check(&truth));
        assert!(!db.check(&atom("falsity")));
    }

    #[test]
    fn vars() {
        let mut db = DB::new();
        assert!(!db.check(&var("X")));
        db.add(fact(atom("truth"), vec![], vec![], vec![]));
        assert!(db.check(&var("X")));
    }

    #[test]
    fn preds() {
        let mut db = DB::new();
        let man_atom = pred("man", vec![atom("socrates")]);
        let man_var = pred("man", vec![var("X")]);
        let thing_atom = pred("thing", vec![atom("socrates")]);
        let thing_var = pred("thing", vec![var("X")]);
        db.add(fact(man_atom.clone(), vec![], vec![], vec![]));
        db.add(fact(thing_var.clone(), vec![], vec![], vec![]));
        assert_eq!(1, db.query(&man_atom).count()); // man(socrates)?
        assert_eq!(1, db.query(&man_var).count()); // man(X)?
        assert_eq!(1, db.query(&thing_atom).count()); // thing(socrates)?
        assert_eq!(1, db.query(&thing_var).count()); // thing(X)?
    }

    #[test]
    fn tic_tac_toe() {
        let mut db = set_up_tic_tac_toe();

        let role_query = pred("role", vec![var("X")]);
        assert_eq!(2, db.query(&role_query).count());

        let input_query = pred("input", vec![var("R"), var("I")]);
        assert_eq!(20, db.query(&input_query).count());

        let base_query = pred("base", vec![var("X")]);
        assert_eq!(29, db.query(&base_query).count());

        let init_query = pred("init", vec![var("X")]);
        let init = db.query(&init_query).collect::<Vec<_>>();
        assert_eq!(10, init.len());

        for expr in init.iter() {
            if let &Pred(_, ref args) = expr {
                db.add(fact(pred("true", args.iter().cloned().collect()), vec![], vec![], vec![]));
            } else { unreachable!() }
        }

        let goal_query = pred("goal", vec![var("R"), var("X")]);
        assert_eq!(2, db.query(&goal_query).count());

        let legal_query = pred("legal", vec![var("R"), var("X")]);
        let legal = db.query(&legal_query).collect::<Vec<_>>();
        assert_eq!(10, legal.len());

        db.add(fact(pred("does", vec![atom("x"), pred("mark", vec![atom("1"), atom("1")])]), vec![], vec![], vec![]));
        db.add(fact(pred("does", vec![atom("o"), atom("noop")]), vec![], vec![], vec![]));

        // let next_query = pred("next", vec![var("X")]);
        // assert_eq!(10, db.query(&next_query).count()); // This would pass if results had no duplicates. (Should they?)
    }

    #[bench]
    fn tic_tac_toe_playthrough(bench: &mut Bencher) {
        use rand::{weak_rng, Rng};

        let mut rng = weak_rng();
        let ggp = GGP::from_rules(set_up_tic_tac_toe());

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

    fn set_up_tic_tac_toe() -> DB<'static> {
        // based on the example in http://games.stanford.edu/index.php/intro-to-gdl
        let mut db = DB::new();

        let roles = ["x", "o"];
        for r in roles.iter() { db.add(fact(pred("role", vec![atom(r)]), vec![], vec![], vec![])) }
        
        db.add(fact(pred("input", vec![var("R"), pred("mark", vec![var("M"), var("N")])]), vec![pred("role", vec![var("R")]), pred("index", vec![var("M")]), pred("index", vec![var("N")])], vec![], vec![]));
        db.add(fact(pred("input", vec![var("R"), atom("noop")]), vec![pred("role", vec![var("R")])], vec![], vec![]));

        for &i in ["1", "2", "3"].iter() { db.add(fact(pred("index", vec![Atom(i.into())]), vec![], vec![], vec![])) }

        for m in ["x", "o", "b"].iter() { db.add(fact(pred("base", vec![pred("cell", vec![var("M"), var("N"), atom(m)])]), vec![pred("index", vec![var("M")]), pred("index", vec![var("N")])], vec![], vec![])) }
        for r in roles.iter() { db.add(fact(pred("base", vec![pred("control", vec![atom(r)])]), vec![], vec![], vec![])) }

        for &x in ["1", "2", "3"].iter() { for &y in ["1", "2", "3"].iter() { db.add(fact(pred("init", vec![pred("cell", vec![Atom(x.into()), Atom(y.into()), atom("b")])]), vec![], vec![], vec![])) } }
        db.add(fact(pred("init", vec![pred("control", vec![atom("x")])]), vec![], vec![], vec![]));

        db.add(fact(pred("legal", vec![var("W"), pred("mark", vec![var("X"), var("Y")])]),
            vec![pred("true", vec![pred("cell", vec![var("X"), var("Y"), atom("b")])]), pred("true", vec![pred("control", vec![var("W")])])],
            vec![], vec![]));
        db.add(fact(pred("legal", vec![atom("x"), atom("noop")]), vec![pred("true", vec![pred("control", vec![atom("o")])])], vec![], vec![]));
        db.add(fact(pred("legal", vec![atom("o"), atom("noop")]), vec![pred("true", vec![pred("control", vec![atom("x")])])], vec![], vec![]));
        
        db.add(fact(pred("next", vec![pred("cell", vec![var("M"), var("N"), var("R")])]),
            vec![pred("does", vec![var("R"), pred("mark", vec![var("M"), var("N")])]),
                pred("true", vec![pred("cell", vec![var("M"), var("N"), atom("b")])])],
            vec![], vec![]));
        db.add(fact(pred("next", vec![pred("cell", vec![var("M"), var("N"), var("W")])]),
            vec![pred("true", vec![pred("cell", vec![var("M"), var("N"), var("W")])])],
            vec![],
            vec![(var("W"), atom("b"))]));
        db.add(fact(pred("next", vec![pred("cell", vec![var("M"), var("N"), atom("b")])]),
            vec![pred("does", vec![var("W"), pred("mark", vec![var("J"), var("K")])]),
                pred("true", vec![pred("cell", vec![var("M"), var("N"), atom("b")])])],
            vec![],
            vec![(var("M"), var("J"))]));
        db.add(fact(pred("next", vec![pred("cell", vec![var("M"), var("N"), atom("b")])]),
            vec![pred("does", vec![var("W"), pred("mark", vec![var("J"), var("K")])]),
                pred("true", vec![pred("cell", vec![var("M"), var("N"), atom("b")])])],
            vec![],
            vec![(var("N"), var("K"))]));
        db.add(fact(pred("next", vec![pred("control", vec![atom("o")])]),
            vec![pred("true", vec![pred("control", vec![atom("x")])])],
            vec![], vec![]));
        db.add(fact(pred("next", vec![pred("control", vec![atom("x")])]),
            vec![pred("true", vec![pred("control", vec![atom("o")])])],
            vec![], vec![]));

        db.add(fact(pred("goal", vec![atom("x"), atom("100")]),
            vec![pred("line", vec![atom("x")])],
            vec![pred("line", vec![atom("o")])],
            vec![]));
        db.add(fact(pred("goal", vec![atom("x"), atom("50")]),
            vec![],
            vec![pred("line", vec![atom("x")]), pred("line", vec![atom("o")])],
            vec![]));
        db.add(fact(pred("goal", vec![atom("x"), atom("0")]),
            vec![pred("line", vec![atom("o")])],
            vec![pred("line", vec![atom("x")])],
            vec![]));
        db.add(fact(pred("goal", vec![atom("o"), atom("100")]),
            vec![pred("line", vec![atom("o")])],
            vec![pred("line", vec![atom("x")])],
            vec![]));
        db.add(fact(pred("goal", vec![atom("o"), atom("50")]),
            vec![],
            vec![pred("line", vec![atom("o")]), pred("line", vec![atom("x")])],
            vec![]));
        db.add(fact(pred("goal", vec![atom("o"), atom("0")]),
            vec![pred("line", vec![atom("x")])],
            vec![pred("line", vec![atom("o")])],
            vec![]));

        db.add(fact(pred("line", vec![var("X")]),
            vec![pred("row", vec![var("M"), var("X")])],
            vec![],
            vec![]));
        db.add(fact(pred("line", vec![var("X")]),
            vec![pred("column", vec![var("M"), var("X")])],
            vec![],
            vec![]));
        db.add(fact(pred("line", vec![var("X")]),
            vec![pred("diagonal", vec![var("X")])],
            vec![],
            vec![]));

        db.add(fact(pred("row", vec![var("M"), var("X")]),
            ["1","2","3"].iter().map(|&i| pred("true", vec![pred("cell", vec![var("M"), Atom(i.into()), var("X")])])).collect(),
            vec![], vec![]));
        db.add(fact(pred("column", vec![var("M"), var("X")]),
            ["1","2","3"].iter().map(|&i| pred("true", vec![pred("cell", vec![Atom(i.into()), var("M"), var("X")])])).collect(),
            vec![], vec![]));
        db.add(fact(pred("diagonal", vec![var("X")]),
            ["1","2","3"].iter().map(|&i| pred("true", vec![pred("cell", vec![Atom(i.into()), Atom(i.into()), var("X")])])).collect(),
            vec![], vec![]));
        db.add(fact(pred("diagonal", vec![var("X")]),
            ["1","2","3"].iter().zip(["3","2","1"].iter()).map(|(&i, &j)| pred("true", vec![pred("cell", vec![Atom(i.into()), Atom(j.into()), var("X")])])).collect(),
            vec![], vec![]));

        db.add(fact(atom("terminal"), vec![pred("line", vec![var("W")]), pred("role", vec![var("W")])], vec![], vec![]));
        db.add(fact(atom("terminal"), vec![], vec![atom("open")], vec![]));

        db.add(fact(atom("open"), vec![pred("true", vec![pred("cell", vec![var("M"), var("N"), atom("b")])])], vec![], vec![]));

        db
    }
}
