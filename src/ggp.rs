// A general game player

use std::iter::*;
use std::collections::HashMap;
use std::fmt::{Display, Formatter, Error};
use std::result::Result;
use self::Expr::*;
use self::{ValExpr as V};
use std::mem::replace;

#[derive(Clone, Debug)]
struct Assignments {
    vars: HashMap<String, usize>,
    vals: Vec<ValExpr>
}

impl Display for Assignments {
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
enum ValExpr {
    Atom(String),
    Var(usize),
    Pred(String, Box<[ValExpr]>)
}

impl Display for ValExpr {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        match self {
            &V::Atom(ref s) => s.fmt(fmt),
            &V::Var(ref i) => write!(fmt, "?{}", i),
            &V::Pred(ref name, ref args) => {
                try!(write!(fmt, "{}(", name));
                let mut first = true;
                for arg in args.iter() {
                    if !first { try!(write!(fmt, ", ")) }
                    try!(arg.fmt(fmt));
                    first = false
                }
                write!(fmt, ")")
            }
        }
    }
}

#[derive(Clone)]
pub enum Expr {
    Atom(String),
    Var(String),
    Pred(String, Box<[Expr]>)
}

impl Display for Expr {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        match self {
            &Atom(ref s) => s.fmt(fmt),
            &Var(ref s) => s.fmt(fmt),
            &Pred(ref name, ref args) => {
                try!(write!(fmt, "{}(", name));
                let mut first = true;
                for arg in args.iter() {
                    if !first { try!(write!(fmt, ", ")) }
                    try!(arg.fmt(fmt));
                    first = false
                }
                write!(fmt, ")")
            }
        }
    }
}

#[derive(Clone)]
pub struct Fact {
    cons: Expr,
    pos: Box<[Expr]>,
    neg: Box<[Expr]>,
    distinct: Box<[(Expr, Expr)]>
}

impl Display for Fact {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        try!(write!(fmt, "{}", self.cons));
        if !self.pos.is_empty() || !self.neg.is_empty() || !self.distinct.is_empty() {
            try!(write!(fmt, " :- "));
        }
        let mut first = true;
        for p in self.pos.iter() {
            if !first { try!(write!(fmt, " & ")) }
            first = false;
            try!(write!(fmt, "{}", p));
        }
        for n in self.neg.iter() {
            if !first { try!(write!(fmt, " & ")) }
            first = false;
            try!(write!(fmt, "~{}", n));
        }
        for &(ref l, ref r) in self.distinct.iter() {
            if !first { try!(write!(fmt, " & ")) }
            first = false;
            try!(write!(fmt, "distinct({}, {})", l, r));
        }
        Ok(())
    }
}
impl Assignments {
    fn new() -> Assignments { Assignments { vars: HashMap::new(), vals: vec![] } }
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
    fn to_val(&mut self, expr: &Expr, suf: Option<usize>) -> V {
        match expr {
            &Atom(ref x) => V::Atom(x.clone()),
            &Var(ref x) => {
                let vals = &mut self.vals;
                let x = match suf { Some(n) => format!("{}${}", x, n), None => x.clone() };
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
    fn from_val(&self, val: &V) -> Expr {
        match self.get_val(val) {
            V::Atom(x) => Atom(x),
            V::Var(i) => Var(format!("?{}", i)),
            V::Pred(name, args) => Pred(name, args.iter().map(|arg| self.from_val(arg)).collect::<Vec<_>>().into_boxed_slice())
        }
    }
    fn unify(mut self, left: &Expr, l_depth: Option<usize>, right: &Expr, r_depth: Option<usize>) -> Option<Assignments> {
        let l_val = self.to_val(left, l_depth);
        let r_val = self.to_val(right, r_depth);
        if self.unify_val(&l_val, &r_val) {
            Some(self)
        } else {
            None
        }
    }
    fn unify_val(&mut self, left: &V, right: &V) -> bool {
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
pub struct DB { facts: HashMap<String, Vec<Fact>> }

impl Display for DB {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        for f in self.facts.values().flat_map(|x| x.iter()) { try!(writeln!(fmt, "{}", f)) }
        Ok(())
    }
}

impl DB {
    pub fn new() -> DB { DB { facts: HashMap::new() } }
    pub fn query<'a>(&'a self, expr: &'a Expr) -> Box<Iterator<Item=Expr> + 'a> {
        Box::new(self.query_inner(expr, Assignments::new(), None).map(move |mut asg| { let val = asg.to_val(expr, None); asg.from_val(&val) }))
    }
    pub fn check(&self, e: &Expr) -> bool {
        self.query(e).next().is_some()
    }
    pub fn add(&mut self, f: Fact) {
        let e = match f.cons {
            Atom(ref n) | Pred(ref n, _) => self.facts.entry(n.clone()),
            Var(_) => panic!("Can't add a variable as a fact.")
        };
        e.or_insert(vec![]).push(f)
    }
    fn query_inner<'a>(&'a self, expr: &'a Expr, asg: Assignments, depth: Option<usize>) -> Box<Iterator<Item=Assignments> + 'a> {
        let relevant : Box<Iterator<Item=&'a Vec<Fact>> + 'a> = match expr {
            &Var(_) => Box::new(self.facts.values()),
            &Atom(ref n) | &Pred(ref n, _) => Box::new(self.facts.get(n).into_iter())
        };
        Box::new(relevant.flat_map(|x| x.iter()).flat_map(move |&Fact { ref cons, ref pos, ref neg, ref distinct }| {
            let r_depth = Some(asg.vars.len());
            let pos = pos.iter().fold(Box::new(asg.clone().unify(expr, depth, &cons, r_depth).into_iter()) as Box<Iterator<Item=Assignments> + 'a>, move |asgs, p| Box::new(asgs.flat_map(move |asg| self.query_inner(p, asg, r_depth))));
            let neg = neg.iter().fold(pos, move |asgs, n| Box::new(asgs.filter(move |asg| self.query_inner(n, asg.clone(), r_depth).next().is_none())));
            distinct.iter().fold(neg, move |asgs, &(ref l, ref r)| Box::new(asgs.filter(move |asg| asg.clone().unify(l, r_depth, r, r_depth).is_none())))
        }))
    }
}

struct GGP {
    base: DB,
    cur: DB
}

impl GGP {
    fn roles(&self) -> Vec<Expr> {
        let query = Pred("role".into(), Box::new([Var("X".into())]));
        let ret = self.cur.query(&query).map(|x| match x { Pred(_, args) => args[1].clone(), _ => unreachable!() }).collect();
        ret
    }
    fn legal_moves_for(&self, r: &Expr) -> Vec<Expr> {
        let query = Pred("legal".into(), Box::new([r.clone(), Var("X".into())]));
        let ret = self.cur.query(&query).map(|x| match x { Pred(_, args) => args[1].clone(), _ => unreachable!() }).collect();
        ret
    }
    fn play(&mut self, moves: &[(Expr, Expr)]) {
        let mut db = replace(&mut self.cur, self.base.clone());
        for &(ref r, ref m) in moves.iter() {
            db.add(Fact { cons: Pred("does".into(), Box::new([r.clone(), m.clone()])), pos: Box::new([]), neg: Box::new([]), distinct: Box::new([]) });
        }
        let next_query = Pred("next".into(), Box::new([Var("X".into())]));
        for next in db.query(&next_query) {
            match next {
                Pred(_, args) => self.cur.add(Fact { cons: Pred("true".into(), args.clone()), pos: Box::new([]), neg: Box::new([]), distinct: Box::new([]) }),
                _ => unreachable!()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::Expr::*;

    fn fact(cons: Expr, pos: Vec<Expr>, neg: Vec<Expr>, distinct: Vec<(Expr, Expr)>) -> Fact { Fact { cons: cons, pos: pos.into_boxed_slice(), neg: neg.into_boxed_slice(), distinct: distinct.into_boxed_slice() } }
    fn pred(name: &str, args: Vec<Expr>) -> Expr { Pred(name.to_string(), args.into_boxed_slice()) }
    fn atom(x: &str) -> Expr { Atom(x.to_string()) }
    fn var(x: &str) -> Expr { Var(x.to_string()) }

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
        // based on the example in http://games.stanford.edu/index.php/intro-to-gdl
        let mut db = DB::new();

        let roles = ["x", "o"];
        for r in roles.iter() { db.add(fact(pred("role", vec![atom(r)]), vec![], vec![], vec![])) }
        
        db.add(fact(pred("input", vec![var("R"), pred("mark", vec![var("M"), var("N")])]), vec![pred("role", vec![var("R")]), pred("index", vec![var("M")]), pred("index", vec![var("N")])], vec![], vec![]));
        db.add(fact(pred("input", vec![var("R"), atom("noop")]), vec![pred("role", vec![var("R")])], vec![], vec![]));

        for i in 1..4 { db.add(fact(pred("index", vec![Atom(i.to_string())]), vec![], vec![], vec![])) }

        for m in ["x", "o", "b"].iter() { db.add(fact(pred("base", vec![pred("cell", vec![var("M"), var("N"), atom(m)])]), vec![pred("index", vec![var("M")]), pred("index", vec![var("N")])], vec![], vec![])) }
        for r in roles.iter() { db.add(fact(pred("base", vec![pred("control", vec![atom(r)])]), vec![], vec![], vec![])) }

        for x in 1..4 { for y in 1..4 { db.add(fact(pred("init", vec![pred("cell", vec![Atom(x.to_string()), Atom(y.to_string()), atom("b")])]), vec![], vec![], vec![])) } }
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
            (1..4).map(|i| pred("true", vec![pred("cell", vec![var("M"), Atom(i.to_string()), var("X")])])).collect(),
            vec![], vec![]));
        db.add(fact(pred("column", vec![var("M"), var("X")]),
            (1..4).map(|i| pred("true", vec![pred("cell", vec![Atom(i.to_string()), var("M"), var("X")])])).collect(),
            vec![], vec![]));
        db.add(fact(pred("diagonal", vec![var("X")]),
            (1..4).map(|i| pred("true", vec![pred("cell", vec![Atom(i.to_string()), Atom(i.to_string()), var("X")])])).collect(),
            vec![], vec![]));
        db.add(fact(pred("diagonal", vec![var("X")]),
            (1..4).map(|i| pred("true", vec![pred("cell", vec![Atom(i.to_string()), Atom((4-i).to_string()), var("X")])])).collect(),
            vec![], vec![]));

        db.add(fact(atom("terminal"), vec![pred("line", vec![var("W")])], vec![], vec![]));
        db.add(fact(atom("terminal"), vec![], vec![atom("open")], vec![]));

        db.add(fact(atom("open"), vec![pred("true", vec![pred("cell", vec![var("M"), var("N"), atom("b")])])], vec![], vec![]));

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
}
