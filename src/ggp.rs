// A general game player

use std::iter::*;
use std::collections::HashMap;
use std::fmt::{Display, Formatter, Error};
use std::result::Result;
use self::Expr::*;
use self::{ValExpr as V};

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
pub enum ValExpr {
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
    fn to_val(&mut self, expr: &Expr) -> V {
        match expr {
            &Atom(ref x) => V::Atom(x.clone()),
            &Var(ref x) => {
                let vals = &mut self.vals;
                V::Var(*self.vars.entry(x.clone()).or_insert_with(|| { let i = vals.len(); vals.push(V::Var(i)); i }))
            },
            &Pred(ref name, ref args) => {
                let mut v_args = Vec::with_capacity(args.len());
                for arg in args.iter() {
                    v_args.push(self.to_val(arg));
                }
                V::Pred(name.clone(), v_args.into_boxed_slice())
            }
        }
    }
    fn from_val(&self, val: &V) -> Expr {
        match self.get_val(val) {
            V::Atom(x) => Atom(x),
            V::Var(i) => Var(i.to_string()),
            V::Pred(name, args) => Pred(name, args.iter().map(|arg| self.from_val(arg)).collect::<Vec<_>>().into_boxed_slice())
        }
    }
    fn unify(mut self, left: &Expr, right: &Expr) -> Option<Assignments> {
        let l_val = self.to_val(left);
        let r_val = self.to_val(right);
        if self.unify_val(&l_val, &r_val) { Some(self) } else { None }
    }
    fn unify_val(&mut self, left: &V, right: &V) -> bool {
        match (self.get_val(left), self.get_val(right)) {
           (V::Atom(x), V::Atom(y)) => x == y,
           (V::Pred(l_name, l_args), V::Pred(r_name, r_args)) => l_name == r_name && l_args.iter().zip(r_args.iter()).all(|(l, r)| self.unify_val(l, r)),
           (V::Var(i), V::Var(j)) => if i == j { true } else { false },
           (V::Var(i), x) | (x, V::Var(i)) => { self.vals[i] = x; true },
           (V::Atom(_), V::Pred(_, _)) | (V::Pred(_, _), V::Atom(_)) => false
        }
    }
}

#[derive(Clone)]
pub struct DB { facts: Vec<Fact> }

impl DB {
    pub fn new() -> DB { DB { facts: vec![] } }
    pub fn query<'a>(&'a self, expr: &'a Expr) -> Box<Iterator<Item=Expr> + 'a> {
        Box::new(self.query_inner(expr, Assignments::new()).map(move |mut asg| { let val = asg.to_val(expr); asg.from_val(&val) }))
    }
    pub fn check(&self, e: &Expr) -> bool {
        self.query(e).next().is_some()
    }
    pub fn add(&mut self, f: Fact) {
        self.facts.push(f)
    }
    fn query_inner<'a>(&'a self, expr: &'a Expr, asg: Assignments) -> Box<Iterator<Item=Assignments> + 'a> {
        Box::new(self.facts.iter().flat_map(move |&Fact { ref cons, ref pos, ref neg, ref distinct }| {
            let pos = pos.iter().fold(Box::new(asg.clone().unify(expr, cons).into_iter()) as Box<Iterator<Item=Assignments> + 'a>, move |asgs, p| Box::new(asgs.flat_map(move |asg| self.query_inner(p, asg))));
            let neg = neg.iter().fold(pos, move |asgs, n| Box::new(asgs.filter(move |asg| self.query_inner(n, asg.clone()).next().is_none())));
            distinct.iter().fold(neg, move |asgs, &(ref l, ref r)| Box::new(asgs.filter(move |asg| asg.clone().unify(l, r).is_none())))
        }))
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
        
        for r in roles.iter() { db.add(fact(pred("input", vec![atom(r), pred("mark", vec![var("M"), var("N")])]), vec![pred("role", vec![var("R")]), pred("index", vec![var("M")]), pred("index", vec![var("N")])], vec![], vec![])) }
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
        db.add(fact(pred("diagonal", vec![var("M"), var("X")]),
            (1..4).map(|i| pred("true", vec![pred("cell", vec![Atom(i.to_string()), Atom(i.to_string()), var("X")])])).collect(),
            vec![], vec![]));
        db.add(fact(pred("diagonal", vec![var("M"), var("X")]),
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

        // TODO build "true" from "init", then check the legal/next/etc.
    }
}
