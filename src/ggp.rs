// A general game player

use std::iter::*;
use std::collections::HashMap;
use std::result::Result;

#[derive(Clone, Debug, PartialEq)]
struct Assignments {
    vars: HashMap<String, usize>,
    vals: Vec<Result<String, usize>> // TODO should I allow for variables to be equal to expressions?
} 

#[derive(Clone)]
pub enum Expr {
    Atom(String),
    Var(String),
    Pred(String, Box<[Expr]>)
}

#[derive(Clone)]
pub struct Fact {
    cons: Expr,
    pos: Box<[Expr]>,
    neg: Box<[Expr]>
}

impl Assignments {
    fn new() -> Assignments { Assignments { vars: HashMap::new(), vals: vec![] } }
    fn get_val(&self, mut i: usize) -> Result<String, usize> {
        loop {
            match &self.vals[i] {
                &Ok(ref x) => return Ok(x.clone()),
                &Err(j) => if i == j { return Err(j) } else { i = j }
            }
        }
    }
    fn unify(&self, left: &Expr, right: &Expr) -> Option<Assignments> {
        match (left, right) {
           (&Expr::Atom(_), &Expr::Pred(_, _)) | (&Expr::Pred(_, _), &Expr::Atom(_)) | (&Expr::Var(_), &Expr::Pred(_, _)) | (&Expr::Pred(_, _), &Expr::Var(_))  => None,
           (&Expr::Atom(ref x), &Expr::Atom(ref y)) => if x == y { Some(self.clone()) } else { None },
           (&Expr::Atom(ref x), &Expr::Var(ref y)) | (&Expr::Var(ref y), &Expr::Atom(ref x)) => {
               match self.vars.get(y) {
                   None => { let mut ret = self.clone(); ret.vars.insert(y.clone(), ret.vals.len()); ret.vals.push(Ok(x.clone())); Some(ret) },
                   Some(&i) => match self.get_val(i) {
                       Err(i) => { let mut ret = self.clone(); ret.vals[i] = Ok(x.clone()); Some(ret) },
                       Ok(ref v) => if x == v { Some(self.clone()) } else { None }
                   }
               }
           },
           (&Expr::Var(ref x), &Expr::Var(ref y)) => {
               match (self.vars.get(x), self.vars.get(y)) {
                   (None, None) => { let mut ret = self.clone(); let i = ret.vals.len(); ret.vars.insert(x.clone(), i); ret.vars.insert(y.clone(), i); ret.vals.push(Err(i)); Some(ret) },
                   (None, Some(&i)) => { let mut ret = self.clone(); ret.vars.insert(x.clone(), i); Some(ret) },
                   (Some(&i), None) => { let mut ret = self.clone(); ret.vars.insert(y.clone(), i); Some(ret) },
                   (Some(&i), Some(&j)) => match (self.get_val(i), self.get_val(j)) {
                       (Ok(x), Ok(y)) => if x == y { Some(self.clone()) } else { None },
                       (Ok(_), Err(j)) => { let mut ret = self.clone(); ret.vals[j] = Err(i); Some(ret) },
                       (Err(i), Ok(_)) => { let mut ret = self.clone(); ret.vals[i] = Err(j); Some(ret) },
                       (Err(i), Err(j)) => if i == j { Some(self.clone()) } else { None }
                   }
               }
           },
           (&Expr::Pred(ref l_name, ref l_args), &Expr::Pred(ref r_name, ref r_args)) => if l_name == r_name {
               l_args.iter().zip(r_args.iter()).fold(Some(self.clone()), |masg, (l, r)| masg.and_then(|asg| asg.unify(l, r)))
           } else { None }
        }
    }
}

#[derive(Clone)]
pub struct DB { facts: Vec<Fact> }

impl DB {
    pub fn new() -> DB { DB { facts: vec![] } }
    pub fn query<'a>(&'a self, expr: &'a Expr) -> Box<Iterator<Item=Assignments> + 'a> {
        self.query_inner(expr, Assignments::new())
    }
    pub fn check(&self, e: &Expr) -> bool {
        self.query(e).next().is_some()
    }
    pub fn add(&mut self, f: Fact) {
        self.facts.push(f)
    }
    fn query_inner<'a>(&'a self, expr: &'a Expr, asg: Assignments) -> Box<Iterator<Item=Assignments> + 'a> {
        Box::new(self.facts.iter().flat_map(move |&Fact { ref cons, ref pos, ref neg }| {
            let pos = pos.iter().fold(Box::new(asg.unify(expr, cons).into_iter()) as Box<Iterator<Item=Assignments> + 'a>, move |asgs, p| Box::new(asgs.flat_map(move |asg| self.query_inner(p, asg))));
            neg.iter().fold(pos, move |asgs, n| Box::new(asgs.filter(move |asg| self.query_inner(n, asg.clone()).next().is_none())))
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atoms() {
        let mut db = DB::new();
        let truth = Expr::Atom("truth".to_string());
        db.add(Fact { cons: truth.clone(), pos: Box::new([]), neg: Box::new([]) });
        assert!(db.check(&truth));
        assert!(!db.check(&Expr::Atom("falsity".to_string())));
    }

    #[test]
    fn vars() {
        let mut db = DB::new();
        assert!(!db.check(&Expr::Var("X".to_string())));
        db.add(Fact { cons: Expr::Atom("truth".to_string()), pos: Box::new([]), neg: Box::new([]) });
        assert!(db.check(&Expr::Var("X".to_string())));
    }

    #[test]
    fn preds() {
        let man_atom = Expr::Pred("man".to_string(), Box::new([Expr::Atom("socrates".to_string())]));
        let man_var = Expr::Pred("man".to_string(), Box::new([Expr::Var("X".to_string())]));
        let thing_atom = Expr::Pred("thing".to_string(), Box::new([Expr::Atom("socrates".to_string())]));
        let thing_var = Expr::Pred("thing".to_string(), Box::new([Expr::Var("X".to_string())]));
        let mut db = DB::new();
        db.add(Fact { pos: Box::new([]), cons: Expr::Pred("man".to_string(), Box::new([Expr::Atom("socrates".to_string())])), neg: Box::new([]) }); // man(socrates)
        db.add(Fact { pos: Box::new([]), cons: Expr::Pred("thing".to_string(), Box::new([Expr::Var("X".to_string())])), neg: Box::new([]) }); // thing(Y)
        assert_eq!(1, db.query(&man_atom).count()); // man(socrates)?
        assert_eq!(1, db.query(&man_var).count()); // man(X)?
        assert_eq!(1, db.query(&thing_atom).count()); // thing(socrates)?
        assert_eq!(1, db.query(&thing_var).count()); // thing(X)?
    }

    #[test]
    fn tic_tac_toe() {
        let roles = [Expr::Pred("role".to_string(), Box::new([Expr::Atom("x".to_string())])), Expr::Pred("role".to_string(), Box::new([Expr::Atom("x".to_string())]))];
        let mut db = DB::new();
        for r in roles.into_iter().cloned() { db.add(Fact { cons: r, pos: Box::new([]), neg: Box::new([]) }) }
    }
}
