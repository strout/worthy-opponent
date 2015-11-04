// A general game player

use std::iter::{empty, once};
use std::collections::{HashSet, HashMap};

type Atom = String;

type Var = String;

enum Value {
    Free,
    Bound(Expr)
}

type Assignments = HashMap<Var, Value>;

enum Expr {
    Atom(Atom),
    Var(Var)
}

enum Fact {
    Atom(Atom)
}

struct DB { atoms: HashSet<Atom> }

impl DB {
    fn run<'a> (&'a self, e: &'a Expr) -> Box<Iterator<Item=Assignments> + 'a> {
        match *e {
            Expr::Atom(ref x) => Box::new(once(HashMap::new()).filter(move |_| self.atoms.contains(x))),
            Expr::Var(ref x) => Box::new(once({let mut h = HashMap::with_capacity(1); h.insert(x.clone(), Value::Free); h}))
        }
    }
    fn add(&mut self, f: Fact) {
        match f {
            Fact::Atom(x) => { self.atoms.insert(x); }
        }
    }
}
