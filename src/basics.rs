use rand::{Rand, Rng};
use std::mem;
pub use self::Color::*;

#[cfg(test)]
use quickcheck::{Gen, Arbitrary};

#[derive(Debug)]
pub struct History {
    items: Vec<Space>,
    black : Option<Box<History>>,
    white : Option<Box<History>>,
    none : Option<Box<History>>
}

impl Clone for History {
    fn clone(self: &History) -> History {
        History { items: self.items.clone(), black: self.black.clone(), white: self.white.clone(), none: self.none.clone() }
    }
    fn clone_from(self: &mut History, other: &History) {
        self.items.clone_from(&other.items);
        self.black.clone_from(&other.black);
        self.white.clone_from(&other.white);
        self.none.clone_from(&other.none);
    }
}

impl History {
    pub fn new() -> History { History { items: Vec::new(), black: None, white: None, none: None } }
    pub fn from_iter<I>(iter: I) -> History where I: Iterator<Item=Space> { History { items: iter.collect(), black: None, white: None, none: None } }
    pub fn contains<'a, I>(mut self: &History, iter: I) -> bool where I: Iterator<Item=&'a Space> {
        let mut i = 0;
        for &item in iter {
            match self.items.get(i) {
                Some(&s) => {
                    if item != s { return false }
                    i += 1;
                }
                None => { // go to next
                    let branch = match item {
                        None => &self.none,
                        Some(Black) => &self.black,
                        Some(White) => &self.white
                    };
                    match *branch {
                        None => return false,
                        Some(ref h2) => { self = h2 }
                    }
                    i = 0
                }
            }
        }
        true
    }
    pub fn insert<I>(mut self: &mut History, mut iter: I) -> bool where I: Iterator<Item=Space> {
        loop {
            for i in 0..self.items.len() {
                match iter.next() {
                    Some(item) => if item != self.items[i] {
                        let old_items = if i + 1 >= self.items.len() { Vec::new() } else { self.items.split_off(i + 1) };
                        let cur = self.items.pop().unwrap();
                        let new_next = Some(Box::new(History::from_iter(iter)));
                        let mut old_next = mem::replace(self, History::new());
                        self.items = old_next.items;
                        old_next.items = old_items;
                        match item {
                            None => self.none = new_next,
                                 Some(Black) => self.black = new_next,
                                 Some(White) => self.white = new_next
                        }
                        match cur {
                            None => self.none = Some(Box::new(old_next)),
                                 Some(Black) => self.black = Some(Box::new(old_next)),
                                 Some(White) => self.white = Some(Box::new(old_next))
                        }
                        return true
                    },
                        None => return false
                }
            }
            let temp = self;
            let branch = match iter.next() {
                None => return false,
                     Some(None) => &mut temp.none,
                     Some(Some(Black)) => &mut temp.black,
                     Some(Some(White)) => &mut temp.white
            };
            match *branch {
                None => { *branch = Some(Box::new(History::from_iter(iter))); return true }
                Some(ref mut next) => { self = next }
            }
        }
    }
}

pub type Space = Option<Color>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color { Black, White }

impl Color {
    pub fn enemy(self) -> Color {
        match self {
            Black => White,
            White => Black
        }
    }
}

impl Rand for Color {
    fn rand<R: Rng>(rng: &mut R) -> Color {
        if rng.gen() { Black } else { White }
    }
}

#[cfg(test)]
impl Arbitrary for Color {
    fn arbitrary<G: Gen>(g: &mut G) -> Color {
        if bool::arbitrary(g) { Black } else { White }
    }
    fn shrink(self: &Color) -> Box<Iterator<Item=Color>> {
        Box::new(match *self {
            Black => true,
            White => false
        }.shrink().map(|b| if b { Black } else { White }))
    }
}

#[cfg(test)]
mod tests {
    use quickcheck::*;
    use super::*;
    use test::Bencher;
    use rand;

    #[test]
    fn history_present_iff_added() {
        fn test(xss: Vec<Vec<Space>>, new: Vec<Space>) -> TestResult {
            if new.len() == 0 { return TestResult::discard() }
            let mut h = History::new();
            for xs in xss.iter() { if xs.len() != new.len() { return TestResult::discard() }; h.insert(xs.iter().cloned()); }
            TestResult::from_bool(h.contains(new.iter().cloned()) == xss.contains(&new))
        }
        quickcheck(test as fn(Vec<Vec<Space>>, Vec<Space>) -> TestResult);
    }

    #[bench]
    fn history_insert_1000_24(bench: &mut Bencher) {
        let mut rng = rand::weak_rng();
        bench.iter(|| {
            let mut vs = Vec::<Vec<Option<Color>>>::with_capacity(1000);
            for _ in 0..1000 { vs.push(rng.gen_iter().take(24).collect()) }
            let mut h = History::new();
            for v in vs.iter() { h.insert(v.iter().cloned()); }
        });
    }

    #[bench]
    fn hashset_insert_1000_24(bench: &mut Bencher) {
        use std::collections::HashSet;
        let mut rng = rand::weak_rng();
        bench.iter(|| {
            let mut vs = Vec::<Vec<Option<Color>>>::with_capacity(1000);
            for _ in 0..1000 { vs.push(rng.gen_iter().take(24).collect()) }
            let mut h = HashSet::new();
            for v in vs.iter() { h.insert(v.clone()); }
        });
    }

    #[bench]
    fn history_insert_1000_19x19(bench: &mut Bencher) {
        let mut rng = rand::weak_rng();
        bench.iter(|| {
            let mut vs = Vec::<Vec<Option<Color>>>::with_capacity(1000);
            for _ in 0..1000 { vs.push(rng.gen_iter().take(361).collect()) }
            let mut h = History::new();
            for v in vs.iter() { h.insert(v.iter().cloned()); }
        });
    }

    #[bench]
    fn hashset_insert_1000_19x19(bench: &mut Bencher) {
        use std::collections::HashSet;
        let mut rng = rand::weak_rng();
        bench.iter(|| {
            let mut vs = Vec::<Vec<Option<Color>>>::with_capacity(1000);
            for _ in 0..1000 { vs.push(rng.gen_iter().take(361).collect()) }
            let mut h = HashSet::new();
            for v in vs.iter() { h.insert(v.clone()); }
        });
    }
}
