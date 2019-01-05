use rand::Rng;
use rand::distributions::{Distribution, Standard};
use std::mem;
pub use self::Space::*;

#[cfg(test)]
use quickcheck::{Gen, Arbitrary};

#[derive(Debug, Clone)]
pub struct History {
    items: Vec<Space>,
    black : Option<Box<History>>,
    white : Option<Box<History>>,
    empty : Option<Box<History>>
}

impl History {
    pub fn new() -> History { History { items: Vec::new(), black: None, white: None, empty: None } }
    pub fn from_iter<I>(iter: I) -> History where I: Iterator<Item=Space> { History { items: iter.collect(), black: None, white: None, empty: None } }
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
                        Empty => &self.empty,
                        Black => &self.black,
                        White => &self.white
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
                            Empty => self.empty = new_next,
                            Black => self.black = new_next,
                            White => self.white = new_next
                        }
                        match cur {
                            Empty => self.empty = Some(Box::new(old_next)),
                            Black => self.black = Some(Box::new(old_next)),
                            White => self.white = Some(Box::new(old_next))
                        }
                        return true
                    },
                        None => return false
                }
            }
            let temp = self;
            let branch = match iter.next() {
                None => return false,
                     Some(Empty) => &mut temp.empty,
                     Some(Black) => &mut temp.black,
                     Some(White) => &mut temp.white
            };
            match *branch {
                None => { *branch = Some(Box::new(History::from_iter(iter))); return true }
                Some(ref mut next) => { self = next }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Space { Empty, Black, White }

impl Space {
    pub fn enemy(&self) -> Space {
        match *self {
            Empty => Empty,
            Black => White,
            White => Black
        }
    }
    pub fn is_empty(&self) -> bool {
        *self == Empty
    }
    pub fn is_filled(&self) -> bool {
        *self != Empty
    }
}

impl Distribution<Space> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Space {
        match rng.gen_range(0, 3) {
            0 => Empty,
            1 => Black,
            2 => White,
            _ => unreachable!()
        }
    }
}

#[cfg(test)]
impl Arbitrary for Space {
    fn arbitrary<G: Gen>(g: &mut G) -> Space {
        match usize::arbitrary(g) % 3 { 0 => Empty, 1 => Black, 2 => White, _ => unreachable!() }
    }
    fn shrink(self: &Space) -> Box<Iterator<Item=Space>> {
        use std::iter::{once, empty};
        match *self {
            Black => Box::new(once(Empty)),
            White => Box::new(once(Empty)),
            Empty => Box::new(empty())
        }
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
            TestResult::from_bool(h.contains(new.iter()) == xss.contains(&new))
        }
        quickcheck(test as fn(Vec<Vec<Space>>, Vec<Space>) -> TestResult);
    }

    #[bench]
    fn history_insert_19x19(bench: &mut Bencher) {
        let mut rng = rand::weak_rng();
        let mut h = History::new();
        bench.iter(|| {
            h.insert(rng.gen_iter().take(361))
        });
    }

    #[bench]
    fn hashset_insert_19x19(bench: &mut Bencher) {
        use std::collections::HashSet;
        let mut rng = rand::weak_rng();
        let mut h = HashSet::<Vec<Space>>::new();
        bench.iter(|| {
            h.insert(rng.gen_iter().take(361).collect())
        });
    }
}
