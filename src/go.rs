const SIZE : usize = 9;

use bit_set::BitSet;
use game::Game;
use basics::*;

type Pos = usize;

fn neighbors(p: Pos) -> [Option<usize>; 4] {
    [if p % SIZE > 0 { Some(p - 1) } else { None },
        if p % SIZE < SIZE - 1 { Some(p + 1) } else { None },
            if p / SIZE > 0 { Some(p - SIZE) } else { None },
                if p / SIZE < SIZE - 1 { Some(p + SIZE) } else { None }]
}

fn owner_inner(c: Space, p: Pos, b: &[Space], visited: &mut BitSet, white: bool, black: bool) -> (bool, bool, usize) {
    let c2 = b[p];
    let (mut black, mut white) = match (c, c2) {
        (Empty, White) => (black, true),
        (Empty, Black) => (true, white),
        (Empty, Empty) => (black, white),
        (c1, _) => (c1 == Black, c1 == White) // TODO this comparison will run WAY too many times!
    };
    if c == c2 {
        if visited.contains(p) { return (black, white, 0) }
        visited.insert(p);
        let mut sum = 0;
        for &p in neighbors(p).into_iter().flat_map(|x| x) {
            let (b, w, s) = owner_inner(c, p, b, visited, white, black);
            black = b;
            white = w;
            sum += s;
        }
        (black, white, sum + 1)
    } else { (black, white, 0) }
}

fn score(board: &[Space]) -> (usize, usize) {
    let mut b = 0;
    let mut w = 0;
    let mut visited = BitSet::with_capacity(SIZE * SIZE);
    for p in 0..SIZE * SIZE {
        if !visited.contains(p) {
            match owner_inner(board[p], p, board, &mut visited, false, false) {
                (true, false, n) => b += n,
                    (false, true, n) => w += n,
                    (_, _, _) => {}
            }
        }
    }
    (b, w)
}

fn capture_inner(c: Space, p: Pos, b: &[Space], captured: &mut BitSet) -> bool {
    if !captured.contains(p) {
        match b[p] {
            Empty => return false,
            c2 => if c == c2 {
                captured.insert(p);
                if neighbors(p).into_iter().flat_map(|x| x).any(|&p| !capture_inner(c, p, b, captured)) { return false }
            }
        }
    }
    true
}

fn capture(c: Space, p: Pos, b: &[Space]) -> BitSet {
    let mut captured = BitSet::with_capacity(SIZE * SIZE);
    if !capture_inner(c, p, b, &mut captured) { captured.clear(); }
    captured
}

const UPPER_LETTERS : &'static str = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
const DOWN_LETTERS : &'static str = "abcdefghjklmnopqrstuvwxyz";
 
fn parse_pos(s: &str) -> Option<Pos> {
    if s.len() < 1 { return None }
    if s == "XX" || s == "xx" { return Some(SIZE * SIZE) }
    let y = s[1..].parse::<usize>().ok();
    let xch = s.chars().nth(0);
    xch.and_then(|xch| y.and_then(|y| UPPER_LETTERS.chars().position(|c| c == xch).or_else(|| DOWN_LETTERS.chars().position(|c| c == xch)).map(|x| (y - 1) * SIZE + x)))
}

fn to_letter(n : usize) -> char {
    UPPER_LETTERS.chars().nth(n).unwrap()
}

fn print_pos(p: usize) {
    if p == SIZE * SIZE { print!("XX") } else { print!("{}{}", to_letter(p % SIZE), 1 + p / SIZE); }
}

fn print_board(b: &[Space]) {
    print!("   ");
    for i in 0..SIZE { print!("{} ", to_letter(i)) };
    println!("");
    for y in (0..SIZE).rev() {
        print!("{: >2} ", y + 1);
        for x in 0..SIZE {
            print!("{} ", match b[x + y * SIZE] { Empty => '.', Black => 'X', White => 'O'});
        }
        println!("{}", y + 1);
    }
    print!("   ");
    for i in 0..SIZE { print!("{} ", to_letter(i)) };
    println!("");
}

#[derive(Clone)]
pub struct Go {
    board: Vec<Space>,
    history: History,
    current: Space,
    passes: usize
}

impl Go {
    fn legal(&self, p: Pos) -> bool {
        if self.board[p].is_filled() { return false }
        let mut board = self.board.clone();
        board[p] = self.current;
        let oc = self.current.enemy();
        let mut captured = BitSet::with_capacity(SIZE * SIZE);
        for &p in neighbors(p).into_iter().flat_map(|x| x) {
            captured.union_with(&capture(oc, p, &board[..]))
        }
        for p in captured.into_iter() { board[p] = Empty; }
        let self_captured = capture(self.current, p, &board[..]);
        for p in self_captured.into_iter() { board[p] = Empty; }
        !self.history.contains(board.iter())
    }
    fn try_play(&mut self, p: Pos) -> bool {
        if self.board[p].is_filled() { return false }
        self.board[p] = self.current;
        let oc = self.current.enemy();
        let mut captured = BitSet::with_capacity(SIZE * SIZE);
        for &p in neighbors(p).into_iter().flat_map(|x| x) {
            captured.union_with(&capture(oc, p, &self.board[..]))
        }
        for p in captured.iter() { self.board[p] = Empty; }
        let self_captured = capture(self.current, p, &self.board[..]);
        for p in self_captured.iter() { self.board[p] = Empty; }
        let ok = self.history.insert(self.board.iter().cloned());
        if !ok {
            for p in self_captured.iter() { self.board[p] = self.current; }
            for p in captured.iter() { self.board[p] = oc; }
            self.board[p] = Empty;
        }
        ok
    }
    fn weigh_move(&self, p: Pos, check: bool) -> u32 {
        if neighbors(p).into_iter().flat_map(|x| x).any(|&p| {
            let x = self.board[p];
            match x {
                Empty => false,
                c => neighbors(p).into_iter().filter_map(|&x| x).filter(|&p| self.board[p] != c.enemy()).count() == 1
            }
        }) {
            if !check || self.legal(p) {
                10
            } else {
                0
            }
        } else {
            1
        }
    }
}

impl Game for Go {
    type Move = usize;
    fn init() -> Go {
        Go { board: vec![Empty; SIZE * SIZE], history: History::new(), current: Black, passes: 0 }
    }
    fn payoff(&self) -> Option<f64> {
        if self.passes > 1 {
            let (b, w) = score(&self.board[..]);
            if b == w { Some(0.5) }
            else if (b > w) ^ (self.current == Black) { Some(0.0) }
            else { Some(1.0) }
        } else { None }
    }
    fn legal_moves(&self) -> Vec<(usize, u32)> {
        let max = SIZE * SIZE;
        let mut moves = (0..max).filter_map(|i| if self.legal(i) { Some((i, self.weigh_move(i, false))) } else { None }).collect::<Vec<_>>();
        moves.push((max, 1));
        moves
    }
    fn playout_moves(&self) -> Vec<(usize, u32)> {
        let max = SIZE * SIZE;
        let mut moves = self.board.iter().enumerate().filter_map(|(i, x)| if x.is_empty() { Some((i, self.weigh_move(i, true))) } else { None }).collect::<Vec<_>>();
        moves.push((max, 1));
        moves
    }
    fn play(&mut self, &act: &usize) {
        if act >= SIZE * SIZE || !self.try_play(act) {
            self.passes += 1
        } else {
            self.passes = 0
        }
        self.current = self.current.enemy();
    }
    fn print(&self) {
        print_board(&self.board[..]);
    }
    fn parse_move(string: &str) -> Option<usize> {
        parse_pos(string)
    }
    fn print_move(&mv: &usize) { print_pos(mv) }
}

#[cfg(test)]
mod tests {
    use game::Game;
    use super::*;
    use test::Bencher;
    use rand::{weak_rng, Rng};

    #[bench]
    fn play_legal_move(bench: &mut Bencher) {
        let mut rng = weak_rng();
        let mut go = Go::init();
        bench.iter(|| {
            let mvs = go.legal_moves();
            let mv = rng.choose(&mvs[..]).unwrap().item;
            go.play(&mv);
        });
    }

    #[bench]
    fn play_playout_move(bench: &mut Bencher) {
        let mut rng = weak_rng();
        let mut go = Go::init();
        bench.iter(|| {
            let mvs = go.playout_moves();
            let mv = rng.choose(&mvs[..]).unwrap().item;
            go.play(&mv);
        });
    }
}
