const DEFAULT_SIZE : usize = 9;

use bit_set::BitSet;
use game::Game;
use basics::*;

#[derive(Debug)]
struct BoardOf<T> {
    dat: Vec<T>,
    size: usize,
    komi: f32
}

impl<T: Clone> Clone for BoardOf<T> {
    fn clone(self: &BoardOf<T>) -> BoardOf<T> {
        BoardOf { dat: self.dat.clone(), size: self.size, komi: self.komi }
    }
    fn clone_from(self: &mut BoardOf<T>, other: &BoardOf<T>) {
        self.dat.clone_from(&other.dat);
        self.size = other.size;
        self.komi = other.komi;
    }
}

type Board = BoardOf<Space>;

type Pos = usize;

fn neighbors(p: Pos, sz: usize) -> [Option<usize>; 4] {
    [if p % sz > 0 { Some(p - 1) } else { None },
        if p % sz < sz - 1 { Some(p + 1) } else { None },
            if p / sz > 0 { Some(p - sz) } else { None },
                if p / sz < sz - 1 { Some(p + sz) } else { None }]
}

fn owner_inner(c: Space, p: Pos, b: &Board, visited: &mut BitSet, white: bool, black: bool) -> (bool, bool, usize) {
    let c2 = b.dat[p];
    let (mut black, mut white) = match (c, c2) {
        (None, Some(White)) => (black, true),
            (None, Some(Black)) => (true, white),
            (None, None) => (black, white),
            (Some(c1), _) => (c1 == Black, c1 == White) // TODO this comparison will run WAY too many times!
    };
    if c == c2 {
        if visited.contains(&p) { return (black, white, 0) }
        visited.insert(p);
        let mut sum = 0;
        for &p in neighbors(p, b.size).into_iter().flat_map(|x| x) {
            let (b, w, s) = owner_inner(c, p, b, visited, white, black);
            black = b;
            white = w;
            sum += s;
        }
        (black, white, sum + 1)
    } else { (black, white, 0) }
}

fn score(board: &Board) -> (f32, f32) {
    let mut b : f32 = 0.0;
    let mut w : f32 = 0.0;
    let mut visited = BitSet::with_capacity(board.size * board.size);
    for p in 0..board.size * board.size {
        if !visited.contains(&p) {
            match owner_inner(board.dat[p], p, board, &mut visited, false, false) {
                (true, false, n) => b += n as f32,
                    (false, true, n) => w += n as f32,
                    (_, _, _) => {}
            }
        }
    }
    (b, w + board.komi)
}

fn capture(c: Color, p: Pos, b: &Board) -> BitSet {
    let mut captured = BitSet::with_capacity(b.size * b.size);
    let mut stack = vec![p];
    loop {
        match stack.pop() {
            None => return captured,
                 Some(p) => if !captured.contains(&p) {
                     match b.dat[p] {
                         None => return BitSet::new(),
                         Some(c2) => if c == c2 {
                             captured.insert(p);
                             stack.extend(neighbors(p, b.size).iter().filter_map(Option::clone))
                         }
                     }
                 }
        }
    }
}

fn play(c: Color, p: Pos, board: &mut Board, h: &mut History) -> bool {
    if board.dat[p].is_some() { return false }
    board.dat[p] = Some(c);
    let oc = c.enemy();
    let mut captured = BitSet::with_capacity(board.size * board.size);
    for &p in neighbors(p, board.size).into_iter().flat_map(|x| x) {
        captured.union_with(&capture(oc, p, &board))
    }
    for p in captured.iter() { board.dat[p] = None; }
    let self_captured = capture(c, p, &board);
    for p in self_captured.iter() { board.dat[p] = None; }
    let ok = h.insert(board.dat.iter().cloned());
    if !ok {
        for p in self_captured.iter() { board.dat[p] = Some(c); }
        for p in captured.iter() { board.dat[p] = Some(oc); }
        board.dat[p] = None;
    }
    ok
}

const UPPER_LETTERS : &'static str = "ABCDEFGHJKLMNOPQRSTUVWXYZ";

fn to_letter(n : usize) -> char {
    UPPER_LETTERS.chars().nth(n).unwrap()
}

fn print_board(b: &Board) {
    print!("   ");
    for i in 0..b.size { print!("{} ", to_letter(i)) };
    println!("");
    for y in (0..b.size).rev() {
        print!("{: >2} ", y + 1);
        for x in 0..b.size {
            print!("{} ", match b.dat[x + y * b.size] { None => '.', Some(Black) => 'X', Some(White) => 'O'});
        }
        println!("{}", y + 1);
    }
    print!("   ");
    for i in 0..b.size { print!("{} ", to_letter(i)) };
    println!("");
}

fn make_board(sz: usize, komi: f32) -> Board {
    let len = sz * sz;
    BoardOf { dat: vec![None; len], size: sz, komi: komi }
}

pub type GoState = (Board, History, Color, usize, bool);

impl Game for GoState {
    fn init() -> GoState {
        (make_board(DEFAULT_SIZE, 0.0), History::new(), Black, 0, false)
    }
    fn payoff(&self) -> Option<f64> {
        if self.4 {
            Some(1.0) // opponent made an illegal play; we win
        } else if self.3 > 1 {
            let (b, w) = score(&self.0);
            if b == w { Some(0.5) }
            else if (b > w) ^ (self.2 == Black) { Some(0.0) }
            else { Some(1.0) }
        } else { None }
    }
    fn legal_moves(&self) -> Vec<usize> {
        let max = self.0.size * self.0.size;
        let mut moves = self.0.dat.iter().enumerate().filter_map(|(i, x)| if x.is_none() { Some(i) } else { None }).collect::<Vec<_>>();
        moves.push(max);
        moves
    }
    fn play(&mut self, act: usize) {
        let sz = self.0.size;
        if act >= sz * sz {
            self.3 += 1
        } else {
            self.3 = 0;
            self.4 = !play(self.2, act, &mut self.0, &mut self.1);
        }
        self.2 = self.2.enemy();
    }
    fn print(&self) {
        print_board(&self.0);
    }
}
