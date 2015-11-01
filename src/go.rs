const SIZE : usize = 9;

use bit_set::BitSet;
use game::Game;
use basics::*;
use rand::distributions::Weighted;

#[derive(Debug)]
struct BoardOf<T> {
    dat: Vec<T>,
    komi: f32
}

impl<T: Clone> Clone for BoardOf<T> {
    fn clone(self: &BoardOf<T>) -> BoardOf<T> {
        BoardOf { dat: self.dat.clone(), komi: self.komi }
    }
    fn clone_from(self: &mut BoardOf<T>, other: &BoardOf<T>) {
        self.dat.clone_from(&other.dat);
        self.komi = other.komi;
    }
}

type Board = BoardOf<Space>;

type Pos = usize;

fn neighbors(p: Pos) -> [Option<usize>; 4] {
    [if p % SIZE > 0 { Some(p - 1) } else { None },
        if p % SIZE < SIZE - 1 { Some(p + 1) } else { None },
            if p / SIZE > 0 { Some(p - SIZE) } else { None },
                if p / SIZE < SIZE - 1 { Some(p + SIZE) } else { None }]
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
        for &p in neighbors(p).into_iter().flat_map(|x| x) {
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
    let mut visited = BitSet::with_capacity(SIZE * SIZE);
    for p in 0..SIZE * SIZE {
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
    let mut captured = BitSet::with_capacity(SIZE * SIZE);
    let mut stack = vec![p];
    loop {
        match stack.pop() {
            None => return captured,
                 Some(p) => if !captured.contains(&p) {
                     match b.dat[p] {
                         None => return BitSet::new(),
                         Some(c2) => if c == c2 {
                             captured.insert(p);
                             stack.extend(neighbors(p).iter().filter_map(Option::clone))
                         }
                     }
                 }
        }
    }
}

fn legal(c: Color, p: Pos, board: &Board, h: &History) -> bool {
    if board.dat[p].is_some() { return false }
    let mut board = board.clone();
    board.dat[p] = Some(c);
    let oc = c.enemy();
    let mut captured = BitSet::with_capacity(SIZE * SIZE);
    for &p in neighbors(p).into_iter().flat_map(|x| x) {
        captured.union_with(&capture(oc, p, &board))
    }
    for p in captured.iter() { board.dat[p] = None; }
    let self_captured = capture(c, p, &board);
    for p in self_captured.iter() { board.dat[p] = None; }
    let bad = h.contains(board.dat.iter());
    !bad
}

fn play(c: Color, p: Pos, board: &mut Board, h: &mut History) -> bool {
    if board.dat[p].is_some() { return false }
    board.dat[p] = Some(c);
    let oc = c.enemy();
    let mut captured = BitSet::with_capacity(SIZE * SIZE);
    for &p in neighbors(p).into_iter().flat_map(|x| x) {
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

fn weigh_move(g: &GoState, p: Pos, check: bool) -> u32 {
    if neighbors(p).into_iter().filter_map(|&x| x).any(|p| {
        let x = g.0.dat[p];
        match x {
            None => false,
            Some(c) => neighbors(p).into_iter().filter_map(|&x| x).all(|p| g.0.dat[p] == Some(c.enemy()))
        }
    }) {
        if !check || legal(g.2, p, &g.0, &g.1) {
            10
        } else {
            0
        }
    } else {
        1
    }
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

fn print_board(b: &Board) {
    print!("   ");
    for i in 0..SIZE { print!("{} ", to_letter(i)) };
    println!("");
    for y in (0..SIZE).rev() {
        print!("{: >2} ", y + 1);
        for x in 0..SIZE {
            print!("{} ", match b.dat[x + y * SIZE] { None => '.', Some(Black) => 'X', Some(White) => 'O'});
        }
        println!("{}", y + 1);
    }
    print!("   ");
    for i in 0..SIZE { print!("{} ", to_letter(i)) };
    println!("");
}

fn make_board(komi: f32) -> Board {
    let len = SIZE * SIZE;
    BoardOf { dat: vec![None; len], komi: komi }
}

pub type GoState = (Board, History, Color, usize, bool);

impl Game for GoState {
    fn init() -> GoState {
        (make_board(0.0), History::new(), Black, 0, false)
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
    fn legal_moves(&self) -> Vec<Weighted<usize>> {
        let max = SIZE * SIZE;
        let mut moves = (0..max).filter_map(|i| if legal(self.2, i, &self.0, &self.1) { Some(Weighted { weight: weigh_move(&self, i, false), item: i }) } else { None }).collect::<Vec<_>>();
        moves.push(Weighted { weight: 1, item: max });
        moves
    }
    fn playout_moves(&self) -> Vec<Weighted<usize>> {
        let max = SIZE * SIZE;
        let mut moves = self.0.dat.iter().enumerate().filter_map(|(i, x)| if x.is_none() { Some(Weighted { weight: weigh_move(&self, i, true), item: i }) } else { None }).collect::<Vec<_>>();
        moves.push(Weighted { weight: 1, item: max });
        moves
    }
    fn play(&mut self, act: usize) {
        if act >= SIZE * SIZE {
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
    fn parse_move(string: &str) -> usize {
        parse_pos(string).expect("Bad move.")
    }
    fn print_move(mv: usize) { print_pos(mv) }
}
