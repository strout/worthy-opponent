const DEFAULT_SIZE : usize = 19;
const DEFAULT_KOMI : f32 = 7.5;

use std::io;
use bit_set::BitSet;
use game::Game;
use basics::*;

#[derive(Debug)]
struct BoardOf<T> {
    dat: Vec<T>, // TODO tighten visibility
    size: usize, // TODO tighten visibility
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

fn has_liberties_inner(c: Color, p: Pos, b: &Board, visited: &mut BitSet) -> bool {
    let mut stack = vec![p];
    loop {
        match stack.pop() {
            None => return false,
                 Some(p) => if !visited.contains(&p) {
                     visited.insert(p);
                     match b.dat[p] {
                         None => return true,
                              Some(c2) => if c == c2 {
                                  stack.extend(neighbors(p, b.size).into_iter().filter_map(Option::clone))
                              }
                     }
                 }
        }
    }
}

fn has_liberties(p: Pos, b: &Board) -> bool {
    let mut visited = BitSet::with_capacity(b.size * b.size);
    match b.dat[p] {
        None => false,
             Some(c) => has_liberties_inner(c, p, b, &mut visited)
    }
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

fn play(c: Color, mv: Option<Pos>, board: &mut Board, h: &mut History) -> bool {
    let p = match mv {
        None => return true,
             Some(p) => p
    };
    if board.dat[p].is_some() { return false }
    board.dat[p] = Some(c);
    let oc = c.enemy();
    let mut captured = BitSet::with_capacity(board.size * board.size);
    for &p in neighbors(p, board.size).into_iter().flat_map(|x| x) {
        captured.union_with(&capture(oc, p, &board))
    }
    for p in captured.iter() {
        board.dat[p] = None;
    }
    let ok = has_liberties(p, &board) && h.insert(board.dat.iter().cloned());
    if !ok {
        board.dat[p] = None;
        for p in captured.iter() { board.dat[p] = Some(oc); }
    }
    ok
}

fn legal(c: Color, mv: Option<Pos>, board: &Board, h: &History) -> bool {
    let p = match mv {
        None => return true,
             Some(p) => p
    };
    if board.dat[p].is_some() { return false }
    let mut nb = board.clone();
    nb.dat[p] = Some(c);
    let oc = c.enemy();
    for &p in neighbors(p, nb.size).into_iter().flat_map(|x| x) {
        for p in capture(oc, p, &nb).iter() {
            nb.dat[p] = None;
        }
    }
    has_liberties(p, &nb) && !h.contains(nb.dat.iter().cloned())
}

const UPPER_LETTERS : &'static str = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
const DOWN_LETTERS : &'static str = "abcdefghjklmnopqrstuvwxyz";

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

fn parse_pos(s: &str, sz: usize) -> Option<Pos> {
    if s.len() < 1 { return None }
    let y = s[1..].parse::<usize>().ok();
    let xch = s.chars().nth(0);
    xch.and_then(|xch| y.and_then(|y| UPPER_LETTERS.chars().position(|c| c == xch).or_else(|| DOWN_LETTERS.chars().position(|c| c == xch)).map(|x| (y - 1) * sz + x)))
}

fn hu_pos(sz: usize) -> Option<Pos> {
    print!("Enter coordinate or anything else to pass\n");
    let stdin = io::stdin();
    let mut line = String::new();
    stdin.read_line(&mut line).unwrap();
    parse_pos(line.trim(), sz)
}

fn make_board(sz: usize, komi: f32) -> Board {
    let len = sz * sz;
    BoardOf { dat: vec![None; len], size: sz, komi: komi }
}

/* TODO
fn console() {
    let mut c = Black;
    let mut passed = false;
    let mut board = make_board(DEFAULT_SIZE, DEFAULT_KOMI);
    let mut h = History::new();
    let (sendcmd, recvcmd) = channel();
    let (sendmv, recvmv) = channel();
    thread::spawn(|| think(recvcmd, sendmv));
    loop {
        print_board(&board);
        let mv = if c == Black {
            if cfg!(feature = "human_black") {
                hu_pos(3)
            } else {
                thread::sleep_ms(THINK_MS);
                sendcmd.send(Cmd::Gen).unwrap();
                recvmv.recv().unwrap()
            }
        } else {
            if cfg!(feature = "human_white") {
                hu_pos(3)
            } else {
                thread::sleep_ms(THINK_MS);
                sendcmd.send(Cmd::Gen).unwrap();
                recvmv.recv().unwrap()
            }
        };
        sendcmd.send(Cmd::Move(c, mv)).unwrap();
        let passed2 = mv.is_none() || !play(c, mv, &mut board, &mut h);
        if passed && passed2 { break }
        c = c.enemy();
        passed = passed2;
    }
    print!("-------------------------------------\n");
    print_board(&board);
    print!("{:?}\n", score(&board));
}

fn main() {
    if cfg!(feature = "gtp") { gtp() } else { console() }
}
*/

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
    fn history_insert_1000(bench: &mut Bencher) {
        let mut rng = rand::weak_rng();
        bench.iter(|| {
                let mut vs = Vec::<Vec<Option<Color>>>::with_capacity(1000);
                for _ in 0..1000 { vs.push(rng.gen_iter().take(361).collect()) }
                let mut h = History::new();
                for v in vs.iter() { h.insert(v.iter().cloned()); }
                });
    }

#[bench]
    fn hashset_insert_1000(bench: &mut Bencher) {
        use std::collections::HashSet;
        let mut rng = rand::weak_rng();
        bench.iter(|| {
                let mut vs = Vec::<Vec<Option<Color>>>::with_capacity(1000);
                for _ in 0..1000 { vs.push(rng.gen_iter().take(361).collect()) }
                let mut h = HashSet::new();
                for v in vs.iter() { h.insert(v.clone()); }
                });
    }

#[bench]
    fn play_out_bench(bench: &mut Bencher) {
        let mut rng = rand::weak_rng();
        bench.iter(|| play_out(&mut rng, Black, false, &mut make_board(19, 7.5)));
    }
}

pub type GoState = (Board, History, Color, usize);

impl Game for GoState {
    fn init() -> GoState {
        (make_board(19, 0.0), History::new(), Black, 0)
    }
    fn payoff(&self) -> Option<f64> {
        if self.3 > 1 {
            let (b, w) = score(&self.0);
            if b == w { Some(0.5) }
            else if (b > w) ^ (self.2 == Black) { Some(0.0) }
            else { Some(1.0) }
        } else { None }
    }
    fn legal_moves(&self) -> Vec<usize> {
        let max = self.0.size * self.0.size;
        let mut moves = self.0.dat.iter().enumerate().filter(|&(_, x)| x.is_none()).map(|(i, _)| i).collect::<Vec<_>>();
        moves.push(max);
        moves
    }
    fn play(&mut self, act: usize) {
        let c = self.2;
        let sz = self.0.size;
        let pass = act >= sz * sz;
        let ok = play(self.2, if pass { None } else { Some(act) }, &mut self.0, &mut self.1);
        self.2 = self.2.enemy();
        self.3 = if pass || !ok { self.3 + 1 } else { 0 }; 
    }
    fn print(&self) {
        print_board(&self.0);
    }
}
