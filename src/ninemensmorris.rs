use game::Game;
use basics::*;
use bit_set::BitSet;
use std::str::FromStr;
use std::fmt::{self, Display, Formatter};

#[derive(Debug, Clone)]
pub struct NineMensMorris {
    board: [Space; 24],
    turn: usize,
    black_pieces: usize,
    white_pieces: usize,
    history: History
}

#[derive(Debug, Clone, PartialEq)]
pub struct Move { from: Option<usize>, to: usize, remove: Option<usize> }

impl FromStr for Move {
    type Err = ();
    fn from_str(s: &str) -> Result<Move, ()> {
        let mut ss = s.split(',');
        let f: isize = try!(ss.next().and_then(|s| s.parse().ok()).ok_or(()));
        let t: usize = try!(ss.next().and_then(|s| s.parse().ok()).ok_or(()));
        let r: isize = try!(ss.next().and_then(|s| s.parse().ok()).ok_or(()));
        Ok(Move {
            from: if f < 0 { None } else { Some(f as usize) },
            to: t,
            remove: if r < 0 { None } else { Some(r as usize) }
        })
    }
}

impl Display for Move {
    fn fmt(&self, formatter: &mut Formatter) -> Result<(), fmt::Error> {
        try!(self.from.map(|x| x as isize).unwrap_or(-1).fmt(formatter));
        try!(','.fmt(formatter));
        try!(self.to.fmt(formatter));
        try!(','.fmt(formatter));
        self.remove.map(|x| x as isize).unwrap_or(-1).fmt(formatter)
    }
}

pub static MILLS_BY_SPACE : [[[usize; 2]; 2]; 24] = [
    [[1,2], [9,21]], // 0
    [[0,2], [4,7]], // 1
    [[0,1], [14,23]], // 2
    [[4,5], [10,18]], // 3
    [[3,5], [1,7]], // 4
    [[3,4], [13,20]], // 5
    [[7,8], [11,15]], // 6
    [[6,8], [1,4]], // 7
    [[6,7], [12,17]], // 8
    [[10,11], [0,21]], // 9
    [[9,11], [3,18]], // 10
    [[9,10], [6,15]], // 11
    [[13,14], [8,17]], // 12
    [[12,14], [5,20]], // 13
    [[12,13], [2,23]], // 14
    [[16,17], [6,11]], // 15
    [[15,17], [19,22]], // 16
    [[15,16], [8,12]], // 17
    [[19,20], [3,10]], // 18
    [[18,20], [16,22]], // 19
    [[18,19], [5,13]], // 20
    [[22,23], [0,9]], // 21
    [[21,23], [16,19]], // 22
    [[21,22], [2,14]] // 23
];

pub static ADJACENT_SPACES : [&'static [usize]; 24] = [
    &[1,9], // 0
    &[0,2,4], // 1
    &[1,14], // 2
    &[4,10], // 3
    &[1,3,5,7], // 4
    &[4,13], // 5
    &[7,11], // 6
    &[4,6,8], // 7
    &[7,12], // 8
    &[0,10,21], // 9
    &[3,9,11,18], // 10
    &[6,10,15], // 11
    &[8,13,17], // 12
    &[5,12,14,20], // 13
    &[2,13,23], // 14
    &[11,16], // 15
    &[15,17,19], // 16
    &[12,16], // 17
    &[10,19], // 18
    &[16,18,20,22], // 19
    &[13,19], // 20
    &[9,22], // 21
    &[19,21,23], // 22
    &[14,22] // 23
];

impl NineMensMorris {
    fn current_player(&self) -> Space { if self.turn % 2 == 0 { Black } else { White } }
    fn forms_mill(&self, x: usize) -> bool { self.possible_mills_for(x).len() > 0 }
    fn forms_mill_without(&self, x: usize, y: usize) -> bool { self.possible_mills_for(x).iter().any(|m| !m.contains(&y)) }
    fn possible_mills_for(&self, x: usize) -> Vec<&'static [usize; 2]> {
        let c = self.current_player();
        MILLS_BY_SPACE[x].iter().filter(|m| m.iter().all(|&y| c == self.board[y])).collect()
    }
    fn removable_pieces(&self) -> BitSet {
        let c = self.current_player().enemy();
        let mut in_mills = BitSet::with_capacity(24);
        let mut out_of_mills = BitSet::with_capacity(24);
        for i in 0..self.board.len() {
            if self.board[i] == c {
                if MILLS_BY_SPACE[i].iter().any(|m| m.iter().all(|&y| c == self.board[y])) {
                    in_mills.insert(i);
                } else {
                    out_of_mills.insert(i);
                }
            }
        }
        if out_of_mills.is_empty() { in_mills } else { out_of_mills }
    }
    fn adjacent_free(&self, x: usize) -> BitSet {
        let mut ret = ADJACENT_SPACES[x].iter().cloned().collect::<BitSet>();
        for (i, x) in self.board.iter().enumerate() { if x.is_filled() { ret.remove(i); } };
        ret
    }
    fn moves(&self, c: Space) -> Vec<Move> {
        let mut ret = vec![];
        if self.turn < 18 {
           for d in self.board.iter().enumerate().filter_map(|(i, x)| if x.is_empty() { Some(i) } else { None }) {
               if self.forms_mill(d) {
                   for r in self.removable_pieces().iter() {
                       ret.push(Move { from: None, to: d, remove: Some(r) });
                   }
               } else {
                   ret.push(Move { from: None, to: d, remove: None });
               }
           }
        } else {
           for s in self.board.iter().enumerate().filter_map(|(i, &x)| if x == c { Some(i) } else { None }) {
               for d in if self.pieces_for(c) == 3 { self.board.iter().enumerate().filter_map(|(i, x)| if x.is_empty() { Some(i) } else { None }).collect() } else { self.adjacent_free(s) }.into_iter() {
                   if self.forms_mill_without(d, s) {
                       for r in self.removable_pieces().iter() {
                           ret.push(Move { from: Some(s), to: d, remove: Some(r) });
                       }
                   } else {
                       ret.push(Move { from: Some(s), to: d, remove: None });
                   }
               }
           }
        }
        ret
    }
    fn pieces_for(&self, c: Space) -> usize { match c { Black => self.black_pieces, White => self.white_pieces, Empty => panic!("Bad argument.") } }
    fn weigh_move(&self, m: &Move) -> u32 {
        if let &Move { remove: Some(_), .. } = m { return 10 }
        1
    }
}

impl Game for NineMensMorris {
    type Move = Move;
    fn init() -> NineMensMorris {
        NineMensMorris { board: [Empty; 24], turn: 0, black_pieces: 0, white_pieces: 0, history: History::new() }
    }
    fn payoff(&self) -> Option<f64> {
        if self.turn < 18 {
            None
        } else {
            let c = self.current_player();
            let mine = self.pieces_for(c);
            let no_adjacent_moves = || {
                let mut my_pieces = self.board.iter().enumerate().filter_map(|(i, &x)| if x == c { Some(i) } else { None });
                my_pieces.all(|s| ADJACENT_SPACES[s].iter().all(|&x| self.board[x].is_filled()))
            };
            if self.pieces_for(c.enemy()) <= 2 { Some(1.0) }
            else if mine <= 2 { Some(0.0) }
            else if mine > 3 && no_adjacent_moves() { Some(0.0) }
            else if self.history.contains(self.board.iter()) { Some(0.5) }
            else { None }
        }
    }
    fn legal_moves(&self) -> Vec<(Move, u32)> {
        self.moves(self.current_player()).into_iter().map(|m| {
            let w = self.weigh_move(&m);
            (m, w)
        }).collect()
    }
    fn play(&mut self, &Move { from, to, remove }: &Move) {
        let mut removed = 0;
        let mut added = 1;
        let c = self.current_player();
        if let Some(x) = from { self.history.insert(self.board.iter().cloned()); self.board[x] = Empty; added -= 1; }
        self.board[to] = c;
        if let Some(x) = remove { self.board[x] = Empty; removed += 1; }
        self.turn += 1;
        if c == Black { self.black_pieces += added; self.white_pieces -= removed }
        else { self.black_pieces -= removed; self.white_pieces += added }
    }
    fn print(&self) {
        let disp = |x| match x { Empty => ' ', Black => 'X', White => 'O' };
        println!("{}----{}----{}", disp(self.board[0]), disp(self.board[1]), disp(self.board[2]));
        println!("|         |");
        println!("| {}--{}--{} |", disp(self.board[3]), disp(self.board[4]), disp(self.board[5]));
        println!("| |     | |");
        println!("| | {}{}{} | |", disp(self.board[6]), disp(self.board[7]), disp(self.board[8]));
        println!("{}-{}-{} {}-{}-{}", disp(self.board[9]), disp(self.board[10]), disp(self.board[11]), disp(self.board[12]), disp(self.board[13]), disp(self.board[14]));
        println!("| | {}{}{} | |", disp(self.board[15]), disp(self.board[16]), disp(self.board[17]));
        println!("| |     | |");
        println!("| {}--{}--{} |", disp(self.board[18]), disp(self.board[19]), disp(self.board[20]));
        println!("|         |");
        println!("{}----{}----{}", disp(self.board[21]), disp(self.board[22]), disp(self.board[23]));
    }
    fn parse_move(string: &str) -> Option<Move> { string.split(';').last().and_then(|s| s.parse().ok()) }
    fn print_move(mv: &Move) { print!("{}", mv) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_spaces_in_seen_by_four() {
        let mut counts = vec![0; 24];
        for &x in MILLS_BY_SPACE.iter().flat_map(|ms| ms.iter()).flat_map(|m| m.iter()) {
            counts[x] += 1;
        }
        assert_eq!(&[4; 24][..], &counts[..]);
    }

    #[test]
    fn adjacency_is_mutual() {
        for (x, adj) in ADJACENT_SPACES.iter().enumerate() {
            for &y in adj.iter() {
                assert!(ADJACENT_SPACES[y].contains(&x))
            }
        }
    }
}
