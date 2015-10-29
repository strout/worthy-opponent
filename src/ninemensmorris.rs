use game::Game;
use bit_set::BitSet;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct NineMensMorris {
    board: [Option<bool>; 24],
    turn: usize,
    history: HashSet<[Option<bool>; 24]>
}

static mills : [[usize; 3]; 16] = [
    [0,1,2], [3,4,5], [6,7,8],
    [9,10,11], [12,13,14],
    [15,16,17], [18,19,20], [21,22,23],
    [0,9,21], [3,10,18], [6,11,15],
    [1,4,7], [16,19,22],
    [2,14,23], [5,13,20], [8,12,17]
];

// TODO is there a way toreturn an iterator of this? Prob not.
fn mills_containing(x: usize) -> Vec<&'static [usize; 3]> {
    mills.iter().filter(|m| m.contains(&x)).collect()
}

fn adjacent_spaces_to(x: usize) -> BitSet {
    let mut spaces = mills.iter().flat_map(|m| m.windows(2)).filter(|w| w.contains(&x)).flat_map(|w| w).cloned().collect::<BitSet>();
    spaces.remove(&x);
    spaces
}

impl NineMensMorris {
    fn current_player(&self) -> bool { self.turn % 2 == 0 }
    fn forms_mill(&self, x: usize) -> bool { self.possible_mills_for(x).len() > 0 }
    fn forms_mill_without(&self, x: usize, y: usize) -> bool { self.possible_mills_for(x).iter().filter(|m| !m.contains(&y)).count() > 0 }
    fn possible_mills_for(&self, x: usize) -> Vec<&'static [usize; 3]> {
        let c = Some(self.current_player());
        mills_containing(x).into_iter().filter(|m| m.iter().all(|&y| x == y || c == self.board[y])).collect()
    }
    fn removable_pieces(&self) -> BitSet {
        let c = Some(!self.current_player());
        let in_mills = mills.iter().filter(|m| m.iter().all(|&x| c == self.board[x])).flat_map(|m| m).cloned().collect::<BitSet>();
        let mut out_of_mills = self.board.iter().enumerate().filter(|&(_, &x)| x == c).map(|(i, _)| i).collect::<BitSet>();
        out_of_mills.difference_with(&in_mills);
        if out_of_mills.is_empty() { in_mills } else { out_of_mills }
    }
    fn adjacent_free(&self, x: usize) -> BitSet {
        let mut ret = adjacent_spaces_to(x);
        for (i, x) in self.board.iter().enumerate() { if x.is_some() { ret.remove(&i); } };
        ret
    }
}

impl Game for NineMensMorris {
    fn init() -> NineMensMorris {
        NineMensMorris { board: [None; 24], turn: 0, history: HashSet::new() }
    }
    fn payoff(&self) -> Option<f64> {
        if self.turn < 18 {
            None
        } else {
            let mine = Some(!self.current_player());
            let yours = Some(!self.current_player());
            if self.board.iter().filter(|&&x| x == yours).count() <= 2 { Some(1.0) }
            else if self.history.contains(&self.board) { Some(0.5) }
            else if self.board.iter().filter(|&&x| x == mine).count() <= 2 || self.legal_moves().is_empty() { Some(0.0) }
            else { None }
        }
    }
    fn legal_moves(&self) -> Vec<usize> {
        let mut ret = vec![];
        if self.turn < 18 {
           for d in self.board.iter().enumerate().filter(|&(_, x)| x.is_none()).map(|(i, _)| i) {
               if self.forms_mill(d) {
                   for r in self.removable_pieces().iter() {
                       ret.push(d + 24 * r);
                   }
               } else {
                   ret.push(d);
               }
           }
        } else {
           let c = Some(self.current_player());
           for s in self.board.iter().enumerate().filter(|&(_, &x)| x == c).map(|(i, _)| i) {
               for d in if self.board.iter().filter(|&&x| x == c).count() == 3 { self.board.iter().enumerate().filter(|&(_, &x)| x.is_none()).map(|(i, _)| i).collect() } else { self.adjacent_free(s) }.into_iter() {
                   if self.forms_mill_without(d, s) {
                       for r in self.removable_pieces().iter() {
                           ret.push(s + 24 * d + 24 * 24 * r);
                       }
                   } else {
                       ret.push(s + 24 * d);
                   }
               }
           }
        }
        ret
    }
    fn play(&mut self, act: usize) {
        if self.turn < 18 {
            let d = act % 24;
            let r = act / 24;
            if self.forms_mill(d) {
                self.board[r] = None;
            }
            self.board[d] = Some(self.current_player());
        } else {
            self.history.insert(self.board.clone());
            let s = act % 24;
            let rd = act / 24;
            let d = rd % 24;
            let r = rd / 24;
            if self.forms_mill(d) {
                self.board[r] = None;
            }
            self.board[d] = Some(self.current_player());
            self.board[s] = None;
        }
        self.turn += 1;
    }
    fn print(&self) {
        let disp = |x| match x { None => ' ', Some(true) => 'X', Some(false) => 'O' };
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
}
