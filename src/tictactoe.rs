use game::Game;
use basics::*;
use rand::distributions::Weighted;

#[derive(Clone, Debug)]
pub struct TicTacToe {
    board: [Space; 9],
    current: Space
}

static GROUPS : [[usize; 3]; 8] = [[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [2,4,6]];

static GROUPS_BY_SPACE : [&'static [[usize; 2]]; 9] = [
    &[[1, 2], [3, 6], [4, 8]], // 0
    &[[0, 2], [4, 7]], // 1
    &[[0, 1], [5, 8], [4, 6]], // 2
    &[[4, 5], [0, 6]], // 3
    &[[3, 5], [1, 7], [0, 8], [2, 6]], // 4
    &[[3, 4], [2, 8]], // 5
    &[[7, 8], [0, 3], [2, 4]], // 6
    &[[6, 8], [1, 4]], // 7
    &[[6, 7], [2, 5], [0, 4]] // 8
];

impl TicTacToe {
    fn winner(&self) -> Space {
        for grp in GROUPS.iter() {
            let x = self.board[grp[0]];
            if x.is_filled() && x == self.board[grp[1]] && x == self.board[grp[2]] { return x }
        }
        Empty
    }
    fn weight_for(&self, x: usize) -> u32 {
        let groups = GROUPS_BY_SPACE[x];
        if groups.iter().any(|grp| self.board[grp[0]].is_filled() && self.board[grp[0]] == self.board[grp[1]]) {
            1000
        } else {
            1
        }
    }
}

impl Game for TicTacToe {
    type Move = usize;
    fn init() -> TicTacToe {
       TicTacToe { board: [Empty; 9], current: Black }
    }
    fn payoff(&self) -> Option<f64> {
       match self.winner() {
           Empty => if self.board.iter().all(|x| x.is_filled()) { Some(0.5) } else { None },
           x => Some(if x == self.current { 1.0 } else { 0.0 }),
       }
    }
    fn legal_moves(&self) -> Vec<Weighted<usize>> {
       self.board.iter().enumerate().filter_map(|(i, x)| if x.is_empty() { Some(Weighted { weight: self.weight_for(i), item: i }) } else { None }).collect::<Vec<_>>()
    }
    fn play(&mut self, &act: &usize) {
       self.board[act] = self.current;
       self.current = self.current.enemy();
    }
    fn print(&self) {
       let disp = |x| match x { Empty => ' ', Black => 'X', White => 'O' };
       println!("{}|{}|{}", disp(self.board[0]), disp(self.board[1]), disp(self.board[2]));
       println!("-+-+-");
       println!("{}|{}|{}", disp(self.board[3]), disp(self.board[4]), disp(self.board[5]));
       println!("-+-+-");
       println!("{}|{}|{}", disp(self.board[6]), disp(self.board[7]), disp(self.board[8]));
    }
    fn parse_move(string: &str) -> usize { string.parse().unwrap() }
    fn print_move(mv: &usize) { print!("{}", mv) }
}
