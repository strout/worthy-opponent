use game::Game;
use basics::*;

#[derive(Clone, Debug)]
pub struct TicTacToe {
    board: [Space; 9],
    current: Color
}

static GROUPS : [[usize; 3]; 8] = [[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [2,4,6]];

impl TicTacToe {
    fn winner(&self) -> Option<Color> {
       for grp in GROUPS.iter() {
           let x = self.board[grp[0]];
           if x.is_some() && x == self.board[grp[1]] && x == self.board[grp[2]] { return x }
       }
       None
    }
}

impl Game for TicTacToe {
    fn init() -> TicTacToe {
       TicTacToe { board: [None; 9], current: Black }
    }
    fn payoff(&self) -> Option<f64> {
       match self.winner() {
           Some(x) => Some(if x == self.current { 1.0 } else { 0.0 }),
           None => if self.board.iter().all(|x| x.is_some()) { Some(0.5) } else { None }
       }
    }
    fn legal_moves(&self) -> Vec<usize> {
       self.board.iter().enumerate().filter(|&(_, x)| x.is_none()).map(|(i, _)| i).collect::<Vec<_>>()
    }
    fn play(&mut self, act: usize) {
       self.board[act] = Some(self.current);
       self.current = self.current.enemy();
    }
    fn print(&self) {
       let disp = |x| match x { None => ' ', Some(Black) => 'X', Some(White) => 'O' };
       println!("{}|{}|{}", disp(self.board[0]), disp(self.board[1]), disp(self.board[2]));
       println!("-+-+-");
       println!("{}|{}|{}", disp(self.board[3]), disp(self.board[4]), disp(self.board[5]));
       println!("-+-+-");
       println!("{}|{}|{}", disp(self.board[6]), disp(self.board[7]), disp(self.board[8]));
    }
}
