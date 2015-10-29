use game::Game;

#[derive(Clone, Debug)]
pub struct TicTacToe {
    board: [Option<bool>; 9],
    current: bool
}

impl TicTacToe {
    fn winner(&self) -> Option<bool> {
       if self.board[0] != None && self.board[0] == self.board[1] && self.board[0] == self.board[2] { return self.board[0] }
       if self.board[3] != None && self.board[3] == self.board[4] && self.board[3] == self.board[5] { return self.board[3] }
       if self.board[6] != None && self.board[6] == self.board[7] && self.board[6] == self.board[8] { return self.board[6] }
       if self.board[0] != None && self.board[0] == self.board[3] && self.board[0] == self.board[6] { return self.board[0] }
       if self.board[1] != None && self.board[1] == self.board[4] && self.board[1] == self.board[7] { return self.board[1] }
       if self.board[2] != None && self.board[2] == self.board[5] && self.board[2] == self.board[8] { return self.board[2] }
       if self.board[0] != None && self.board[0] == self.board[4] && self.board[0] == self.board[8] { return self.board[0] }
       if self.board[2] != None && self.board[2] == self.board[4] && self.board[2] == self.board[6] { return self.board[2] }
       None
    }
}

impl Game for TicTacToe {
    fn init() -> TicTacToe {
       TicTacToe { board: [None; 9], current: true }
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
       self.current = !self.current;
    }
    fn print(&self) {
       let disp = |x| match x { None => ' ', Some(true) => 'X', Some(false) => 'O' };
       println!("{}|{}|{}", disp(self.board[0]), disp(self.board[1]), disp(self.board[2]));
       println!("-+-+-");
       println!("{}|{}|{}", disp(self.board[3]), disp(self.board[4]), disp(self.board[5]));
       println!("-+-+-");
       println!("{}|{}|{}", disp(self.board[6]), disp(self.board[7]), disp(self.board[8]));
    }
}
