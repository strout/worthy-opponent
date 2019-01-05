use game::Game;
use basics::*;
use std::iter::repeat;
use std::mem::uninitialized;

const WIDTH : usize = 7;
const HEIGHT : usize = 6;

pub struct ConnectFour {
    board: [Space; WIDTH * HEIGHT],
    free: usize
}

impl Clone for ConnectFour {
    fn clone(&self) -> ConnectFour {
        let mut b = unsafe /* hold on to your hat, 'cause here we go */ { uninitialized::<[Space; WIDTH * HEIGHT]>() };
        for (new, old) in b.iter_mut().zip(self.board.iter()) { *new = *old }
        ConnectFour { board: b, free: self.free }
    }
}

static GROUPS : [[usize; 4]; (HEIGHT - 3) * WIDTH + (WIDTH - 3) * HEIGHT + (HEIGHT - 3) * (WIDTH - 3) * 2] = [
    [WIDTH * 0 + 0, WIDTH * 1 + 0, WIDTH * 2 + 0, WIDTH * 3 + 0],
    [WIDTH * 1 + 0, WIDTH * 2 + 0, WIDTH * 3 + 0, WIDTH * 4 + 0],
    [WIDTH * 2 + 0, WIDTH * 3 + 0, WIDTH * 4 + 0, WIDTH * 5 + 0],
    [WIDTH * 0 + 1, WIDTH * 1 + 1, WIDTH * 2 + 1, WIDTH * 3 + 1],
    [WIDTH * 1 + 1, WIDTH * 2 + 1, WIDTH * 3 + 1, WIDTH * 4 + 1],
    [WIDTH * 2 + 1, WIDTH * 3 + 1, WIDTH * 4 + 1, WIDTH * 5 + 1],
    [WIDTH * 0 + 2, WIDTH * 1 + 2, WIDTH * 2 + 2, WIDTH * 3 + 2],
    [WIDTH * 1 + 2, WIDTH * 2 + 2, WIDTH * 3 + 2, WIDTH * 4 + 2],
    [WIDTH * 2 + 2, WIDTH * 3 + 2, WIDTH * 4 + 2, WIDTH * 5 + 2],
    [WIDTH * 0 + 3, WIDTH * 1 + 3, WIDTH * 2 + 3, WIDTH * 3 + 3],
    [WIDTH * 1 + 3, WIDTH * 2 + 3, WIDTH * 3 + 3, WIDTH * 4 + 3],
    [WIDTH * 2 + 3, WIDTH * 3 + 3, WIDTH * 4 + 3, WIDTH * 5 + 3],
    [WIDTH * 0 + 4, WIDTH * 1 + 4, WIDTH * 2 + 4, WIDTH * 3 + 4],
    [WIDTH * 1 + 4, WIDTH * 2 + 4, WIDTH * 3 + 4, WIDTH * 4 + 4],
    [WIDTH * 2 + 4, WIDTH * 3 + 4, WIDTH * 4 + 4, WIDTH * 5 + 4],
    [WIDTH * 0 + 5, WIDTH * 1 + 5, WIDTH * 2 + 5, WIDTH * 3 + 5],
    [WIDTH * 1 + 5, WIDTH * 2 + 5, WIDTH * 3 + 5, WIDTH * 4 + 5],
    [WIDTH * 2 + 5, WIDTH * 3 + 5, WIDTH * 4 + 5, WIDTH * 5 + 5],
    [WIDTH * 0 + 6, WIDTH * 1 + 6, WIDTH * 2 + 6, WIDTH * 3 + 6],
    [WIDTH * 1 + 6, WIDTH * 2 + 6, WIDTH * 3 + 6, WIDTH * 4 + 6],
    [WIDTH * 2 + 6, WIDTH * 3 + 6, WIDTH * 4 + 6, WIDTH * 5 + 6],
    [WIDTH * 0 + 0, WIDTH * 0 + 1, WIDTH * 0 + 2, WIDTH * 0 + 3],
    [WIDTH * 0 + 1, WIDTH * 0 + 2, WIDTH * 0 + 3, WIDTH * 0 + 4],
    [WIDTH * 0 + 2, WIDTH * 0 + 3, WIDTH * 0 + 4, WIDTH * 0 + 5],
    [WIDTH * 0 + 3, WIDTH * 0 + 4, WIDTH * 0 + 5, WIDTH * 0 + 6],
    [WIDTH * 1 + 0, WIDTH * 1 + 1, WIDTH * 1 + 2, WIDTH * 1 + 3],
    [WIDTH * 1 + 1, WIDTH * 1 + 2, WIDTH * 1 + 3, WIDTH * 1 + 4],
    [WIDTH * 1 + 2, WIDTH * 1 + 3, WIDTH * 1 + 4, WIDTH * 1 + 5],
    [WIDTH * 1 + 3, WIDTH * 1 + 4, WIDTH * 1 + 5, WIDTH * 1 + 6],
    [WIDTH * 2 + 0, WIDTH * 2 + 1, WIDTH * 2 + 2, WIDTH * 2 + 3],
    [WIDTH * 2 + 1, WIDTH * 2 + 2, WIDTH * 2 + 3, WIDTH * 2 + 4],
    [WIDTH * 2 + 2, WIDTH * 2 + 3, WIDTH * 2 + 4, WIDTH * 2 + 5],
    [WIDTH * 2 + 3, WIDTH * 2 + 4, WIDTH * 2 + 5, WIDTH * 2 + 6],
    [WIDTH * 3 + 0, WIDTH * 3 + 1, WIDTH * 3 + 2, WIDTH * 3 + 3],
    [WIDTH * 3 + 1, WIDTH * 3 + 2, WIDTH * 3 + 3, WIDTH * 3 + 4],
    [WIDTH * 3 + 2, WIDTH * 3 + 3, WIDTH * 3 + 4, WIDTH * 3 + 5],
    [WIDTH * 3 + 3, WIDTH * 3 + 4, WIDTH * 3 + 5, WIDTH * 3 + 6],
    [WIDTH * 4 + 0, WIDTH * 4 + 1, WIDTH * 4 + 2, WIDTH * 4 + 3],
    [WIDTH * 4 + 1, WIDTH * 4 + 2, WIDTH * 4 + 3, WIDTH * 4 + 4],
    [WIDTH * 4 + 2, WIDTH * 4 + 3, WIDTH * 4 + 4, WIDTH * 4 + 5],
    [WIDTH * 4 + 3, WIDTH * 4 + 4, WIDTH * 4 + 5, WIDTH * 4 + 6],
    [WIDTH * 5 + 0, WIDTH * 5 + 1, WIDTH * 5 + 2, WIDTH * 5 + 3],
    [WIDTH * 5 + 1, WIDTH * 5 + 2, WIDTH * 5 + 3, WIDTH * 5 + 4],
    [WIDTH * 5 + 2, WIDTH * 5 + 3, WIDTH * 5 + 4, WIDTH * 5 + 5],
    [WIDTH * 5 + 3, WIDTH * 5 + 4, WIDTH * 5 + 5, WIDTH * 5 + 6],
    [WIDTH * 3 + 0, WIDTH * 2 + 1, WIDTH * 1 + 2, WIDTH * 0 + 3],
    [WIDTH * 3 + 1, WIDTH * 2 + 2, WIDTH * 1 + 3, WIDTH * 0 + 4],
    [WIDTH * 3 + 2, WIDTH * 2 + 3, WIDTH * 1 + 4, WIDTH * 0 + 5],
    [WIDTH * 3 + 3, WIDTH * 2 + 4, WIDTH * 1 + 5, WIDTH * 0 + 6],
    [WIDTH * 4 + 0, WIDTH * 3 + 1, WIDTH * 2 + 2, WIDTH * 1 + 3],
    [WIDTH * 4 + 1, WIDTH * 3 + 2, WIDTH * 2 + 3, WIDTH * 1 + 4],
    [WIDTH * 4 + 2, WIDTH * 3 + 3, WIDTH * 2 + 4, WIDTH * 1 + 5],
    [WIDTH * 4 + 3, WIDTH * 3 + 4, WIDTH * 2 + 5, WIDTH * 1 + 6],
    [WIDTH * 5 + 0, WIDTH * 4 + 1, WIDTH * 3 + 2, WIDTH * 2 + 3],
    [WIDTH * 5 + 1, WIDTH * 4 + 2, WIDTH * 3 + 3, WIDTH * 2 + 4],
    [WIDTH * 5 + 2, WIDTH * 4 + 3, WIDTH * 3 + 4, WIDTH * 2 + 5],
    [WIDTH * 5 + 3, WIDTH * 4 + 4, WIDTH * 3 + 5, WIDTH * 2 + 6],
    [WIDTH * 3 + 0, WIDTH * 2 + 1, WIDTH * 1 + 2, WIDTH * 0 + 3],
    [WIDTH * 3 + 1, WIDTH * 2 + 2, WIDTH * 1 + 3, WIDTH * 0 + 4],
    [WIDTH * 3 + 2, WIDTH * 2 + 3, WIDTH * 1 + 4, WIDTH * 0 + 5],
    [WIDTH * 3 + 3, WIDTH * 2 + 4, WIDTH * 1 + 5, WIDTH * 0 + 6],
    [WIDTH * 4 + 0, WIDTH * 3 + 1, WIDTH * 2 + 2, WIDTH * 1 + 3],
    [WIDTH * 4 + 1, WIDTH * 3 + 2, WIDTH * 2 + 3, WIDTH * 1 + 4],
    [WIDTH * 4 + 2, WIDTH * 3 + 3, WIDTH * 2 + 4, WIDTH * 1 + 5],
    [WIDTH * 4 + 3, WIDTH * 3 + 4, WIDTH * 2 + 5, WIDTH * 1 + 6],
    [WIDTH * 5 + 0, WIDTH * 4 + 1, WIDTH * 3 + 2, WIDTH * 2 + 3],
    [WIDTH * 5 + 1, WIDTH * 4 + 2, WIDTH * 3 + 3, WIDTH * 2 + 4],
    [WIDTH * 5 + 2, WIDTH * 4 + 3, WIDTH * 3 + 4, WIDTH * 2 + 5],
    [WIDTH * 5 + 3, WIDTH * 4 + 4, WIDTH * 3 + 5, WIDTH * 2 + 6]
];

impl ConnectFour {
    fn winner(&self) -> Space {
        for grp in GROUPS.iter() {
            let x = self.board[grp[0]];
            if x.is_filled() && x == self.board[grp[1]] && x == self.board[grp[2]] && x == self.board[grp[3]] { return x }
        }
        Empty
    }
    fn weight_for(&self, _: usize) -> u32 {
        1
    }
    fn current_player(&self) -> Space {
        if self.free % 2 == 1 { Black } else { White }
    }
    fn lowest_free(&self, c: usize) -> Option<usize> {
        for r in 0..HEIGHT { let m = r * WIDTH + c; if self.board[m] == Empty { return Some(m) } }
        None
    }
}

impl Game for ConnectFour {
    type Move = usize;
    fn init() -> ConnectFour {
       ConnectFour { board: [Empty; WIDTH * HEIGHT], free: WIDTH * HEIGHT }
    }
    fn payoff(&self) -> Option<f64> {
       match self.winner() {
           Empty => if self.free == 0 { Some(0.5) } else { None },
           x => Some(if x == self.current_player() { 1.0 } else { 0.0 }),
       }
    }
    fn legal_moves(&self) -> Vec<(usize, u32)> {
       (0..WIDTH).filter_map(|c| self.lowest_free(c).map(|m| (m, self.weight_for(m)))).collect()
    }
    fn play(&mut self, &act: &usize) {
       self.board[act] = self.current_player();
       self.free -= 1;
    }
    fn print(&self) {
       let disp = |x| match x { Empty => ' ', Black => 'X', White => 'O' };
       for r in (0..HEIGHT).rev() {
           println!("|{}|", (0..WIDTH).map(|c| disp(self.board[r * WIDTH + c])).collect::<String>());
       }
       println!("+{}+", repeat('-').take(WIDTH).collect::<String>());
    }
    fn parse_move(string: &str) -> Option<usize> { string.split(',').last().and_then(|s| s.parse().ok()) }
    fn print_move(mv: &usize) { print!("{}", mv) }
}
