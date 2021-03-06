use std::fmt::Display;
use std::str::FromStr;

pub trait Game : Clone {
    type Move : Clone + Display + FromStr + PartialEq + Send;
    fn init() -> Self;
    fn payoff(&self) -> Option<f64>;
    fn legal_moves(&self) -> Vec<(Self::Move, u32)>;
    fn playout_moves(&self) -> Vec<(Self::Move, u32)> { self.legal_moves() }
    fn play(&mut self, &Self::Move);
    fn print(&self);
    fn parse_move(string: &str) -> Option<Self::Move>;
    fn print_move(mv: &Self::Move);
}
