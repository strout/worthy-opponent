pub trait Game : Clone {
    fn init() -> Self;
    fn payoff(&self) -> Option<f64>;
    fn legal_moves(&self) -> Vec<usize>;
    fn play(&mut self, usize);
    fn print(&self);
    fn parse_move(string: &str) -> usize {
        string.parse().unwrap()
    }
}
