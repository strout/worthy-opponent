use rand::distributions::Weighted;

pub trait Game : Clone {
    fn init() -> Self;
    fn payoff(&self) -> Option<f64>;
    fn legal_moves(&self) -> Vec<Weighted<usize>>;
    fn play(&mut self, usize);
    fn print(&self);
    fn parse_move(string: &str) -> usize {
        string.parse().unwrap()
    }
    fn print_move(mv: usize) {
        print!("{}", mv)
    }
}
