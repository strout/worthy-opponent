use rand::distributions::Weighted;

pub trait Game : Clone {
    type Move : Clone;
    fn init() -> Self;
    fn payoff(&self) -> Option<f64>;
    fn legal_moves(&self) -> Vec<Weighted<Self::Move>>;
    fn playout_moves(&self) -> Vec<Weighted<Self::Move>> { self.legal_moves() }
    fn play(&mut self, &Self::Move);
    fn print(&self);
    fn parse_move(string: &str) -> Self::Move;
    fn print_move(mv: &Self::Move);
}
