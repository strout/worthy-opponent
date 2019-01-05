#[macro_use]
extern crate criterion;
extern crate rand;
extern crate worthy_opponent;

use ::rand::{Rng, FromEntropy, rngs::SmallRng, distributions::Standard, seq::SliceRandom};

use std::collections::HashMap;

use criterion::Criterion;

use worthy_opponent::basics::{Space, History};
use worthy_opponent::game::Game;
use worthy_opponent::go::*;
use worthy_opponent::ggp::*;
use worthy_opponent::labeler::Labeler;

fn history_insert_19x19(c: &mut Criterion) {
    let mut rng = SmallRng::from_entropy();
    c.bench_function("history insert 19x19", move |b| b.iter(||
    {
        let mut h = History::new();
        for _ in 0..200 {
            h.insert(rng.sample_iter(&Standard).take(361));
        }
    }));
}

fn hashset_insert_19x19(c: &mut Criterion) {
    use std::collections::HashSet;
    let mut rng = SmallRng::from_entropy();
    c.bench_function("hashset insert 19x19", move |b| b.iter(||
    {
        let mut h = HashSet::<Vec<Space>>::new();
        for _ in 0..200 {
            h.insert(rng.sample_iter(&Standard).take(361).collect());
        }
    }));
}

criterion_group!(basics, history_insert_19x19, hashset_insert_19x19);

fn play_legal_move(c: &mut Criterion) {
    let mut rng = SmallRng::from_entropy();
    c.bench_function("go: play legal move", move |b| b.iter(||
    {
        let mut go = Go::init();
        for _ in 0..100 {
            let mvs = go.legal_moves();
            let mv = mvs.choose(&mut rng).unwrap().0;
            go.play(&mv);
        }
    }));
}

fn play_playout_move(c: &mut Criterion) {
    let mut rng = SmallRng::from_entropy();
    c.bench_function("go: play playout move", move |b| b.iter(||
    {
        let mut go = Go::init();
        for _ in 0..100 {
            let mvs = go.playout_moves();
            let mv = mvs.choose(&mut rng).unwrap().0;
            go.play(&mv);
        }
    }));
}

criterion_group!(go, play_legal_move, play_playout_move);

fn set_up_tic_tac_toe() -> (DB, Labeler<'static>, HashMap<&'static str, usize>) {
    // based on the example in http://games.stanford.edu/index.php/intro-to-gdl
    let rules = "(
(role x)
(role o)

(init (cell 1 1 b))
(init (cell 1 2 b))
(init (cell 1 3 b))
(init (cell 2 1 b))
(init (cell 2 2 b))
(init (cell 2 3 b))
(init (cell 3 1 b))
(init (cell 3 2 b))
(init (cell 3 3 b))
(init (control x))

(<= (legal ?w (mark ?x ?y))
    (true (cell ?x ?y b))
    (true (control ?w)))
(<= (legal x noop)
    (true (control o)))
(<= (legal o noop)
    (true (control x)))

(<= (next (cell ?m ?n x))
    (does x (mark ?m ?n))
    (true (cell ?m ?n b)))
(<= (next (cell ?m ?n o))
    (does o (mark ?m ?n))
    (true (cell ?m ?n b)))
(<= (next (cell ?m ?n ?w))
    (true (cell ?m ?n ?w))
    (distinct ?w b))
(<= (next (cell ?m ?n b))
    (does ?w (mark ?j ?k))
    (true (cell ?m ?n b))
    (or (distinct ?m ?j) (distinct ?n ?k)))
(<= (next (control x))
    (true (control o)))
(<= (next (control o))
    (true (control x)))

(<= (line ?x) (row ?m ?x))
(<= (line ?x) (column ?m ?x))
(<= (line ?x) (diagonal ?x))

(<= (row ?m ?x)
    (true (cell ?m 1 ?x))
    (true (cell ?m 2 ?x))
    (true (cell ?m 3 ?x)))

(<= (column ?n ?x)
    (true (cell 1 ?n ?x))
    (true (cell 2 ?n ?x))
    (true (cell 3 ?n ?x)))

(<= (diagonal ?x)
    (true (cell 1 1 ?x))
    (true (cell 2 2 ?x))
    (true (cell 3 3 ?x)))
(<= (diagonal ?x)
    (true (cell 1 3 ?x))
    (true (cell 2 2 ?x))
    (true (cell 3 1 ?x)))

(<= (goal x 100)
    (line x))
(<= (goal x 50)
    (not (line x))
    (not (line o))
    (not open))
(<= (goal x 0)
    (line o))
(<= (goal o 100)
    (line o))
(<= (goal o 50)
    (not (line x))
    (not (line o))
    (not open))
(<= (goal o 0)
    (line x))

(<= terminal
    (line x))
(<= terminal
    (line o))
(<= terminal
    (not open))

(<= open
    (true (cell ?m ?n b)))
    )";
    let sexp = parse_sexpr(rules).unwrap().0;
    sexpr_to_db(&sexp).unwrap()
}

fn tic_tac_toe_playthrough(c: &mut Criterion) {
    use rand::{seq::SliceRandom, FromEntropy, rngs::SmallRng};

    let mut rng = SmallRng::from_entropy();
    let (db, labeler, lens) = set_up_tic_tac_toe();
    let ggp = GGP::from_rules(db, &labeler, &lens).unwrap();

    assert!(!ggp.is_done());
    c.bench_function("ggp: tic tac toe playthrough", move |b| b.iter(|| {
        let mut ggp = ggp.clone();
        let roles = ggp.roles();
        while !ggp.is_done() {
            let moves = roles.iter().map(|&r| {
                let all = ggp.legal_moves_for(r);
                assert!(!all.is_empty());
                (r, all.choose(&mut rng).unwrap().clone())
            }).collect::<Vec<_>>();
            ggp.play(&moves[..]);
        }
    }));
}

criterion_group!(ggp, tic_tac_toe_playthrough);

criterion_main!(basics, go, ggp);
