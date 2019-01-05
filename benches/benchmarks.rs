#[macro_use]
extern crate criterion;
extern crate rand;
extern crate worthy_opponent;

use ::rand::{Rng, FromEntropy, rngs::SmallRng, distributions::Standard, seq::SliceRandom};

use criterion::Criterion;

use worthy_opponent::basics::{Space, History};
use worthy_opponent::game::Game;
use worthy_opponent::go::*;

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

criterion_main!(basics, go);
