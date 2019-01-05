#[macro_use]
extern crate criterion;
extern crate rand;
extern crate worthy_opponent;

use ::rand::{Rng, FromEntropy, rngs::SmallRng, distributions::Standard};

use criterion::Criterion;

use worthy_opponent::basics::{Space, History};

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
criterion_main!(basics);
