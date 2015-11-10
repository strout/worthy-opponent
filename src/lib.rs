#![cfg_attr(test, feature(test))]

extern crate rand;
extern crate bit_set;

#[cfg(test)]
extern crate test;
#[cfg(test)]
extern crate quickcheck;

mod basics;
pub mod connectfour;
pub mod game;
pub mod ggp;
pub mod go;
pub mod ninemensmorris;
pub mod tictactoe;
