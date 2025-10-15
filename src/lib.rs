#![feature(impl_trait_in_assoc_type)]

// mod anf;
mod card;
mod index;
// mod onitama;
// mod ply;
// mod onitama2;
pub mod onitama_simd;
mod proj;
// mod table;

// use anf::Anf;
// use board::Board;

// mod board;

// type Table = Anf;

// impl Table {
//     pub fn losses(&self) -> Self {
//         // self is the wins for player 1.
//         let wins = self;

//         // check if player 2 can only get wins for player 1
//         let mut loss = !Table::default();

//         for from in 0..Board::BOARD_SIZE {
//             if from != 0 {
//                 let (from, to) = (1 << from, 1 << from >> 1);
//                 loss = &loss & &!wins.map(|b| b.backward(from, to))
//             }
//             if from != Board::BOARD_SIZE - 1 {
//                 let (from, to) = (1 << from, 1 << from << 1);
//                 loss = &loss & &!wins.map(|b| b.backward(from, to))
//             }
//         }

//         loss
//     }
// }

// #[cfg(test)]
// mod tests {
//     use crate::{board::Board, Table};

//     #[test]
//     fn do_steps() {
//         let all = Board::generate_all();
//         println!("total states: {}", all.len());
//         let mut anf = Table::default();
//         for i in 0.. {
//             anf = anf.losses();
//             if i % 2 == 0 {
//                 anf = !anf;
//                 println!(
//                     "term count: {}, num wins: {}",
//                     all.len() - anf.len(),
//                     anf.count(&all)
//                 );
//                 anf = !anf;
//             } else {
//                 println!("term count: {}, num losses: {}", anf.len(), anf.count(&all));
//             }
//             // println!("table: {:#?}", anf);
//         }
//     }
// }
