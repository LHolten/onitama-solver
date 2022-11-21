use std::ops::{Index, IndexMut};

use crate::onitama::Board;

pub struct Table {
    all_cards: u16,
    kings_cards: Vec<KingCards>,
}

struct KingCards {
    pawns: Vec<bool>,
}

impl Index<&Board> for Table {
    type Output = bool;

    fn index(&self, index: &Board) -> &Self::Output {
        let index_no_pawns = index.no_pawns();
        let pos = Board::generate_no_pawns(self.all_cards)
            .position(|b| b == index_no_pawns)
            .unwrap();
        let pos2 = Board::generate_with_pawns(index_no_pawns)
            .position(|b| &b == index)
            .unwrap();
        &self.kings_cards[pos].pawns[pos2]
    }
}

impl IndexMut<&Board> for Table {
    fn index_mut(&mut self, index: &Board) -> &mut Self::Output {
        let index_no_pawns = index.no_pawns();
        let pos = Board::generate_no_pawns(self.all_cards)
            .position(|b| b == index_no_pawns)
            .unwrap();
        let pos2 = Board::generate_with_pawns(index_no_pawns)
            .position(|b| &b == index)
            .unwrap();
        &mut self.kings_cards[pos].pawns[pos2]
    }
}

#[derive(PartialEq, Eq)]
pub enum Res {
    NotLoss,
    Loss,
}

impl Table {
    pub fn populate(&mut self) {
        loop {
            let mut done = true;
            for board_no_pawns in Board::generate_no_pawns(self.all_cards) {
                for board in Board::generate_with_pawns(board_no_pawns) {
                    if self.search(&board, 1) == Res::Loss {
                        done &= self[&board]; // all values were true
                        self[&board] = true;
                    }
                }
            }
            if done {
                break;
            }
        }
    }

    pub fn search(&self, board0: &Board, depth: u8) -> Res {
        if depth == 0 {
            return if self[board0] {
                Res::Loss
            } else {
                Res::NotLoss
            };
        }
        // all the next states have to be wins
        'outer: for board1 in board0.generate_next() {
            let Some(board1) = board1 else {
                return Res::NotLoss;
            };
            if self.search(&board1, 0) == Res::Loss {
                return Res::NotLoss;
            }
            // find any next state that is a loss
            for board2 in board1.generate_next() {
                let Some(board2) = board2 else {
                    continue 'outer;
                };
                if self.search(&board2, depth - 1) == Res::Loss {
                    continue 'outer;
                }
            }
            return Res::NotLoss;
        }
        return Res::Loss;
    }
}
