//! list of things to prevent:
//! - temple capture 1 and 2 plies
//!   - keep in mind that if there is a pawn on the temple it is not a threat
//! - temple capture 3 plies
//!   - move pawn from temple (no king threat), or move king in position (without threat)
//! - king capture 1 and -1 plies
//!   - two pawns attack with side card
//!
//! these things need everything except pawns[0]
//!

use bit_iter::BitIter;

use crate::{
    card::{cards_mask, get_bitmap},
    index::{Empty, Indexer},
    proj,
};

#[derive(PartialEq, Eq, Default, Clone, Copy)]
pub struct Board1 {
    cards0: u16,
    side_card: u8,
    kings: [u8; 2],
    pawns1: u32,
}

// team 0 is at the bottom, so that they can use the cards unrotated
const TEMPLES: [u8; 2] = [22, 2];
const TABLE_MASK: u32 = (1 << 25) - 1;
const NUM_PAWNS_MASK: u32 = 0b11111; // five options, because there can be 0..=4 pawns

impl Board1 {
    pub fn index1(all_cards: u16, pawns1_len: u8) -> impl Indexer<Item = Self> {
        type B = Board1;

        let cards0_mask = move |b: &B| all_cards & !(1 << b.side_card);
        let king1_mask = TABLE_MASK & !(1 << TEMPLES[0]);
        let king0_mask = TABLE_MASK & !(1 << TEMPLES[1]);
        let pawns1_mask = |b: &B| TABLE_MASK & !(1 << b.kings[0]) & !(1 << b.kings[1]);

        Empty::default()
            .choose_one(proj!(|b: B| b.side_card), all_cards as u32)
            .choose(2, proj!(|b: B| b.cards0), cards0_mask)
            .choose_one(proj!(|b: B| b.kings[1]), king1_mask)
            .choose_one(proj!(|b: B| b.kings[0]), king0_mask)
            .choose(pawns1_len, proj!(|b: B| b.pawns1), pawns1_mask)
    }
}

#[derive(PartialEq, Eq, Default, Clone, Copy)]
pub struct Board {
    pawns: [u32; 2],
    cards: [u16; 2],
    kings: [u8; 2],
}

impl Board {
    // Second half of the index, the size of this table depends on the first index
    pub fn index2(
        all_cards: u16,
        first: Board1,
        pawns0_len: u8,
        has_pawn0_on_temple1: bool,
    ) -> Option<impl Indexer<Item = Self> + Clone> {
        type B = Board;
        let king0 = 1 << first.kings[0];
        let king1 = 1 << first.kings[1];
        let cards1 = all_cards & !first.cards0 & !(1 << first.side_card);
        let mut cards1_iter = BitIter::from(cards1);
        let old1_cards = 1 << first.side_card | 1 << cards1_iter.next().unwrap();
        let old2_cards = 1 << first.side_card | 1 << cards1_iter.next().unwrap();

        // king0 is not allowed to attack king1
        let king1_attacked = king1 | cards_mask::<true>(first.kings[1], first.cards0);
        if king1_attacked & king0 != 0 {
            return None;
        }

        // if opp attacks temple they need to have a pawn on temple to prevent win
        let temple0_attacked = cards_mask::<false>(TEMPLES[0], cards1);
        let need_pawn1_on_temple0 = king1 & temple0_attacked != 0;
        let has_pawn1_on_temple0 = first.pawns1 & 1 << TEMPLES[0] != 0;
        if need_pawn1_on_temple0 && !has_pawn1_on_temple0 {
            return None;
        }

        // make sure opp doesn't have double attack with any old cards on our king
        let pieces1 = first.pawns1 | king1;
        let pieces1_attack = BitIter::from(pieces1).fold(0, |union, offset| {
            union | cards_mask::<true>(offset as u8, cards1)
        });

        // check if previous state must have been a win in 1 by king capture
        let king0_old1_attacked = cards_mask::<false>(first.kings[0], old1_cards);
        let king0_old2_attacked = cards_mask::<false>(first.kings[0], old2_cards);
        if (king0_old1_attacked & pieces1).count_ones() >= 2
            && (king0_old2_attacked & pieces1).count_ones() >= 2
        {
            return None;
        }

        // check for temple win in 1 ply
        let temple1_attacked = cards_mask::<true>(TEMPLES[1], first.cards0);
        let need_pawn0_on_temple1 = king0 & temple1_attacked != 0;
        if need_pawn0_on_temple1 && !has_pawn0_on_temple1 {
            return None;
        }

        // check if we are in checkmate
        let king0_attacked = cards_mask::<false>(first.kings[0], cards1);
        let options = cards_mask::<false>(first.kings[0], first.cards0);
        if (king0_attacked & pieces1).count_ones() >= 2 && options & !pieces1_attack == 0 {
            return None;
        }

        let pawns0 = if has_pawn0_on_temple1 {
            1 << TEMPLES[1]
        } else {
            // if there is no pawn on temple, then we need to block jump squares
            let mut cards0_iter = BitIter::from(first.cards0);
            let options1 = cards_mask::<false>(first.kings[0], 1 << cards0_iter.next().unwrap());
            let options2 = cards_mask::<false>(first.kings[0], 1 << cards0_iter.next().unwrap());
            let mut cards0_iter = BitIter::from(first.cards0);
            let rev_2 = cards_mask::<true>(
                TEMPLES[1],
                1 << first.side_card | 1 << cards0_iter.next().unwrap(),
            );
            let rev_1 = cards_mask::<true>(
                TEMPLES[1],
                1 << first.side_card | 1 << cards0_iter.next().unwrap(),
            );
            (options1 & rev_1 | options2 & rev_2) & !pieces1_attack
        };

        // any required pawns should not interfer with existing pieces
        if (pieces1 | king1_attacked) & pawns0 != 0 {
            return None;
        }

        // we can not have to many pieces
        let pawns0_required = pawns0.count_ones() as u8;
        if pawns0_required > pawns0_len {
            return None;
        }

        let pawns0_mask =
            TABLE_MASK & !king1_attacked & !king0 & !(1 << TEMPLES[1]) & !first.pawns1 & !pawns0;

        // TODO: add required pawns
        let iter = Empty(Board {
            pawns: [pawns0, first.pawns1],
            cards: [first.cards0, cards1],
            kings: first.kings,
        })
        .choose(
            pawns0_len - pawns0_required,
            proj!(|b: B| b.pawns[0]),
            pawns0_mask,
        );
        Some(iter)
    }
}

#[test]
fn count_perft2() {
    // OX BOAR HORSE ELEPHANT CRAB
    let all_cards = 0b0000000000011111;
    let mut total = 0;
    for pawns1_len in 0..=4 {
        for board_no_pawns in Board1::index1(all_cards, pawns1_len) {
            for pawns0_len in 0..=4 {
                'first: {
                    let Some(indexer) = Board::index2(all_cards, board_no_pawns.clone(), pawns0_len, false) else {
                        break 'first;
                    };
                    let board = indexer.clone().into_iter().next().unwrap();
                    total += indexer.index(&board).total;
                }

                'second: {
                    let Some(indexer) = Board::index2(all_cards, board_no_pawns.clone(), pawns0_len, true) else {
                        break 'second;
                    };
                    let board = indexer.clone().into_iter().next().unwrap();
                    total += indexer.index(&board).total;
                }
            }
        }
    }

    dbg!(total);
}
