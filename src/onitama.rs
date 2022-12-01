use crate::{
    card::cards_mask,
    index::{Empty, Indexer},
    proj,
};

#[derive(PartialEq, Eq, Default, Clone, Copy)]
pub struct Board {
    pawns: [u32; 2],
    cards: [u16; 2],
    kings: [u8; 2],
}

#[derive(Clone, Default)]
pub struct BoardIncomplete {
    pawn_n: [u8; 2], // number of pawns for each side
    cards: [u16; 2],
    king1: u8,
}

// team 0 is at the bottom, so that they can use the cards unrotated
const TEMPLES: [u8; 2] = [22, 2];
const TABLE_MASK: u32 = (1 << 25) - 1;
const NUM_PAWNS_MASK: u32 = 0b11111; // five options, because there can be 0..=4 pawns

impl BoardIncomplete {
    // Index boards that are not win in 1
    // we could save a little more by restricting opp king loc
    // but that would need an additional indirection
    pub fn index1(all_cards: u16) -> impl Indexer<Item = Self> {
        type B = BoardIncomplete;

        let cards1_mask = move |b: &B| all_cards & !b.cards[0];
        let king1_mask = TABLE_MASK & !(1 << TEMPLES[0]);

        Empty::default()
            .choose(2, proj!(|b: B| b.cards[0]), all_cards)
            .choose(2, proj!(|b: B| b.cards[1]), cards1_mask)
            .choose_one(proj!(|b: B| b.king1), king1_mask)
            .choose_one(proj!(|b: B| b.pawn_n[0]), NUM_PAWNS_MASK)
            .choose_one(proj!(|b: B| b.pawn_n[1]), NUM_PAWNS_MASK)
    }

    pub fn is_lost(&self) -> bool {
        let king1_temple_attack = cards_mask::<false>(TEMPLES[0], self.cards[1]);
        (1 << self.king1) & king1_temple_attack != 0
    }
}

impl Board {
    pub fn incomplete(&self) -> BoardIncomplete {
        BoardIncomplete {
            pawn_n: [
                self.pawns[0].count_ones() as u8,
                self.pawns[1].count_ones() as u8,
            ],
            cards: self.cards,
            king1: self.kings[1],
        }
    }

    // Second half of the index, the size of this table depends on the first index
    pub fn index2(first: BoardIncomplete) -> impl Indexer<Item = Self> {
        type B = Board;

        let temple1 = 1 << TEMPLES[1];
        let king0 = |b: &B| 1 << b.kings[0];
        let king1 = 1 << first.king1;
        let attack_temple1 = cards_mask::<true>(TEMPLES[1], first.cards[0]);
        let attack_kings1 = cards_mask::<true>(first.king1, first.cards[0]);

        let kings0_mask = TABLE_MASK & !temple1 & !king1 & !attack_temple1 & !attack_kings1;
        let pawns0_mask = move |b: &B| TABLE_MASK & !king1 & !king0(b) & !attack_kings1;
        let pawns1_mask = move |b: &B| TABLE_MASK & !king1 & !king0(b) & !b.pawns[0];

        Empty(Board {
            pawns: [0; 2],
            cards: first.cards,
            kings: [0, first.king1],
        })
        .choose_one(proj!(|b: B| b.kings[0]), kings0_mask)
        .choose(first.pawn_n[0], proj!(|b: B| b.pawns[0]), pawns0_mask)
        .choose(first.pawn_n[1], proj!(|b: B| b.pawns[1]), pawns1_mask)
    }

    // // generate pawn moves that do not result in obv win for opp
    // fn generate_next_pawn(self) -> impl Iterator<Item = Board> {
    //     // if king is threatened we only need to consider defense moves

    //     // let card_mask = self.cards[0];
    //     // Choose::<1, true, u16>::new(!card_mask).flat_map(|card| {
    //     //     let from_mask = self.pawns[0] | 1 << self.kings[0];
    //     //     Choose::<1, true, u32>::new(!from_mask).flat_map(|from_mask| {
    //     //         let from = from_mask.trailing_zeros() as u8;
    //     //         let to_mask = cards_mask(from, card, false);
    //     //         Choose::<1, true, u32>::new(!to_mask).map(|to|{

    //     //         })
    //     //     })
    //     // })
    //     todo!()
    // }

    // // generate king moves that do not result in obv win for opp
    // // this might return a board that is obv loss for opp (temple threat)
    // // in that case we return None
    // fn generate_next_king(self) -> impl Iterator<Item = Option<Board>> {
    //     todo!()
    // }

    // // combine the above two functions
    // pub fn generate_next(&self) -> impl Iterator<Item = Option<Board>> {
    //     todo!()
    // }
}
