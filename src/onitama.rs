use crate::{
    card::cards_mask,
    index::{Choose, Index},
};

#[derive(PartialEq, Eq)]
pub struct Board {
    pawns: [u32; 2],
    cards: [u16; 2],
    kings: [u8; 2],
}

impl Board {
    const TEMPLES: [u8; 2] = [2, 22];
    const TABLE_MASK: u32 = (1 << 25) - 1;

    // Index boards that are not win in 1
    // we could save a little more by restricting opp king loc
    // but that would need an additional indirection
    pub fn index1(&self, all_cards: u16) -> Index {
        let cards0_mask = all_cards;
        let cards1_mask = all_cards & !self.cards[0];
        let kings1_mask = Self::TABLE_MASK & !(1 << Self::TEMPLES[0]);
        let pawns0_count = self.pawns[0].count_ones() as u8;

        Index::default()
            .choose_exact(self.cards[0], cards0_mask)
            .choose_exact(self.cards[1], cards1_mask)
            .choose_one(self.kings[1], kings1_mask)
            .apply(pawns0_count, 5)
    }

    // Second half of the index, the size of this table depends on the first index
    pub fn index2(&self) -> Index {
        let kings0_mask = Self::TABLE_MASK
            & !(1 << Self::TEMPLES[1])
            & !cards_mask::<true>(Self::TEMPLES[1], self.cards[0])
            & !cards_mask::<true>(self.kings[1], self.cards[0]);
        let kings_mask = Self::TABLE_MASK & !(1 << self.kings[0]) & !(1 << self.kings[1]);
        let pawns0_skip = kings_mask & !cards_mask::<true>(self.kings[1], self.cards[0]);
        let pawns1_skip = kings_mask & !self.pawns[0];

        Index::default()
            .choose_one(self.kings[0], kings0_mask)
            .choose_exact(self.pawns[0], pawns0_skip)
            .choose_at_most::<4>(self.pawns[1], pawns1_skip)
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
