struct Board {
    pawns: [u32; 2],
    cards: [u16; 2],
    kings: [u8; 2],
}

struct Choose<const MAX: usize, const EXACT: bool, N> {
    current: N,
    skip: N,
}

impl<const MAX: usize, const EXACT: bool, N> Choose<MAX, EXACT, N> {
    pub fn new(skip: N) -> Self {
        todo!()
    }
}

impl<const MAX: usize, const EXACT: bool, N> Iterator for Choose<MAX, EXACT, N> {
    type Item = N;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

fn cards_mask(offset: u8, cards: u16, inv: bool) -> u32 {
    todo!()
}

impl Board {
    const SIZE: u8 = 5;
    const TEMPLES: [u8; 2] = [2, 22];
    const TABLE_MASK: u32 = (1 << 25) - 1;

    // generates all boards that are not win in 1.2 without pieces
    fn generate_no_pawns(all_cards: u16) -> impl Iterator<Item = Board> {
        let cards0_skip = !all_cards;
        Choose::<2, true, u16>::new(cards0_skip).flat_map(move |cards0| {
            let cards1_skip = !all_cards | cards0;
            Choose::<2, true, u16>::new(cards1_skip).flat_map(move |cards1| {
                let king0_skip = !Self::TABLE_MASK
                    | 1 << Self::TEMPLES[1]
                    | cards_mask(Self::TEMPLES[1], cards0, true);
                Choose::<1, true, u32>::new(king0_skip).flat_map(move |king0_mask| {
                    let king0 = king0_mask.trailing_zeros() as u8;
                    let king1_skip = !Self::TABLE_MASK
                        | 1 << Self::TEMPLES[0]
                        | cards_mask(king0, cards0, false)
                        | cards_mask(Self::TEMPLES[0], cards1, false);
                    Choose::<1, true, u32>::new(king1_skip).map(move |king1_mask| {
                        let king1 = king1_mask.trailing_zeros() as u8;
                        Board {
                            pawns: [0; 2],
                            cards: [cards0, cards1],
                            kings: [king0, king1],
                        }
                    })
                })
            })
        })
    }

    // add pawns to boards with kings and cards
    fn generate_with_pawns(self) -> impl Iterator<Item = Board> {
        let king_mask = !Self::TABLE_MASK | 1 << self.kings[0] | 1 << self.kings[1];
        let pieces0_skip = king_mask | cards_mask(self.kings[1], self.cards[0], true);
        Choose::<4, false, u32>::new(pieces0_skip).flat_map(move |pieces0| {
            let pieces1_skip = king_mask | pieces0;
            Choose::<4, false, u32>::new(pieces1_skip).map(move |pieces1| Board {
                pawns: [pieces0, pieces1],
                cards: self.cards,
                kings: self.kings,
            })
        })
    }
}
