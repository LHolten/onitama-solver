use bit_iter::BitIter;
use seq_macro::seq;

// TODO: precalculate with every possible offset if that is faster

#[inline]
pub(crate) fn cards_mask<const S: bool>(offset: u8, cards: u16) -> u32 {
    let bitmap = get_bitmap::<S>(cards);
    offset_mask(offset, bitmap)
}

#[inline]
pub(crate) fn offset_mask(offset: u8, mask: u32) -> u32 {
    let new = mask & BOARD_MASK[offset as usize];
    ((new as u64) << 12 >> offset) as u32
}

/// this method is used to get the orginal mask after using [offset_mask]
#[inline]
pub(crate) fn undo_offset(offset: u8, mask: u32) -> u32 {
    ((mask as u64) << offset >> 12) as u32
}

#[allow(clippy::unusual_byte_groupings)]
const BOARD_MASK: [u32; 25] = [
    0b00000_00000_00111_00111_00111,
    0b00000_00000_01111_01111_01111,
    0b00000_00000_11111_11111_11111,
    0b00000_00000_11110_11110_11110,
    0b00000_00000_11100_11100_11100,
    0b00000_00111_00111_00111_00111,
    0b00000_01111_01111_01111_01111,
    0b00000_11111_11111_11111_11111,
    0b00000_11110_11110_11110_11110,
    0b00000_11100_11100_11100_11100,
    0b00111_00111_00111_00111_00111,
    0b01111_01111_01111_01111_01111,
    0b11111_11111_11111_11111_11111,
    0b11110_11110_11110_11110_11110,
    0b11100_11100_11100_11100_11100,
    0b00111_00111_00111_00111_00000,
    0b01111_01111_01111_01111_00000,
    0b11111_11111_11111_11111_00000,
    0b11110_11110_11110_11110_00000,
    0b11100_11100_11100_11100_00000,
    0b00111_00111_00111_00000_00000,
    0b01111_01111_01111_00000_00000,
    0b11111_11111_11111_00000_00000,
    0b11110_11110_11110_00000_00000,
    0b11100_11100_11100_00000_00000,
];

// card masks are ordered from top to bottom
pub(crate) fn get_bitmap<const S: bool>(cards: u16) -> u32 {
    #[allow(clippy::unusual_byte_groupings)]
    const CARD_MAP_0: [u32; 16] = [
        0b00000_00100_00010_00100_00000,
        0b00000_00100_01010_00000_00000,
        0b00000_00100_01000_00100_00000,
        0b00000_01010_01010_00000_00000,
        0b00000_00100_10001_00000_00000,
        0b00100_00000_00000_00100_00000,
        0b00000_01010_00000_01010_00000,
        0b00000_00100_00000_01010_00000,
        0b00000_10001_00000_01010_00000,
        0b00000_01010_00000_00100_00000,
        0b00000_01000_10000_00010_00000,
        0b00000_00010_00001_01000_00000,
        0b00000_01000_01010_00010_00000,
        0b00000_00010_01010_01000_00000,
        0b00000_01000_00010_01000_00000,
        0b00000_00010_01000_00010_00000,
    ];
    const CARD_MAP_1: [u32; 16] = seq!(C in 0..16 {
        [
            #(reverse_bitmap(CARD_MAP_0[C]),)*
        ]
    });

    let mut mask = 0;
    BitIter::from(cards).for_each(|card| mask |= [CARD_MAP_0, CARD_MAP_1][S as usize][card]);
    mask
}

#[inline]
const fn reverse_bitmap(board: u32) -> u32 {
    board.reverse_bits() >> (32 - 25)
}

#[cfg(test)]
mod tests {
    use super::cards_mask;

    #[test]
    pub fn test() {
        let cards = 1;
        let res10 = 0b00000_10000_01000_10000_00000;
        let res12 = 0b00000_00100_00010_00100_00000;
        let res10_rev = 0b00000_10000_00000_10000_00000;
        let res3_rev = 0b00100_00010_00000_00000_00000;
        let mask10 = cards_mask::<false>(10, cards);
        let mask10_rev = cards_mask::<true>(10, cards);
        let mask12 = cards_mask::<false>(12, cards);
        let mask3_rev = cards_mask::<true>(3, cards);
        // println!("{mask3_rev:025b}");
        assert_eq!(res12, mask12);
        assert_eq!(res10, mask10);
        assert_eq!(res10_rev, mask10_rev);
        assert_eq!(res3_rev, mask3_rev);
    }
}
