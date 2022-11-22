use bit_iter::BitIter;
use seq_macro::seq;

// TODO: precalculate with every possible offset if that is faster

#[inline]
pub(crate) fn single_mask<const S: bool>(cards: u16, offset: u8) -> u32 {
    let bitmap = get_bitmap::<S>(cards);
    let new = bitmap & BOARD_MASK[offset as usize];
    ((new as u64) << offset >> 14) as u32
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

pub(crate) fn get_bitmap<const S: bool>(cards: u16) -> u32 {
    #[allow(clippy::unusual_byte_groupings)]
    const CARD_MAP_0: [u32; 16] = [
        0b00000_00000_01001_00000_00000,
        0b00100_00000_00010_00000_00100,
        0b00000_01010_00000_01010_00000,
        0b00000_01000_00010_01000_00000,
        0b00010_01000_00000_01000_00010,
        0b00000_00110_00000_00110_00000,
        0b00000_00010_01000_00010_00000,
        0b00000_00100_00010_00100_00000,
        0b00100_00010_00000_01000_00000,
        0b00000_00110_00000_01100_00000,
        0b00000_00100_01010_00000_00000,
        0b00000_01010_00000_00100_00000,
        0b00000_01000_00000_00010_00100,
        0b00000_01100_00000_00110_00000,
        0b00000_00000_01010_00100_00000,
        0b00000_00100_00000_01010_00000,
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
    board.reverse_bits() >> 3
}
