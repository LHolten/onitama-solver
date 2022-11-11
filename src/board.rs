use std::{fmt::Debug, iter::from_fn, ops::BitAnd};

// Default value is all zeros and allowes everything
#[derive(Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Board {
    // both zero, means we don't care
    // both ones is illegal.
    sides: [u32; 2],
}

const PIECE_MASK: u32 = (1 << 3) - 1;

impl Debug for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Board")
            .field("me_", &format!("{:03b}", self.sides[0] & PIECE_MASK))
            .field("you", &format!("{:03b}", self.sides[1] & PIECE_MASK))
            .finish()
    }
}

impl Board {
    pub fn check(self) -> Option<Self> {
        let [me, you] = self.sides;
        if me.count_ones() > Self::MAX_PIECES {
            return None;
        }
        if you.count_ones() > Self::MAX_PIECES {
            return None;
        }
        (me & you == 0).then_some(self)
    }

    // if self is less specific (less pieces) than rhs
    pub fn includes(&self, rhs: &Self) -> bool {
        let [a1, a2] = self.sides;
        let [b1, b2] = rhs.sides;
        a1 & !b1 == 0 && a2 & !b2 == 0
    }
}

impl BitAnd for &Board {
    type Output = Option<Board>;

    fn bitand(self, rhs: Self) -> Self::Output {
        let [a1, a2] = self.sides;
        let [b1, b2] = rhs.sides;
        let new = Board {
            sides: [a1 | b1, a2 | b2],
        };
        new.check()
    }
}

const ME: usize = 0;
const YOU: usize = 1;

impl Board {
    pub const BOARD_SIZE: usize = 10;
    pub const MAX_PIECES: u32 = 4;

    pub fn backward(&self, from: u32, to: u32) -> Option<[Self; 2]> {
        if self.sides[YOU] & to != 0 {
            // assert that YOU is not required.
            // `to` should be able to be me
            return None;
        }
        if (self.sides[ME] | self.sides[YOU]) & from != 0 {
            // assert that neither ME or YOU is required.
            // `from` should be able to be empty
            return None;
        }

        let mut new = *self;

        // `from` can only be me
        // it was already checked that YOU is not required.
        new.sides[ME] |= from;

        // `to` can be empty or you
        // this is represented as two boards, one with everything, and one with just me
        // `new` is already able to be me at `to`
        // we don't know if me is possible though, so we do both
        new.sides[ME] |= to;
        let new_backup = new;
        new.sides[ME] &= !to;

        Some([new.swap(), new_backup.swap()])
    }

    pub fn swap(self) -> Self {
        Self {
            sides: [self.sides[1], self.sides[0]],
        }
    }

    pub fn generate_all() -> Vec<Self> {
        let mut list = Vec::new();
        for me in Self::generate_combinations(0) {
            for you in Self::generate_combinations(me) {
                list.push(Self { sides: [me, you] })
            }
        }
        list
    }

    pub fn generate_combinations(mut skip: u32) -> impl Iterator<Item = u32> {
        skip |= !((1 << Self::BOARD_SIZE) - 1);
        let mut prev = None;
        from_fn(move || {
            if let Some(p) = prev {
                let p = next(p, skip, Self::MAX_PIECES);
                prev = (p != 0).then_some(p);
            } else {
                prev = Some(0);
            }
            prev
        })
    }
}

pub fn next(board: u32, skip: u32, max: u32) -> u32 {
    let res = if board.count_ones() < max {
        1
    } else {
        board & board.wrapping_neg()
    };
    res.wrapping_add(board | skip) & !skip
}
