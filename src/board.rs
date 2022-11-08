use std::{fmt::Debug, ops::BitAnd};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Board {
    // it is illegal for both sides to have the same bit zeroed
    // if both sides are one, then the square is also allowed to be empty
    sides: [u32; 2],
}

const PIECE_MASK: u32 = (1 << 5) - 1;

impl Debug for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Board")
            .field("me_", &format!("{:05b}", self.sides[0] & PIECE_MASK))
            .field("you", &format!("{:05b}", self.sides[1] & PIECE_MASK))
            .finish()
    }
}

// by default we allow everything
impl Default for Board {
    fn default() -> Self {
        Self {
            sides: [u32::MAX, u32::MAX],
        }
    }
}

const MAX_PIECES: u32 = 2;

impl Board {
    pub fn check(self) -> Option<Self> {
        let [me, you] = self.sides;
        if (me & !you).count_ones() > MAX_PIECES {
            return None;
        }
        if (you & !me).count_ones() > MAX_PIECES {
            return None;
        }
        (me | you == u32::MAX).then_some(self)
    }
}

impl BitAnd for &Board {
    type Output = Option<Board>;

    fn bitand(self, rhs: Self) -> Self::Output {
        let [a1, a2] = self.sides;
        let [b1, b2] = rhs.sides;
        let new = Board {
            sides: [a1 & b1, a2 & b2],
        };
        new.check()
    }
}

const ME: usize = 0;
const YOU: usize = 1;

impl Board {
    pub fn backward(&self, from: u32, to: u32) -> Option<[Self; 2]> {
        if self.sides[ME] & to == 0 {
            // `to` should be able to be me
            return None;
        }
        if self.sides[ME] & self.sides[YOU] & from == 0 {
            // `from` should be able to be empty
            return None;
        }

        let mut new = *self;

        // `from` can only be me
        new.sides[ME] |= from;
        new.sides[YOU] &= !from;

        // `to` can be empty or you
        // this is represented as two boards, one with everything, and one with just me
        // `new` is already able to be me at `to`
        // we don't know if you is possible though, so we do both
        new.sides[YOU] |= to;
        let new_backup = new;
        new.sides[YOU] &= !to;

        Some([new.swap(), new_backup.swap()])
    }

    pub fn swap(self) -> Self {
        Self {
            sides: [self.sides[1], self.sides[0]],
        }
    }
}
