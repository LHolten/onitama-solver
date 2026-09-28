use std::sync::atomic::Ordering;

use bit_iter::BitIter;

use super::{Accum, Spread, SpreadStatus};

impl Accum<'_> {
    /// Add what this move says about the source chunk.
    ///
    /// A forward move leaves `pieces1` alone unless it captures, so the king
    /// positions of the source and the target run in the same order: within a
    /// row the two indices differ by a constant, and a capture only splits that
    /// into two such runs. Walking those runs means the source index is a
    /// counter rather than a lookup, so the inner loop never builds a
    /// [`KingPos`] or a lookup table at all.
    pub fn accumulate(self) {
        let (from, to) = self.step;
        let rows_new = self.new.pieces0 & !(1 << 22);
        let rows_old = self.old.pieces0 & !(1 << 22);
        let cols_new = self.new.pieces1 & !(1 << 2);
        let cols_old = self.old.pieces1 & !(1 << 2);

        let width_new = cols_new.count_ones() as usize;
        let width_old = cols_old.count_ones() as usize;
        // a capture takes one column out of the middle, which is where the two
        // runs meet; without one the whole row is a single run
        let split = if width_new == width_old {
            width_new
        } else {
            (cols_old & (1 << to) - 1).count_ones() as usize
        };

        // hoisted so the inner runs do not re-derive them from the `Vec`
        let target = self.target;
        let slice = &mut self.slice[..];

        let mut new_i = 0;
        for king0 in BitIter::from(rows_new) {
            let old0 = if king0 == to {
                if from == 22 {
                    // there is no way we came from the temple
                    new_i += width_new;
                    continue;
                }
                from
            } else {
                king0
            };
            let base = (rows_old & (1 << old0) - 1).count_ones() as usize * width_old;

            // without a capture the whole row is one run and the two indices
            // just walk together; with one, the source is a step ahead after
            // the column that was taken
            for k in 0..split {
                let new_val = unsafe { target.get_unchecked(new_i + k) };
                let s = unsafe { slice.get_unchecked_mut(base + k) };
                // if new state is not won, then old state is not lost
                *s |= !new_val.load(Ordering::Relaxed) & self.mask;
            }
            for k in split..width_new {
                let new_val = unsafe { target.get_unchecked(new_i + k) };
                let s = unsafe { slice.get_unchecked_mut(base + k + 1) };
                *s |= !new_val.load(Ordering::Relaxed) & self.mask;
            }
            new_i += width_new;
        }
    }
}

impl Spread<'_> {
    /// `UP` is the `go_up` form, which adds a `pieces0` piece on `from`.
    ///
    /// That is the only way the king can be standing where the move came from,
    /// so for an ordinary backward move the test for it is not just hoisted out
    /// of the walk, it is gone: `from` is a `pieces1` square and `king0` only
    /// ever visits `pieces0`. That also leaves the rows in the same order, so
    /// the row index is a counter, and which source column each target column
    /// came from is settled once for the move.
    pub fn spreadout<const UP: bool>(self) -> SpreadStatus {
        let (from, to) = self.step;
        let rows_new = self.new.pieces0 & !(1 << 22);
        let rows_old = self.old.pieces0 & !(1 << 22);
        let cols_new = self.new.pieces1 & !(1 << 2);
        let cols_old = self.old.pieces1 & !(1 << 2);
        let width_new = cols_new.count_ones() as usize;
        let width_old = cols_old.count_ones() as usize;

        // where each target column came from. `NONE` is the moved piece having
        // come off the temple, which the source has no column for.
        const NONE: u8 = u8::MAX;
        let mut came_from = [NONE; 8];
        for (slot, square) in came_from.iter_mut().zip(BitIter::from(cols_new)) {
            let square = if square == to { from } else { square };
            if cols_old & 1 << square != 0 {
                *slot = (cols_old & (1 << square) - 1).count_ones() as u8;
            }
        }

        let expanded = self.expanded;
        let target = self.target;

        let mut progress = false;
        let mut could = false;
        let mut new_i = 0;
        let mut row = 0;
        for king0 in BitIter::from(rows_new) {
            if UP && king0 == from {
                // this king was added, so this oldk did not exist
                new_i += width_new;
                continue;
            }
            // only the `go_up` form moves the rows around
            let base = if UP {
                (rows_old & (1 << king0) - 1).count_ones() as usize * width_old
            } else {
                row * width_old
            };
            row += 1;

            for &col in &came_from[..width_new] {
                if col == NONE {
                    new_i += 1;
                    continue;
                }

                let (tmp, may) = unsafe { *expanded.get_unchecked(base + col as usize) };
                let new_val = unsafe { target.get_unchecked(new_i) };
                new_i += 1;
                let fetch = new_val.load(Ordering::Relaxed);
                // `tmp` is a subset of `may`, so this also covers the case where
                // there is nothing to do right now
                if fetch | may == fetch {
                    continue;
                }
                could = true;
                if fetch | tmp == fetch {
                    continue;
                }

                // if accum state is lost, then new state is won
                new_val.fetch_or(tmp, Ordering::Relaxed);
                progress = true;
            }
        }

        SpreadStatus { progress, could }
    }
}
