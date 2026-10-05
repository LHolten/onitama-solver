use std::{
    simd::{cmp::SimdPartialOrd, Select, Simd},
    slice::from_raw_parts,
    sync::atomic::Ordering,
};

use bit_iter::BitIter;

use super::{Chunk, PreparedMove, Spread, SpreadStatus, PAD};

/// Add what every move says about the source chunk.
///
/// The moves come off the card offsets, so every move of one offset carries the
/// same card mask, and what each contributes to a slot is `!target & mask`.
/// Those or together, and `(!a & m) | (!b & m)` is `!(a & b) & m`, so a whole
/// offset is one and per move and a single not and mask at the end of it. And
/// is associative, so that accumulator stays in a register instead of being an
/// or back into `status` per move, and the ors across offsets stay in one too,
/// which leaves one store per row for the whole chunk.
///
/// Both dimensions keep their order apart from the squares the move touches.
/// The row the king moved off is the only one that does not map to itself, and
/// the columns are the target's with the captured one taken out, which is a
/// rotate and a select rather than a second walk. The slot that select leaves
/// holding its neighbour is the one with the king on the captured square: this
/// move takes that king for every card it has, so the slot is won and anything
/// claiming it is not lost is true there.
pub fn accumulate(chunk: Chunk, moves: &[PreparedMove], status: &mut [u32]) {
    let lane = Simd::<u32, PAD>::from_array(std::array::from_fn(|k| k as u32));

    // `moves` is built offset by offset, so an offset is a run of it. Taking
    // those on the outside keeps the handful of chunks one offset reads hot
    // across the rows, instead of walking every chunk again for each row
    for group in moves.chunk_by(|a, b| a.mask == b.mask) {
        let card = Simd::splat(group[0].mask);

        for (r, king0) in BitIter::from(chunk.rows).enumerate() {
            let mut all = Simd::splat(u32::MAX);

            for mv in group {
                let (from, to) = mv.step;
                // the king moved exactly when it was the piece on `from`
                let new0 = if king0 == from { to } else { king0 };
                if new0 == 22 {
                    // onto its own temple, so it has already won and there is
                    // no state over there to read. Leaving `all` alone drops
                    // this move out of the offset, which is what that means
                    continue;
                }

                let width_new = mv.width_new as usize;
                let tbase = (mv.rows_new & (1 << new0) - 1).count_ones() as usize * width_new;
                let src: Simd<u32, PAD> =
                    Simd::from_slice(unsafe { from_raw_parts(mv.src().add(tbase), PAD) });

                let shifted = src.rotate_elements_right::<1>();
                all &= lane.simd_lt(Simd::splat(mv.split as u32)).select(src, shifted);
            }

            // if no move of this offset is won, then the state is not lost
            let row = unsafe { status.get_unchecked_mut(r * PAD..r * PAD + PAD) };
            (Simd::from_slice(row) | (!all & card)).copy_to_slice(row);
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
        let Chunk {
            rows: rows_old,
            cols: cols_old,
            width: width_old,
            ..
        } = self.old;
        let rows_new = self.new.pieces0 & !(1 << 22);
        let cols_new = self.new.pieces1 & !(1 << 2);
        let width_new = cols_new.count_ones() as usize;

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
                (rows_old & (1 << king0) - 1).count_ones() as usize * PAD
            } else {
                row * PAD
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
