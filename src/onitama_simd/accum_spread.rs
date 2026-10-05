use std::{
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        num::SimdUint,
        simd_swizzle, Mask, Select, Simd,
    },
    slice::from_raw_parts,
    sync::atomic::Ordering,
};

use bit_iter::BitIter;

use super::{Chunk, PreparedMove, Spread, SpreadStatus, MAX_ROWS, PAD};

/// Add what every move says about the source chunk.
///
/// The moves come off the card offsets, so every move of one offset carries the
/// same card mask, and what each contributes to a slot is `!target & mask`.
/// Those or together, and `(!a & m) | (!b & m)` is `!(a & b) & m`, so a whole
/// offset is one and per move and a single not and mask at the end of it. And
/// is associative, so that accumulator stays in a register instead of being an
/// or back into `status` per move, and the ors across offsets stay in one too,
/// which leaves one store per vector of rows for the whole chunk.
///
/// Both dimensions keep their order apart from the squares the move touches.
/// The row the king moved off is the only one that does not map to itself, and
/// the columns are the target's with the captured one taken out, which is a
/// rotate and a select rather than a second walk. The slot that select leaves
/// holding its neighbour is the one with the king on the captured square: this
/// move takes that king for every card it has, so the slot is won and anything
/// claiming it is not lost is true there.
pub fn accumulate(chunk: Chunk, moves: &[PreparedMove], status: &mut [u32]) {
    /// Rows taken at a time, so one vector holds this many of them side by
    /// side. Every row of a chunk fits in one at the sizes this is built for.
    const ROWS: usize = 4;
    /// Slots in that vector.
    const WIDE: usize = PAD * ROWS;

    // the square of each row, so the walks below index it instead of stepping a
    // bit iterator they cannot start in the middle of
    let mut kings = [0usize; MAX_ROWS];
    for (slot, king0) in kings.iter_mut().zip(BitIter::from(chunk.rows)) {
        *slot = king0;
    }
    let height = chunk.height;

    // `moves` is built offset by offset, so an offset is a run of it. Taking
    // those on the outside keeps the handful of chunks one offset reads hot
    // across the rows, instead of walking every chunk again for each row
    for group in moves.chunk_by(|a, b| a.mask == b.mask) {
        let card = Simd::<u32, WIDE>::splat(group[0].mask);

        for gs in (0..height).step_by(ROWS) {
            let mut all = Simd::<u32, WIDE>::splat(u32::MAX);

            for mv in group {
                let (from, to) = mv.step;
                let width_new = mv.width_new as usize;

                // Only the rows differ between these, and only in where they
                // read: the columns permute the same way for all of them, so
                // the rotate and the select below are one go for the lot.
                //
                // A row that is not there, and one whose king moved onto its
                // own temple, read nothing and stay all ones. That is the
                // identity of the and, which is what dropping them means.
                let mut part = [Simd::<u32, PAD>::splat(u32::MAX); ROWS];
                for (i, slot) in part.iter_mut().enumerate() {
                    let r = gs + i;
                    if r >= height {
                        break;
                    }
                    let king0 = kings[r];
                    // the king moved exactly when it was the piece on `from`
                    let new0 = if king0 == from { to } else { king0 };
                    if new0 == 22 {
                        continue;
                    }
                    let tbase = (mv.rows_new & (1 << new0) - 1).count_ones() as usize * width_new;
                    // a row is read whole however short it is, which the `PAD`
                    // of slack on the table is there to let the last one do
                    *slot = Simd::from_slice(unsafe { from_raw_parts(mv.src().add(tbase), PAD) });
                }

                let low = simd_swizzle!(part[0], part[1], [0, 1, 2, 3, 4, 5, 6, 7]);
                let high = simd_swizzle!(part[2], part[3], [0, 1, 2, 3, 4, 5, 6, 7]);
                let src: Simd<u32, WIDE> = simd_swizzle!(
                    low,
                    high,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                );
                // the rotate is per row, so it stays inside each row's slots
                let shifted = simd_swizzle!(
                    src,
                    [3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14]
                );
                // the same columns in every row, which is the move's mask once
                // per row. It is `PAD` wide and never carries, so spreading it
                // over the rows is a multiply
                let keep = Mask::<i32, WIDE>::from_bitmask(mv.keep as u64 * 0x1111);
                all &= keep.select(src, shifted);
            }

            // if no move of this offset is won, then the state is not lost
            let row = unsafe { status.get_unchecked_mut(gs * PAD..gs * PAD + WIDE) };
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
            ..
        } = self.old;
        let rows_new = self.new.pieces0 & !(1 << 22);
        let cols_new = self.new.pieces1 & !(1 << 2);
        let width_new = cols_new.count_ones() as usize;

        // Which slot each target column reads is a rotate of a run of them. The
        // move takes the piece out of one column and puts it back in another,
        // so the columns between the two shift along by one and every column
        // outside stays where it was. `p` is the column it arrives in, which is
        // the one that reads the slot `f` it left.
        //
        // A piece on the temple has no column, so `from` there means `p` reads
        // nothing and `to` there means the piece leaves the chunk and there is
        // no arrival column at all. Both only ever drop lanes, so they are a
        // bound on the run and a bit out of `live` rather than a case.
        let gone = from == 2;
        let f = if gone {
            width_new.saturating_sub(1)
        } else {
            (cols_old & (1 << from) - 1).count_ones() as usize
        };
        let p = if to == 2 {
            width_new
        } else {
            (cols_new & (1 << to) - 1).count_ones() as usize
        };

        let lane = Simd::<u64, PAD>::from_array(std::array::from_fn(|k| k as u64));
        let at_p = lane.simd_eq(Simd::splat(p as u64));
        // the run is the columns strictly between the two, which is the half
        // open interval on whichever side the piece moved
        let run = if p > f {
            lane.simd_ge(Simd::splat(f as u64)) & lane.simd_lt(Simd::splat(p as u64))
        } else {
            lane.simd_gt(Simd::splat(p as u64)) & lane.simd_le(Simd::splat(f as u64))
        };
        let live = lane.simd_lt(Simd::splat(width_new as u64)) & !(at_p & Mask::splat(gone));

        let target = self.target.as_ptr();
        // the pair is one word, so a row of them is one vector
        let row_at = |i: usize| unsafe { self.expanded.as_ptr().add(i).cast::<u64>() };
        // the target word in both halves, so one and against the pair asks of
        // the bound and of the losses at the same time
        const LOW: u64 = u32::MAX as u64;
        let zero = Simd::<u64, PAD>::splat(0);

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

            let src = Simd::<u64, PAD>::from_slice(unsafe { from_raw_parts(row_at(base), PAD) });
            let shifted = if p > f {
                src.rotate_elements_left::<1>()
            } else {
                src.rotate_elements_right::<1>()
            };
            let moved = Simd::splat(unsafe { *row_at(base + f) });
            let pair = at_p.select(moved, run.select(shifted, src));

            // the row of the target is contiguous, and the table carries `PAD`
            // of slack precisely so the last one can be read whole
            let fetch: Simd<u32, PAD> =
                Simd::from_slice(unsafe { from_raw_parts(target.cast::<u32>().add(new_i), PAD) });
            let fetch = fetch.cast::<u64>();
            let both = fetch | fetch << Simd::splat(32);
            new_i += width_new;

            // what the pair says that the target does not have yet, in both
            // halves at once. The losses are a subset of the bound, so the high
            // half empty means the low one is too and the row is finished
            let news = pair & !both;
            let can = (news & Simd::splat(LOW << 32)).simd_ne(zero) & live;
            if !can.any() {
                continue;
            }
            could = true;

            let todo = (news & Simd::splat(LOW)).simd_ne(zero) & live;
            if !todo.any() {
                continue;
            }
            progress = true;
            // the write is an atomic read modify write per column, which no
            // vector does. By here it is the rare case though: a row with
            // nothing to say leaves above, all of it at once
            for col in 0..PAD {
                if todo.test(col) {
                    let slot = unsafe { &*target.add(new_i - width_new + col) };
                    slot.fetch_or(pair[col] as u32, Ordering::Relaxed);
                }
            }
        }

        SpreadStatus { progress, could }
    }
}
