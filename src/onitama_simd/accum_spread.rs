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

/// `src` read through `idx`, with every lane `keep` leaves out zeroed.
///
/// One instruction where the permutation it stands for is a rotate, a broadcast
/// and two selects, and the zeroing is the same instruction's mask rather than
/// an and per test that reads the result.
#[inline(always)]
fn pick(src: Simd<u64, PAD>, idx: Simd<u64, PAD>, keep: Mask<i64, PAD>) -> Simd<u64, PAD> {
    #[cfg(target_feature = "avx512vl")]
    unsafe {
        use std::arch::x86_64::_mm256_maskz_permutexvar_epi64;
        _mm256_maskz_permutexvar_epi64(keep.to_bitmask() as u8, idx.into(), src.into()).into()
    }
    #[cfg(not(target_feature = "avx512vl"))]
    {
        let mut out = [0; PAD];
        for (k, slot) in out.iter_mut().enumerate() {
            if keep.test(k) {
                *slot = src[idx[k] as usize];
            }
        }
        Simd::from_array(out)
    }
}

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

    // The rows on the outside and the offsets within, so what every offset
    // contributes to these rows adds up in a register and `status` is written
    // once. It is also the only write, which is what lets the caller hand this
    // whatever it last held rather than zeroing it first.
    for gs in (0..height).step_by(ROWS) {
        let mut acc = Simd::<u32, WIDE>::splat(0);

        // `moves` is built offset by offset, so an offset is a run of it
        for group in moves.chunk_by(|a, b| a.mask == b.mask) {
            let card = Simd::<u32, WIDE>::splat(group[0].mask);
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
                    // `r` is bounded by `height`, which is `MAX_ROWS` at
                    // most, but that is not something the walk makes obvious
                    let king0 = unsafe { *kings.get_unchecked(r) };
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
                let keep = Mask::<i32, WIDE>::from_bitmask(mv.keep as u64);
                all &= keep.select(src, shifted);
            }

            // if no move of this offset is won, then the state is not lost
            acc |= !all & card;
        }

        let row = unsafe { status.get_unchecked_mut(gs * PAD..gs * PAD + WIDE) };
        acc.copy_to_slice(row);
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
        // open interval on whichever side the piece moved, and it steps towards
        // the slot the piece left
        let (run, step) = if p > f {
            let run = lane.simd_ge(Simd::splat(f as u64)) & lane.simd_lt(Simd::splat(p as u64));
            (run, lane + Simd::splat(1))
        } else {
            let run = lane.simd_gt(Simd::splat(p as u64)) & lane.simd_le(Simd::splat(f as u64));
            (run, lane - Simd::splat(1))
        };
        let live = lane.simd_lt(Simd::splat(width_new as u64)) & !(at_p & Mask::splat(gone));
        // which slot every column reads, which is the whole mapping and the
        // same for every row, so it is settled once for the move
        let idx = at_p.select(Simd::splat(f as u64), run.select(step, lane));

        let target = self.target.as_ptr();
        // the pair is one word, so a row of them is one vector
        let row_at = |i: usize| unsafe { self.expanded.as_ptr().add(i).cast::<u64>() };
        // the target word in both halves, so one and against the pair asks of
        // the bound and of the losses at the same time
        const LOW: u64 = u32::MAX as u64;
        let zero = Simd::<u64, PAD>::splat(0);

        // every row's news folded together, so what the move still owes and
        // whether anything went out at all are two tests after the walk rather
        // than a compare, a mask test and a fold per row
        let mut seen = Simd::<u64, PAD>::splat(0);
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
            // the dead columns come back zeroed, which is what every test below
            // would otherwise have to mask off one by one
            let pair = pick(src, idx, live);

            // the row of the target is contiguous, and the table carries `PAD`
            // of slack precisely so the last one can be read whole
            let fetch: Simd<u32, PAD> =
                Simd::from_slice(unsafe { from_raw_parts(target.cast::<u32>().add(new_i), PAD) });
            let fetch = fetch.cast::<u64>();
            let both = fetch | fetch << Simd::splat(32);
            new_i += width_new;

            // what the pair says that the target does not have yet, in both
            // halves at once. The two are disjoint, since the bound has what
            // is going out taken off it already, so nothing in either means
            // the row has nothing to say and each half then wants its own test
            let news = pair & !both;
            if !news.simd_ne(zero).any() {
                continue;
            }
            seen |= news;

            let todo = (news & Simd::splat(LOW)).simd_ne(zero);
            if !todo.any() {
                continue;
            }
            // The write is an atomic read modify write per column, which no
            // vector does. By here it is the rare case though: a row with
            // nothing to say leaves above, all of it at once.
            //
            // It has to be the exchange and not a store of the read or'd with
            // the losses. Two chunks spread into the same one, and a store
            // built on a stale read drops the other's bits; the move that lost
            // them can have retired on a read taken before the store landed,
            // and then nobody writes them ever again.
            // Off the bitmask rather than off the mask itself, which tests a
            // lane by pulling it into a register and comparing it.
            //
            // The column still has to be the constant the unrolling makes it,
            // and not a bit walked out of the mask: a lane picked by a runtime
            // index is one the whole vector has to be on the stack for, and
            // that spill lands on every row rather than on the few that write
            let bits = todo.to_bitmask();
            for col in 0..PAD {
                if bits >> col & 1 != 0 {
                    let slot = unsafe { &*target.add(new_i - width_new + col) };
                    slot.fetch_or(pair[col] as u32, Ordering::Relaxed);
                }
            }
        }

        SpreadStatus {
            // the losses that went out, and the bound still owed beyond them
            progress: (seen & Simd::splat(LOW)).simd_ne(zero).any(),
            could: (seen & Simd::splat(LOW << 32)).simd_ne(zero).any(),
        }
    }
}
