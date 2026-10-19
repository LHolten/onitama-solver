use std::{
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        num::SimdUint,
        Mask, Select, Simd,
    },
    slice::from_raw_parts,
    sync::atomic::Ordering,
};

use bit_iter::BitIter;

use super::{Chunk, PreparedMove, Spread, SpreadStatus, TeamLayout, LANES, LANE_COL, LANE_ROW, ROW, SIDE};

/// The words of `low` and then `high` that `idx` names, a lane each.
///
/// One instruction, which takes the index modulo the two vectors' length.
#[inline(always)]
pub fn permute(low: Simd<u32, LANES>, high: Simd<u32, LANES>, idx: Simd<u32, LANES>) -> Simd<u32, LANES> {
    #[cfg(target_feature = "avx512f")]
    unsafe {
        use std::arch::x86_64::_mm512_permutex2var_epi32;
        _mm512_permutex2var_epi32(low.into(), idx.into(), high.into()).into()
    }
    #[cfg(not(target_feature = "avx512f"))]
    {
        Simd::from_array(std::array::from_fn(|l| {
            let i = idx[l] as usize % (2 * LANES);
            if i < LANES {
                low[i]
            } else {
                high[i - LANES]
            }
        }))
    }
}

/// The words of `src` that `idx` names, one per lane, with every lane `keep`
/// leaves out zeroed.
///
/// The zeroing is the same instruction's mask rather than an and per test that
/// reads the result.
#[inline(always)]
fn pick(src: Simd<u64, ROW>, idx: Simd<u64, ROW>, keep: Mask<i64, ROW>) -> Simd<u64, ROW> {
    #[cfg(target_feature = "avx512f")]
    unsafe {
        use std::arch::x86_64::_mm512_maskz_permutexvar_epi64;
        _mm512_maskz_permutexvar_epi64(keep.to_bitmask() as u8, idx.into(), src.into()).into()
    }
    #[cfg(not(target_feature = "avx512f"))]
    {
        Simd::from_array(std::array::from_fn(|k| if keep.test(k) { src[idx[k] as usize % ROW] } else { 0 }))
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
/// which leaves one store per vector of slots for the whole chunk.
///
/// Both dimensions keep their order apart from the squares the move touches.
/// The row the king moved off is the only one that does not map to itself, and
/// the columns are the target's with the captured one taken out, which is a
/// rotate and a select rather than a second walk. The slot that select leaves
/// holding its neighbour is the one with the king on the captured square: this
/// move takes that king for every card it has, so the slot is won and anything
/// claiming it is not lost is true there.
pub fn accumulate(chunk: Chunk, moves: &[PreparedMove], status: &mut [u32]) {
    let Chunk {
        rows, height, width, ..
    } = chunk;
    let slots = height * width;

    // the square of each row's king, five bits each, for the slots to shift
    // theirs out of: built in a register, since a vector read of words just
    // stored one by one waits for every one of them to reach the cache
    let mut kings = 0;
    for (i, king0) in BitIter::from(rows).enumerate() {
        kings |= (king0 as u32) << 5 * i;
    }

    // The slots on the outside and the offsets within, so what every offset
    // contributes to these slots adds up in a register and `status` is written
    // once. It is also the only write, which is what lets the caller hand this
    // whatever it last held rather than zeroing it first.
    for start in (0..slots).step_by(LANES) {
        let mut acc = Simd::<u32, LANES>::splat(0);
        // the square of the row every slot is in, the column it is, and the
        // slots that are there
        let row = Simd::<u32, LANES>::from_slice(&LANE_ROW[width][start..start + LANES]);
        let col = Simd::<u32, LANES>::from_slice(&LANE_COL[width][start..start + LANES]);
        let king = Simd::splat(kings) >> row * Simd::splat(5) & Simd::splat(31);
        let live = ((1u32 << (slots - start).min(LANES)) - 1) as u16;

        // `moves` is built offset by offset, so an offset is a run of it
        for group in moves.chunk_by(|a, b| a.mask == b.mask) {
            let card = Simd::<u32, LANES>::splat(group[0].mask);
            let mut all = Simd::<u32, LANES>::splat(u32::MAX);

            for mv in group {
                let (from, to) = mv.step;

                // Every slot reads one word of the target, so the whole move
                // is one permute out of a register, or two at ten men, and the
                // row remap, the column rotate and the select all go into the
                // index. That index is worked out for every slot at once.
                //
                // The king moved exactly when it was the piece on `from`, and
                // where its row lands is its rank among the target's rows.
                let moved = king.simd_eq(Simd::splat(from as u32));
                let new0 = moved.select(Simd::splat(to as u32), king);
                // no square is past 24, so the shift never runs out of bits
                let below = Simd::splat(u32::MAX) << new0;
                let TeamLayout { pieces0: rows_new, pieces1: cols_new } = mv.new;
                let width_new = cols_new.count_ones();
                let rank = (!below & Simd::splat(rows_new)).count_ones();
                let idx = rank * Simd::splat(width_new) + col;
                // the columns past the break read the one before
                let keep = Mask::<i32, LANES>::from_bitmask((mv.keep >> start) as u64);
                let idx = keep.select(idx, idx - Simd::splat(1));

                // the target's own length, which is all of it there is
                let size = (rows_new.count_ones() * width_new) as usize;
                let target = unsafe { from_raw_parts(mv.src(), size) };
                // A slot that is not there reads nothing and stays all ones.
                // That is the identity of the and, which is what dropping it
                // means.
                all = Mask::<i32, LANES>::from_bitmask(live as u64).select(all & read(target, idx), all);
            }

            // if no move of this offset is won, then the state is not lost
            acc |= !all & card;
        }

        let out = unsafe { status.get_unchecked_mut(start..start + LANES) };
        acc.copy_to_slice(out);
    }
}

/// The words of `target` that `idx` names, a lane each.
///
/// The target is at most two vectors long, so this is a permute out of them,
/// and only reads the second one when the target reaches into it. The lanes
/// past the target are masked off, so they neither read nor fault even at the
/// end of the table.
#[inline(always)]
fn read(target: &[u32], idx: Simd<u32, LANES>) -> Simd<u32, LANES> {
    #[cfg(target_feature = "avx512f")]
    unsafe {
        use std::arch::x86_64::{_mm512_maskz_loadu_epi32, _mm512_permutexvar_epi32};
        // a target is at most 25 words, so the shift stays in the word and
        // its halves are the two vectors' lanes
        let lanes = (1u32 << target.len()) - 1;
        let low = _mm512_maskz_loadu_epi32(lanes as u16, target.as_ptr().cast());
        if target.len() <= LANES {
            _mm512_permutexvar_epi32(idx.into(), low).into()
        } else {
            let high = _mm512_maskz_loadu_epi32((lanes >> LANES) as u16, target.as_ptr().add(LANES).cast());
            permute(low.into(), high.into(), idx)
        }
    }
    #[cfg(not(target_feature = "avx512f"))]
    {
        Simd::from_array(std::array::from_fn(|l| target.get(idx[l] as usize).copied().unwrap_or(u32::MAX)))
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
        let TeamLayout {
            pieces0: rows_new,
            pieces1: cols_new,
        } = self.new;
        let width_new = cols_new.count_ones() as usize;

        // Which slot each target column reads is a rotate of a run of them. The
        // move takes the piece out of one column and puts it back in another,
        // so the columns between the two shift along by one and every column
        // outside stays where it was. `p` is the column it arrives in, which is
        // the one that reads the slot `f` it left.
        let f = (cols_old & (1 << from) - 1).count_ones() as usize;
        let p = (cols_new & (1 << to) - 1).count_ones() as usize;

        let lane = Simd::<u64, ROW>::from_array(std::array::from_fn(|k| k as u64));
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
        let live = lane.simd_lt(Simd::splat(width_new as u64));
        // which slot every column reads, which is the whole mapping and the
        // same for every row, so it is settled once for the move
        let idx = at_p.select(Simd::splat(f as u64), run.select(step, lane));

        let target = self.target.as_ptr();
        // the target word in both halves, so one and against the pair asks of
        // the bound and of the losses at the same time
        const LOW: u64 = u32::MAX as u64;
        let zero = Simd::<u64, ROW>::splat(0);

        // every row's news folded together, so what the move still owes and
        // whether anything went out at all are two tests after the walk rather
        // than a compare, a mask test and a fold per row
        let mut seen = Simd::<u64, ROW>::splat(0);
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
                (rows_old & (1 << king0) - 1).count_ones() as usize
            } else {
                row
            };
            row += 1;

            // the dead columns come back zeroed, which is what every test below
            // would otherwise have to mask off one by one
            let pair = pick(unsafe { *self.expanded.get_unchecked(base) }, idx, live);

            // The row of the target is contiguous, and read in at most four
            // words. A wider read runs into the line of the next chunk, which
            // nothing prefetched, and waits for it, masked or not: the mask
            // only drops the words, the load still covers all of them. So the
            // fifth word, which only ten men have, is a load of its own.
            let row_new = unsafe { from_raw_parts(target.cast::<u32>().add(new_i), width_new) };
            let mut fetch = Simd::<u32, 4>::load_or_default(&row_new[..width_new.min(4)]).resize::<ROW>(0);
            if width_new > 4 {
                fetch[4] = row_new[4];
            }
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
            for col in 0..SIDE {
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
