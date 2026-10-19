use std::{
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        num::{SimdInt, SimdUint},
        simd_swizzle, Mask, Select, Simd,
    },
    slice::from_raw_parts,
    sync::atomic::Ordering,
};

use bit_iter::BitIter;

use super::{BLOCK_MASK, Chunk, PreparedMove, Spread, SpreadStatus, TeamLayout, LANES, LANE_COL, LANE_ROW, ROW, SIDE};

/// The words of `low` and then `high` that `idx` names, a lane each, the
/// index taken modulo the two vectors' length.
#[inline(always)]
pub fn permute(low: Simd<u32, LANES>, high: Simd<u32, LANES>, idx: Simd<u32, LANES>) -> Simd<u32, LANES> {
    #[cfg(target_feature = "avx512f")]
    unsafe {
        use std::arch::x86_64::_mm512_permutex2var_epi32;
        _mm512_permutex2var_epi32(low.into(), idx.into(), high.into()).into()
    }
    // which of the two the word is in is bit 4 of the index
    #[cfg(not(target_feature = "avx512f"))]
    {
        let upper = (idx & Simd::splat(LANES as u32)).simd_ne(Simd::splat(0));
        upper.select(permute16(high, idx), permute16(low, idx))
    }
}

/// The words of `table` that `idx` names, a lane each, the index taken modulo
/// sixteen.
///
/// This and [`permute`] are the only places that name a target, because a
/// dynamic word permute is the one thing the compiler does not find on its
/// own: a lane by lane lookup becomes a gather out of a copy on the stack. The
/// portable form is the byte swizzle, which costs a multiply, an add, a compare
/// and a masked `vpermb` where AVX-512 has one `vpermd`. AVX2 has the permute
/// of a half, so there it is one per half and a select on bit 3.
#[inline(always)]
pub fn permute16(table: Simd<u32, LANES>, idx: Simd<u32, LANES>) -> Simd<u32, LANES> {
    #[cfg(target_feature = "avx512f")]
    unsafe {
        use std::arch::x86_64::_mm512_permutexvar_epi32;
        _mm512_permutexvar_epi32(idx.into(), table.into()).into()
    }
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    unsafe {
        use std::{arch::x86_64::_mm256_permutevar8x32_epi32, mem::transmute};
        let [t0, t1]: [Simd<u32, 8>; 2] = transmute(table);
        let idx: [Simd<u32, 8>; 2] = transmute(idx);
        transmute(idx.map(|i| {
            use std::arch::x86_64::{_mm256_blendv_ps, _mm256_castps_si256, _mm256_castsi256_ps, _mm256_slli_epi32};
            let lo = _mm256_permutevar8x32_epi32(t0.into(), i.into());
            let hi = _mm256_permutevar8x32_epi32(t1.into(), i.into());
            // the select reads the sign, so bit 3 is shifted up to it
            let upper = _mm256_castsi256_ps(_mm256_slli_epi32::<28>(i.into()));
            Simd::<u32, 8>::from(_mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(lo), _mm256_castsi256_ps(hi), upper)))
        }))
    }
    // the four byte indices of the word, as no byte of it carries
    #[cfg(not(target_feature = "avx2"))]
    {
        use std::simd::ToBytes;
        let bytes = (idx & Simd::splat(LANES as u32 - 1)) * Simd::splat(0x0404_0404) + Simd::splat(0x0302_0100);
        Simd::from_ne_bytes(table.to_ne_bytes().swizzle_dyn(bytes.to_ne_bytes()))
    }
}

/// The words of `table` that `idx` names, a lane each, where every index is
/// below [`ROW`].
///
/// [`permute16`] with the table in its first half, but knowing the second is
/// never read: with AVX2 that is one `vpermd` per half of the index, against
/// two and a select.
#[inline(always)]
fn lookup8(table: Simd<u32, ROW>, idx: Simd<u32, LANES>) -> Simd<u32, LANES> {
    #[cfg(target_feature = "avx512f")]
    unsafe {
        use std::arch::x86_64::_mm512_permutexvar_epi32;
        _mm512_permutexvar_epi32(idx.into(), table.resize::<LANES>(0).into()).into()
    }
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    unsafe {
        use std::{arch::x86_64::_mm256_permutevar8x32_epi32, mem::transmute};
        let idx: [Simd<u32, ROW>; 2] = transmute(idx);
        transmute(idx.map(|i| Simd::<u32, ROW>::from(_mm256_permutevar8x32_epi32(table.into(), i.into()))))
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        permute16(table.resize::<LANES>(0), idx)
    }
}

/// Each half of `src` permuted within itself, by the same half of `idx`, with
/// every lane `keep` leaves out zeroed.
///
/// One instruction with AVX-512, whose index then carries the half in its
/// fourth bit, and whose mask does the zeroing rather than an and per test
/// that reads the result. With AVX2 a half is a register, so it is a `vpermd`
/// each, which takes the index modulo the half anyway, and there is no mask:
/// the lanes to zero are pointed at the last of their half instead, which the
/// rows are built with zero in. See [`SPARE`].
/// The lane of each half of a spread row that no chunk is wide enough to
/// use, and that is kept zero for [`pick`] to point the dead columns at.
pub const SPARE: usize = ROW - 1;
const _: () = assert!(SIDE <= SPARE);

#[inline(always)]
fn pick(src: Simd<u32, LANES>, idx: Simd<u32, LANES>, keep: Mask<i32, LANES>) -> Simd<u32, LANES> {
    #[cfg(target_feature = "avx512f")]
    unsafe {
        use std::arch::x86_64::_mm512_maskz_permutexvar_epi32;
        _mm512_maskz_permutexvar_epi32(keep.to_bitmask() as u16, idx.into(), src.into()).into()
    }
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    unsafe {
        use std::{arch::x86_64::_mm256_permutevar8x32_epi32, mem::transmute};
        let src: [Simd<u32, ROW>; 2] = transmute(src);
        let idx: [Simd<u32, ROW>; 2] = transmute(idx);
        let words: Simd<u32, LANES> =
            transmute([0, 1].map(|h| Simd::<u32, ROW>::from(_mm256_permutevar8x32_epi32(src[h].into(), idx[h].into()))));
        let _ = keep;
        words
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        keep.select(permute16(src, idx), Simd::splat(0))
    }
}

/// `Block::invert`, a lane at a time.
#[inline(always)]
pub fn invert(x: Simd<u32, LANES>) -> Simd<u32, LANES> {
    const MASK0: u32 = 0b001001_001001_001001_001001_001001;
    const MASK1: u32 = 0b000111_000111_000111_000111_000111;
    let (m0, m1) = (Simd::splat(MASK0), Simd::splat(MASK1));
    let x = (x & m0 << 1) << 1 | (x & m0 << 2) >> 1 | (x & m0);
    (x & m1) << 3 | (x & m1 << 3) >> 3
}

/// The bits of the accumulator that cannot change a state `wins` leaves open.
///
/// A slot's not lost states are `invert(expand(invert(acc)))`, which is an or
/// of two rotations of every bit, so a bit only matters if one of the two
/// states it lands on is not won yet. That is the same rotations with an and.
#[inline(always)]
pub fn settled(wins: Simd<u32, LANES>) -> Simd<u32, LANES> {
    let w = invert(wins);
    let rot = |x: Simd<u32, LANES>, by: u32| (x << by | x >> (30 - by)) & Simd::splat(BLOCK_MASK);
    invert(rot(w, 10) & rot(w, 20))
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
///
/// Returns the bits of the moves that can still change something, the same way
/// the spread does. A move whose every read is all ones where it matters adds
/// nothing, and as the targets and `wins` only ever gain bits, it never will:
/// the caller stops reading it.
pub fn accumulate(chunk: Chunk, moves: &[PreparedMove], status: &mut [u32], wins: &[u32]) -> u32 {
    let Chunk {
        rows, height, width, ..
    } = chunk;
    let slots = height * width;

    // the square of each row's king, five bits each, for the slots to shift
    // theirs out of: built in a register, since a vector read of words just
    // stored one by one waits for every one of them to reach the cache
    let mut kings = [0; ROW];
    for (i, king0) in BitIter::from(rows).enumerate() {
        kings[i] = king0 as i32;
    }
    let kings = Simd::from_array(kings);
    #[cfg(not(target_feature = "avx512f"))]
    let row_number = Simd::<i32, ROW>::from_array(std::array::from_fn(|r| r as i32));

    let mut matter = Simd::<u32, LANES>::splat(0);
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
        #[cfg(not(target_feature = "avx512f"))]
        let col_i = col.cast::<i32>();
        // the square of every slot's king, which AVX-512 counts the rows below
        // of directly
        #[cfg(target_feature = "avx512f")]
        let king = lookup8(kings.cast::<u32>(), row);
        // the slots past the chunk are all won, so nothing there matters
        let settled = settled(Simd::from_slice(&wins[start..start + LANES]));

        // `moves` is built offset by offset, so an offset is a run of it
        for group in moves.chunk_by(|a, b| a.mask == b.mask) {
            let card = Simd::<u32, LANES>::splat(group[0].mask);
            let open = card & !settled;
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
                // Those are this chunk's with the one piece moved, so any
                // other king's rank is its row here, one up if the piece
                // landed below it, and one down if it left from below it: two
                // compares rather than a count of the bits below, which AVX2
                // has no instruction for. That is a property of the row, so it
                // is worked out a row a lane and only then spread over the
                // slots, which is one lookup.
                //
                // Signed, as squares are small, because AVX2 has no unsigned
                // compare.
                let TeamLayout { pieces0: rows_new, pieces1: cols_new } = mv.new;
                let width_new = cols_new.count_ones();
                #[cfg(target_feature = "avx512f")]
                let idx = {
                    let moved = king.simd_eq(Simd::splat(from as u32));
                    let new0 = moved.select(Simd::splat(to as u32), king);
                    let below = Simd::splat(u32::MAX) << new0;
                    let rank = (!below & Simd::splat(rows_new)).count_ones();
                    let idx = rank * Simd::splat(width_new) + col;
                    col.simd_lt(Simd::splat(mv.split)).select(idx, idx - Simd::splat(1))
                };
                #[cfg(not(target_feature = "avx512f"))]
                let (from_v, to_v) = (Simd::splat(from as i32), Simd::splat(to as i32));
                #[cfg(not(target_feature = "avx512f"))]
                let idx = {
                    let moved = kings.simd_eq(from_v);
                    let rank_to = (rows_new & (1 << to) - 1).count_ones() as i32;
                    // the masks are all ones where they hold, which is minus one
                    let up = kings.simd_gt(to_v).to_simd();
                    let down = kings.simd_gt(from_v).to_simd();
                    let rank = moved.select(Simd::splat(rank_to), row_number - up + down);
                    let start_of_row = rank.cast::<u32>() * Simd::splat(width_new);
                    // the columns past the break read the one before
                    let past = col_i.simd_gt(Simd::splat(mv.split as i32 - 1)).to_simd().cast::<u32>();
                    lookup8(start_of_row, row) + col + past
                };

                // the target's own length, which is all of it there is
                let size = (rows_new.count_ones() * width_new) as usize;
                let target = unsafe { from_raw_parts(mv.src(), size) };
                // A slot that is not there reads nothing and stays all ones.
                // That is the identity of the and, which is what dropping it
                // means.
                // The slots past the chunk read whatever the index lands on.
                // Nothing they come out with is kept: they are won in `wins`,
                // which `check_unresolved` takes off `status`, and `open` is
                // empty there, so they do not count as reads that matter.
                let word = read(target, idx);
                all &= word;
                // A coin flip in the first pass, so not a branch, which is
                // what a scalar test turns into whatever it is written as: an
                // or into the lanes that still need it, folded at the end.
                let open_read = (open & !word).simd_ne(Simd::splat(0));
                matter = open_read.select(matter | Simd::splat(mv.bit), matter);
            }

            // if no move of this offset is won, then the state is not lost
            acc |= !all & card;
        }

        let out = unsafe { status.get_unchecked_mut(start..start + LANES) };
        acc.copy_to_slice(out);
    }
    matter.reduce_or()
}

/// A whole chunk, as the two vectors it fits in, with the lanes past it zero.
///
/// Masked loads like [`read`]'s, so the lanes past the chunk neither read nor
/// fault.
#[inline(always)]
pub fn load(chunk: &[u32]) -> [Simd<u32, LANES>; 2] {
    let low = unsafe { load_masked::<0>(chunk.as_ptr(), chunk.len()) };
    // the second load only when there is something for it: with every lane
    // masked off it still costs a microcode assist when the address it would
    // read is not mapped, and an empty slice's is not
    let high = if chunk.len() > LANES {
        unsafe { load_masked::<LANES>(chunk.as_ptr(), chunk.len()) }
    } else {
        Simd::splat(0)
    };
    [low, high]
}

/// The words of `target` that `idx` names, a lane each.
///
/// The target is at most two vectors long, so this is a permute out of them,
/// and only reads the second one when the target reaches into it. The lanes
/// past the target are masked off, so they neither read nor fault even at the
/// end of the table.
#[inline(always)]
fn read(target: &[u32], idx: Simd<u32, LANES>) -> Simd<u32, LANES> {
    let low = unsafe { load_masked::<0>(target.as_ptr(), target.len()) };
    if target.len() <= LANES {
        permute16(low, idx)
    } else {
        permute(low, unsafe { load_masked::<LANES>(target.as_ptr(), target.len()) }, idx)
    }
}

/// The vector of words `START` on from `ptr`, with the lanes at or past `len`
/// zero, which neither read nor fault.
///
/// The mask is a signed compare of the lane numbers against the length, one
/// per vector whatever the target. `load_or_default` compares `usize` lanes,
/// which is four vectors of them for sixteen words and a pack to bring them
/// back down, and a mask out of a bitmask is two more steps without AVX-512.
///
/// # Safety
/// The `len` words from `ptr` on have to be readable.
#[inline(always)]
unsafe fn load_masked<const START: usize>(ptr: *const u32, len: usize) -> Simd<u32, LANES> {
    let lane = Simd::<i32, LANES>::from_array(std::array::from_fn(|l| (START + l) as i32));
    let enable = lane.simd_lt(Simd::splat(len as i32));
    unsafe { Simd::load_select_ptr(ptr.add(START), enable, Simd::splat(0)) }
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

        // a column per lane, the same in both halves. Signed, as columns are
        // small, because AVX2 has no unsigned compare.
        let lane = Simd::<i32, LANES>::from_array(std::array::from_fn(|k| (k % ROW) as i32));
        let (f_v, p_v) = (Simd::splat(f as i32), Simd::splat(p as i32));
        let at_p = lane.simd_eq(p_v);
        // The run is the columns strictly between the two, which is the half
        // open interval on whichever side the piece moved, and it steps towards
        // the slot the piece left. As integers rather than masks: a lane is in
        // `lo..hi` when it is below `hi` and not below `lo`, which is the sign
        // of the differences, and shifted across the word that is minus one,
        // the step. An and of two compares instead goes through packing the
        // masks down to bytes and back without AVX-512.
        let within = |lo: Simd<i32, LANES>, hi: Simd<i32, LANES>| ((lane - hi) & !(lane - lo)) >> 31;
        let idx = if p > f {
            lane - within(f_v, p_v)
        } else {
            lane + within(p_v + Simd::splat(1), f_v + Simd::splat(1))
        };
        let live = Simd::splat(width_new as i32).simd_gt(lane);
        // which slot every column reads, which is the whole mapping and the
        // same for every row, so it is settled once for the move; the second
        // half reads its own half
        let half = Simd::<i32, LANES>::from_array(std::array::from_fn(|k| (k / ROW * ROW) as i32));
        let idx = at_p.select(f_v, idx);
        // without the mask of AVX-512 the dead columns read the spare lane
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        let idx = live.select(idx, Simd::splat(SPARE as i32));
        let idx = (idx + half).cast::<u32>();

        let target = self.target.as_ptr();
        let zero = Simd::<u32, LANES>::splat(0);
        // the lanes of the losses, which are the first half
        const LOW: u64 = (1 << ROW) - 1;

        // every row's news folded together, so what the move still owes and
        // whether anything went out at all are two tests after the walk rather
        // than a compare, a mask test and a fold per row
        let mut seen = Simd::<u32, LANES>::splat(0);
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
            // the mask a signed compare of the lane numbers, where
            // `load_or_default` compares `usize` lanes
            let enable = Simd::<i32, 4>::from_array([0, 1, 2, 3]).simd_lt(Simd::splat(width_new.min(4) as i32));
            let mut fetch = unsafe { Simd::<u32, 4>::load_select_ptr(row_new.as_ptr(), enable, Simd::splat(0)) }.resize::<ROW>(0);
            if width_new > 4 {
                fetch[4] = row_new[4];
            }
            // the target row in both halves, so one and against the pair asks
            // of the bound and of the losses at the same time
            let both = simd_swizzle!(fetch, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]);
            new_i += width_new;

            // what the pair says that the target does not have yet, in both
            // halves at once. The two are disjoint, since the bound has what
            // is going out taken off it already, so nothing in either means
            // the row has nothing to say and each half then wants its own test
            let news = pair & !both;
            if news == zero {
                continue;
            }
            seen |= news;

            let bits = news.resize::<ROW>(0).simd_ne(Simd::splat(0)).to_bitmask();
            if bits == 0 {
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
            for col in 0..SIDE {
                if bits >> col & 1 != 0 {
                    let slot = unsafe { &*target.add(new_i - width_new + col) };
                    slot.fetch_or(pair[col], Ordering::Relaxed);
                }
            }
        }

        let seen = seen.simd_ne(zero).to_bitmask();
        SpreadStatus {
            // the losses that went out, and the bound still owed beyond them
            progress: seen & LOW != 0,
            could: seen & !LOW != 0,
        }
    }
}
