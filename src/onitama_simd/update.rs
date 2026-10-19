use std::{
    iter::zip,
    simd::{Select, Simd},
    sync::atomic::AtomicU32,
};

use bit_iter::BitIter;

use crate::card::offset_mask_fixed as offset_mask;

use super::{
    accum_spread::{accumulate, invert, load, permute, permute16}, locate, prefetch, Block, Chunk, ImmutableUpdate, LocalMem, PreparedMove, Spread,
    SpreadStatus, TeamLayout, Update, BLOCK_MASK, LANES, LANE_COL, LANE_ROW, ROW, SIDE,
};


pub struct UpdateStatus {
    pub(crate) progress: bool,
    pub(crate) unresolved: u64,
    /// Whether the capture predecessors were already written, which means the
    /// chunk does not need the extra pass at the end.
    pub(crate) went_up: bool,
}

impl<'t> Update<'_, 't> {
    pub fn get_unresolved<const COUNT: bool>(&mut self) -> u64 {
        let inv = self.mirror(self.layout);
        self.prepare_forward(self.layout, self.rank, 0);
        self.finish_unresolved::<COUNT>(inv).0
    }

    /// Locate and prefetch the mirrored chunk of `layout`, which need not be
    /// this update's own: see [`Self::update_layout`].
    fn mirror(&self, layout: TeamLayout) -> &'t [AtomicU32] {
        let slice = self.immutable.inv_current.index(layout.invert()).slice;
        prefetch(slice);
        slice
    }

    /// Locate and prefetch every forward move of `layout`, which need not be
    /// this update's own: see [`Self::update_layout`]. `spent` is the moves
    /// [`accumulate`] found can no longer change anything, which are left out.
    fn prepare_forward(&mut self, layout: TeamLayout, rank: usize, spent: u32) {
        let ImmutableUpdate {
            current,
            take_one,
            mask_lookup,
            directions,
            ..
        } = *self.immutable;
        let pieces0 = layout.pieces0;
        let chunk = Chunk::new(layout);

        // every 0 bit means that it could be anything, win loss or draw
        // every 1 bit means that it must be a draw or win
        // we will gradually flip these to 1s, leaving only losses on 0
        // it is initialized to the wins, because those are not lost even when they don't have moves

        // first locate (and prefetch) the target chunk of every move, then do
        // the actual work; see `PreparedMove`
        let moves = &mut self.moves.forward;
        moves.clear();
        // a move per direction and piece at most, so the pushes below need no
        // check of their own
        moves.reserve(directions.count_ones() as usize * SIDE);
        let mut seen = 0u32;
        for offset in BitIter::from(directions) {
            let mask = mask_lookup[offset];
            let to_mask = offset_mask(offset, pieces0);
            // can not move onto your own pieces
            let to_mask = to_mask & !pieces0;

            for to in BitIter::from(to_mask) {
                // shared and wrapping like the bits of the backward moves, see
                // `TableJob::live`
                let bit = 1u32 << (seen & 31);
                seen += 1;
                if bit & spent != 0 {
                    continue;
                }
                let from = to + 12 - offset;

                let new = TeamLayout {
                    pieces0: pieces0 ^ (1 << to) ^ (1 << from),
                    pieces1: layout.pieces1 & !(1 << to),
                };

                // check if we are taking a piece
                let table = if layout.pieces1 & 1 << to != 0 {
                    let Some(table) = take_one else {
                        // there is no way to take a piece when there is only a king to take
                        continue
                    };
                    table
                } else {
                    current
                };

                let width_new = new.pieces1.count_ones();
                // the column a capture takes out of the source, which is where
                // the row breaks; without one it never does
                let split = if width_new == chunk.width as u32 {
                    width_new
                } else {
                    (chunk.cols & (1 << to) - 1).count_ones()
                };
                push_unchecked(moves, PreparedMove {
                    bit,
                    step: (from, to),
                    mask,
                    new,
                    split,
                    // `locate` works out which chunk of it
                    table,
                    target: std::ptr::null(),
                });
            }
        }
        locate::<0>(moves, &mut self.moves.reranker, layout, rank);
    }

    /// Build `status` and `wins` from the moves [`Self::prepare_forward`]
    /// located, and say which of those moves are spent.
    fn finish_unresolved<const COUNT: bool>(&mut self, inv: &[AtomicU32]) -> (u64, u32) {
        let chunk = Chunk::new(self.layout);
        let padded = chunk.padded();
        // the wins first, since they decide which reads still matter. The slots
        // no king position uses are set to won, which makes them resolved and
        // contributes nothing, so the walks below need no bounds of their own
        self.load_stuff(inv, chunk);
        let mem = &mut *self.mem;
        // no clearing first: `accumulate` writes every slot it covers, and it
        // covers the whole of `padded`
        let matter = accumulate(chunk, &self.moves.forward, &mut mem.status, &mem.wins);

        // we expand, marking all states that are not lost because it has the card
        // then we negate to get only the lost states
        mem.status[..padded]
            .iter_mut()
            .for_each(|x| *x = !Block(*x).invert().expand().invert().0);

        (self.check_unresolved::<COUNT>(padded), !matter)
    }

    /// `pre` is this layout's mirrored chunk if the previous update already
    /// located it and ran [`Self::prepare_forward`] for it, and `next` the
    /// layout, with its rank and its spent moves, to do that for in turn. The
    /// return carries that on to the next update. `spent` is this layout's,
    /// and comes back updated.
    pub fn update_layout(
        mut self,
        live: &mut u32,
        spent: &mut u32,
        pre: Option<&'t [AtomicU32]>,
        next: Option<(TeamLayout, usize, u32)>,
    ) -> (UpdateStatus, Option<&'t [AtomicU32]>) {
        let go_up = self.immutable.go_up;

        // The backward moves are located first and read last. Locating one is a
        // multiset ranking, so its target address is only known at the end of a
        // long scalar chain and the line it names is a near certain miss;
        // hoisting the lot up here puts the whole of `get_unresolved` between
        // every one of those prefetches and the read that wants it, instead of
        // only the rest of this same walk.
        let mut scratch = u32::MAX;
        // the `go_up` pass prepares a different set of moves, so the bits would
        // not line up. It only runs once, so there is nothing to remember.
        let live = if go_up { &mut scratch } else { live };
        let inv = pre.unwrap_or_else(|| {
            let inv = self.mirror(self.layout);
            self.prepare_forward(self.layout, self.rank, *spent);
            inv
        });
        // The next mirrored chunk is the first thing its update reads, ahead of
        // the forward moves, since it decides which of their reads matter. So
        // its miss starts here, a whole update ahead of the read.
        let inv_next = next.map(|(next, ..)| self.mirror(next));
        self.prepare_spread(go_up, *live);
        let unresolved;
        (unresolved, *spent) = self.finish_unresolved::<false>(inv);
        // this layout's forward moves are done with, so their buffer is free
        // for the next one's, and everything from here on is distance for them
        if let Some((next, rank, spent)) = next {
            self.prepare_forward(next, rank, spent);
        }

        let res = self.run_spread(go_up, live);

        // Either way out of the work list is worth spreading up on the spot,
        // because `status` and `wins` are still built here: doing it in the
        // extra pass at the end means rebuilding them, which costs a full
        // accumulate and a guaranteed miss on the mirrored chunk.
        let done = !go_up && (unresolved == 0 || !res.could);
        let went_up = done && {
            // this runs once for the chunk, so there is no mask worth keeping
            let mut all = u32::MAX;
            self.prepare_spread(true, all);
            let up = self.run_spread(true, &mut all);

            // `unresolved == 0` means every state is settled, so these losses
            // can never grow again and that was the last word.
            //
            // Otherwise they can still grow, since the chunk is retired for
            // what it spreads, not for what gets spread into it. It is still
            // finished upward if the capture predecessors have subsumed
            // everything it could ever send them, which is the same bound that
            // retired it here, read against the other table.
            unresolved == 0 || !up.could
        };

        let status = UpdateStatus {
            progress: res.progress,
            // a chunk that can never spread anything new again is done, even if
            // some of its own states are still unresolved: it is only kept in
            // the work list to spread. The `go_up` pass has different targets
            // and only runs once, so leave its bookkeeping alone.
            unresolved: if go_up || res.could { unresolved } else { 0 },
            went_up,
        };
        (status, inv_next)
    }

    /// Locate and prefetch every backward move.
    ///
    /// `up` picks the capture predecessors, which live in the table with one
    /// more piece, instead of the ones in this table. `live` is the move mask
    /// as it stands, which never changes between here and [`Self::run_spread`]:
    /// only this layout's own update writes it.
    fn prepare_spread(&mut self, up: bool, live: u32) {
        let layout = self.layout;
        let ImmutableUpdate {
            current,
            leave_one,
            mask_lookup,
            directions,
            ..
        } = *self.immutable;
        let TeamLayout { pieces0, pieces1 } = layout;

        // same two phase split as in `get_unresolved`: locate and prefetch every
        // target chunk before touching any of them.
        // Cards are now inverted, but it is also the other team, so not inverted.
        let moves = &mut self.moves.spread;
        moves.clear();
        // a move per direction and piece at most, so the pushes below need no
        // check of their own
        moves.reserve(directions.count_ones() as usize * SIDE);
        let mut seen = 0u32;
        for offset in BitIter::from(directions) {
            let mask = mask_lookup[offset];

            // these are backwards moves, so `to` is the where the piece came from
            let to_mask = offset_mask(offset, pieces1);
            // can not move onto your own pieces or opp pieces
            let to_mask = to_mask & !pieces0 & !pieces1;

            for to in BitIter::from(to_mask) {
                // the bits wrap, so a chunk with more moves than bits has some
                // of them share one. A shared bit only goes out once every move
                // on it is done, which costs skips but is never wrong, and it
                // spreads the sharing instead of piling it on the last bit
                let bit = 1u32 << (seen & 31);
                seen += 1;
                // a finished move is not even worth locating: `Table::index`
                // runs the multiset ranking, which is the whole reason for the
                // prefetch below
                if bit & !live != 0 {
                    continue;
                }

                let from = to + 12 - offset;

                let mut new = TeamLayout {
                    pieces0,
                    pieces1: pieces1 ^ (1 << to) ^ (1 << from),
                };
                let table = if up {
                    new.pieces0 |= 1 << from;
                    let Some(table) = leave_one else {
                        // there is no larger table, so no progress
                        continue
                    };
                    table
                } else {
                    current
                };

                push_unchecked(moves, PreparedMove {
                    bit,
                    step: (from, to),
                    mask,
                    new,
                    split: 0,
                    // `locate` works out which chunk of it
                    table,
                    target: std::ptr::null(),
                });
            }
        }
        locate::<1>(moves, &mut self.moves.reranker, layout, self.rank);
    }

    /// Spread every backward move [`Self::prepare_spread`] located.
    fn run_spread(&mut self, up: bool, live: &mut u32) -> SpreadStatus {
        let chunk = Chunk::new(self.layout);
        let LocalMem {
            status,
            wins,
            expanded,
        } = &mut *self.mem;
        let moves = &self.moves.spread;


        let mut progress = false;
        // a bit can be shared, so it is not cleared the moment one of its moves
        // is finished: collect the ones that can still contribute instead, and
        // let everything else go at the end. Every bit that was still set got a
        // move, so what is not kept is finished, down to the bits past the move
        // count that never had one to begin with.
        let mut keep = 0u32;
        // `moves` is grouped by offset, so this rebuilds the expansions once per
        // card mask. Both come off the same walk over the chunk: `wins` is kept
        // intact by `get_unresolved` precisely so the bound can be read here.
        //
        // `status` and `wins` are read in the vectors they were written in: a
        // load across two of those stores would wait for both to reach the
        // cache. The rows are cut out of the registers instead, a vector each,
        // which is the load the spread does per row and per move.
        // a column per lane, and the second half is the bound
        #[cfg(target_feature = "avx512f")]
        let pair = Simd::<u32, LANES>::from_array(std::array::from_fn(|k| (k % ROW) as u32));
        #[cfg(target_feature = "avx512f")]
        let high = std::simd::Mask::<i32, LANES>::from_bitmask(!((1 << ROW) - 1) & ((1 << LANES) - 1));
        let mut built = None;
        for mv in moves.iter() {
            if built != Some(mv.mask) {
                built = Some(mv.mask);
                // `Block::expand`, a lane at a time: the bits shifted past the
                // word are the ones the block mask drops anyway
                fn expand<const N: usize>(x: Simd<u32, N>) -> Simd<u32, N> {
                    (x << 10 | x << 20 | x >> 10 | x >> 20) & Simd::splat(BLOCK_MASK)
                }
                // we spread out the loses, these are wins for the previous
                // state. What is going out now is taken off the bound here
                // rather than in the spread: it does not depend on the target,
                // so it is the same for every move that reads this slot and for
                // every row it reads it into
                #[cfg(target_feature = "avx512f")]
                {
                    let mask = Simd::splat(mv.mask);
                    let [(tmp_lo, may_lo), (tmp_hi, may_hi)] = [0, LANES].map(|start| {
                        let s = Simd::from_slice(&status[start..start + LANES]);
                        let w = Simd::from_slice(&wins[start..start + LANES]);
                        let tmp = expand(s & mask);
                        (tmp, expand(!w & mask) & !tmp)
                    });
                    for (row, e) in expanded[..chunk.height].iter_mut().enumerate() {
                        // the lanes past the row read the next one's, which the
                        // spread masks off
                        let idx = Simd::splat((row * chunk.width) as u32) + pair;
                        *e = high.select(permute(may_lo, may_hi, idx), permute(tmp_lo, tmp_hi, idx));
                    }
                }
                // Without AVX-512 the row is built from its own slots, which are
                // contiguous and one register for each half. Cutting it out of
                // the whole chunk instead is a permute of four registers into
                // each, for every row.
                #[cfg(not(target_feature = "avx512f"))]
                {
                    use std::simd::simd_swizzle;
                    let mask = Simd::splat(mv.mask);
                    for (row, e) in expanded[..chunk.height].iter_mut().enumerate() {
                        // the lanes past the row read the next one's, which the
                        // spread masks off
                        let at = row * chunk.width;
                        // a chunk and a row past it fit the scratch
                        debug_assert!(at + ROW <= status.len().min(wins.len()));
                        let s = Simd::<u32, ROW>::from_slice(unsafe { status.get_unchecked(at..at + ROW) });
                        let w = Simd::<u32, ROW>::from_slice(unsafe { wins.get_unchecked(at..at + ROW) });
                        let tmp = expand(s & mask);
                        let may = expand(!w & mask) & !tmp;
                        let row = simd_swizzle!(tmp, may, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
                        // the spare lane of each half zero, for the spread's
                        // dead columns to read
                        let used = Simd::<u32, LANES>::from_array(std::array::from_fn(|l| if l % ROW == super::accum_spread::SPARE { 0 } else { u32::MAX }));
                        *e = row & used;
                    }
                }
            }

            let spread = Spread {
                old: chunk,
                new: mv.new,
                step: mv.step,
                target: mv.target(),
                expanded,
            };
            let res = if up {
                spread.spreadout::<true>()
            } else {
                spread.spreadout::<false>()
            };
            progress |= res.progress;
            if res.could {
                keep |= mv.bit;
            }
        }
        *live = keep;

        SpreadStatus {
            progress,
            could: keep != 0,
        }
    }

    pub fn check_unresolved<const COUNT: bool>(&mut self, padded: usize) -> u64 {
        let mem = &mut *self.mem;

        let mut all_done = BLOCK_MASK;
        let mut total_unresolved = 0;
        // let mut num_done = 0;
        // let mut num_not_done = 0;
        for (s, &w) in zip(&mut mem.status[..padded], &mem.wins[..padded]) {
            let l = *s;
            *s &= !w;
            if COUNT {
                total_unresolved += 30 - ((w | l) & BLOCK_MASK).count_ones() as u64
            } else {
                all_done &= w | l;
            }
            // if (w | l) & BLOCK_MASK == BLOCK_MASK {
            //     inv_slice[kpos.invert()].fetch_or(RESOLVED_BIT, Ordering::Relaxed);
            //     num_done += 1;
            // } else {
            //     num_not_done += 1;
            //     // for mask in mask_iter().take(5) {
            //     //     let mask = Block(mask).invert().expand().invert().0;
            //     //     if (w | l) & mask == mask {
            //     //         self.card_done.fetch_add(1, Ordering::Relaxed);
            //     //     } else {
            //     //         self.card_not_done.fetch_add(1, Ordering::Relaxed);
            //     //     }
            //     // }
            // }
        }
        // if all_done != BLOCK_MASK {
        //     self.block_done.fetch_add(num_done, Ordering::Relaxed);
        //     self.block_not_done
        //         .fetch_add(num_not_done, Ordering::Relaxed);
        // }
        if COUNT {
            total_unresolved
        } else {
            (all_done != BLOCK_MASK) as u64
        }
    }

    /// Read this chunk's wins out of the mirrored chunk.
    ///
    /// The mirror's piece masks are this one's reversed, and inverting a king
    /// position swaps the two kings, so its rows are our columns read backwards
    /// and the other way about. That makes the position it wants a transpose
    /// and two reversals, which is arithmetic on the two counters rather than a
    /// lookup per position.
    fn load_stuff(&mut self, inv: &[AtomicU32], chunk: Chunk) {
        use std::simd::cmp::SimdPartialOrd;
        let (rows, cols) = (chunk.height, chunk.width);

        // The chunk is at most two vectors, which makes the transpose a
        // permute per vector and the write one store each. That store is also
        // the only thing the walks after this read `wins` through: a wide load
        // over a run of narrow stores cannot be forwarded from them and waits
        // for every one of them to reach the cache.
        //
        // So the chunk is read straight into the vectors, although other
        // threads `fetch_or` into it, which is sound for the reason
        // `PreparedMove::src` gives. Going through words on the stack would
        // wait for every store older than these, which this early in the
        // update is all of the backward moves' ranking.
        let [low, high] = load(unsafe { std::slice::from_raw_parts(inv.as_ptr().cast::<u32>(), inv.len()) });
        let (rows_v, cols_v) = (Simd::splat(rows as u32), Simd::splat(cols as u32));
        let lane = Simd::<u32, LANES>::from_array(std::array::from_fn(|l| l as u32));
        for start in (0..chunk.padded()).step_by(LANES) {
            let r = Simd::<u32, LANES>::from_slice(&LANE_ROW[cols][start..start + LANES]);
            let c = Simd::<u32, LANES>::from_slice(&LANE_COL[cols][start..start + LANES]);
            // the slots past the chunk read anywhere: they are overwritten
            let idx = (cols_v - Simd::splat(1) - c) * rows_v + (rows_v - Simd::splat(1) - r);
            // up to eight men the chunk is one vector, and the permute of one
            // is half the work of two without AVX-512
            let x = invert(if inv.len() <= LANES { permute16(low, idx) } else { permute(low, high, idx) });
            // the slots no king position uses are set to won, which makes them
            // resolved and contributes nothing, so the walks below need no
            // bounds of their own
            let used = (lane + Simd::splat(start as u32)).simd_lt(Simd::splat((rows * cols) as u32));
            let x = used.select(x, Simd::splat(BLOCK_MASK));
            x.copy_to_slice(&mut self.mem.wins[start..start + LANES]);
        }
    }
}

/// `vec.push(x)` for a `vec` with room for `x` already, without checking.
#[inline(always)]
fn push_unchecked<T>(vec: &mut Vec<T>, x: T) {
    let len = vec.len();
    debug_assert!(len < vec.capacity());
    unsafe {
        vec.as_mut_ptr().add(len).write(x);
        vec.set_len(len + 1);
    }
}
