use std::{
    iter::zip,
    sync::atomic::{AtomicU32, Ordering},
};

use bit_iter::BitIter;

use crate::card::offset_mask_fixed as offset_mask;

use super::{
    accum_spread::accumulate, locate, prefetch, Block, Chunk, Expanded, ImmutableUpdate, LocalMem, PreparedMove, Spread,
    SpreadStatus, TeamLayout, Update, BLOCK_MASK, PAD,
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
        let inv = self.prepare_forward(self.layout);
        self.finish_unresolved::<COUNT>(inv)
    }

    /// Locate and prefetch the mirrored chunk and every forward move of
    /// `layout`, which need not be this update's own: see
    /// [`Self::update_layout`].
    fn prepare_forward(&mut self, layout: TeamLayout) -> &'t [AtomicU32] {
        let ImmutableUpdate {
            inv_current,
            current,
            take_one,
            mask_lookup,
            directions,
            ..
        } = *self.immutable;
        let pieces0 = layout.pieces0;
        let chunk = Chunk::new(layout);


        // the mirrored chunk is only read at the very end, but its address does
        // not depend on any of the work in between, so start the miss now
        let inv_slice = inv_current.index(layout.invert());
        prefetch(inv_slice.slice);

        // every 0 bit means that it could be anything, win loss or draw
        // every 1 bit means that it must be a draw or win
        // we will gradually flip these to 1s, leaving only losses on 0
        // it is initialized to the wins, because those are not lost even when they don't have moves

        // first locate (and prefetch) the target chunk of every move, then do
        // the actual work; see `PreparedMove`
        let moves = &mut self.moves.forward;
        moves.clear();
        for offset in BitIter::from(directions) {
            let mask = mask_lookup[offset];
            let to_mask = offset_mask(offset, pieces0);
            // can not move onto your own pieces
            let to_mask = to_mask & !pieces0;

            for to in BitIter::from(to_mask) {
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

                let width_new = (new.pieces1 & !(1 << 2)).count_ones();
                // the column a capture takes out of the source, which is where
                // the row breaks; without one it never does
                let split = if width_new == chunk.width as u32 {
                    width_new
                } else {
                    (chunk.cols & (1 << to) - 1).count_ones()
                };
                // the lanes below the break, which is every lane when there
                // is no capture to break the row, in every row of a vector
                let keep = (((1u32 << split) - 1) * 0x1111) as u16;
                moves.push(PreparedMove {
                    bit: 0,
                    step: (from, to),
                    mask,
                    new,
                    rows_new: new.pieces0 & !(1 << 22),
                    width_new: width_new as u8,
                    keep,
                    // `locate` works out which chunk of it
                    table,
                    target: &[],
                });
            }
        }
        locate(moves);
        inv_slice.slice
    }

    /// Build `status` and `wins` from the moves [`Self::prepare_forward`]
    /// located.
    fn finish_unresolved<const COUNT: bool>(&mut self, inv: &[AtomicU32]) -> u64 {
        let chunk = Chunk::new(self.layout);
        let padded = chunk.padded();
        let mem = &mut *self.mem;
        // no clearing first: `accumulate` writes every slot it covers, and it
        // covers the whole of `padded`
        accumulate(chunk, &self.moves.forward, &mut mem.status);

        // we expand, marking all states that are not lost because it has the card
        // then we negate to get only the lost states
        mem.status[..padded]
            .iter_mut()
            .for_each(|x| *x = !Block(*x).invert().expand().invert().0);

        // the slots no king position uses are set to won, which makes them
        // resolved and contributes nothing, so the walks below need no bounds
        // of their own
        self.load_stuff(inv, chunk);

        self.check_unresolved::<COUNT>(padded)
    }

    /// `pre` is what [`Self::prepare_forward`] gave for this layout if the
    /// previous update already ran it, and `next` the layout to run it for in
    /// turn. The return carries that on to the next update.
    pub fn update_layout(
        mut self,
        live: &mut u32,
        pre: Option<&'t [AtomicU32]>,
        next: Option<TeamLayout>,
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
        let inv = pre.unwrap_or_else(|| self.prepare_forward(self.layout));
        self.prepare_spread(go_up, *live);
        let unresolved = self.finish_unresolved::<false>(inv);
        // this layout's forward moves are spent, so their buffer is free for
        // the next one's, and everything from here on is distance for them
        let ahead = next.map(|next| self.prepare_forward(next));

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
        (status, ahead)
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

                moves.push(PreparedMove {
                    bit,
                    step: (from, to),
                    mask,
                    new,
                    rows_new: 0,
                    width_new: 0,
                    keep: 0,
                    // `locate` works out which chunk of it
                    table,
                    target: &[],
                });
            }
        }
        locate(moves);
    }

    /// Spread every backward move [`Self::prepare_spread`] located.
    fn run_spread(&mut self, up: bool, live: &mut u32) -> SpreadStatus {
        let chunk = Chunk::new(self.layout);
        let padded = chunk.padded();
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
        let mut built = None;
        for mv in moves.iter() {
            if built != Some(mv.mask) {
                built = Some(mv.mask);
                let mask = mv.mask;
                let src = zip(&status[..padded], &wins[..padded]);
                for (e, (&s, &w)) in zip(&mut expanded[..padded], src) {
                    // we spread out the loses, these are wins for the previous state
                    let tmp = Block(s & mask).expand().0;
                    *e = Expanded {
                        tmp,
                        // what is going out now is taken off the bound here
                        // rather than in the spread: it does not depend on the
                        // target, so it is the same for every move that reads
                        // this slot and for every row it reads it into
                        may: Block(!w & mask).expand().0 & !tmp,
                    };
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
        let rows = chunk.height;
        let cols = chunk.width;
        use std::simd::{cmp::SimdPartialOrd, Select, Simd, ToBytes};

        // The whole chunk is one vector, which makes the transpose one permute
        // and the write one store. That store is also the only thing the walks
        // after this read `wins` through: a wide load over a run of narrow
        // stores cannot be forwarded from them and waits for every one of them
        // to reach the cache, and those stores wait on the mirrored chunk.
        const WIDE: usize = PAD * PAD;
        debug_assert!(chunk.padded() <= WIDE);
        let lane = Simd::<u32, WIDE>::from_array(std::array::from_fn(|l| l as u32));
        let (r, c) = (lane / Simd::splat(PAD as u32), lane % Simd::splat(PAD as u32));
        let (rows, cols) = (Simd::splat(rows as u32), Simd::splat(cols as u32));
        // the chunk is at most a vector long, and an atomic load a word at
        // a time is what reading it while other threads `fetch_or` into it
        // takes
        let mut words = [0; WIDE];
        for (word, x) in zip(&mut words, inv) {
            *word = x.load(Ordering::Relaxed);
        }
        // As a byte shuffle, which is the one with a run time index that
        // `std::simd` has: four bytes per word. That is one `vpermb` only
        // because `.cargo/config.toml` builds `core` for this machine; with the
        // prebuilt one it is a byte at a time.
        let idx = (cols - Simd::splat(1) - c) * rows + (rows - Simd::splat(1) - r);
        let bytes = (idx * Simd::splat(0x0404_0404) + Simd::splat(0x0302_0100)).to_ne_bytes();
        let x = Simd::<u32, WIDE>::from_ne_bytes(Simd::from_array(words).to_ne_bytes().swizzle_dyn(bytes));
        // `Block::invert`, a lane at a time
        const MASK0: u32 = 0b001001_001001_001001_001001_001001;
        const MASK1: u32 = 0b000111_000111_000111_000111_000111;
        let m0 = Simd::splat(MASK0);
        let m1 = Simd::splat(MASK1);
        let x = (x & m0 << 1) << 1 | (x & m0 << 2) >> 1 | (x & m0);
        let x = (x & m1) << 3 | (x & m1 << 3) >> 3;
        // the slots no king position uses are set to won, which makes them
        // resolved and contributes nothing, so the walks below need no
        // bounds of their own
        let used = r.simd_lt(rows) & c.simd_lt(cols);
        let x = used.select(x, Simd::splat(BLOCK_MASK));
        x.copy_to_slice(&mut self.mem.wins[..WIDE]);
    }
}
