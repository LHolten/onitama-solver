use std::{
    iter::zip,
    simd::{simd_swizzle, Simd},
    sync::atomic::{AtomicU32, Ordering},
};

use bit_iter::BitIter;

use crate::card::offset_mask;

use super::{
    accum_spread::{accumulate, SPARE}, locate, prefetch, Block, Chunk, ImmutableUpdate, LocalMem, PreparedMove, Spread,
    SpreadStatus, TeamLayout, Update, BLOCK_MASK, LANES, ROW, SIDE,
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
        let slice = self.immutable.inv_current.index(layout.invert());
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
        // the spare lane of each half zero, for the spread's dead columns to
        // read
        let used = Simd::<u32, LANES>::from_array(std::array::from_fn(|l| if l % ROW == SPARE { 0 } else { u32::MAX }));
        let mut built = None;
        for mv in moves.iter() {
            if built != Some(mv.mask) {
                built = Some(mv.mask);
                // `Block::expand`, a lane at a time: the bits shifted past the
                // word are the ones the block mask drops anyway
                let expand = |x: Simd<u32, LANES>| (x << 10 | x << 20 | x >> 10 | x >> 20) & Simd::splat(BLOCK_MASK);
                let mask = Simd::splat(mv.mask);
                for (row, e) in expanded[..chunk.height].iter_mut().enumerate() {
                    // A row is built from its own slots, which are contiguous:
                    // the losses to spread in the first half, and the not won
                    // states in the second, expanded together. The lanes past
                    // the row read the next one's, which the spread masks off.
                    let at = row * chunk.width;
                    // a chunk and a row past it fit the scratch
                    debug_assert!(at + ROW <= status.len().min(wins.len()));
                    let s = Simd::<u32, ROW>::from_slice(unsafe { status.get_unchecked(at..at + ROW) });
                    let w = Simd::<u32, ROW>::from_slice(unsafe { wins.get_unchecked(at..at + ROW) });
                    let both = expand(simd_swizzle!(s, !w, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) & mask);
                    // We spread out the losses, these are wins for the previous
                    // state. What is going out now is taken off the bound here
                    // rather than in the spread: it does not depend on the
                    // target, so it is the same for every move that reads this
                    // slot and for every row it reads it into.
                    let going = simd_swizzle!(Simd::splat(0), both, [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]);
                    *e = both & !going & used;
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
        for (s, &w) in zip(&mut mem.status[..padded], &mem.wins[..padded]) {
            let l = *s;
            *s &= !w;
            if COUNT {
                total_unresolved += 30 - ((w | l) & BLOCK_MASK).count_ones() as u64
            } else {
                all_done &= w | l;
            }
        }
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
    /// and the other way about. The slots no king position uses are set to won,
    /// which makes them resolved and contributes nothing, so the walks after
    /// this need no bounds of their own.
    fn load_stuff(&mut self, inv: &[AtomicU32], chunk: Chunk) {
        let (rows, cols) = (chunk.height, chunk.width);
        let wins = &mut self.mem.wins[..chunk.padded()];
        wins.fill(BLOCK_MASK);
        for r in 0..rows {
            for c in 0..cols {
                let x = inv[(cols - 1 - c) * rows + (rows - 1 - r)].load(Ordering::Relaxed);
                wins[r * cols + c] = Block(x).invert().0;
            }
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
