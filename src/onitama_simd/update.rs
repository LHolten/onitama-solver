use std::{
    iter::zip,
    sync::atomic::Ordering,
};

use bit_iter::BitIter;

use crate::card::offset_mask_fixed as offset_mask;

use super::{
    accum_spread::accumulate, prefetch, Block, Chunk, Expanded, ImmutableUpdate, LocalMem, PreparedMove, Spread, SubTable,
    SpreadStatus, TeamLayout, Update, BLOCK_MASK, PAD,
};

pub struct UpdateStatus {
    pub(crate) progress: bool,
    pub(crate) unresolved: u64,
    /// Whether the capture predecessors were already written, which means the
    /// chunk does not need the extra pass at the end.
    pub(crate) went_up: bool,
}

impl Update<'_> {
    pub fn get_unresolved<const COUNT: bool>(&mut self) -> u64 {
        let layout = self.layout;
        let mem = &mut *self.mem;
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
        let padded = chunk.padded();

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
        mem.moves.clear();
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

                let target = table.index(new).slice;
                prefetch(target);
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
                mem.moves.push(PreparedMove {
                    bit: 0,
                    step: (from, to),
                    mask,
                    new,
                    rows_new: new.pieces0 & !(1 << 22),
                    width_new: width_new as u8,
                    keep,
                    target,
                });
            }
        }

        let LocalMem { status, moves, .. } = &mut *mem;
        // no clearing first: `accumulate` writes every slot it covers, and it
        // covers the whole of `padded`
        accumulate(chunk, moves, status);

        // we expand, marking all states that are not lost because it has the card
        // then we negate to get only the lost states
        mem.status[..padded]
            .iter_mut()
            .for_each(|x| *x = !Block(*x).invert().expand().invert().0);

        // the slots no king position uses are set to won, which makes them
        // resolved and contributes nothing, so the walks below need no bounds
        // of their own
        mem.wins[..padded].fill(BLOCK_MASK);
        self.load_stuff(&inv_slice, chunk);

        self.check_unresolved::<COUNT>(padded)
    }

    pub fn update_layout(mut self, live: &mut u32) -> UpdateStatus {
        let go_up = self.immutable.go_up;

        let unresolved = self.get_unresolved::<false>();

        // the `go_up` pass prepares a different set of moves, so the bits would
        // not line up. It only runs once, so there is nothing to remember.
        let mut scratch = u32::MAX;
        let res = self.spread_phase(go_up, if go_up { &mut scratch } else { live });

        // Either way out of the work list is worth spreading up on the spot,
        // because `status` and `wins` are still built here: doing it in the
        // extra pass at the end means rebuilding them, which costs a full
        // accumulate and a guaranteed miss on the mirrored chunk.
        let done = !go_up && (unresolved == 0 || !res.could);
        let went_up = done && {
            // this runs once for the chunk, so there is no mask worth keeping
            let mut all = u32::MAX;
            let up = self.spread_phase(true, &mut all);

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

        UpdateStatus {
            progress: res.progress,
            // a chunk that can never spread anything new again is done, even if
            // some of its own states are still unresolved: it is only kept in
            // the work list to spread. The `go_up` pass has different targets
            // and only runs once, so leave its bookkeeping alone.
            unresolved: if go_up || res.could { unresolved } else { 0 },
            went_up,
        }
    }

    /// Locate, prefetch and then spread every backward move.
    ///
    /// `up` picks the capture predecessors, which live in the table with one
    /// more piece, instead of the ones in this table.
    fn spread_phase(&mut self, up: bool, live: &mut u32) -> SpreadStatus {
        let layout = self.layout;
        let ImmutableUpdate {
            current,
            leave_one,
            mask_lookup,
            directions,
            ..
        } = *self.immutable;
        let TeamLayout { pieces0, pieces1 } = layout;
        let chunk = Chunk::new(layout);

        // same two phase split as in `get_unresolved`: locate and prefetch every
        // target chunk before touching any of them.
        // Cards are now inverted, but it is also the other team, so not inverted.
        let moves = &mut self.mem.moves;
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
                if bit & !*live != 0 {
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

                let target = table.index(new).slice;
                prefetch(target);
                moves.push(PreparedMove {
                    bit,
                    step: (from, to),
                    mask,
                    new,
                    rows_new: 0,
                    width_new: 0,
                    keep: 0,
                    target,
                });
            }
        }

        let padded = chunk.padded();
        let LocalMem {
            status,
            wins,
            expanded,
            moves,
            ..
        } = &mut *self.mem;


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
    fn load_stuff(&mut self, inv_slice: &SubTable, chunk: Chunk) {
        let rows = chunk.height;
        let cols = chunk.width;

        let wins = &mut self.mem.wins[..];
        let inv = inv_slice.slice;

        let mut inv_i = 0;
        for r in 0..cols {
            let ours = cols - 1 - r;
            for c in 0..rows {
                let i = (rows - 1 - c) * PAD + ours;
                let s = unsafe { wins.get_unchecked_mut(i) };
                let x = unsafe { inv.get_unchecked(inv_i) };
                let tmp = x.load(Ordering::Relaxed);
                *s = Block(tmp).invert().0;
                inv_i += 1;
            }
        }
    }
}
