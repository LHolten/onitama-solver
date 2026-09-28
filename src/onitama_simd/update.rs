use std::{
    iter::{repeat, zip},
    sync::atomic::Ordering,
};

use bit_iter::BitIter;

use crate::{
    card::offset_mask_fixed as offset_mask,
    index::{Indexer, InternalIter},
};

use super::{
    prefetch, Accum, Block, ImmutableUpdate, LocalMem, PreparedMove, Spread, SubTable,
    SpreadStatus, TeamLayout, Update, BLOCK_MASK,
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

        // the mirrored chunk is only read at the very end, but its address does
        // not depend on any of the work in between, so start the miss now
        let inv_slice = inv_current.index(layout.invert());
        prefetch(inv_slice.slice);

        layout
            .indexer(current.counts)
            .for_enumerate(|i, oldk| mem.king_lookup[*oldk] = i as u8);

        // every 0 bit means that it could be anything, win loss or draw
        // every 1 bit means that it must be a draw or win
        // we will gradually flip these to 1s, leaving only losses on 0
        // it is initialized to the wins, because those are not lost even when they don't have moves
        mem.status.clear();
        mem.status
            .extend(repeat(0).take(layout.indexer(current.counts).total()));

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
                mem.moves.push(PreparedMove {
                    bit: 0,
                    step: (from, to),
                    mask,
                    new,
                    counts: table.counts,
                    target,
                });
            }
        }

        let LocalMem {
            status,
            king_lookup,
            moves,
            ..
        } = &mut *mem;
        for mv in moves.iter() {
            let accum = Accum {
                new: mv.new,
                counts: mv.counts,
                step: mv.step,
                mask: mv.mask,
                target: mv.target(),
                slice: status,
                king_lookup,
            };
            accum.accumulate();
        }

        // we expand, marking all states that are not lost because it has the card
        // then we negate to get only the lost states
        mem.status
            .iter_mut()
            .for_each(|x| *x = !Block(*x).invert().expand().invert().0);

        mem.wins.resize(layout.indexer(current.counts).total(), 0);
        self.load_stuff(&inv_slice);

        self.check_unresolved::<COUNT>()
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
                // a zero bit is a move past 31, which just never gets skipped
                let bit = 1u32.checked_shl(seen).unwrap_or(0);
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
                    counts: table.counts,
                    target,
                });
            }
        }

        let total = layout.indexer(current.counts).total();
        let LocalMem {
            status,
            wins,
            expanded,
            king_lookup,
            moves,
            ..
        } = &mut *self.mem;
        expanded.resize(total, (0, 0));

        let mut progress = false;
        let mut could = false;
        // `moves` is grouped by offset, so this rebuilds the expansions once per
        // card mask. Both come off the same walk over the chunk: `wins` is kept
        // intact by `get_unresolved` precisely so the bound can be read here.
        let mut built = None;
        for mv in moves.iter() {
            if built != Some(mv.mask) {
                built = Some(mv.mask);
                let mask = mv.mask;
                let src = zip(&status[..total], &wins[..total]);
                for (e, (&s, &w)) in zip(&mut expanded[..total], src) {
                    // we spread out the loses, these are wins for the previous state
                    *e = (Block(s & mask).expand().0, Block(!w & mask).expand().0);
                }
            }

            let spread = Spread {
                new: mv.new,
                counts: mv.counts,
                step: mv.step,
                target: mv.target(),
                expanded,
                king_lookup,
            };
            let res = spread.spreadout();
            progress |= res.progress;
            could |= res.could;
            if !res.could {
                *live &= !mv.bit;
            }
        }

        SpreadStatus { progress, could }
    }

    pub fn check_unresolved<const COUNT: bool>(&mut self) -> u64 {
        let layout = self.layout;
        let mem = &mut *self.mem;
        let current = self.immutable.current;

        let mut all_done = BLOCK_MASK;
        let mut total_unresolved = 0;
        // let mut num_done = 0;
        // let mut num_not_done = 0;
        // the indexer just enumerates 0..total here, so walk the slices directly
        let total = layout.indexer(current.counts).total();
        for (s, &w) in zip(&mut mem.status[..total], &mem.wins[..total]) {
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

    fn load_stuff(&mut self, inv_slice: &SubTable) {
        let mem = &mut *self.mem;

        let inv_indexer = inv_slice.layout.indexer(inv_slice.counts);
        inv_indexer.for_enumerate(|inv_i, kpos| {
            let i = mem.king_lookup[kpos.invert()] as usize;
            let s = unsafe { mem.wins.get_mut(i).unwrap_unchecked() };
            let x = unsafe { inv_slice.slice.get(inv_i).unwrap_unchecked() };
            let tmp = x.load(Ordering::Relaxed);
            *s = Block(tmp).invert().0;
        });
    }
}
