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
    TeamLayout, Update, BLOCK_MASK,
};

pub struct UpdateStatus {
    pub(crate) progress: bool,
    pub(crate) unresolved: u64,
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

        let inv_slice = inv_current.index(layout.invert());
        mem.wins.resize(layout.indexer(current.counts).total(), 0);
        self.load_stuff(&inv_slice);

        self.check_unresolved::<COUNT>()
    }

    // returns whether there was any progress
    pub fn update_layout(mut self) -> UpdateStatus {
        let layout = self.layout;
        let ImmutableUpdate {
            current,
            leave_one,
            go_up,
            mask_lookup,
            directions,
            ..
        } = *self.immutable;
        let TeamLayout { pieces0, pieces1 } = layout;

        let unresolved = self.get_unresolved::<false>();

        // same two phase split as in `get_unresolved`: locate and prefetch every
        // target chunk before touching any of them.
        // Cards are now inverted, but it is also the other team, so not inverted.
        let moves = &mut self.mem.moves;
        moves.clear();
        for offset in BitIter::from(directions) {
            let mask = mask_lookup[offset];

            // these are backwards moves, so `to` is the where the piece came from
            let to_mask = offset_mask(offset, pieces1);
            // can not move onto your own pieces or opp pieces
            let to_mask = to_mask & !pieces0 & !pieces1;

            for to in BitIter::from(to_mask) {
                let from = to + 12 - offset;

                let mut new = TeamLayout {
                    pieces0,
                    pieces1: pieces1 ^ (1 << to) ^ (1 << from),
                };
                let table = if go_up {
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
            wins,
            king_lookup,
            moves,
            ..
        } = &mut *self.mem;

        let mut progress = false;
        // `moves` is grouped by offset, so this rebuilds `wins` once per card mask
        let mut built = None;
        for mv in moves.iter() {
            if built != Some(mv.mask) {
                built = Some(mv.mask);
                // we spread out the loses, these are wins for the previous state
                wins.clear();
                wins.extend(status.iter().map(|x| Block(x & mv.mask).expand().0));
            }

            let spread = Spread {
                new: mv.new,
                counts: mv.counts,
                step: mv.step,
                target: mv.target(),
                slice: wins,
                king_lookup,
            };
            progress |= spread.spreadout();
        }

        UpdateStatus {
            progress,
            unresolved,
        }
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
