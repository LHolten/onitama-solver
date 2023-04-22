use std::{iter::repeat, sync::atomic::Ordering};

use bit_iter::BitIter;

use crate::{
    card::offset_mask_fixed as offset_mask,
    index::{Indexer, InternalIter},
};

use super::{
    Accum, Block, ImmutableUpdate, PawnCount, Spread, SubTable, TeamLayout, Update, BLOCK_MASK,
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
            leave_one,
            go_up,
            mask_lookup,
            directions,
        } = *self.immutable;
        let TeamLayout { pieces0, pieces1 } = layout;
        let PawnCount { count0, count1 } = current.counts;

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

        let inv_slice = inv_current.index(layout.invert());
        mem.wins.resize(layout.indexer(current.counts).total(), 0);
        self.load_stuff(&inv_slice);
        let mem = &mut *self.mem;

        for offset in BitIter::from(directions) {
            let mask = mask_lookup[offset];
            let to_mask = offset_mask(offset, pieces0);
            // can not move onto your own pieces
            let to_mask = to_mask & !pieces0;

            for to in BitIter::from(to_mask) {
                let from = to + 12 - offset;
                let accum = Accum {
                    layout,
                    current,
                    take_one,
                    step: (from, to),
                    slice: &mut mem.status,
                    mask,
                    king_lookup: &mem.king_lookup,
                };
                accum.accumulate();
            }
        }

        // we expand, marking all states that are not lost because it has the card
        // then we negate to get only the lost states
        mem.status
            .iter_mut()
            .for_each(|x| *x = !Block(*x).invert().expand().invert().0);

        self.check_unresolved::<COUNT>(&inv_slice)
    }

    // returns whether there was any progress
    pub fn update_layout(mut self) -> UpdateStatus {
        let layout = self.layout;
        let ImmutableUpdate {
            inv_current,
            current,
            take_one,
            leave_one,
            go_up,
            mask_lookup,
            directions,
        } = *self.immutable;
        let TeamLayout { pieces0, pieces1 } = layout;
        let PawnCount { count0, count1 } = current.counts;

        let unresolved = self.get_unresolved::<false>();
        let mem = &mut *self.mem;

        let mut progress = false;
        // same thing, but cards are now inverted
        // but it is also the other team, so not inverted
        for offset in BitIter::from(directions) {
            let mask = mask_lookup[offset];

            // we spread out the loses, these are wins for the previous state
            mem.wins.clear();
            mem.wins
                .extend(mem.status.iter().map(|x| Block(x & mask).expand().0));

            // these are backwards moves, so `to` is the where the piece came from
            let to_mask = offset_mask(offset, pieces1);
            // can not move onto your own pieces or opp pieces
            let to_mask = to_mask & !pieces0 & !pieces1;

            for to in BitIter::from(to_mask) {
                let from = to + 12 - offset;
                let spread = Spread {
                    layout,
                    current,
                    leave_one,
                    go_up,
                    step: (from, to),
                    slice: &mem.wins,
                    king_lookup: &mem.king_lookup,
                };
                progress |= spread.spreadout();
            }
        }

        UpdateStatus {
            progress,
            unresolved,
        }
    }

    pub fn check_unresolved<const COUNT: bool>(&mut self, inv_slice: &SubTable) -> u64 {
        let layout = self.layout;
        let mem = &mut *self.mem;
        let current = self.immutable.current;

        let mut all_done = BLOCK_MASK;
        let mut total_unresolved = 0;
        // let mut num_done = 0;
        // let mut num_not_done = 0;
        layout.indexer(current.counts).for_enumerate(|i, kpos| {
            let w = Block(mem.wins[i]).invert().0;
            let l = mem.status[i];
            mem.status[i] &= !w;
            if COUNT {
                total_unresolved += 30 - ((w | l) & BLOCK_MASK).count_ones() as u64
            } else {
                all_done &= (w | l);
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
        });
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
        let layout = self.layout;
        let mem = &mut self.mem;
        let current = self.immutable.current;

        let inv_indexer = inv_slice.layout.indexer(inv_slice.counts);
        inv_indexer.for_enumerate(|inv_i, kpos| {
            let i = mem.king_lookup[kpos.invert()] as usize;
            let s = unsafe { mem.wins.get_mut(i).unwrap_unchecked() };
            let x = unsafe { inv_slice.slice.get(inv_i).unwrap_unchecked() };
            let tmp = x.load(Ordering::Relaxed);
            *s = tmp;
        });
    }
}
