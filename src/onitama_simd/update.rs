use std::{iter::repeat, sync::atomic::Ordering};

use bit_iter::BitIter;

use crate::{
    card::offset_mask_fixed as offset_mask,
    index::{Indexer, InternalIter},
};

use super::{
    Accum, Block, ImmutableUpdate, PawnCount, Spread, TeamLayout, Update, BLOCK_MASK, RESOLVED_BIT,
};

impl Update<'_> {
    // returns whether there was any progress
    pub fn update_layout(self) -> bool {
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
        let TeamLayout {
            pieces0, pieces1, ..
        } = layout;
        let PawnCount { count0, count1 } = current.counts;

        let inv_slice = inv_current.index(layout.invert());
        // wins are still inverted here :/
        mem.wins.resize(layout.indexer(current.counts).total(), 0);
        layout.indexer(current.counts).for_enumerate(|i, kpos| {
            let s = unsafe { mem.wins.get_mut(i).unwrap_unchecked() };
            *s = inv_slice[kpos.invert()].load(Ordering::Relaxed);
        });
        let mut resolved: u32 = 0;
        for (i, win) in mem.wins.iter().enumerate() {
            if *win & RESOLVED_BIT != 0 {
                resolved |= 1 << i;
            }
        }
        if go_up {
            resolved = 0;
        }
        if resolved.count_ones() as usize == mem.wins.len() {
            return false;
        }

        // every 0 bit means that it could be anything, win loss or draw
        // every 1 bit means that it must be a draw or win
        // we will gradually flip these to 1s, leaving only losses on 0
        // it is initialized to the wins, because those are not lost even when they don't have moves
        mem.status.clear();
        mem.status
            .extend(repeat(0).take(layout.indexer(current.counts).total()));

        layout
            .indexer(current.counts)
            .for_enumerate(|i, oldk| mem.king_lookup[*oldk] = i as u8);

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

        {
            let mut all_done = BLOCK_MASK;
            let mut num_done = 0;
            let mut num_not_done = 0;
            layout.indexer(current.counts).for_enumerate(|i, kpos| {
                let w = Block(mem.wins[i]).invert().0;
                mem.status[i] &= !w;
                let l = mem.status[i];
                // debug_assert_eq!(w & l, 0);
                all_done &= (w | l);
                if (w | l) & BLOCK_MASK == BLOCK_MASK {
                    inv_slice[kpos.invert()].fetch_or(RESOLVED_BIT, Ordering::Relaxed);
                    num_done += 1;
                } else {
                    num_not_done += 1;
                    // for mask in mask_iter().take(5) {
                    //     let mask = Block(mask).invert().expand().invert().0;
                    //     if (w | l) & mask == mask {
                    //         self.card_done.fetch_add(1, Ordering::Relaxed);
                    //     } else {
                    //         self.card_not_done.fetch_add(1, Ordering::Relaxed);
                    //     }
                    // }
                }
            });
            // if all_done != BLOCK_MASK {
            //     self.block_done.fetch_add(num_done, Ordering::Relaxed);
            //     self.block_not_done
            //         .fetch_add(num_not_done, Ordering::Relaxed);
            // }
        }

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

        progress
    }
}
