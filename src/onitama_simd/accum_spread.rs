use std::sync::atomic::Ordering;

use crate::index::{InternalIter, Indexer};

use super::{Accum, Spread, TeamLayout};

impl Accum<'_> {
    pub fn accumulate(self) {
        let (from, to) = self.step;

        let old = self.layout;

        let new = TeamLayout {
            pieces0: old.pieces0 ^ (1 << to) ^ (1 << from),
            pieces1: old.pieces1 & !(1 << to),
        };

        // check if we are taking a piece
        let table = if old.pieces1 & 1 << to != 0 {
            let Some(table) = self.take_one else {
                // there is no way to take a piece when there is only a king to take
                return
            };
            table
        } else {
            self.current
        };
        let new_slice = table.index(new);

        new.indexer(table.counts).for_enumerate(|new_i, newk| {
            let mut oldk = *newk;
            if newk.king0 as usize == to {
                oldk.king0 = from as u32;
                if oldk.king0 == 22 {
                    // there is no way we came from the temple
                    return;
                }
            }

            let new_val = unsafe { new_slice.slice.get(new_i).unwrap_unchecked() };
            let old_i = self.king_lookup[oldk] as usize;
            debug_assert_eq!(old_i, old.indexer(self.current.counts).index(&oldk));
            let s = unsafe { self.slice.get_mut(old_i).unwrap_unchecked() };
            // if new state is not won, then old state is not lost
            *s |= !new_val.load(Ordering::Relaxed) & self.mask;
        });
    }
}

impl Spread<'_> {
    // returns whether there was any progress
    pub fn spreadout(self) -> bool {
        let (from, to) = self.step;

        let old = self.layout;

        let mut new = TeamLayout {
            pieces0: old.pieces0,
            pieces1: old.pieces1 ^ (1 << to) ^ (1 << from),
        };
        if self.go_up {
            new.pieces0 |= 1 << from;
        }
        let table = if self.go_up {
            let Some(table) = self.leave_one else { 
                // there is no larger table, so no progress
                return false
            };
            table
        } else {
            self.current
        };
        let new_slice = table.index(new);

        let mut progress = false;
        new.indexer(table.counts).for_enumerate(|new_i, newk| {
            let mut oldk = *newk;
            if newk.king1 as usize == to {
                oldk.king1 = from as u32;
                if oldk.king1 == 2 {
                    return;
                }
            }

            if oldk.king0 as usize == from {
                // this king was added, so this oldk did not exist
                return;
            }

            let old_i = self.king_lookup[oldk] as usize;

            let tmp = self.slice[old_i];

            let new_val = unsafe { new_slice.slice.get(new_i).unwrap_unchecked() };
            let fetch = new_val.load(Ordering::Relaxed);
            if fetch | tmp == fetch {
                return;
            }

            // if accum state is lost, then new state is won
            new_val.fetch_or(tmp, Ordering::Relaxed);
            progress = true;
        });

        progress
    }
}