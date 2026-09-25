use std::sync::atomic::Ordering;

use crate::index::InternalIter;

use super::{Accum, Spread};

impl Accum<'_> {
    pub fn accumulate(self) {
        let (from, to) = self.step;

        self.new.indexer(self.counts).for_enumerate(|new_i, newk| {
            let mut oldk = *newk;
            if newk.king0 as usize == to {
                oldk.king0 = from as u32;
                if oldk.king0 == 22 {
                    // there is no way we came from the temple
                    return;
                }
            }

            let new_val = unsafe { self.target.get(new_i).unwrap_unchecked() };
            let old_i = self.king_lookup[oldk] as usize;
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

        let mut progress = false;
        self.new.indexer(self.counts).for_enumerate(|new_i, newk| {
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

            let new_val = unsafe { self.target.get(new_i).unwrap_unchecked() };
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
