use rayon::prelude::*;

use std::{
    cell::RefCell,
    sync::atomic::{AtomicBool, Ordering},
};

use crate::onitama_simd::LocalMem;

use super::{AllTables, ImmutableUpdate, PawnCount, TableJob, Update};

impl<'a> TableJob<'a> {
    pub fn new(tb: &'a AllTables, counts: PawnCount) -> Self {
        let PawnCount { count0, count1 } = counts;

        let mut update = ImmutableUpdate {
            current: tb.index_count(counts),
            inv_current: tb.index_count(counts.invert()),
            take_one: (count1 != 0).then(|| {
                tb.index_count(PawnCount {
                    count0,
                    count1: count1 - 1,
                })
            }),
            leave_one: (count0 + 1 != tb.size).then(|| {
                tb.index_count(PawnCount {
                    count0: count0 + 1,
                    count1,
                })
            }),
            go_up: false,
            mask_lookup: &tb.mask_lookup,
            directions: tb.directions,
        };

        let layouts = counts.indexer().into_iter().collect();
        Self {
            layouts,
            update,
            done: false,
            tb,
        }
    }
}

thread_local! {
    static UPDATE: RefCell<LocalMem> = RefCell::new(LocalMem::new());
}

impl Iterator for TableJob<'_> {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        };
        let mut progress = AtomicBool::new(false);
        #[cfg(feature = "parallell")]
        let iter = self.layouts.par_iter();
        #[cfg(not(feature = "parallell"))]
        let iter = self.layouts.iter();

        iter.for_each(|layout| {
            UPDATE.with(|vals| {
                let mem = &mut *vals.borrow_mut();
                let update = Update {
                    layout: *layout,
                    immutable: &self.update,
                    mem,
                };
                let tmp = update.update_layout();
                progress.fetch_or(tmp, Ordering::Relaxed);
            });
        });
        if self.update.go_up {
            self.done = true;
        } else if !progress.load(Ordering::Relaxed) {
            self.update.go_up = true;
        }
        Some(())
    }
}
