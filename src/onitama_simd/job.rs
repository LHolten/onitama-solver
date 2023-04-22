#[cfg(feature = "parallell")]
use rayon::prelude::ParallelExtend;
#[cfg(feature = "parallell")]
use rayon::prelude::*;

use std::{
    cell::RefCell,
    mem::take,
    sync::atomic::{AtomicBool, Ordering},
};

use crate::{index::Indexer, onitama_simd::LocalMem};

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

        let layouts = counts.into_iter().collect();
        Self {
            layouts,
            is_resolved: Vec::with_capacity(counts.total()),
            resolved: Vec::with_capacity(counts.total()),
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
        if self.update.go_up {
            self.layouts.extend(take(&mut self.resolved));
        }
        #[cfg(feature = "parallell")]
        let iter = self.layouts.par_iter();
        #[cfg(not(feature = "parallell"))]
        let iter = self.layouts.iter();

        let mut progress = AtomicBool::new(false);
        let iter = iter.map(|layout| {
            UPDATE.with(|vals| {
                let mem = &mut *vals.borrow_mut();
                let update = Update {
                    layout: *layout,
                    immutable: &self.update,
                    mem,
                };
                let tmp = update.update_layout();
                progress.fetch_or(tmp.progress, Ordering::Relaxed);
                tmp.resolved
            })
        });

        #[cfg(feature = "parallell")]
        self.is_resolved.par_extend(iter);
        #[cfg(not(feature = "parallell"))]
        self.is_resolved.extend(iter);

        let mut i = 0;
        self.layouts.retain(|layout| {
            let res = self.is_resolved[i];
            if res {
                self.resolved.push(*layout)
            }
            i += 1;
            !res
        });
        self.is_resolved.clear();

        if self.update.go_up {
            self.done = true;
        } else if !progress.load(Ordering::Relaxed) {
            self.update.go_up = true;
        }
        Some(())
    }
}
