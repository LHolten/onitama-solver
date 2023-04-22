#[cfg(feature = "parallell")]
use rayon::prelude::ParallelExtend;
#[cfg(feature = "parallell")]
use rayon::prelude::*;

use std::{
    cell::RefCell,
    mem::take,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

use crate::{index::Indexer, onitama_simd::LocalMem};

use super::{AllTables, Block, ImmutableUpdate, PawnCount, TableJob, Update};

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
            total_unresolved: AtomicU64::new(0),
            tb,
        }
    }

    pub fn count_unresolved(&self) {
        #[cfg(feature = "parallell")]
        let iter = self.layouts.par_iter();
        #[cfg(not(feature = "parallell"))]
        let iter = self.layouts.iter();

        iter.for_each(|layout| {
            UPDATE.with(|vals| {
                let mem = &mut *vals.borrow_mut();
                let mut update = Update {
                    layout: *layout,
                    immutable: &self.update,
                    mem,
                };
                let unresolved = update.get_unresolved::<true>();
                self.total_unresolved
                    .fetch_add(unresolved, Ordering::Relaxed);
            })
        });
    }

    pub fn mark_ez_win(&self) {
        let counts = self.update.current.counts;

        #[cfg(feature = "parallell")]
        let iter = self.layouts.par_iter();
        #[cfg(not(feature = "parallell"))]
        let iter = self.layouts.iter();

        iter.for_each(|layout| {
            self.tb.ez_win_for_each(counts, *layout, &mut |i, mask| {
                // mask is the future cards
                self.tb.index_count(counts).index(*layout)[i]
                    .fetch_or(Block(mask).invert().expand().0, Ordering::Relaxed);
            })
        });
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
                tmp.unresolved == 0
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
            self.layouts.extend(take(&mut self.resolved));
            self.update.go_up = true;
        }
        Some(())
    }
}
