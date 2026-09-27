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

use super::{AllTables, Block, ImmutableUpdate, PawnCount, TableJob, Update, Work};

impl<'a> TableJob<'a> {
    pub fn new(tb: &'a AllTables, counts: PawnCount) -> Self {
        let PawnCount { count0, count1 } = counts;

        let update = ImmutableUpdate {
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

        let mut layouts = Vec::with_capacity(counts.total());
        layouts.extend(counts.into_iter().map(|layout| Work { layout, dead: 0 }));
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

        // reduce per thread instead of hammering one shared cache line per layout
        let total: u64 = iter
            .map(|work| {
                UPDATE.with(|vals| {
                    let mem = &mut *vals.borrow_mut();
                    let mut update = Update {
                        layout: work.layout,
                        immutable: &self.update,
                        mem,
                    };
                    update.get_unresolved::<true>()
                })
            })
            .sum();
        self.total_unresolved.fetch_add(total, Ordering::Relaxed);
    }

    pub fn mark_ez_win(&self) {
        let counts = self.update.current.counts;

        #[cfg(feature = "parallell")]
        let iter = self.layouts.par_iter();
        #[cfg(not(feature = "parallell"))]
        let iter = self.layouts.iter();

        let table = self.tb.index_count(counts);
        iter.for_each(|work| {
            let layout = work.layout;
            // the slice only depends on the layout, so look it up once
            let slice = table.index(layout).slice;
            self.tb.ez_win_for_each(counts, layout, &mut |i, mask| {
                // mask is the future cards
                let val = unsafe { slice.get(i).unwrap_unchecked() };
                val.fetch_or(Block(mask).invert().expand().0, Ordering::Relaxed);
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
        let Self {
            layouts,
            is_resolved,
            update: immutable,
            ..
        } = self;

        #[cfg(feature = "parallell")]
        let iter = layouts.par_iter_mut();
        #[cfg(not(feature = "parallell"))]
        let iter = layouts.iter_mut();

        let progress = AtomicBool::new(false);
        let iter = iter.map(|work| {
            UPDATE.with(|vals| {
                let mem = &mut *vals.borrow_mut();
                let update = Update {
                    layout: work.layout,
                    immutable,
                    mem,
                };
                let tmp = update.update_layout(&mut work.dead);
                // a `fetch_or` per layout ping-pongs this cache line between all
                // threads; a relaxed load keeps the line shared once it is set
                if tmp.progress && !progress.load(Ordering::Relaxed) {
                    progress.store(true, Ordering::Relaxed);
                }
                if tmp.unresolved != 0 {
                    0
                } else if tmp.went_up {
                    2
                } else {
                    1
                }
            })
        });

        #[cfg(feature = "parallell")]
        is_resolved.par_extend(iter);
        #[cfg(not(feature = "parallell"))]
        is_resolved.extend(iter);

        let mut i = 0;
        let resolved = &mut self.resolved;
        self.layouts.retain(|work| {
            let res = self.is_resolved[i];
            // only the ones that have not spread upward yet are worth keeping
            // around for the `go_up` pass
            if res == 1 {
                resolved.push(*work)
            }
            i += 1;
            res == 0
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
