#[cfg(feature = "parallell")]
use rayon::prelude::*;

use std::{
    cell::RefCell,
    mem::take,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

use crate::{index::Indexer, onitama_simd::LocalMem};

use super::{
    iter::LayoutCursor, AllTables, Block, ImmutableUpdate, PawnCount, TableJob, TeamLayout, Update,
    BLOCK,
};

/// Layouts covered by one word of [`TableJob::needs_up`], and the granularity
/// at which a retired run is skipped without being looked at layout by layout.
const GROUP: usize = 64;

/// Words of [`TableJob::needs_up`] per block.
const WORDS: usize = BLOCK / GROUP;

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

        let total = counts.total();
        let blocks = total.div_ceil(BLOCK);
        let padded = blocks * BLOCK;

        // every move of every layout starts out worth running, but the
        // padding at the end of the last block is not a layout at all. The
        // zeroed allocation is untouched pages, so the only cost here is the
        // parallel write that faults them in.
        let mut live = vec![0u32; padded];
        #[cfg(feature = "parallell")]
        live[..total].par_iter_mut().for_each(|x| *x = u32::MAX);
        #[cfg(not(feature = "parallell"))]
        live[..total].fill(u32::MAX);

        Self {
            counts,
            live,
            needs_up: vec![0; padded / 64],
            total,
            update,
            done: false,
            total_unresolved: AtomicU64::new(0),
            tb,
        }
    }

    /// Call `f` on every live layout and add up what it returns.
    ///
    /// A retired layout costs one word of [`TableJob::live`] and nothing else:
    /// its position in the enumeration is only turned back into a layout once a
    /// live word is found, so a sparse late pass never pays for the gaps.
    fn map_live<F>(&self, f: F) -> u64
    where
        F: Fn(TeamLayout) -> u64 + Sync + Send,
    {
        let counts = self.counts;

        #[cfg(feature = "parallell")]
        let iter = self.live.par_chunks(BLOCK);
        #[cfg(not(feature = "parallell"))]
        let iter = self.live.chunks(BLOCK);

        iter.enumerate()
            .map(|(b, live)| {
                let start = b * BLOCK;
                let mut cursor = LayoutCursor::new(counts);
                let mut sum = 0;
                for (g, group) in live.chunks(GROUP).enumerate() {
                    // a retired run is rejected by one scan that stops at the
                    // first live layout, instead of a branch per layout
                    if group.iter().all(|&moves| moves == 0) {
                        continue;
                    }
                    let start = start + g * GROUP;
                    for (j, &moves) in group.iter().enumerate() {
                        if moves != 0 {
                            sum += f(cursor.at(start + j));
                        }
                    }
                }
                sum
            })
            .sum()
    }

    pub fn count_unresolved(&self) {
        // reduce per thread instead of hammering one shared cache line per layout
        let total = self.map_live(|layout| {
            UPDATE.with(|vals| {
                let mem = &mut *vals.borrow_mut();
                let mut update = Update {
                    layout,
                    immutable: &self.update,
                    mem,
                };
                update.get_unresolved::<true>()
            })
        });
        self.total_unresolved.fetch_add(total, Ordering::Relaxed);
    }

    pub fn mark_ez_win(&self) {
        let counts = self.update.current.counts;
        let table = self.tb.index_count(counts);

        self.map_live(|layout| {
            // the slice only depends on the layout, so look it up once
            let slice = table.index(layout).slice;
            self.tb.ez_win_for_each(counts, layout, &mut |i, mask| {
                // mask is the future cards
                let val = unsafe { slice.get(i).unwrap_unchecked() };
                val.fetch_or(Block(mask).invert().expand().0, Ordering::Relaxed);
            });
            0
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

        let progress = AtomicBool::new(false);
        let counts = self.counts;
        let immutable = &self.update;

        #[cfg(feature = "parallell")]
        let iter = self
            .live
            .par_chunks_mut(BLOCK)
            .zip(self.needs_up.par_chunks_mut(WORDS));
        #[cfg(not(feature = "parallell"))]
        let iter = self
            .live
            .chunks_mut(BLOCK)
            .zip(self.needs_up.chunks_mut(WORDS));

        iter.enumerate().for_each(|(b, (live, needs_up))| {
            let start = b * BLOCK;
            let mut cursor = LayoutCursor::new(counts);
            for (g, (group, needs_up)) in live.chunks_mut(GROUP).zip(needs_up).enumerate() {
                // a retired run is rejected by one scan that stops at the first
                // live layout, instead of a branch per layout
                if group.iter().all(|&moves| moves == 0) {
                    continue;
                }
                let start = start + g * GROUP;

                for j in 0..group.len() {
                    // no move left worth running means the layout is retired,
                    // and a retired layout is never unranked into a layout
                    if group[j] == 0 {
                        continue;
                    }
                    let layout = cursor.at(start + j);

                    UPDATE.with(|vals| {
                        let mem = &mut *vals.borrow_mut();
                        let update = Update {
                            layout,
                            immutable,
                            mem,
                        };
                        let tmp = update.update_layout(&mut group[j]);
                        // a `fetch_or` per layout ping-pongs this cache line
                        // between all threads; a relaxed load keeps the line
                        // shared once it is set
                        if tmp.progress && !progress.load(Ordering::Relaxed) {
                            progress.store(true, Ordering::Relaxed);
                        }

                        if tmp.unresolved == 0 {
                            // this is the whole compaction
                            group[j] = 0;
                            // only the ones that have not spread upward yet are
                            // worth bringing back for the `go_up` pass
                            if !tmp.went_up {
                                *needs_up |= 1 << j;
                            }
                        }
                    })
                }
            }
        });

        if self.update.go_up {
            self.done = true;
        } else if !progress.load(Ordering::Relaxed) {
            for (w, word) in self.needs_up.iter_mut().enumerate() {
                let mut bits = take(word);
                while bits != 0 {
                    // the `go_up` pass has its own move set, so the bits do not
                    // line up and every one of them starts out set
                    self.live[w * 64 + bits.trailing_zeros() as usize] = u32::MAX;
                    bits &= bits - 1;
                }
            }
            self.update.go_up = true;
        }
        Some(())
    }
}
