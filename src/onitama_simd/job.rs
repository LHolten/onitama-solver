#[cfg(feature = "parallell")]
use rayon::prelude::*;

use std::{
    cell::RefCell,
    mem::take,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Mutex,
    },
};

use crate::{index::Indexer, onitama_simd::LocalMem};

use super::{
    count_indexer, iter::LayoutCursor, AllTables, Block, ImmutableUpdate, PawnCount, TableJob,
    TeamLayout, Update, BLOCK, PAD,
};

/// Layouts covered by one word of [`TableJob::needs_up`], and the granularity
/// at which a retired run is skipped without being looked at layout by layout.
const GROUP: usize = 64;

/// Words of [`TableJob::needs_up`] per block.
const WORDS: usize = BLOCK / GROUP;

impl<'a> ImmutableUpdate<'a> {
    /// Everything about one table that does not change while it is solved.
    fn new(tb: &'a AllTables, counts: PawnCount) -> Self {
        let PawnCount { count0, count1 } = counts;
        ImmutableUpdate {
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
        }
    }
}

impl<'a> TableJob<'a> {
    pub fn new(tb: &'a AllTables, counts: PawnCount) -> Self {
        let update = ImmutableUpdate::new(tb, counts);

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

    /// Blocks of layouts this job covers, which is what seeding is split into.
    pub fn blocks(&self) -> usize {
        self.live.len() / BLOCK
    }

    /// Write the wins in one of one block, which is the seed of the solve.
    ///
    /// A block rather than a whole table, so that seeding every table is one
    /// flat walk: the small tables have nowhere near enough layouts to fill the
    /// machine on their own, and a table at a time is a barrier at the end of
    /// each.
    ///
    /// A chunk's cards are gathered before the write, so this is one write per
    /// king position rather than one per card, and a plain store rather than an
    /// or into what is there: this runs before any table is solved, so the
    /// tables still hold the zeros they were allocated with, and a layout is
    /// the only one that writes its own chunk.
    pub fn seed_block(&self, b: usize) {
        let counts = self.update.current.counts;
        let table = self.tb.index_count(counts);

        let start = b * BLOCK;
        let mut cursor = LayoutCursor::new(counts);
        for (g, group) in self.live[start..][..BLOCK].chunks(GROUP).enumerate() {
            // a retired run is rejected by one scan that stops at the first
            // live layout, instead of a branch per layout
            if group.iter().all(|&moves| moves == 0) {
                continue;
            }
            let start = start + g * GROUP;
            for (j, &moves) in group.iter().enumerate() {
                if moves == 0 {
                    continue;
                }
                let layout = cursor.at(start + j);
                // the slice only depends on the layout, so look it up once
                let slice = table.index(layout).slice;

                let mut acc = [0u32; PAD * PAD];
                // mask is the future cards
                self.tb.ez_win_for_each(counts, layout, &mut |i, mask| {
                    acc[i] |= Block(mask).invert().expand().0;
                });

                for (val, &new) in slice.iter().zip(&acc) {
                    if new != 0 {
                        val.store(new, Ordering::Relaxed);
                    }
                }
            }
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

        let progress = AtomicBool::new(false);
        let counts = self.counts;
        let immutable = &self.update;

        let block = |b: usize, live: &mut [u32], needs_up: &mut [u64]| {
            let start = b * BLOCK;
            let mut cursor = LayoutCursor::new(counts);
            // no move left worth running means the layout is retired, and a
            // retired layout is never unranked into a layout
            let find = |live: &[u32], from: usize| {
                live[from..].iter().position(|&moves| moves != 0).map(|k| from + k)
            };

            UPDATE.with(|vals| {
                let mem = &mut *vals.borrow_mut();
                // One layout ahead, so the forward moves of the next one are
                // located while this one is still being worked on. Their lines
                // come from all over the table, and cheap as locating them is,
                // that leaves too little between the prefetch and the read
                // when it is done for the layout's own update alone.
                let mut cur = find(live, 0).map(|i| (i, cursor.at(start + i)));
                let mut pre = None;
                while let Some((i, layout)) = cur {
                    let next = find(live, i + 1).map(|k| (k, cursor.at(start + k)));
                    let update = Update {
                        layout,
                        immutable,
                        mem: &mut *mem,
                    };
                    let ahead = next.map(|(_, layout)| layout);
                    let (tmp, inv) = update.update_layout(&mut live[i], pre, ahead);
                    pre = inv;
                    // a `fetch_or` per layout ping-pongs this cache line
                    // between all threads; a relaxed load keeps the line
                    // shared once it is set
                    if tmp.progress && !progress.load(Ordering::Relaxed) {
                        progress.store(true, Ordering::Relaxed);
                    }

                    if tmp.unresolved == 0 {
                        // this is the whole compaction
                        live[i] = 0;
                        // only the ones that have not spread upward yet are
                        // worth bringing back for the `go_up` pass
                        if !tmp.went_up {
                            needs_up[i / GROUP] |= 1 << (i % GROUP);
                        }
                    }
                    cur = next;
                }
            })
        };

        // The blocks are handed out in order, so every thread works in the same
        // narrow stretch of the table at once. The solve is bound by how many
        // lines it pulls out of memory, and a chunk is read by the moves of many
        // layouts: in step, the ones of neighbouring layouts land close enough
        // together in time to find the line still in the shared cache, where a
        // stretch of the table per thread gives each of them a sliver of it.
        // That is a sixth fewer lines from memory, and a tenth off the solve.
        let blocks: Vec<_> = self
            .live
            .chunks_mut(BLOCK)
            .zip(self.needs_up.chunks_mut(WORDS))
            .map(Mutex::new)
            .collect();
        let next = AtomicUsize::new(0);
        let work = || loop {
            let b = next.fetch_add(1, Ordering::Relaxed);
            let Some(slot) = blocks.get(b) else { break };
            // only ever this thread's, so never contended
            let mut guard = slot.lock().unwrap();
            let (live, needs_up) = &mut *guard;
            block(b, live, needs_up);
        };
        #[cfg(feature = "parallell")]
        rayon::broadcast(|_| work());
        #[cfg(not(feature = "parallell"))]
        work();

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

impl AllTables {
    /// Add up the states that ended up neither won nor lost.
    ///
    /// This rebuilds every chunk from scratch, which is why it is a separate
    /// pass and not something the solve keeps track of: a chunk leaves the work
    /// list for what it can still spread, not for being settled, so the solve
    /// never finds out what the ones it dropped early settled on.
    pub fn count_unresolved(&self) -> u64 {
        count_indexer(self.size)
            .into_iter()
            .map(|counts: PawnCount| {
                let immutable = ImmutableUpdate::new(self, counts);
                let total = counts.total();

                #[cfg(feature = "parallell")]
                let blocks = (0..total.div_ceil(BLOCK)).into_par_iter();
                #[cfg(not(feature = "parallell"))]
                let blocks = 0..total.div_ceil(BLOCK);

                blocks
                    .map(|b| {
                        let start = b * BLOCK;
                        // every layout is visited, so the cursor only ever steps
                        let mut cursor = LayoutCursor::new(counts);
                        let mut sum = 0;
                        for i in start..(start + BLOCK).min(total) {
                            let layout = cursor.at(i);
                            sum += UPDATE.with(|vals| {
                                let mem = &mut *vals.borrow_mut();
                                let mut update = Update {
                                    layout,
                                    immutable: &immutable,
                                    mem,
                                };
                                update.get_unresolved::<true>()
                            });
                        }
                        sum
                    })
                    .sum::<u64>()
            })
            .sum()
    }
}
