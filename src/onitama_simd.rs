#![allow(unused)]
mod accum_spread;
mod job;

use rayon::prelude::*;

use std::{
    alloc::Layout,
    array,
    cell::{LazyCell, RefCell},
    iter::{repeat, repeat_with, zip},
    mem::transmute,
    ops::{BitAnd, Index, IndexMut, Shr},
    process::exit,
    sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
};

use bit_iter::BitIter;

use crate::{
    card::{get_one_bitmap, offset_mask_fixed as offset_mask},
    index::{Empty, Indexer, InternalIter},
    proj,
};

pub const TABLE_MASK: u32 = (1 << 25) - 1;
pub const BLOCK_MASK: u32 = (1 << 30) - 1;
pub const RESOLVED_BIT: u32 = 1 << 30;

fn count_indexer(size: u32) -> impl Indexer<Item = PawnCount> {
    let count_mask = (1u32 << size) - 1;
    Empty::default()
        .choose_one(proj!(|c: PawnCount| c.count0), (count_mask, size))
        .choose_one(proj!(|c: PawnCount| c.count1), (count_mask, size))
}

// number of pawns of each player
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct PawnCount {
    pub(crate) count0: u32,
    pub(crate) count1: u32,
}

impl PawnCount {
    fn invert(self) -> Self {
        Self {
            count0: self.count1,
            count1: self.count0,
        }
    }

    pub(crate) fn indexer(self) -> impl Indexer<Item = TeamLayout> {
        type L = TeamLayout;
        let pieces1_mask = |l: &L| TABLE_MASK & !l.pieces0;
        Empty(Default::default())
            .choose(self.count0 + 1, proj!(|l: L| l.pieces0), (TABLE_MASK, 25))
            .choose(
                self.count1 + 1,
                proj!(|l: L| l.pieces1),
                (pieces1_mask, 24 - self.count0),
            )
    }
}

// only contains erased piece positions
// we don't know which pieces are the kings
#[derive(Debug, Default, Clone, Copy)]
pub struct TeamLayout {
    pub(crate) pieces0: u32,
    pub(crate) pieces1: u32,
}

impl TeamLayout {
    fn indexer(self, counts: PawnCount) -> impl Indexer<Item = KingPos> {
        debug_assert_eq!(self.pieces0.count_ones(), counts.count0 + 1);
        debug_assert_eq!(self.pieces1.count_ones(), counts.count1 + 1);

        Empty::default()
            .choose_one(
                proj!(|p: KingPos| p.king0),
                (
                    self.pieces0 & !(1 << 22),
                    counts.count0 + (self.pieces0 & 1 << 22 == 0) as u32,
                ),
            )
            .choose_one(
                proj!(|p: KingPos| p.king1),
                (
                    self.pieces1 & !(1 << 2),
                    counts.count1 + (self.pieces1 & 1 << 2 == 0) as u32,
                ),
            )
    }

    fn counts(self) -> PawnCount {
        PawnCount {
            count0: self.pieces0.count_ones() - 1,
            count1: self.pieces1.count_ones() - 1,
        }
    }

    fn invert(self) -> Self {
        TeamLayout {
            pieces0: self.pieces1.reverse_bits() >> 7,
            pieces1: self.pieces0.reverse_bits() >> 7,
        }
    }
}

// the positions of the kings
#[derive(Debug, Default, Clone, Copy)]
pub struct KingPos {
    king0: u32,
    king1: u32,
}

impl KingPos {
    fn invert(self) -> Self {
        Self {
            king0: 24 - self.king1,
            king1: 24 - self.king0,
        }
    }
}

// contains all the results up to some number of pieces
pub struct AllTables {
    size: u32,
    cards: Cards,
    mask_lookup: [u32; 25],
    list: Box<[Table]>,
    block_done: AtomicU64,
    block_not_done: AtomicU64,
    card_done: AtomicU64,
    card_not_done: AtomicU64,
}

impl AllTables {
    fn index_count(&self, counts: PawnCount) -> &Table {
        let indexer = count_indexer(self.size);
        let i = indexer.index(&counts);
        unsafe { self.list.get(i).unwrap_unchecked() }
    }

    fn count_ones(&self) -> u64 {
        self.list.iter().map(|x| x.count_ones()).sum()
    }

    fn len(&self) -> u64 {
        self.list.iter().map(|l| l.list.len() as u64).sum()
    }
}

#[derive(Debug)]
pub struct Table {
    counts: PawnCount,
    chunk_size: usize,
    list: Box<[AtomicU32]>,
}

impl Table {
    fn index(&self, layout: TeamLayout) -> SubTable<'_> {
        let indexer = self.counts.indexer();
        let i = indexer.index(&layout);

        let slice = self
            .list
            .get(self.chunk_size * i..self.chunk_size * (i + 1));
        let slice = unsafe { slice.unwrap_unchecked() };
        SubTable {
            layout,
            slice,
            counts: self.counts,
        }
    }

    fn count_ones(&self) -> u64 {
        self.list
            .iter()
            .map(|x| x.load(Ordering::Relaxed).bitand(BLOCK_MASK).count_ones() as u64)
            .sum()
    }
}

// contains results for only one layout
#[derive(Debug, Clone, Copy)]
struct SubTable<'a> {
    counts: PawnCount,
    layout: TeamLayout,
    slice: &'a [AtomicU32],
}

impl Index<KingPos> for SubTable<'_> {
    type Output = AtomicU32;

    fn index(&self, index: KingPos) -> &Self::Output {
        let indexer = self.layout.indexer(self.counts);
        let i = indexer.index(&index);
        unsafe { self.slice.get(i).unwrap_unchecked() }
    }
}

#[derive(Debug)]
struct KingLookup {
    list: [[u8; 25]; 25],
}

impl Index<KingPos> for KingLookup {
    type Output = u8;

    fn index(&self, index: KingPos) -> &Self::Output {
        let t = unsafe { self.list.get(index.king0 as usize).unwrap_unchecked() };
        unsafe { t.get(index.king1 as usize).unwrap_unchecked() }
    }
}

impl IndexMut<KingPos> for KingLookup {
    fn index_mut(&mut self, index: KingPos) -> &mut Self::Output {
        let t = unsafe { self.list.get_mut(index.king0 as usize).unwrap_unchecked() };
        unsafe { t.get_mut(index.king1 as usize).unwrap_unchecked() }
    }
}

pub struct LocalMem {
    wins: Vec<u32>,
    status: Vec<u32>,
    king_lookup: KingLookup,
    lookup: [Vec<u32>; 5],
}

impl LocalMem {
    const fn new() -> Self {
        Self {
            wins: vec![],
            status: vec![],
            king_lookup: KingLookup {
                list: [[0; 25]; 25],
            },
            lookup: [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        }
    }
}

pub struct ImmutableUpdate<'a> {
    inv_current: &'a Table,
    current: &'a Table,
    take_one: Option<&'a Table>,
    leave_one: Option<&'a Table>,
    go_up: bool,
    mask_lookup: &'a [u32; 25],
}

pub struct Update<'a> {
    layout: TeamLayout,
    immutable: &'a ImmutableUpdate<'a>,
    mem: &'a mut LocalMem,
}

#[derive(Debug)]
pub struct Accum<'a> {
    layout: TeamLayout,
    current: &'a Table,
    take_one: Option<&'a Table>,
    mask: u32,
    step: (usize, usize),
    slice: &'a mut [u32],
    king_lookup: &'a KingLookup,
}

pub struct Spread<'a> {
    layout: TeamLayout,
    current: &'a Table,
    leave_one: Option<&'a Table>,
    go_up: bool,
    step: (usize, usize),
    slice: &'a [u32],
    king_lookup: &'a KingLookup,
}

impl AllTables {
    // returns whether there was any progress
    pub fn update_layout(&self, update: &mut Update<'_>) -> bool {
        let layout = update.layout;
        let mem = &mut *update.mem;
        let ImmutableUpdate {
            inv_current,
            current,
            take_one,
            leave_one,
            go_up,
            mask_lookup,
        } = *update.immutable;
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

        for (card, mask) in zip(self.cards.iter(), mask_iter()) {
            let directions = card.bitmap::<false>();

            for offset in BitIter::from(directions) {
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
        for (card, mask) in zip(self.cards.iter(), mask_iter()) {
            // we spread out the loses, these are wins for the previous state
            mem.wins.clear();
            mem.wins
                .extend(mem.status.iter().map(|x| Block(x & mask).expand().0));
            // same thing, but cards are now inverted
            // but it is also the other team, so not inverted
            let directions = card.bitmap::<false>();

            for offset in BitIter::from(directions) {
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
        }

        progress
    }

    pub fn mark_ez_win(&self, counts: PawnCount) {
        for layout in counts.indexer() {
            self.ez_win_for_each(counts, layout, &mut |i, mask| {
                // mask is the future cards
                self.index_count(counts).index(layout)[i]
                    .fetch_or(Block(mask).invert().expand().0, Ordering::Relaxed);
            })
        }
    }

    pub fn ez_win_for_each(
        &self,
        counts: PawnCount,
        layout: TeamLayout,
        f: &mut impl FnMut(KingPos, u32),
    ) {
        let TeamLayout {
            pieces0, pieces1, ..
        } = layout;

        for (card, mask) in zip(self.cards.iter(), mask_iter()) {
            // from where can you attack the temple?
            let from_mask = offset_mask(2, card.bitmap::<false>());

            layout.indexer(counts).for_enumerate(|i, kpos| {
                if 1 << kpos.king1 & from_mask == 0 || 1 << 2 & pieces1 != 0 {
                    // no attack on temple
                    let from_mask = offset_mask(kpos.king0 as usize, card.bitmap::<false>());
                    if from_mask & pieces1 == 0 {
                        // no attack on king
                        return;
                    }
                }
                f(*kpos, mask);
            })
        }
    }

    pub fn build(size: u32, cards: u16) -> Self {
        let mut mask_lookup = [0; 25];
        for (mask, card) in zip(mask_iter(), Cards(cards).iter()) {
            for offset in BitIter::from(card.bitmap::<false>()) {
                mask_lookup[offset] |= mask
            }
        }
        let tb = Self {
            size,
            cards: Cards(cards),
            mask_lookup,
            list: count_indexer(size)
                .into_iter()
                .map(|counts: PawnCount| {
                    let chunk_size = (counts.count0 + 1) as usize * (counts.count1 + 1) as usize;
                    let num_chunks = counts.indexer().total();
                    let list = repeat_with(|| AtomicU32::new(0))
                        .take(chunk_size * num_chunks)
                        .collect();
                    Table {
                        counts,
                        chunk_size,
                        list,
                    }
                })
                .collect(),
            block_done: Default::default(),
            block_not_done: Default::default(),
            card_done: Default::default(),
            card_not_done: Default::default(),
        };

        for counts in count_indexer(size) {
            tb.mark_ez_win(counts);
        }

        println!("{} wins and {} total", tb.count_ones(), tb.len() * 30);

        for counts in count_indexer(size) {
            let counts: PawnCount = counts;
            if counts.count0 < counts.count1 {
                continue;
            }
            let mut jobs = vec![TableJob::new(&tb, counts)];
            if counts.count0 > counts.count1 {
                jobs.push(TableJob::new(&tb, counts.invert()));
            }

            let mut any_progress = true;
            let mut iters = 0;
            while any_progress {
                any_progress = false;
                for job in &mut jobs {
                    any_progress |= job.next().is_some();
                }
                iters += 1;
            }

            println!("finished {counts:?} in {iters} iterations");
            println!("{} wins", tb.index_count(counts).count_ones());
        }

        tb
    }
}

struct TableJob<'a> {
    tb: &'a AllTables,
    layouts: Vec<TeamLayout>,
    update: ImmutableUpdate<'a>,
    done: bool,
}

fn mask_iter() -> impl Iterator<Item = u32> {
    let mut mask = 0b000000_000000_010100_000000_101011;
    std::iter::repeat_with(move || {
        let res = mask;
        mask = mask << 6 | mask >> 24;
        res & BLOCK_MASK
    })
}

#[derive(Debug, Clone, Copy)]
struct Block(u32);

impl Block {
    pub fn invert(self) -> Self {
        let data = self.0;
        const MASK0: u32 = 0b001001_001001_001001_001001_001001;
        const MASK1: u32 = 0b000111_000111_000111_000111_000111;
        let data = (data & MASK0 << 1) << 1 | (data & MASK0 << 2) >> 1 | (data & MASK0);
        let data = (data & MASK1) << 3 | (data & MASK1 << 3) >> 3;
        Self(data)
    }

    pub fn expand(self) -> Self {
        let data = self.0 as u64;
        let data = data << 10 | data << 20;
        let data = data | data >> 30;
        Self(data as u32 & BLOCK_MASK)
    }
}

#[derive(Debug, Clone, Copy)]
struct Cards(u16);

impl Cards {
    fn iter(self) -> impl Iterator<Item = Card> {
        BitIter::from(self.0).map(Card)
    }
}

#[derive(Debug, Clone, Copy)]
struct Card(usize);

impl Card {
    fn bitmap<const S: bool>(self) -> u32 {
        get_one_bitmap::<S>(self.0)
    }
}

fn pretty(layout: TeamLayout, kpos: KingPos) {
    let TeamLayout { pieces0, pieces1 } = layout;
    let KingPos { king0, king1 } = kpos;
    debug_assert_eq!(pieces0 & pieces1, 0);
    debug_assert_ne!(1 << king0 & pieces0, 0);
    debug_assert_ne!(1 << king1 & pieces1, 0);

    println!("----- x side");
    for y in 0..5 {
        for x in 0..5 {
            let i = 24 - 5 * y - x;
            if king0 == i {
                print!("O")
            } else if king1 == i {
                print!("X")
            } else if 1 << i & pieces0 != 0 {
                print!("o")
            } else if 1 << i & pieces1 != 0 {
                print!("x")
            } else {
                print!(".")
            }
        }
        println!()
    }
    println!("----- o side")
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use crate::{index::Indexer, onitama_simd::Block};

    use super::{mask_iter, AllTables, PawnCount, TeamLayout};

    #[test]
    fn build_tb() {
        let tb = AllTables::build(3, 0b11111);
        println!("{} total", tb.len() * 30);
        println!(
            "{} blocks done, {} blocks not done",
            tb.block_done.load(Ordering::Relaxed),
            tb.block_not_done.load(Ordering::Relaxed)
        );
        println!(
            "{} cards done, {} cards not done",
            tb.card_done.load(Ordering::Relaxed),
            tb.card_not_done.load(Ordering::Relaxed)
        );
        // assert_eq!(tb.count_ones(), 6752579);
        assert_eq!(tb.count_ones(), 831344251);
    }

    #[test]
    fn counts0() {
        for layout in PawnCount::default().indexer() {
            dbg!(layout);
        }
    }

    #[test]
    fn mask_test() {
        for mask in mask_iter().take(5) {
            assert_eq!(mask, Block(mask).invert().0)
        }
    }

    // #[test]
    // fn what() {
    //     let layout = TeamLayout {
    //         counts: PawnCount {
    //             count0: 0,
    //             count1: 1,
    //         },
    //         pieces0: 1,
    //         pieces1: 6,
    //     };
    //     assert_eq!(
    //         layout.indexer().total(),
    //         layout.indexer().into_iter().count()
    //     )
    // }
}
