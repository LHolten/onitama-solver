#![allow(unused)]

use std::{
    cell::LazyCell,
    iter::zip,
    mem::transmute,
    ops::{BitAnd, Index, Shr},
    sync::atomic::{AtomicU32, Ordering},
};

use bit_iter::BitIter;

use crate::{
    card::{get_one_bitmap, offset_mask_fixed as offset_mask},
    index::{Empty, Indexer},
    onitama2::TABLE_MASK,
    proj,
};

fn count_indexer(size: u8) -> impl Indexer<Item = PawnCount> {
    let count_mask = (1u32 << size) - 1;
    Empty::default()
        .choose_one(proj!(|c: PawnCount| c.count0), (count_mask, size as u32))
        .choose_one(proj!(|c: PawnCount| c.count1), (count_mask, size as u32))
}

// number of pawns of each player
#[derive(Debug, Default, Clone, Copy)]
struct PawnCount {
    count0: u8,
    count1: u8,
}

impl PawnCount {
    fn invert(self) -> Self {
        Self {
            count0: self.count1,
            count1: self.count0,
        }
    }

    fn indexer(&self) -> impl Indexer<Item = TeamLayout> {
        type L = TeamLayout;
        let pieces1_mask = |l: &L| TABLE_MASK & !l.pieces0;
        Empty(TeamLayout {
            counts: *self,
            ..Default::default()
        })
        .choose(self.count0 + 1, proj!(|l: L| l.pieces0), (TABLE_MASK, 25))
        .choose(
            self.count1 + 1,
            proj!(|l: L| l.pieces1),
            (pieces1_mask, 24 - self.count0 as u32),
        )
    }
}

// only contains erased piece positions
// we don't know which pieces are the kings
#[derive(Debug, Default, Clone, Copy)]
struct TeamLayout {
    counts: PawnCount,
    pieces0: u32,
    pieces1: u32,
}

impl TeamLayout {
    fn indexer(self) -> impl Indexer<Item = KingPos> {
        Empty::default()
            .choose_one(
                proj!(|p: KingPos| p.king0),
                (self.pieces0, self.counts.count0 as u32 + 1),
            )
            .choose_one(
                proj!(|p: KingPos| p.king1),
                (self.pieces1, self.counts.count1 as u32 + 1),
            )
    }

    fn invert(self) -> Self {
        TeamLayout {
            counts: self.counts.invert(),
            pieces0: self.pieces1.reverse_bits() >> 7,
            pieces1: self.pieces0.reverse_bits() >> 7,
        }
    }
}

// the positions of the kings
#[derive(Debug, Default, Clone, Copy)]
struct KingPos {
    king0: u8,
    king1: u8,
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
struct AllTables {
    size: u8,
    cards: Cards,
    list: Box<[Box<[AtomicU32]>]>,
}

impl AllTables {
    fn index_count(&self, counts: PawnCount) -> &[AtomicU32] {
        let indexer = count_indexer(self.size);
        let i = indexer.index(&counts);
        &self.list[i]
    }

    fn index(&self, layout: TeamLayout) -> SubTable<'_> {
        let counts = layout.counts;

        let indexer = counts.indexer();
        let i = indexer.index(&layout);

        let king_indexer = layout.indexer();
        let step_size = king_indexer.total();

        let slice = &self.index_count(counts)[step_size * i..step_size * (i + 1)];
        SubTable { layout, slice }
    }

    fn count_ones(&self) -> u64 {
        self.list
            .iter()
            .flat_map(|l| l.iter())
            .map(|x| x.load(Ordering::Relaxed).bitand((1 << 30) - 1).count_ones() as u64)
            .sum()
    }

    fn len(&self) -> u64 {
        self.list.iter().map(|l| l.len() as u64).sum()
    }
}

// contains results for only one layout
#[derive(Debug, Clone, Copy)]
struct SubTable<'a> {
    layout: TeamLayout,
    slice: &'a [AtomicU32],
}

impl Index<KingPos> for SubTable<'_> {
    type Output = AtomicU32;

    fn index(&self, index: KingPos) -> &Self::Output {
        let indexer = self.layout.indexer();
        let i = indexer.index(&index);
        &self.slice[i]
    }
}

#[derive(Debug)]
struct Accum<'a> {
    layout: TeamLayout,
    mask: u32,
    step: (usize, usize),
    slice: &'a mut [u32],
}

struct Spread<'a> {
    layout: TeamLayout,
    step: (usize, usize),
    slice: &'a [u32],
}

impl AllTables {
    pub fn accumulate(&self, mut accum: Accum<'_>) {
        let (from, to) = accum.step;

        let old = accum.layout;

        let new_slice = LazyCell::new(|| {
            let new = TeamLayout {
                counts: PawnCount {
                    count0: old.counts.count0,
                    count1: old.counts.count1 - (old.pieces1 & 1 << to != 0) as u8,
                },
                pieces0: old.pieces0 ^ (1 << to) ^ (1 << from),
                pieces1: old.pieces1 & !(1 << to),
            };
            self.index(new)
        });

        for (i, oldk) in old.indexer().into_iter().enumerate() {
            if oldk.king1 as usize == to {
                // king is gone, so old state is not lost
                accum.slice[i] |= accum.mask;
                continue;
            }

            let mut newk = oldk;
            if oldk.king0 as usize == from {
                newk.king0 = to as u8
            }
            // if new state is not won, then old state is not lost
            accum.slice[i] |= !new_slice[newk].load(Ordering::Relaxed) & accum.mask;
        }
    }

    // returns whether there was any progress
    pub fn spreadout(&self, mut spread: Spread<'_>) -> bool {
        let (from, to) = spread.step;

        let old = spread.layout;

        let mut new = TeamLayout {
            counts: old.counts,
            pieces0: old.pieces0,
            pieces1: old.pieces1 ^ (1 << to) ^ (1 << from),
        };
        let new_slice = LazyCell::new(|| self.index(new));

        let mut progress = false;
        for (i, oldk) in old.indexer().into_iter().enumerate() {
            debug_assert_ne!(oldk.king0 as usize, to);

            let mut newk = oldk;
            if oldk.king1 as usize == from {
                newk.king1 = to as u8
            }
            // if accum state is lost, then new state is won
            let fetch = new_slice[newk].fetch_or(spread.slice[i], Ordering::Relaxed);
            if fetch | spread.slice[i] != fetch {
                progress = true;
            }
        }

        if new.pieces1.count_ones() == self.size as u32 {
            return progress;
        }

        new.pieces1 |= 1 << from;
        new.counts.count1 += 1;
        let new_slice = LazyCell::new(|| self.index(new));

        for (i, oldk) in old.indexer().into_iter().enumerate() {
            debug_assert_ne!(oldk.king0 as usize, to);

            let mut newk = oldk;
            if oldk.king1 as usize == from {
                newk.king1 = to as u8
            }
            // if accum state is lost, then new state is won
            new_slice[newk].fetch_or(spread.slice[i], Ordering::Relaxed);
        }
        progress
    }

    // returns whether there was any progress
    pub fn update_layout(&self, layout: TeamLayout) -> bool {
        let TeamLayout {
            pieces0, pieces1, ..
        } = layout;

        // every 0 bit means that it could be anything, win loss or draw
        // every 1 bit means that it must be a draw or win
        // we will gradually flip these to 1s, leaving only losses on 0
        // it is initialized to the wins, because those are not lost even when they don't have moves
        let inv_layout = layout.invert();
        let inv_slice = self.index(inv_layout);
        let mut status: Box<[u32]> = layout
            .indexer()
            .into_iter()
            .map(|kpos| inv_slice[kpos.invert()].load(Ordering::Relaxed))
            .collect();

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
                        step: (from, to),
                        slice: &mut status,
                        mask,
                    };
                    self.accumulate(accum);
                }
            }
        }

        // we expand, marking all states that are not lost because it has the card
        // then we negate to get only the lost states
        status
            .iter_mut()
            .for_each(|x| *x = !Block(*x).invert().expand().invert().0);

        let mut progress = false;
        for (card, mask) in zip(self.cards.iter(), mask_iter()) {
            // we spread out the loses, these are wins for the previous state
            let tmp: Box<[u32]> = status
                .iter()
                .map(|x| Block(x & mask).expand().0 & ((1 << 30) - 1))
                .collect();
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
                        step: (from, to),
                        slice: &tmp,
                    };
                    progress |= self.spreadout(spread);
                }
            }
        }
        progress
    }

    pub fn mark_ez_win(&self, counts: PawnCount) {
        for layout in counts.indexer() {
            let TeamLayout {
                pieces0, pieces1, ..
            } = layout;

            for (card, mask) in zip(self.cards.iter(), mask_iter()) {
                let from_mask = offset_mask(2, card.bitmap::<false>());

                for (i, kpos) in layout.indexer().into_iter().enumerate() {
                    if 1 << kpos.king1 & !from_mask == 0 {
                        self.index(layout).slice[i]
                            .fetch_or(Block(mask).expand().0, Ordering::Relaxed);
                        continue;
                    }
                    let from_mask = offset_mask(kpos.king0 as usize, card.bitmap::<false>());
                    if from_mask & pieces1 != 0 {
                        let r = self.index(layout);
                        let m = layout.indexer().total();
                        r.slice[i].fetch_or(Block(mask).expand().0, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    pub fn build(size: u8, cards: u16) -> Self {
        let tb = Self {
            size,
            cards: Cards(cards),
            list: count_indexer(size)
                .into_iter()
                .map(|counts| {
                    counts
                        .indexer()
                        .into_iter()
                        .flat_map(|layout| layout.indexer().into_iter().map(|_| AtomicU32::new(0)))
                        .collect()
                })
                .collect(),
        };

        for counts in count_indexer(size) {
            tb.mark_ez_win(counts);
        }

        println!("{} wins and {} total", tb.count_ones(), tb.len() * 30);

        for counts in count_indexer(size) {
            let mut iters = 0;
            let mut progress = true;
            while progress {
                progress = false;
                for layout in counts.indexer() {
                    progress |= tb.update_layout(layout);
                }
                iters += 1;
            }
            println!("finished {counts:?} in {iters} iterations");
        }

        tb
    }
}

fn mask_iter() -> impl Iterator<Item = u32> {
    let mut mask = 0b000000_000000_010100_000000_101011;
    std::iter::repeat_with(move || {
        let res = mask;
        mask = mask << 6 | mask >> 24;
        res
    })
}

#[derive(Debug, Clone, Copy)]
struct Block(u32);

impl Block {
    pub fn invert(self) -> Self {
        let data = self.0;
        const MASK0: u32 = 0b001001_001001_001001_001001_001001;
        const MASK1: u32 = 0b000111_000111_000111_000111_000111;
        let data = (data & MASK0) << 1 | (data & MASK0 << 1) >> 1 | (data & MASK0 << 2);
        let data = (data & MASK1) << 3 | (data & MASK1 << 3) >> 3;
        Self(data)
    }

    pub fn expand(self) -> Self {
        let data = self.0 as u64;
        // debug_assert_eq!(data & ((1 << 30) - 1), 0);
        let data = data << 10 | data << 20;
        let data = data | data >> 30;
        Self(data as u32)
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

#[cfg(test)]
mod tests {
    use crate::index::Indexer;

    use super::{AllTables, PawnCount, TeamLayout};

    #[test]
    fn build_tb() {
        let tb = AllTables::build(2, 0b11111);
        assert_eq!(tb.count_ones(), 5697226);
        println!("{} total", tb.len() * 30)
    }

    #[test]
    fn counts0() {
        for layout in PawnCount::default().indexer() {
            dbg!(layout);
        }
    }

    #[test]
    fn what() {
        let layout = TeamLayout {
            counts: PawnCount {
                count0: 0,
                count1: 1,
            },
            pieces0: 1,
            pieces1: 6,
        };
        assert_eq!(
            layout.indexer().total(),
            layout.indexer().into_iter().count()
        )
    }
}
