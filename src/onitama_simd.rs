#![allow(unused)]

use std::{
    iter::zip,
    mem::transmute,
    ops::{Index, Shr},
    sync::atomic::{AtomicU32, Ordering},
};

use bit_iter::BitIter;

use crate::{
    card::{get_one_bitmap, offset_mask},
    index::{Empty, Indexer},
    onitama2::TABLE_MASK,
    proj,
};

fn count_indexer(size: u8) -> impl Indexer<Item = PawnCount> {
    let count_mask = (1u32 << size) - 1;
    Empty::default()
        .choose_one(proj!(|c: PawnCount| c.count0), count_mask)
        .choose_one(proj!(|c: PawnCount| c.count1), count_mask)
}

// number of pawns of each player
#[derive(Debug, Default, Clone, Copy)]
struct PawnCount {
    count0: u8,
    count1: u8,
}

impl PawnCount {
    fn indexer(&self) -> impl Indexer<Item = TeamLayout> {
        type L = TeamLayout;
        let pieces1_mask = |l: &L| TABLE_MASK & !l.pieces0;
        Empty::default()
            .choose(self.count0 + 1, proj!(|l: L| l.pieces0), TABLE_MASK)
            .choose(self.count1 + 1, proj!(|l: L| l.pieces1), pieces1_mask)
    }
}

// only contains erased piece positions
// we don't know which pieces are the kings
#[derive(Debug, Default, Clone, Copy)]
struct TeamLayout {
    pieces0: u32,
    pieces1: u32,
}

impl TeamLayout {
    fn indexer(self) -> impl Indexer<Item = KingPos> {
        Empty::default()
            .choose_one(proj!(|p: KingPos| p.king0), self.pieces0)
            .choose_one(proj!(|p: KingPos| p.king1), self.pieces1)
    }

    fn counts(self) -> PawnCount {
        PawnCount {
            count0: self.pieces0.count_ones() as u8 - 1,
            count1: self.pieces1.count_ones() as u8 - 1,
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
struct KingPos {
    king0: u8,
    king1: u8,
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
        let counts = layout.counts();

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
            .map(|x| x.load(Ordering::Relaxed).count_ones() as u64)
            .sum()
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
    slice: &'a mut [Block],
}

struct Spread<'a> {
    layout: TeamLayout,
    step: (usize, usize),
    slice: &'a [Block],
}

impl AllTables {
    pub fn accumulate(&self, mut accum: Accum<'_>) {
        let (from, to) = accum.step;

        let old = accum.layout;

        let new = TeamLayout {
            pieces0: old.pieces0 ^ (1 << to) ^ (1 << from),
            pieces1: old.pieces1 & !(1 << to),
        };

        for (i, oldk) in old.indexer().into_iter().enumerate() {
            if oldk.king1 as usize == to {
                // king is gone, so old state is not lost
                accum.slice[i].0 |= accum.mask;
                continue;
            }

            let mut newk = oldk;
            if oldk.king0 as usize == from {
                newk.king0 = to as u8
            }
            // if new state is not won, then old state is not lost
            accum.slice[i].0 |= !self.index(new)[newk].load(Ordering::Relaxed) & accum.mask;
        }
    }

    // returns whether there was any progress
    pub fn spreadout(&self, mut spread: Spread<'_>) -> bool {
        let (from, to) = spread.step;

        let old = spread.layout;

        let mut new = TeamLayout {
            pieces0: old.pieces0,
            pieces1: old.pieces1 ^ (1 << to) ^ (1 << from),
        };

        let mut progress = false;
        for (i, oldk) in old.indexer().into_iter().enumerate() {
            debug_assert_ne!(oldk.king0 as usize, to);

            let mut newk = oldk;
            if oldk.king1 as usize == from {
                newk.king1 = to as u8
            }
            // if accum state is lost, then new state is won
            let fetch = self.index(new)[newk].fetch_or(!spread.slice[i].0, Ordering::Relaxed);
            if fetch | !spread.slice[i].0 != fetch {
                progress = true;
            }
        }

        if new.pieces1.count_ones() == self.size as u32 {
            return progress;
        }

        new.pieces1 |= 1 << from;

        for (i, oldk) in old.indexer().into_iter().enumerate() {
            debug_assert_ne!(oldk.king0 as usize, to);

            let mut newk = oldk;
            if oldk.king1 as usize == from {
                newk.king1 = to as u8
            }
            // if accum state is lost, then new state is won
            self.index(new)[newk].fetch_or(!spread.slice[i].0, Ordering::Relaxed);
        }
        progress
    }

    // returns whether there was any progress
    pub fn update_layout(&self, layout: TeamLayout) -> bool {
        let TeamLayout { pieces0, pieces1 } = layout;

        // every 0 bit means that it could be anything, win loss or draw
        // every 1 bit means that it must be a draw or win
        // we will gradually flip these to 1s, leaving only losses on 0
        let mut status = vec![Block(0); layout.indexer().total()].into_boxed_slice();

        for (card, mask) in zip(self.cards.iter(), mask_iter()) {
            let directions = card.bitmap::<false>();

            for offset in BitIter::from(directions) {
                let to_mask = offset_mask(offset, pieces0);
                // can not move onto your own pieces
                let to_mask = to_mask & !pieces0;

                for to in BitIter::from(to_mask) {
                    let from = to + offset - 12;
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

        status
            .iter_mut()
            .for_each(|x| *x = x.invert().expand().invert());

        let mut progress = false;
        for (card, mask) in zip(self.cards.iter(), mask_iter()) {
            let tmp: Box<[Block]> = status.iter().map(|x| Block(x.0 & mask).expand()).collect();
            // same thing, but cards are now inverted
            // but it is also the other team, so not inverted
            let directions = card.bitmap::<false>();

            for offset in BitIter::from(directions) {
                // these are backwards moves, so `to` is the where the piece came from
                let to_mask = offset_mask(offset, pieces1);
                // can not move onto your own pieces or opp pieces
                let to_mask = to_mask & !pieces0 & !pieces1;

                for to in BitIter::from(to_mask) {
                    let from = to + offset - 12;
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
            let TeamLayout { pieces0, pieces1 } = layout;

            for (card, mask) in zip(self.cards.iter(), mask_iter()) {
                let from_mask = offset_mask(22, card.bitmap::<false>());

                for (i, kpos) in layout.indexer().into_iter().enumerate() {
                    if 1 << kpos.king1 & !from_mask == 0 {
                        self.index(layout).slice[i]
                            .fetch_or(Block(mask).expand().0, Ordering::Relaxed);
                        continue;
                    }
                    let from_mask = offset_mask(kpos.king0 as usize, card.bitmap::<false>());
                    if from_mask & pieces1 != 0 {
                        self.index(layout).slice[i]
                            .fetch_or(Block(mask).expand().0, Ordering::Relaxed);
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
            let mut progress = true;
            while progress {
                progress = false;
                for layout in counts.indexer() {
                    progress |= tb.update_layout(layout);
                }
            }
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
    use super::{AllTables, PawnCount};

    #[test]
    fn build_tb() {
        let tb = AllTables::build(2, 0b11111);
        assert_eq!(tb.count_ones(), 10617550);
        // println!("{} wins", tb.count_ones())
    }

    #[test]
    fn counts0() {
        for layout in PawnCount::default().indexer() {
            dbg!(layout);
        }
    }
}
