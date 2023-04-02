use std::intrinsics::assume;

use std::iter::{self, once, Once};

use bit_iter::BitIter;
use seq_macro::seq;

use crate::proj::{CountOnes, Mask, Proj};

pub trait Indexer: IntoIterator + Sized + Clone {
    fn index(&self, item: &Self::Item) -> usize;

    fn total(&self) -> usize;

    fn choose_one<V, M>(self, proj: V, mask: M) -> Flatten<Self, V, M, ChooseOne>
    where
        V: Proj<Self::Item, Output = u8>,
        M: Mask<Self::Item>,
    {
        Flatten {
            outer: self,
            proj,
            mask,
            gen: ChooseOne,
        }
    }

    fn choose<V, M>(self, n: u8, proj: V, mask: M) -> Flatten<Self, V, M, ChooseExact>
    where
        V: Proj<Self::Item>,
        M: Mask<Self::Item>,
    {
        Flatten {
            outer: self,
            proj,
            mask,
            gen: ChooseExact { count: n },
        }
    }
}

#[derive(Default, Clone)]
pub struct Empty<T>(pub T);

impl<T> IntoIterator for Empty<T> {
    type Item = T;
    type IntoIter = Once<T>;

    fn into_iter(self) -> Self::IntoIter {
        once(self.0)
    }
}

impl<T: Clone> Indexer for Empty<T> {
    fn index(&self, _: &Self::Item) -> usize {
        0
    }

    fn total(&self) -> usize {
        1
    }
}

#[derive(Clone)]
pub struct Flatten<I, V, M, G> {
    outer: I,
    proj: V,
    mask: M,
    gen: G,
}

impl<I: IntoIterator, V, M, G> IntoIterator for Flatten<I, V, M, G>
where
    V: Proj<I::Item>,
    M: Mask<I::Item>,
    G: Gen<M::Output, V::Output>,
    I::Item: Clone,
{
    type Item = I::Item;
    type IntoIter = impl Iterator<Item = Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.outer.into_iter().flat_map(move |board: I::Item| {
            let mask: M::Output = self.mask.get_mask(&board);
            debug_assert_eq!(mask.count_ones(), self.mask.get_size());

            self.gen.gen_iter(mask).map(move |field: V::Output| {
                let mut new = board.clone();
                *(self.proj).proj_mut(&mut new) = field;
                new
            })
        })
    }
}

impl<I: Indexer, V, M, G> Indexer for Flatten<I, V, M, G>
where
    V: Proj<I::Item>,
    M: Mask<I::Item>,
    G: Gen<M::Output, V::Output>,
    I::Item: Clone,
{
    fn index(&self, item: &I::Item) -> usize {
        let outer_index = self.outer.index(item);
        let mask: M::Output = self.mask.get_mask(item);
        debug_assert_eq!(mask.count_ones(), self.mask.get_size());

        let field: &V::Output = self.proj.proj_ref(item);
        let gen_index = self.gen.index(mask, field);
        outer_index * self.gen.total(self.mask.get_size()) + gen_index
    }

    fn total(&self) -> usize {
        self.outer.total() * self.gen.total(self.mask.get_size())
    }
}

trait Gen<M, F>: Clone {
    type GenIter: Iterator<Item = F>;
    fn gen_iter(&self, mask: M) -> Self::GenIter;
    fn index(&self, mask: M, field: &F) -> usize;
    fn total(&self, mask_size: u32) -> usize;
}

#[derive(Clone, Copy)]
pub struct ChooseOne;

impl Gen<u32, u8> for ChooseOne {
    type GenIter = impl Iterator<Item = u8>;

    fn gen_iter(&self, mask: u32) -> Self::GenIter {
        debug_assert_ne!(mask, 0);

        BitIter::from(mask).map(|offset| offset as u8)
    }

    fn index(&self, mask: u32, offset: &u8) -> usize {
        debug_assert_eq!((1 << *offset) & !mask, 0);

        let mask_less = (1 << *offset) - 1;
        (mask_less & mask).count_ones() as usize
    }

    fn total(&self, mask_size: u32) -> usize {
        mask_size as usize
    }
}

#[derive(Clone, Copy)]
pub struct ChooseExact {
    count: u8,
}

macro_rules! gen_impl {
    ($($t:ty)*) => {$(
        impl Gen<$t, $t> for ChooseExact {
            type GenIter = impl Iterator<Item = $t>;

            fn gen_iter(&self, mask: $t) -> Self::GenIter {
                debug_assert!(self.count < 6);
                debug_assert!(self.count <= mask.count_ones() as u8);

                let mut lookup: [$t; 6] = [0; 6];
                let mut entry = !mask;
                for i in 0..=self.count as usize {
                    lookup[self.count as usize - i] = entry;
                    entry |= (!entry) & (!entry).wrapping_neg();
                }

                let mut curr: $t = 0;
                let mut curr_or_skip = !mask;
                let mut init = false;
                iter::from_fn(move || {
                    let lowest = curr & curr.wrapping_neg();

                    let new = curr_or_skip.wrapping_add(lowest);
                    let bits = (new & mask).count_ones();

                    unsafe {assume(bits <= 4)}
                    curr_or_skip = new | lookup[bits as usize];

                    curr = curr_or_skip & mask;

                    let done = init & (new & mask == 0);
                    init = true;
                    (!done).then_some(curr)
                })
            }

            fn index(&self, mask: $t, vals: &$t) -> usize {
                debug_assert_eq!(vals.count_ones() as u8, self.count);
                debug_assert_eq!(vals & !mask, 0);

                index_exact(*vals as u32, mask as u32)
            }

            fn total(&self, mask_size: u32) -> usize {
                comb_exact(mask_size, self.count as u32)
            }
        }
    )*}
}

gen_impl! { u16 u32 }

fn index_exact(vals: u32, mask: u32) -> usize {
    let mut i = 0;
    for (count, offset) in BitIter::from(vals).enumerate() {
        let mask_less = (1 << offset) - 1;
        let num_less = (mask_less & mask).count_ones();
        i += comb_exact(num_less, count as u32 + 1);
    }
    i
}

fn comb_exact(num_less: u32, count: u32) -> usize {
    const fn comb_exact_inner(num_less: usize, count: usize) -> usize {
        if count > num_less {
            return 0;
        }

        if count == 0 {
            return 1;
        }

        num_less * comb_exact_inner(num_less - 1, count - 1) / count
    }

    const RES: [[usize; 6]; 26] = seq!(num_less in 0..26 {
        [#(
            seq!(count in 0..6 {
                [#(comb_exact_inner(num_less, count),)*]
            })
        ,)*]
    });

    RES[num_less as usize][count as usize]
}

#[cfg(test)]
mod tests {

    use crate::proj;

    use super::{ChooseExact, ChooseOne, Empty, Gen, Indexer};

    #[test]
    fn comb_some() {
        let mask = 0b101111u16;
        let indexer = ChooseExact { count: 2 };
        indexer
            .gen_iter(mask)
            .for_each(|x| println!("{x:06b} with id {}", indexer.index(mask, &x)))
    }

    #[test]
    fn comb_one() {
        let mask = 0b101111u32;
        let indexer = ChooseOne;
        indexer
            .gen_iter(mask)
            .for_each(|x| println!("{x} with id {}", indexer.index(mask, &x)))
    }

    #[test]
    fn comb_two() {
        #[derive(Clone, Default)]
        struct Two {
            one: u32,
            two: u32,
            three: u32,
        }
        let indexer = Empty::default()
            .choose(0, proj!(|b: Two| b.one), 0b1111u32)
            .choose(2, proj!(|b: Two| b.two), (|b: &Two| 0b1111u32 & !b.one, 4))
            .choose(
                0,
                proj!(|b: Two| b.three),
                (|b: &Two| 0b1111u32 & !b.one & !b.two, 2),
            );
        for (i, x) in indexer.clone().into_iter().enumerate() {
            assert_eq!(i, indexer.index(&x))
        }
    }
}
