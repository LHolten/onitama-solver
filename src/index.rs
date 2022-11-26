use std::intrinsics::assume;

use std::{
    iter::{self, once, Once},
    marker::PhantomData,
};

use bit_iter::BitIter;

use crate::proj::{Mask, Proj};

pub struct Index {
    pub index: usize,
    pub total: usize,
}

pub trait Indexer: IntoIterator + Sized {
    fn index(self, board: &Self::Item) -> Index;

    fn choose_one<V, M>(self, proj: V, mask: M) -> Flatten<Self, V, M, ChooseOne>
    where
        V: Proj<Self::Item, Output = u8>,
        M: Mask<Self::Item, Output = u32>,
    {
        Flatten {
            inner: self,
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
            inner: self,
            proj,
            mask,
            gen: ChooseExact { count: n },
        }
    }
}

#[derive(Default)]
pub struct Empty<T>(PhantomData<T>);

impl<T: Default> IntoIterator for Empty<T> {
    type Item = T;
    type IntoIter = Once<T>;

    fn into_iter(self) -> Self::IntoIter {
        once(Default::default())
    }
}

impl<T: Default> Indexer for Empty<T> {
    fn index(self, _: &Self::Item) -> Index {
        Index { index: 0, total: 1 }
    }
}

pub struct Flatten<I, V, M, G> {
    inner: I,
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
        self.inner.into_iter().flat_map(move |board: I::Item| {
            let mask: M::Output = self.mask.get_mask(&board);
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
    fn index(self, board: &I::Item) -> Index {
        let this = self.inner.index(board);
        let mask: M::Output = self.mask.get_mask(board);
        let field: &V::Output = self.proj.proj_ref(board);
        let other = self.gen.index(mask, field);
        Index {
            index: this.index * other.total as usize + other.index as usize,
            total: this.total * other.total as usize,
        }
    }
}

trait Gen<M, F> {
    type GenIter: Iterator<Item = F>;
    fn gen_iter(&self, mask: M) -> Self::GenIter;
    fn index(&self, mask: M, field: &F) -> Index;
}

pub struct ChooseOne;

impl Gen<u32, u8> for ChooseOne {
    type GenIter = impl Iterator<Item = u8>;

    fn gen_iter(&self, mask: u32) -> Self::GenIter {
        BitIter::from(mask).map(|offset| offset as u8)
    }

    fn index(&self, mask: u32, offset: &u8) -> Index {
        let mask_less = (1 << *offset) - 1;
        Index {
            index: (mask_less & mask).count_ones() as usize,
            total: mask.count_ones() as usize,
        }
    }
}

pub struct ChooseExact {
    count: u8,
}

macro_rules! gen_impl {
    ($($t:ty)*) => {$(
        impl Gen<$t, $t> for ChooseExact {
            type GenIter = impl Iterator<Item = $t>;

            fn gen_iter(&self, mask: $t) -> Self::GenIter {
                debug_assert!(self.count < 5);
                assert!(self.count <= mask.count_ones() as u8);

                let mut lookup: [$t; 5] = [0; 5];
                let mut entry = !mask;
                for i in 0..=self.count as usize {
                    lookup[self.count as usize - i] = entry;
                    entry |= (!entry) & (!entry).wrapping_neg();
                }

                let mut curr_or_skip = !mask;
                let mut curr: $t = 0;
                iter::from_fn(move || {
                    let lowest = curr & curr.wrapping_neg();

                    let (new, done) = curr_or_skip.overflowing_add(lowest);
                    let bits = (new & mask).count_ones();

                    unsafe {assume(bits <= 4)}
                    curr_or_skip = new | lookup[bits as usize];

                    curr = curr_or_skip & mask;
                    (!done).then_some(curr)
                })
            }

            fn index(&self, mask: $t, vals: &$t) -> Index {
                Index {
                    index: index_exact(*vals as u32, mask as u32),
                    total: comb_exact(mask.count_ones(), vals.count_ones()),
                }
            }
        }
    )*}
}

gen_impl! { u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize }

pub fn index_exact(vals: u32, mask: u32) -> usize {
    debug_assert_eq!(vals & !mask, 0);

    let mut i = 0;
    for (count, offset) in BitIter::from(vals).enumerate() {
        let mask_less = (1 << offset) - 1;
        let num_less = (mask_less & mask).count_ones();
        i += comb_exact(num_less, count as u32 + 1);
    }
    i
}

pub fn comb_exact(num_less: u32, count: u32) -> usize {
    if count as u32 > num_less {
        return 0;
    }

    let n: u64 = (0..count as u64).map(|i| num_less as u64 - i).product();
    let d: u64 = (1..=count as u64).product();
    (n / d) as usize
}

#[cfg(test)]
mod tests {

    use super::{ChooseExact, ChooseOne, Gen};

    #[test]
    fn comb_some() {
        let indexer = ChooseExact { count: 3 };
        let mask = 0b101111u16;
        indexer
            .gen_iter(mask)
            .for_each(|x| println!("{x:06b} with id {}", indexer.index(mask, &x).index))
    }

    #[test]
    fn comb_one() {
        let indexer = ChooseOne;
        let mask = 0b101111u32;
        indexer
            .gen_iter(mask)
            .for_each(|x| println!("{x} with id {}", indexer.index(mask, &x).index))
    }
}
