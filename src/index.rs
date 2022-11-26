use std::{
    iter::{self, once, Once},
    marker::PhantomData,
};

use bit_iter::BitIter;

use crate::proj::{Mask, Proj};

#[derive(Default)]
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

    fn choose_exact<V, M>(self, proj: V, mask: M, count: u8) -> Flatten<Self, V, M, ChooseExact>
    where
        V: Proj<Self::Item>,
        M: Mask<Self::Item>,
    {
        Flatten {
            inner: self,
            proj,
            mask,
            gen: ChooseExact { count },
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
        Index::default()
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

impl Gen<u16, u16> for ChooseExact {
    type GenIter = impl Iterator<Item = u16>;

    fn gen_iter(&self, mask: u16) -> Self::GenIter {
        debug_assert!(self.count < 5);

        let mut lookup = [0u16; 5];
        let mut low_mask = 0;
        for i in 0..=self.count as usize {
            lookup[self.count as usize - i] = low_mask;
            let high_bits = mask & !low_mask;
            low_mask |= high_bits & high_bits.wrapping_neg();
        }

        println!("{:?}", lookup);

        let mut curr = 0u16;
        iter::from_fn(move || {
            let lowest = curr & curr.wrapping_neg();

            let (new, done) = (curr | !mask).overflowing_add(lowest);

            curr = new & mask;
            curr |= lookup[curr.count_ones() as usize];

            (!done).then_some(curr)
        })
    }

    fn index(&self, mask: u16, vals: &u16) -> Index {
        Index {
            index: index_exact(*vals as u32, mask as u32),
            total: comb_exact(mask.count_ones(), vals.count_ones()),
        }
    }
}

macro_rules! gen_impl {
    ($($t:ty)*) => {$(
        impl Gen<$t, $t> for ChooseExact {
            type GenIter = impl Iterator<Item = $t>;

            fn gen_iter(&self, mask: $t) -> Self::GenIter {
                // TODO: fix this
                BitIter::from(mask).map(|offset| offset as $t)
                // BitIter::from(mask).flat_map(|offset| {
                //     let mask_less = (1 << offset) - 1;
                //     let new_mask = mask & mask_less;
                //     todo!()
                // })
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

gen_impl! { u8 u32 u64 u128 usize i8 i16 i32 i64 i128 isize }

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

    use super::{comb_exact, ChooseExact, Gen};

    #[test]
    fn comb_some() {
        ChooseExact { count: 2 }
            .gen_iter(0b1111u16)
            .for_each(|x| println!("{x:04b}"))
    }
}
