use std::iter::{once, Once};

use bit_iter::BitIter;

use crate::proj::{CountOnes, Mask, Proj};

pub(crate) trait Indexer: Clone + IntoIterator {
    fn index(&self, item: &Self::Item) -> usize;

    fn total(&self) -> usize;

    fn choose_one<V, M>(self, proj: V, mask: M) -> Flatten<Self, V, M, ChooseOne>
    where
        V: Proj<Self::Item, Output = u32>,
        M: Mask<Self::Item>,
    {
        Flatten {
            outer: self,
            proj,
            mask,
            gen: ChooseOne,
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

impl<I: IntoIterator + Clone, V, M, G> IntoIterator for Flatten<I, V, M, G>
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

impl Gen<u32, u32> for ChooseOne {
    type GenIter = impl Iterator<Item = u32>;

    fn gen_iter(&self, mask: u32) -> Self::GenIter {
        BitIter::from(mask).map(|offset| offset as u32)
    }

    fn index(&self, mask: u32, offset: &u32) -> usize {
        debug_assert_eq!((1 << *offset) & !mask, 0);

        let mask_less = (1 << *offset) - 1;
        (mask_less & mask).count_ones() as usize
    }

    fn total(&self, mask_size: u32) -> usize {
        mask_size as usize
    }
}
