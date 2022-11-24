use bit_iter::BitIter;

#[derive(Default)]
pub struct Index {
    pub index: usize,
    pub total: usize,
}

pub trait Choose<T> {
    fn apply(self, index: T, total: T) -> Self;
    fn choose_one(self, val: u8, mask: T) -> Self;
    fn choose_exact(self, vals: T, mask: T) -> Self;
    fn choose_at_most<const NUM: u32>(self, vals: T, mask: T) -> Self;
}

macro_rules! index_impl {
    ($($t:ty)*) => {$(
        impl Choose<$t> for Index {
            fn apply(self, n: $t, d: $t) -> Self {
                Self {
                    index: self.index * d as usize + n as usize,
                    total: self.total * d as usize
                }
            }
            fn choose_one(self, val: u8, mask: $t) -> Self {
                let mask_less = (1 << val) - 1;
                let num_less = (mask_less & mask).count_ones();
                let num_total = mask.count_ones();
                self.apply(num_less, num_total)
            }
            fn choose_exact(self, vals: $t, mask: $t) -> Self {
                let index = index_exact(vals as u32, mask as u32);
                let total = comb_exact(mask.count_ones(), vals.count_ones());
                self.apply(index, total)
            }
            fn choose_at_most<const NUM: u32>(self, vals: $t, mask: $t) -> Self {
                let index = index_at_most::<NUM>(vals as u32, mask as u32);
                let total = comb_at_most(mask.count_ones(), NUM);
                self.apply(index, total)
            }
        }
    )*}
}

index_impl! { u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize }

pub fn index_at_most<const MAX: u32>(vals: u32, mask: u32) -> u64 {
    debug_assert!(vals.count_ones() <= MAX as u32);
    debug_assert_eq!(vals & !mask, 0);

    let mut i = 0;
    let mut max = MAX;
    for offset in BitIter::from(vals).rev() {
        let mask_less = (1 << offset) - 1;
        let num_less = (mask_less & mask).count_ones();
        i += comb_at_most(num_less, max);
        max -= 1;
    }
    i
}

pub fn comb_at_most(num_less: u32, max: u32) -> u64 {
    (0..=max).map(|count| comb_exact(num_less, count)).sum()
}

pub fn index_exact(vals: u32, mask: u32) -> u64 {
    debug_assert_eq!(vals & !mask, 0);

    let mut i = 0;
    for (count, offset) in BitIter::from(vals).enumerate() {
        let mask_less = (1 << offset) - 1;
        let num_less = (mask_less & mask).count_ones();
        i += comb_exact(num_less, count as u32 + 1);
    }
    i
}

pub fn comb_exact(num_less: u32, count: u32) -> u64 {
    if count as u32 > num_less {
        return 0;
    }

    let n: u64 = (0..count as u64).map(|i| num_less as u64 - i).product();
    let d: u64 = (1..=count as u64).product();
    n / d
}

#[cfg(test)]
mod tests {

    use super::comb_exact;

    #[test]
    fn comb_some() {
        assert_eq!(comb_exact(0, 0), 1);
        assert_eq!(comb_exact(1, 0), 1);
        assert_eq!(comb_exact(1, 1), 2);
        assert_eq!(comb_exact(2, 1), 3);
        assert_eq!(comb_exact(2, 2), 4);
        assert_eq!(comb_exact(3, 2), 7);

        assert_eq!(comb_exact(1, 2), 2);
    }
}
