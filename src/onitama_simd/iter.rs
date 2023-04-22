use std::iter::Take;

use bit_iter::BitIter;
use seq_macro::seq;

use crate::index::{Indexer, InternalIter};

use super::{PawnCount, TeamLayout};

pub struct TeamLayoutIter(TeamLayout);

impl Iterator for TeamLayoutIter {
    type Item = TeamLayout;

    fn next(&mut self) -> Option<Self::Item> {
        let TeamLayout {
            pieces0: mut ones,
            pieces1: mut twos,
        } = self.0;

        let l1 = ones.lowest_bit();
        let l2 = twos.lowest_bit();

        let p1 = ones.pivot();
        let p2 = twos.pivot();

        let pivot = if l2 < l1 || p1 & twos != 0 { p2 } else { p1 };
        let swap = pivot | if l2 < l1 || pivot & ones != 0 { l2 } else { l1 };

        if swap & twos != 0 {
            twos ^= swap;
        }
        if swap & ones != 0 {
            ones ^= swap;
        }

        let mask = pivot - 1;

        let twos_diff = (twos & mask).count_ones();
        let ones_diff = (ones & mask).count_ones();

        let old = self.0;
        self.0.pieces1 = (twos & !mask) | ((1 << twos_diff) - 1);
        self.0.pieces0 = (ones & !mask) | ((1 << ones_diff) - 1) << twos_diff;

        Some(old)
    }
}

impl IntoIterator for PawnCount {
    type Item = TeamLayout;

    type IntoIter = Take<TeamLayoutIter>;

    fn into_iter(self) -> Self::IntoIter {
        let (count0, count1) = (self.count0 + 1, self.count1 + 1);
        let init = TeamLayout {
            pieces0: ((1 << count0) - 1) << count1,
            pieces1: (1 << count1) - 1,
        };
        TeamLayoutIter(init).take(self.total())
    }
}

impl InternalIter for PawnCount {
    fn for_each<F>(self, mut f: F)
    where
        F: for<'a> FnMut(&'a mut Self::Item),
    {
        for mut x in self {
            f(&mut x)
        }
    }
}

impl Indexer for PawnCount {
    fn index(&self, item: &Self::Item) -> usize {
        ranking(item.pieces0, item.pieces1)
    }

    fn total(&self) -> usize {
        combinations(25, self.count0 as i32 + 1, self.count1 as i32 + 1)
    }
}

pub fn g((mut ones, mut twos): (u32, u32)) -> (u32, u32) {
    let target_ones = ones.count_ones();
    let target_twos = twos.count_ones();

    if ones.lowest_bit() < twos.lowest_bit() {
        ones += ones.lowest_bit();
        if ones & twos != 0 {
            ones = ones.clear_lowest();
            twos += twos.lowest_bit();
            let good = ones & twos;
            twos ^= good;
            // ones &= !twos;
            // ones ^= twos.lowest_bit();
            // twos ^= twos.lowest_bit() & ones;
        }

        // ones |= ones + ones.lowest_bit();
        // if ones & twos != 0 {
        //     // pivot2
        // } else {
        //     // pivot 1
        // }
    } else {
        twos += twos.lowest_bit();
        ones &= !twos;

        // pivot 2
        // twos |= twos + twos.lowest_bit();
        // twos ^= twos.lowest_bit();
        // if ones & twos != 0 {
        //     ones &= !twos;
        //     ones ^= twos.lowest_bit();
        // }
    }

    let twos_diff = target_twos - twos.count_ones();
    let ones_diff = target_ones - ones.count_ones();
    twos |= (1 << twos_diff) - 1;
    ones |= ((1 << ones_diff) - 1) << twos_diff;
    (ones, twos)
}

trait IntExt {
    fn lowest_bit(self) -> Self;
    fn clear_lowest(self) -> Self;
    fn pivot(self) -> Self;
    fn leading_zero_offset(self) -> u32;
}

impl IntExt for u32 {
    #[inline]
    fn lowest_bit(self) -> Self {
        self & self.wrapping_neg()
    }

    #[inline]
    fn clear_lowest(self) -> Self {
        self & self.wrapping_sub(1)
    }

    #[inline]
    fn pivot(self) -> Self {
        (self + self.lowest_bit()) & !self
    }

    #[inline]
    fn leading_zero_offset(self) -> u32 {
        32 - self.leading_zeros()
    }
}

fn ranking(s_1: u32, s_2: u32) -> usize {
    let mut r: usize = 0;
    let mut ctr = 0usize;

    BitIter::from(s_1 | s_2).for_each(|i| {
        let cond = s_1 & (1 << i) != 0;
        ctr += if cond { 27 } else { 27 * 7 };
        let j = 1 + i + ctr + 27 * (1 + 7);
        let x = unsafe { *COMB.get(j).unwrap_unchecked() };
        r += if cond { x.0 } else { x.1 } as usize;
    });
    r
}

fn unranking(mut r: usize, n: i32, mut ones: i32, mut twos: i32) -> (u32, u32) {
    let (mut s_1, mut s_2) = (0, 0);

    for i in (0..n).rev() {
        let value0 = combinations(i, ones, twos);
        let value1 = combinations(i, ones - 1, twos);
        if r >= value0 + value1 {
            twos -= 1;
            s_2 |= 1 << i;
            r -= value0 + value1;
        } else if r >= value0 {
            ones -= 1;
            s_1 |= 1 << i;
            r -= value0;
        }
    }
    (s_1, s_2)
}

fn combinations_old(n: i32, k1: i32, k2: i32) -> i32 {
    let k3 = n - k1 - k2;
    if k1 < 0 || k2 < 0 || k3 < 0 {
        return 0;
    }

    let mut res = (k3..n).map(|i| i + 1).product();
    res /= (0..k1).map(|i| i + 1).product::<i32>();
    res /= (0..k2).map(|i| i + 1).product::<i32>();
    res
}

const fn comb_exact_inner(n: i32, k1: i32, k2: i32) -> usize {
    let k3 = n - k1 - k2;
    if k1 < 0 || k2 < 0 || k3 < 0 {
        return 0;
    }

    if k1 > 0 {
        return n as usize * comb_exact_inner(n - 1, k1 - 1, k2) / k1 as usize;
    }
    if k2 > 0 {
        return n as usize * comb_exact_inner(n - 1, k1, k2 - 1) / k2 as usize;
    }
    1
}

const fn comb_exact_inner2(i: i32) -> u32 {
    let n = i % 27;
    let k1 = i / 27 % 7;
    let k2 = i / 27 / 7;
    comb_exact_inner(n - 1, k1 - 1, k2 - 1) as u32
}

const COMB: [(u32, u32); 27 * 7 * 7 + 27] = seq!(i in 0..1350 {
    [#(
        #[allow(clippy::identity_op)]
        #[allow(clippy::erasing_op)]
        #[allow(clippy::eq_op)]
        {
            let v1 = comb_exact_inner2(i);
            let v2 = comb_exact_inner2(i - 27) + comb_exact_inner2(i);
            (v1, v2)
        }
    ,)*]
});

fn combinations(n: i32, k1: i32, k2: i32) -> usize {
    let i = 1 + n + 27 * (1 + k1 + 7 * (1 + k2));
    unsafe { COMB.get(i as usize).unwrap_unchecked().0 as usize }
}

fn iter(s_1: u32, s_2: u32, i: i32, ones: i32, twos: i32, f: &mut impl FnMut(u32, u32)) {
    if ones == 0 && twos == 0 {
        return f(s_1, s_2);
    }

    for j in (ones + twos - 1)..i {
        if ones > 0 {
            iter(s_1 | 1 << j, s_2, j, ones - 1, twos, f);
        }
        if twos > 0 {
            iter(s_1, s_2 | 1 << j, j, ones, twos - 1, f);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::{
        index::Indexer,
        onitama_simd::{
            iter::{combinations, combinations_old, g, ranking, unranking},
            PawnCount,
        },
    };

    #[test]
    fn test_counts() {
        for n in -1..26 {
            for k1 in -1..3 {
                for k2 in -1..3 {
                    assert_eq!(
                        combinations_old(n, k1, k2) as usize,
                        combinations(n, k1, k2)
                    )
                }
            }
        }
        println!("{}", combinations(25, 5, 5))
    }

    #[test]
    fn test_same() {
        let counts = PawnCount {
            count0: 1,
            count1: 1,
        };

        assert_eq!(combinations(25, 2, 2), counts.total());

        let mut set = HashSet::new();

        for layout in counts {
            let idx = ranking(layout.pieces0, layout.pieces1);
            // println!("{idx}");
            assert!(set.insert(idx));
            assert_eq!(unranking(idx, 25, 2, 2), (layout.pieces0, layout.pieces1))
            // assert_eq!(idx, original(layout.pieces0, layout.pieces1))
        }
        // assert_eq!(
        //     original(0b0001, 0b1000),
        //     ranking_multinomial_coefficients(0b0001, 0b1000)
        // )
    }

    #[test]
    fn test_iter() {
        let (mut a0, mut a1) = (0b1100, 0b0011);
        for _ in 0..combinations(8, 2, 2) {
            let mut str = String::new();
            for i in 0..8 {
                if a0 & 1 << i != 0 {
                    str.insert(0, '1')
                } else if a1 & 1 << i != 0 {
                    str.insert(0, '2')
                } else {
                    str.insert(0, '0')
                }
            }
            println!("{str}");
            (a0, a1) = g((a0, a1));
        }
        // let mut i = 0;
        // iter(0, 0, 8, 2, 2, &mut |a0, a1| {
        //     let (b0, b1) = unranking(i, 8, 2, 2);
        //     assert_eq!((a0, a1), (b0, b1));
        //     let mut str = String::new();
        //     for i in 0..8 {
        //         if a0 & 1 << i != 0 {
        //             str.insert(0, '1')
        //         } else if a1 & 1 << i != 0 {
        //             str.insert(0, '2')
        //         } else {
        //             str.insert(0, '0')
        //         }
        //     }
        //     println!("{str}");
        //     // println!("{a0:08b}, {a1:08b}");
        //     i += 1;
        // });

        // assert_eq!(i, combinations(8, 2, 2));
    }
}
