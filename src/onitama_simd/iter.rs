use std::iter::Take;
#[cfg(all(target_feature = "avx512f", target_feature = "avx512cd"))]
use std::simd::{cmp::SimdPartialEq, num::SimdUint, Select, Simd};

use bit_iter::BitIter;
use seq_macro::seq;

use crate::index::{Indexer, InternalIter};

use super::{PawnCount, TeamLayout};

pub struct TeamLayoutIter(TeamLayout);

/// Walks the enumeration, skipping forward to any position asked for.
///
/// One step is much cheaper than one unranking, but a skip over a long run of
/// retired layouts is not worth walking, so that restarts from an unranking
/// instead. Early passes are dense and pay only steps; late ones are sparse and
/// pay about one unranking per live layout.
pub struct LayoutCursor {
    counts: PawnCount,
    /// The position `iter` is about to yield.
    at: usize,
    iter: TeamLayoutIter,
}

impl LayoutCursor {
    /// The longest skip still worth walking, in steps per unranking.
    const GAP: usize = 4;

    pub fn new(counts: PawnCount) -> Self {
        Self {
            counts,
            // no position, so the first `at` always unranks
            at: usize::MAX,
            iter: TeamLayoutIter(TeamLayout::default()),
        }
    }

    pub fn at(&mut self, i: usize) -> TeamLayout {
        if i < self.at || i - self.at > Self::GAP {
            self.iter = TeamLayoutIter(self.counts.nth_layout(i));
        } else {
            for _ in self.at..i {
                self.iter.next();
            }
        }
        self.at = i + 1;
        self.iter.next().unwrap()
    }
}

impl PawnCount {
    /// How many states of the table for these counts are still in play.
    ///
    /// A chunk has a slot for every king position, but one with a king on the
    /// opponent's temple is a game that is already
    /// [`over`](super::KingPos::over), so a layout with a piece there has one
    /// fewer choice for that king. The
    /// layouts that do have one are just a multinomial with that square spoken
    /// for, which makes the whole sum closed form instead of a walk over every
    /// layout.
    pub fn states(self) -> u64 {
        let (k0, k1) = (self.count0 as i32 + 1, self.count1 as i32 + 1);
        let comb = |n, a, b| combinations(n, a, b) as u64;
        let (a, b) = (k0 as u64, k1 as u64);

        // every layout, minus the ones that lose a king position on either
        // temple, plus back the ones that lose both
        a * b * comb(25, k0, k1) + comb(23, k0 - 1, k1 - 1)
            - b * comb(24, k0 - 1, k1)
            - a * comb(24, k0, k1 - 1)
    }

    /// The layout the enumeration yields at position `i`.
    ///
    /// The enumeration is in ranking order, so this inverts the ranking instead
    /// of walking up to `i`. That makes recording block seeds cost one unranking
    /// per block rather than one step per layout, and lets the blocks be built in
    /// any order.
    pub fn nth_layout(self, i: usize) -> TeamLayout {
        let (pieces0, pieces1) = unranking(i, 25, self.count0 as i32 + 1, self.count1 as i32 + 1);
        TeamLayout { pieces0, pieces1 }
    }
}

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

pub(super) fn ranking(s_1: u32, s_2: u32) -> usize {
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

/// [`ranking`] of sixteen layouts at once, one per lane.
///
/// The scalar walk is a chain of a bit pick, a test and a table read per piece,
/// and a chunk has dozens of moves to locate, none of which depend on each
/// other. So the lanes take the walk in step, one set bit of every layout per
/// round.
///
/// The table read is what does not go wide: a gather of sixteen is microcode
/// on this machine, and slower than the scalar walk it replaces. But what it
/// reads is a multinomial, `C(i, k1, k2)` for the squares below the piece and
/// the pieces of each side so far, and that factors as `binom(i, k1 + k2) *
/// binom(k1 + k2, k1)`. In round `n` every lane is on its `n + 1`st piece, so
/// `k1 + k2` is the same in all of them, and both factors are lookups in a
/// table of that round small enough to sit in registers: 25 squares for the
/// first, which is one two source permute, and a row of Pascal's triangle for
/// the second, which is one permute.
///
/// A lane that runs out of pieces early just stops adding.
#[cfg(all(target_feature = "avx512f", target_feature = "avx512cd"))]
pub fn ranking16(s_1: Simd<u32, 16>, s_2: Simd<u32, 16>) -> Simd<u32, 16> {
    use super::accum_spread::{permute, permute16};

    // `binom(i, m)` for every square `i`, as the two halves of one table
    let squares = |m: usize, i: Simd<u32, 16>| -> Simd<u32, 16> {
        let [lo, hi] = SQUARES[m].map(Simd::<u32, 16>::from_array);
        permute(lo, hi, i)
    };
    let pascal = |row: &[u32; 16], k: Simd<u32, 16>| -> Simd<u32, 16> { permute16(Simd::from_array(*row), k) };

    let zero = Simd::splat(0);
    let one = Simd::splat(1);
    let mut x = s_1 | s_2;
    let mut k1 = zero;
    let mut r = zero;
    for n in 0..PIECES {
        let live = x.simd_ne(zero);
        if !live.any() {
            break;
        }
        let low = x & (zero - x);
        let i = Simd::splat(31) - low.leading_zeros();
        let ours = (s_1 & low).simd_ne(zero);
        k1 += ours.select(one, zero);

        // `x.0` in the scalar walk, which a piece of either side adds
        let both = squares(n + 1, i) * pascal(&PASCAL[n + 1], k1);
        // and `x.1` adds `C(i, k1 - 1, k2)` on top, which is the same
        // factoring one piece down
        let theirs = squares(n, i) * pascal(&PASCAL_BELOW[n], k1);
        let term = both + ours.select(zero, theirs);
        r += live.select(term, zero);
        x &= x - one;
    }
    r
}

/// The most pieces a layout has, both sides and both kings.
#[cfg(all(target_feature = "avx512f", target_feature = "avx512cd"))]
const PIECES: usize = 2 * super::SIDE;

#[cfg(all(target_feature = "avx512f", target_feature = "avx512cd"))]
const fn binom(n: usize, k: usize) -> u32 {
    if k > n {
        return 0;
    }
    comb_exact_inner(n as i32, k as i32, 0) as u32
}

/// `binom(i, m)` at `[m][i / 16][i % 16]`.
#[cfg(all(target_feature = "avx512f", target_feature = "avx512cd"))]
static SQUARES: [[[u32; 16]; 2]; PIECES + 1] = {
    let mut t = [[[0; 16]; 2]; PIECES + 1];
    let mut m = 0;
    while m <= PIECES {
        let mut i = 0;
        while i < 32 {
            t[m][i / 16][i % 16] = binom(i, m);
            i += 1;
        }
        m += 1;
    }
    t
};

/// `binom(m, k)` at `[m][k]`.
#[cfg(all(target_feature = "avx512f", target_feature = "avx512cd"))]
static PASCAL: [[u32; 16]; PIECES + 1] = {
    let mut t = [[0; 16]; PIECES + 1];
    let mut m = 0;
    while m <= PIECES {
        let mut k = 0;
        while k < 16 {
            t[m][k] = binom(m, k);
            k += 1;
        }
        m += 1;
    }
    t
};

/// `binom(m, k - 1)` at `[m][k]`, which is nothing for `k == 0`.
#[cfg(all(target_feature = "avx512f", target_feature = "avx512cd"))]
static PASCAL_BELOW: [[u32; 16]; PIECES + 1] = {
    let mut t = [[0; 16]; PIECES + 1];
    let mut m = 0;
    while m <= PIECES {
        let mut k = 1;
        while k < 16 {
            t[m][k] = binom(m, k - 1);
            k += 1;
        }
        m += 1;
    }
    t
};

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
        index::{Indexer, InternalIter},
        onitama_simd::{
            iter::{combinations, combinations_old, g, ranking, unranking},
            PawnCount,
        },
    };

    #[test]
    fn probe_runs() {
        for (c0, c1) in [(1u32, 1u32), (2, 2), (3, 3), (3, 2)] {
            let counts = PawnCount { count0: c0, count1: c1 };
            let (mut n, mut runs0, mut runs1) = (0u64, 0u64, 0u64);
            let (mut last0, mut last1) = (u32::MAX, u32::MAX);
            for l in counts {
                n += 1;
                if l.pieces0 != last0 { runs0 += 1; last0 = l.pieces0 }
                if l.pieces1 != last1 { runs1 += 1; last1 = l.pieces1 }
            }
            println!(
                "counts ({c0},{c1}): {n} layouts | pieces0 runs {runs0} (avg {:.1}) | pieces1 runs {runs1} (avg {:.1})",
                n as f64 / runs0 as f64,
                n as f64 / runs1 as f64,
            );
        }
    }

    /// `states` has to agree with counting the king positions of every layout
    /// that are still in play, which is the walk it replaces.
    #[test]
    fn test_states() {
        for (count0, count1) in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)] {
            let counts = PawnCount { count0, count1 };
            let mut walked = 0u64;
            for layout in counts {
                layout.indexer(counts).for_each(|kpos| walked += !kpos.over() as u64);
            }
            assert_eq!(counts.states(), walked, "{counts:?}");
        }
    }

    /// The enumeration has to be in ranking order for `nth_layout` to agree
    /// with walking, which is what the work blocks rely on.
    #[test]
    fn test_nth_layout() {
        for (count0, count1) in [(0, 0), (1, 1), (2, 1), (1, 2), (2, 2), (3, 2)] {
            let counts = PawnCount { count0, count1 };
            for (i, layout) in counts.into_iter().enumerate() {
                assert_eq!(counts.index(&layout), i);
                let seed = counts.nth_layout(i);
                assert_eq!((seed.pieces0, seed.pieces1), (layout.pieces0, layout.pieces1));
            }
        }
    }

    /// The lanes have to rank exactly as the scalar walk does, including a mix
    /// of piece counts in one vector and lanes with no pieces at all.
    #[cfg(all(target_feature = "avx512f", target_feature = "avx512cd"))]
    #[test]
    fn test_ranking16() {
        use std::simd::Simd;
        let mut all = vec![(0u32, 0u32)];
        for (count0, count1) in [(0, 0), (1, 0), (0, 2), (2, 1), (3, 3), (3, 2), (2, 3)] {
            all.extend(PawnCount { count0, count1 }.into_iter().map(|l| (l.pieces0, l.pieces1)));
        }
        // a stride that is prime to everything keeps the counts mixed per vector
        let picked: Vec<_> = (0..all.len()).step_by(7).map(|i| all[i * 7919 % all.len()]).collect();
        for batch in picked.chunks(16) {
            let mut s_1 = [0; 16];
            let mut s_2 = [0; 16];
            for (k, &(a, b)) in batch.iter().enumerate() {
                (s_1[k], s_2[k]) = (a, b);
            }
            let r = super::ranking16(Simd::from_array(s_1), Simd::from_array(s_2));
            for k in 0..16 {
                assert_eq!(r[k] as usize, ranking(s_1[k], s_2[k]), "{:b} {:b}", s_1[k], s_2[k]);
            }
        }
    }

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
