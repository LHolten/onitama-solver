use std::iter::Take;
use std::simd::Simd;

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

/// [`ranking`] of the layouts one move away from a parent, off the parent's.
///
/// Every piece adds a term that depends on its square and on how many pieces
/// of each side are at or below it. When one piece moves within its side, a
/// piece below both squares sees the same pieces either way, and so does one
/// above both. The ones in between see one piece of the mover's side fewer
/// below them if it moved up past them, and one more if it moved down. So the
/// change is the moved piece's own term, old and new, and for the pieces in
/// between their terms with that count shifted, against the plain ones.
///
/// A capture also takes a piece of the other side away, which every piece
/// above it sees, and the `go_up` form of a backward move adds one of the
/// mover's opponent where the mover left: the same again, with the counts
/// shifted the other way, and a term for the piece that came or went.
///
/// The parent's pieces are walked once, into prefix sums of their plain terms
/// and of the shifted ones less the plain: a move is then a few differences of
/// those, and the one or two new terms, with no walk of its own.
pub(super) struct Reranker {
    /// `u32::MAX` while it is for no layout.
    rank: u32,
    /// Both sides' pieces of the parent, and how many.
    all: u32,
    pieces: usize,
    /// Of the parent's lowest `k` pieces, at `k`: per shift of [`SHIFTS`], the
    /// shifted terms less their plain ones, and then the plain terms, in the
    /// lanes of [`TERMS`].
    sums: [Simd<u32, 8>; PIECES + 1],
}

/// What a piece of either side adds to [`ranking`]'s count of the pieces at or
/// below it.
const STEP: [usize; 2] = [27, 27 * 7];

/// The shifts of the counts the moves of either side make, for the pieces they
/// pass or that are above them: either side's count down and up by one, and
/// both changed at once, for a capture or an uncapture moving down past them.
const SHIFTS: [isize; 5] = [-27, 27, -27 * 7, 27 * 7, 27 - 27 * 7];

/// Which of [`SHIFTS`] a move of either side wants: its own count down and up,
/// both at once, and the other side's change above a capture or uncapture,
/// which is the other side down for the first and the first up for the second.
const USES: [[usize; 4]; 2] = [[0, 1, 4, 2], [2, 3, 4, 1]];

/// The lane of [`TERMS`] with the plain term.
const PLAIN: usize = SHIFTS.len();

/// What a piece of a side adds to [`ranking`] at an index of [`COMB`], per side
/// and index: the term with the count moved by every shift of [`SHIFTS`] less
/// the plain one, and the plain one in lane [`PLAIN`]. A shifted count past the
/// table is one no move ever reads, as it would need a piece more than the side
/// has, so it reads zero.
static TERMS: [[[u32; 8]; COMB.len()]; 2] = {
    const fn at(j: isize, side: usize) -> u32 {
        if j < 0 || j as usize >= COMB.len() {
            0
        } else if side == 0 {
            COMB[j as usize].0
        } else {
            COMB[j as usize].1
        }
    }
    let mut t = [[[0; 8]; COMB.len()]; 2];
    let mut side = 0;
    while side < 2 {
        let mut j = 0;
        while j < COMB.len() {
            let plain = at(j as isize, side);
            let mut n = 0;
            while n < SHIFTS.len() {
                t[side][j][n] = at(j as isize + SHIFTS[n], side).wrapping_sub(plain);
                n += 1;
            }
            t[side][j][PLAIN] = plain;
            j += 1;
        }
        side += 1;
    }
    t
};

/// What a piece of `side` on `square` adds to [`ranking`] of `layout`.
#[inline(always)]
fn term(layout: TeamLayout, square: u32, side: usize) -> u32 {
    let upto = (2u32 << square).wrapping_sub(1);
    let ctr = STEP[0] * (layout.pieces0 & upto).count_ones() as usize + STEP[1] * (layout.pieces1 & upto).count_ones() as usize;
    let j = 1 + square as usize + ctr + 27 * (1 + 7);
    debug_assert!(j < COMB.len());
    let x = unsafe { *COMB.get_unchecked(j) };
    if side == 0 { x.0 } else { x.1 }
}

impl Default for Reranker {
    fn default() -> Self {
        Reranker {
            rank: u32::MAX,
            all: 0,
            pieces: 0,
            sums: [Simd::splat(0); PIECES + 1],
        }
    }
}

impl Reranker {
    /// For the moves of either side from `parent`, which is at `rank`.
    #[cfg(test)]
    pub(super) fn new(parent: TeamLayout, rank: usize) -> Self {
        let mut this = Reranker::default();
        this.fill(parent, rank);
        this
    }

    /// Make this the one for the moves of either side from `parent`, which is
    /// at `rank`, in place: it is too large to build and then move.
    pub(super) fn fill(&mut self, parent: TeamLayout, rank: usize) {
        let TeamLayout { pieces0, pieces1 } = parent;
        let all = pieces0 | pieces1;
        self.rank = rank as u32;
        self.all = all;
        self.pieces = all.count_ones() as usize;
        // only the entries up to the piece count are ever read
        self.sums[0] = Simd::splat(0);
        let mut sum = Simd::splat(0);
        let mut ctr = 0;
        for (k, i) in BitIter::from(all).enumerate() {
            let side = (pieces0 & 1 << i == 0) as usize;
            ctr += STEP[side];
            let j = 1 + i + ctr + 27 * (1 + 7);
            // a layout has at most this many pieces, and its counts are in the
            // table
            debug_assert!(k < PIECES && j < COMB.len());
            sum += Simd::from_array(unsafe { *TERMS.get_unchecked(side).get_unchecked(j) });
            unsafe { *self.sums.get_unchecked_mut(k + 1) = sum };
        }
    }

    /// Whether this is for `parent` at `rank`.
    pub(super) fn is_for(&self, parent: TeamLayout, rank: usize) -> bool {
        self.rank == rank as u32 && self.all == parent.pieces0 | parent.pieces1
    }

    /// The shifted terms less the plain ones of the `n`th shift `MOVER` uses,
    /// of the parent's pieces from the `lo`th up to before the `hi`th.
    #[inline(always)]
    fn shifted<const MOVER: usize>(&self, n: usize, lo: usize, hi: usize) -> u32 {
        self.sum(USES[MOVER][n], lo, hi)
    }

    /// Lane `lane` of the sums of the parent's pieces from the `lo`th up to
    /// before the `hi`th.
    #[inline(always)]
    fn sum(&self, lane: usize, lo: usize, hi: usize) -> u32 {
        debug_assert!(lo <= hi && hi <= PIECES);
        unsafe { self.sums.get_unchecked(hi)[lane].wrapping_sub(self.sums.get_unchecked(lo)[lane]) }
    }

    /// The plain term of the parent's `k`th piece.
    #[inline(always)]
    fn own(&self, k: usize) -> u32 {
        self.sum(PLAIN, k, k + 1)
    }

    /// The parent's pieces below `square`, which is the index of the one on it.
    #[inline(always)]
    fn below(&self, square: u32) -> usize {
        (self.all & square.wrapping_sub(1)).count_ones() as usize
    }

    /// The rank of `new`, which differs from the parent by one piece of
    /// `MOVER` moving, by `changed0` and `changed1`: the squares of each side
    /// that changed. That is a move within the side when the other side has
    /// none, and a capture or a `go_up` uncapture otherwise.
    #[inline(always)]
    pub(super) fn rank<const MOVER: usize>(&self, new: TeamLayout, changed0: u32, changed1: u32) -> usize {
        let (mine, other, moved) = if MOVER == 0 {
            (changed0, changed1, new.pieces0)
        } else {
            (changed1, changed0, new.pieces1)
        };
        let to = mine & moved;
        let from = mine & !moved;
        let at = self.below(from);
        let landed = self.below(to);
        let mut r = self.rank.wrapping_sub(self.own(at)).wrapping_add(term(new, to.trailing_zeros(), MOVER));
        if other == 0 {
            // a move within the side
            r = r.wrapping_add(if to > from {
                self.shifted::<MOVER>(0, at + 1, landed)
            } else {
                self.shifted::<MOVER>(1, landed, at)
            });
        } else if MOVER == 0 {
            // a capture on `to`, which is the `landed`th piece of the parent
            r = r.wrapping_sub(self.own(landed));
            r = r.wrapping_add(if to > from {
                self.shifted::<MOVER>(0, at + 1, landed).wrapping_add(self.shifted::<MOVER>(3, landed + 1, self.pieces))
            } else {
                self.shifted::<MOVER>(2, landed + 1, at).wrapping_add(self.shifted::<MOVER>(3, at + 1, self.pieces))
            });
        } else {
            // a piece of the other side put back where this one left
            r = r.wrapping_add(term(new, from.trailing_zeros(), 0));
            r = r.wrapping_add(if to > from {
                self.shifted::<MOVER>(2, at + 1, landed).wrapping_add(self.shifted::<MOVER>(3, landed, self.pieces))
            } else {
                self.shifted::<MOVER>(1, landed, at).wrapping_add(self.shifted::<MOVER>(3, at + 1, self.pieces))
            });
        }
        r as usize
    }
}

/// The most pieces a layout has, both sides and both kings.
const PIECES: usize = 2 * super::SIDE;

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
            iter::{combinations, combinations_old, g, ranking, unranking, Reranker},
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

    /// Every move the solve ranks off a parent has to rank the same as from
    /// scratch: one piece of either side to a free square, a capture by the
    /// first side, and the second side moving with a piece of the first put
    /// back where it left.
    #[test]
    fn test_reranking() {
        use crate::onitama_simd::TeamLayout;
        for (count0, count1) in [(1, 1), (2, 1), (1, 3), (3, 2), (2, 3), (3, 3), (4, 4)] {
            let counts = PawnCount { count0, count1 };
            // the largest only sparsely, as it is most of the layouts
            let stride = if count0 + count1 > 6 { 37 * 41 } else { 37 };
            for (rank, old) in counts.into_iter().enumerate().step_by(stride) {
                assert_eq!(rank, ranking(old.pieces0, old.pieces1));
                let check = |side: usize, new: TeamLayout| {
                    let (c0, c1) = (new.pieces0 ^ old.pieces0, new.pieces1 ^ old.pieces1);
                    let reranker = Reranker::new(old, rank);
                    let got = if side == 0 { reranker.rank::<0>(new, c0, c1) } else { reranker.rank::<1>(new, c0, c1) };
                    assert_eq!(got, ranking(new.pieces0, new.pieces1), "{old:?} {new:?}");
                };
                let free = !(old.pieces0 | old.pieces1) & (1 << 25) - 1;
                for from in 0..25 {
                    for to in (0..25).filter(|&t| free & 1 << t != 0) {
                        let step = 1 << from | 1 << to;
                        if old.pieces0 & 1 << from != 0 {
                            check(0, TeamLayout { pieces0: old.pieces0 ^ step, pieces1: old.pieces1 });
                        }
                        if old.pieces1 & 1 << from != 0 {
                            check(1, TeamLayout { pieces0: old.pieces0, pieces1: old.pieces1 ^ step });
                            check(1, TeamLayout { pieces0: old.pieces0 | 1 << from, pieces1: old.pieces1 ^ step });
                        }
                    }
                    for to in (0..25).filter(|&t| old.pieces1 & 1 << t != 0) {
                        if old.pieces0 & 1 << from != 0 && old.pieces1.count_ones() > 1 {
                            let step = 1 << from | 1 << to;
                            check(0, TeamLayout { pieces0: old.pieces0 ^ step, pieces1: old.pieces1 & !(1 << to) });
                        }
                    }
                }
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
