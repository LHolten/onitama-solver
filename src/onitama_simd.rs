#![allow(dead_code)]
mod accum_spread;
mod iter;
mod job;
mod update;

use std::{
    alloc::Layout,
    fmt,
    iter::zip,
    mem::size_of,
    ops::{BitAnd, Deref, Index, IndexMut},
    sync::atomic::{AtomicU32, AtomicU64, Ordering},
    time::Instant,
};

use bit_iter::BitIter;

use crate::{
    card::{get_one_bitmap, offset_mask_fixed as offset_mask},
    index::{Empty, Indexer, InternalIter},
    proj,
};

pub const TABLE_MASK: u32 = (1 << 25) - 1;
pub const BLOCK_MASK: u32 = (1 << 30) - 1;
pub const RESOLVED_BIT: u32 = 1 << 30;

fn count_indexer(size: u32) -> impl Indexer<Item = PawnCount> {
    let count_mask = (1u32 << size) - 1;
    Empty::default()
        .choose_one(proj!(|c: PawnCount| c.count0), (count_mask, size))
        .choose_one(proj!(|c: PawnCount| c.count1), (count_mask, size))
}

// number of pawns of each player
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct PawnCount {
    pub(crate) count0: u32,
    pub(crate) count1: u32,
}

impl PawnCount {
    fn invert(self) -> Self {
        Self {
            count0: self.count1,
            count1: self.count0,
        }
    }
}

// only contains erased piece positions
// we don't know which pieces are the kings
#[derive(Debug, Default, Clone, Copy)]
pub struct TeamLayout {
    pub(crate) pieces0: u32,
    pub(crate) pieces1: u32,
}

impl TeamLayout {
    fn indexer(self, counts: PawnCount) -> impl Indexer<Item = KingPos> {
        debug_assert_eq!(self.pieces0.count_ones(), counts.count0 + 1);
        debug_assert_eq!(self.pieces1.count_ones(), counts.count1 + 1);

        Empty::default()
            .choose_one(
                proj!(|p: KingPos| p.king0),
                (
                    self.pieces0 & !(1 << 22),
                    counts.count0 + (self.pieces0 & 1 << 22 == 0) as u32,
                ),
            )
            .choose_one(
                proj!(|p: KingPos| p.king1),
                (
                    self.pieces1 & !(1 << 2),
                    counts.count1 + (self.pieces1 & 1 << 2 == 0) as u32,
                ),
            )
    }

    fn counts(self) -> PawnCount {
        PawnCount {
            count0: self.pieces0.count_ones() - 1,
            count1: self.pieces1.count_ones() - 1,
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
pub struct KingPos {
    king0: u32,
    king1: u32,
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
pub struct AllTables {
    size: u32,
    cards: Cards,
    mask_lookup: [u32; 25],
    directions: u32,
    list: Box<[Table]>,
    pub block_done: AtomicU64,
    pub block_not_done: AtomicU64,
    pub card_done: AtomicU64,
    pub card_not_done: AtomicU64,
    pub win_in1: u64,
}

impl AllTables {
    fn index_count(&self, counts: PawnCount) -> &Table {
        let indexer = count_indexer(self.size);
        let i = indexer.index(&counts);
        unsafe { self.list.get(i).unwrap_unchecked() }
    }

    pub fn count_ones(&self) -> u64 {
        self.list.iter().map(|x| x.count_ones()).sum()
    }

    pub fn len(&self) -> u64 {
        let mut total = 0;
        for table in self.list.iter() {
            for layout in table.counts {
                total += layout.indexer(table.counts).total() as u64;
            }
        }
        total
    }
}

/// The size of a cache line.
///
/// A chunk (all king positions of one layout) is read and written as a unit, so
/// aligning the table to this means a chunk that fits in a line only touches
/// one, instead of straddling two.
const LINE: usize = 64;

/// The size of a huge page.
const HUGE: usize = 2 << 20;

/// A zeroed, cache line aligned `[AtomicU32]`.
///
/// Tables that are at least a huge page big get aligned to one instead, and ask
/// to be backed by huge pages: they are read in essentially random order, so
/// with 4K pages almost every lookup costs a page walk.
pub struct AlignedTable {
    ptr: *mut AtomicU32,
    len: usize,
}

// the contents are atomics, so sharing them across threads is fine
unsafe impl Send for AlignedTable {}
unsafe impl Sync for AlignedTable {}

impl AlignedTable {
    fn size(len: usize) -> usize {
        len * size_of::<AtomicU32>()
    }

    fn layout(len: usize) -> Layout {
        Layout::from_size_align(Self::size(len), LINE).unwrap()
    }

    /// Map the table directly instead of going through the allocator.
    ///
    /// `alloc_zeroed` only reaches `calloc` for alignments up to `MIN_ALIGN`
    /// (16); above that it allocates and then memsets, which both costs
    /// gigabytes of writes up front and faults every page in as a 4K page
    /// before `madvise` gets a chance to say otherwise. A fresh anonymous
    /// mapping is already zeroed, page aligned and lazily faulted.
    #[cfg(target_os = "linux")]
    fn new_zeroed(len: usize) -> Self {
        let size = Self::size(len);
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(
            ptr != libc::MAP_FAILED,
            "could not map table of {len} entries"
        );

        // must happen before the pages are touched, otherwise they are already
        // split into 4K pages and only khugepaged can undo that. This is a
        // hint, so there is nothing useful to do if the kernel declines.
        if size >= HUGE {
            unsafe { libc::madvise(ptr, size, libc::MADV_HUGEPAGE) };
        }

        Self {
            ptr: ptr as *mut AtomicU32,
            len,
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn new_zeroed(len: usize) -> Self {
        let ptr = unsafe { std::alloc::alloc_zeroed(Self::layout(len)) } as *mut AtomicU32;
        assert!(!ptr.is_null(), "could not allocate table of {len} entries");
        Self { ptr, len }
    }
}

impl Drop for AlignedTable {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, Self::size(self.len));
        }
        #[cfg(not(target_os = "linux"))]
        unsafe {
            std::alloc::dealloc(self.ptr as *mut u8, Self::layout(self.len))
        }
    }
}

impl Deref for AlignedTable {
    type Target = [AtomicU32];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl fmt::Debug for AlignedTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deref().fmt(f)
    }
}

#[derive(Debug)]
pub struct Table {
    counts: PawnCount,
    chunk_size: usize,
    list: AlignedTable,
}

impl Table {
    fn index(&self, layout: TeamLayout) -> SubTable<'_> {
        let i = self.counts.index(&layout);

        let slice = self
            .list
            .get(self.chunk_size * i..self.chunk_size * (i + 1));
        let slice = unsafe { slice.unwrap_unchecked() };
        SubTable {
            layout,
            slice,
            counts: self.counts,
        }
    }

    fn count_ones(&self) -> u64 {
        self.list
            .iter()
            .map(|x| x.load(Ordering::Relaxed).bitand(BLOCK_MASK).count_ones() as u64)
            .sum()
    }
}

// contains results for only one layout
#[derive(Debug, Clone, Copy)]
pub struct SubTable<'a> {
    counts: PawnCount,
    layout: TeamLayout,
    slice: &'a [AtomicU32],
}

impl Index<KingPos> for SubTable<'_> {
    type Output = AtomicU32;

    fn index(&self, index: KingPos) -> &Self::Output {
        let indexer = self.layout.indexer(self.counts);
        let i = indexer.index(&index);
        unsafe { self.slice.get(i).unwrap_unchecked() }
    }
}

#[derive(Debug)]
struct KingLookup {
    list: [[u8; 25]; 25],
}

impl Index<KingPos> for KingLookup {
    type Output = u8;

    fn index(&self, index: KingPos) -> &Self::Output {
        let t = unsafe { self.list.get(index.king0 as usize).unwrap_unchecked() };
        unsafe { t.get(index.king1 as usize).unwrap_unchecked() }
    }
}

impl IndexMut<KingPos> for KingLookup {
    fn index_mut(&mut self, index: KingPos) -> &mut Self::Output {
        let t = unsafe { self.list.get_mut(index.king0 as usize).unwrap_unchecked() };
        unsafe { t.get_mut(index.king1 as usize).unwrap_unchecked() }
    }
}

pub struct LocalMem {
    wins: Vec<u32>,
    status: Vec<u32>,
    /// Per king position, `status` and the not won states expanded for the
    /// card mask of the current move group. Interleaved because [`Spread`]
    /// wants both at the same index. See [`Spread::expanded`].
    expanded: Vec<(u32, u32)>,
    king_lookup: KingLookup,
    moves: Vec<PreparedMove>,
}

impl LocalMem {
    const fn new() -> Self {
        Self {
            wins: vec![],
            status: vec![],
            expanded: vec![],
            king_lookup: KingLookup {
                list: [[0; 25]; 25],
            },
            moves: vec![],
        }
    }
}

/// A move whose target chunk has already been located.
///
/// Locating it means running the multiset [`iter::ranking`], so the address of
/// the (usually DRAM resident) chunk is only known at the very end of a long
/// dependency chain. Resolving every move of a layout up front lets all of
/// those misses be prefetched and overlap with each other.
pub struct PreparedMove {
    /// Which bit of `TableJob::live` this move owns. Kept on the move itself so
    /// that skipping one during preparation does not shift the rest.
    bit: u32,
    step: (usize, usize),
    mask: u32,
    new: TeamLayout,
    counts: PawnCount,
    // borrows from the tables in `AllTables`, which outlive every update
    target: *const [AtomicU32],
}

impl PreparedMove {
    fn target(&self) -> &[AtomicU32] {
        unsafe { &*self.target }
    }
}

#[inline(always)]
fn prefetch(slice: &[AtomicU32]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        let ptr = slice.as_ptr();
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
        // a chunk is at most 64 bytes, so it spans at most two lines
        _mm_prefetch(ptr.wrapping_add(slice.len() - 1) as *const i8, _MM_HINT_T0);
    }
}

pub struct ImmutableUpdate<'a> {
    inv_current: &'a Table,
    current: &'a Table,
    take_one: Option<&'a Table>,
    leave_one: Option<&'a Table>,
    go_up: bool,
    mask_lookup: &'a [u32; 25],
    directions: u32,
}

pub struct Update<'a> {
    layout: TeamLayout,
    immutable: &'a ImmutableUpdate<'a>,
    mem: &'a mut LocalMem,
}

#[derive(Debug)]
pub struct Accum<'a> {
    new: TeamLayout,
    counts: PawnCount,
    mask: u32,
    step: (usize, usize),
    target: &'a [AtomicU32],
    slice: &'a mut [u32],
    king_lookup: &'a KingLookup,
}

pub struct Spread<'a> {
    new: TeamLayout,
    counts: PawnCount,
    step: (usize, usize),
    target: &'a [AtomicU32],
    /// Per king position, what to spread and the most that could ever be
    /// spread here.
    ///
    /// The second is not just what it spreads now: a state only ever goes from
    /// unresolved to won or to lost, so the losses it can still gain are
    /// bounded by the states that are not won. That bound shrinks over time
    /// while the target only gains bits, so once the target subsumes it the
    /// chunk is done contributing, for good.
    expanded: &'a [(u32, u32)],
    king_lookup: &'a KingLookup,
}

/// What a [`Spread`] did, and whether it could ever do anything again.
pub struct SpreadStatus {
    pub(crate) progress: bool,
    pub(crate) could: bool,
}

impl AllTables {
    pub fn ez_win_for_each(
        &self,
        counts: PawnCount,
        layout: TeamLayout,
        f: &mut impl FnMut(usize, u32),
    ) {
        let pieces1 = layout.pieces1;

        for (card, mask) in zip(self.cards.iter(), mask_iter()) {
            let bitmap = card.bitmap::<false>();
            // from where can you attack the temple?
            let from_mask = offset_mask(2, bitmap);

            layout.indexer(counts).for_enumerate(|i, kpos| {
                if 1 << kpos.king1 & from_mask == 0 || 1 << 2 & pieces1 != 0 {
                    // no attack on temple
                    let from_mask = offset_mask(kpos.king0 as usize, bitmap);
                    if from_mask & pieces1 == 0 {
                        // no attack on king
                        return;
                    }
                }
                f(i, mask);
            })
        }
    }

    pub fn build(size: u32, cards: u16) -> Self {
        let mut mask_lookup = [0; 25];
        let mut directions = 0;
        for (mask, card) in zip(mask_iter(), Cards(cards).iter()) {
            for offset in BitIter::from(card.bitmap::<false>()) {
                mask_lookup[offset] |= mask
            }
            directions |= card.bitmap::<false>();
        }

        let mut tb = Self {
            size,
            cards: Cards(cards),
            mask_lookup,
            directions,
            list: count_indexer(size)
                .into_iter()
                .map(|counts: PawnCount| {
                    let chunk_size = (counts.count0 + 1) as usize * (counts.count1 + 1) as usize;
                    let num_chunks = counts.total();
                    let len = chunk_size * num_chunks;
                    let list = AlignedTable::new_zeroed(len);
                    Table {
                        counts,
                        chunk_size,
                        list,
                    }
                })
                .collect(),
            block_done: Default::default(),
            block_not_done: Default::default(),
            card_done: Default::default(),
            card_not_done: Default::default(),
            win_in1: 0,
        };

        let mut win_in1 = 0;
        let mut schedule = vec![];
        for counts in count_indexer(size) {
            let counts: PawnCount = counts;
            if counts.count0 < counts.count1 {
                continue;
            }
            let mut jobs = vec![TableJob::new(&tb, counts)];
            if counts.count0 > counts.count1 {
                jobs.push(TableJob::new(&tb, counts.invert()));
            }
            for job in &jobs {
                job.mark_ez_win();
                win_in1 += job.update.current.count_ones();
            }
            schedule.push(jobs)
        }

        for mut jobs in schedule {
            let mut any_progress = true;
            let mut iters = 0;
            while any_progress {
                any_progress = false;
                for job in &mut jobs {
                    any_progress |= job.next().is_some();
                }
                iters += 1;
            }


            println!(
                "finished {:?} in {iters} iterations",
                jobs[0].update.current.counts
            );
        }

        tb.win_in1 = win_in1;
        tb
    }
}

/// Layouts per work block.
///
/// Each block is one parallel task and one recorded seed, so this trades
/// scheduling overhead against how much of the enumeration has to be replayed
/// to reach the live layouts of a sparse late pass.
const BLOCK: usize = 1024;

struct TableJob<'a> {
    tb: &'a AllTables,
    /// What the positions in `live` are indices into, so a live one can be
    /// unranked back into a layout.
    counts: PawnCount,
    /// Per layout, one bit per backward move: still worth running. A bit is
    /// cleared once that move's target has subsumed everything this chunk
    /// could ever spread there, and a layout is in the work list exactly while
    /// any of its bits is set, so this is the work list and the per move state
    /// in one. Moves past bit 31 are never skipped; there are at most 32 for
    /// the sizes that fit in memory anyway, since each of the 8 offsets
    /// contributes at most one per piece.
    live: Vec<u32>,
    /// One bit per layout: retired, but its capture predecessors still have to
    /// be written by the `go_up` pass.
    needs_up: Vec<u64>,
    total: usize,
    update: ImmutableUpdate<'a>,
    done: bool,
}

fn mask_iter() -> impl Iterator<Item = u32> {
    let mut mask = 0b000000_000000_010100_000000_101011;
    std::iter::repeat_with(move || {
        let res = mask;
        mask = mask << 6 | mask >> 24;
        res & BLOCK_MASK
    })
}

#[derive(Debug, Clone, Copy)]
struct Block(u32);

impl Block {
    pub fn invert(self) -> Self {
        let data = self.0;
        const MASK0: u32 = 0b001001_001001_001001_001001_001001;
        const MASK1: u32 = 0b000111_000111_000111_000111_000111;
        let data = (data & MASK0 << 1) << 1 | (data & MASK0 << 2) >> 1 | (data & MASK0);
        let data = (data & MASK1) << 3 | (data & MASK1 << 3) >> 3;
        Self(data)
    }

    pub fn expand(self) -> Self {
        let data = self.0 as u64;
        let data = data << 10 | data << 20;
        let data = data | data >> 30;
        Self(data as u32 & BLOCK_MASK)
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

fn pretty(layout: TeamLayout, kpos: KingPos) {
    let TeamLayout { pieces0, pieces1 } = layout;
    let KingPos { king0, king1 } = kpos;
    debug_assert_eq!(pieces0 & pieces1, 0);
    debug_assert_ne!(1 << king0 & pieces0, 0);
    debug_assert_ne!(1 << king1 & pieces1, 0);

    println!("----- x side");
    for y in 0..5 {
        for x in 0..5 {
            let i = 24 - 5 * y - x;
            if king0 == i {
                print!("O")
            } else if king1 == i {
                print!("X")
            } else if 1 << i & pieces0 != 0 {
                print!("o")
            } else if 1 << i & pieces1 != 0 {
                print!("x")
            } else {
                print!(".")
            }
        }
        println!()
    }
    println!("----- o side")
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use crate::onitama_simd::Block;

    use super::{mask_iter, AllTables, PawnCount};

    #[test]
    fn build_tb() {
        let tb = AllTables::build(3, 0b11111);
        let wins = tb.count_ones();
        let total = tb.len() * 30;
        println!("{wins} total wins");
        println!("{} wins in 1", tb.win_in1);
        println!("{} not win in 1", total - tb.win_in1);
        let unresolved = tb.count_unresolved();
        println!("{unresolved} unresolved states");
        println!(
            "{} resolved, not win in 1",
            total - tb.win_in1 - unresolved
        );
        println!(
            "{} blocks done, {} blocks not done",
            tb.block_done.load(Ordering::Relaxed),
            tb.block_not_done.load(Ordering::Relaxed)
        );
        println!(
            "{} cards done, {} cards not done",
            tb.card_done.load(Ordering::Relaxed),
            tb.card_not_done.load(Ordering::Relaxed)
        );
        // assert_eq!(wins, 6752579);
        assert_eq!(wins, 831344251);
        // assert_eq!(wins, 27126107221);
    }

    #[test]
    fn counts0() {
        for layout in PawnCount::default() {
            dbg!(layout);
        }
    }

    #[test]
    fn mask_test() {
        for mask in mask_iter().take(5) {
            assert_eq!(mask, Block(mask).invert().0)
        }
    }
}
