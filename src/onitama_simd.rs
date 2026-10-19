pub mod accum_spread;
mod iter;
mod job;
mod update;

use std::{
    fmt,
    iter::zip,
    mem::size_of,
    ops::{BitAnd, Deref, Index},
    simd::Simd,
    sync::atomic::{AtomicU32, Ordering},
};

#[cfg(feature = "parallell")]
use rayon::prelude::*;

use bit_iter::BitIter;

use crate::{
    card::{get_one_bitmap, offset_mask},
    index::{Empty, Indexer},
    proj,
};

pub const TABLE_MASK: u32 = (1 << 25) - 1;
pub const BLOCK_MASK: u32 = (1 << 30) - 1;

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
            .choose_one(proj!(|p: KingPos| p.king0), (self.pieces0, counts.count0 + 1))
            .choose_one(proj!(|p: KingPos| p.king1), (self.pieces1, counts.count1 + 1))
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

#[cfg(test)]
impl TeamLayout {
    fn counts(self) -> PawnCount {
        PawnCount {
            count0: self.pieces0.count_ones() - 1,
            count1: self.pieces1.count_ones() - 1,
        }
    }
}

#[cfg(test)]
impl KingPos {
    /// Whether a king stands on the opponent's temple, so the game is already
    /// over.
    ///
    /// These still get a slot, which keeps every chunk of a table the full
    /// rectangle of king positions, but no state still in play depends on what
    /// it holds. It holds all ones from the seed on, which is what keeps it out
    /// of the way: see [`Seeds`].
    fn over(self) -> bool {
        self.king0 == 22 || self.king1 == 2
    }
}

// contains all the results up to some number of pieces
pub struct AllTables {
    size: u32,
    /// The most pieces a table built here has, both kings counted: see
    /// [`AllTables::build_upto`].
    pieces: u32,
    mask_lookup: [u32; 25],
    directions: u32,
    seeds: Seeds,
    list: Box<[Table]>,
}

impl AllTables {
    fn index_count(&self, counts: PawnCount) -> &Table {
        let indexer = count_indexer(self.size);
        let i = indexer.index(&counts);
        unsafe { self.list.get(i).unwrap_unchecked() }
    }

    pub fn count_ones(&self) -> u64 {
        self.built().map(|x| x.count_ones()).sum()
    }

    pub fn len(&self) -> u64 {
        self.built().map(|table| table.counts.states()).sum()
    }

    /// Whether the table for these counts is one this was built with.
    fn has(&self, counts: PawnCount) -> bool {
        counts.count0 + counts.count1 + 2 <= self.pieces
    }

    /// The tables this was built with, which are all of them unless it was
    /// built with a limit.
    fn built(&self) -> impl Iterator<Item = &Table> {
        self.list.iter().filter(|table| self.has(table.counts))
    }

    /// The won states per table, for every table this was built with.
    #[cfg(test)]
    fn table_wins(&self) -> Vec<((u32, u32), u64)> {
        self.built()
            .map(|table| ((table.counts.count0, table.counts.count1), table.count_ones()))
            .collect()
    }
}

/// The size of a huge page.
const HUGE: usize = 2 << 20;

/// The most pieces a side has, which is all Onitama gives it: a chunk is at
/// most this many king positions each way.
const SIDE: usize = 5;

/// Slots of a chunk the walks over all of it take at a time, which is one
/// vector of words.
///
/// The local buffers hold a chunk the way the table does, row after row at
/// its own width, so up to eight men a chunk is one of these and at ten two.
const LANES: usize = 16;

/// The most slots a chunk has, rounded up to whole [`LANES`].
const SLOTS: usize = 2 * LANES;

/// The lanes of a row in [`Spread::spreadout`], which is the vector a row of a
/// chunk fits in at any width. Also the slack a table carries past its end.
const ROW: usize = 8;
const _: () = assert!(2 * ROW == LANES && SIDE <= ROW);

/// The row and the column of every slot of a chunk, per row length, for the
/// walks that take a chunk [`LANES`] slots at a time.
static LANE_ROW: [[u32; SLOTS]; SIDE + 1] = lanes(true);
static LANE_COL: [[u32; SLOTS]; SIDE + 1] = lanes(false);

const fn lanes(row: bool) -> [[u32; SLOTS]; SIDE + 1] {
    let mut table = [[0; SLOTS]; SIDE + 1];
    let mut width = 1;
    while width <= SIDE {
        let mut slot = 0;
        while slot < SLOTS {
            table[width][slot] = (if row { slot / width } else { slot % width }) as u32;
            slot += 1;
        }
        width += 1;
    }
    table
}

/// Slots the vectorised walks over the local buffers take at a time.
const GROUP: usize = 8;

/// A buffer for a chunk, aligned so it starts on a line.
///
/// Far longer than a chunk can need, which is not slack for it but for the
/// vectoriser: the length bounds how long it can tell the walks over this might
/// run, and under eight groups it decides a whole vector at a time is not
/// worth it.
///
/// That bound is the whole reason for the length, so it is the shortest one
/// that clears it rather than anything tuned. It is worth re-checking that the
/// wide loop is still there after a toolchain bump; being wrong about it is a
/// slowdown and nothing else.
#[repr(align(64))]
struct Padded<T>([T; 8 * GROUP]);

impl<T> std::ops::Deref for Padded<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Padded<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

/// Where the kernel reports whether transparent huge pages are allowed.
#[cfg(target_os = "linux")]
const THP_SETTING: &str = "/sys/kernel/mm/transparent_hugepage/enabled";

/// Whether the kernel will hand out transparent huge pages at all.
///
/// The file names every mode and brackets the active one, so `madvise` and
/// `always` both leave this build with huge pages and only `never` does not.
#[cfg(target_os = "linux")]
fn thp_allowed(setting: &str) -> bool {
    !setting.contains("[never]")
}

/// Refuse to run without transparent huge pages.
///
/// `madvise` will not say so: under `never` it still sets the flag and simply
/// never collapses anything, and the tables then run several times slower on
/// page walks alone. That is a slowdown rather than a failure, which is exactly
/// the kind of thing that gets measured without being noticed, so check the
/// setting that `madvise` is quiet about and stop instead.
#[cfg(target_os = "linux")]
fn require_hugepages() {
    let setting = std::fs::read_to_string(THP_SETTING).unwrap_or_else(|err| {
        panic!("this needs transparent huge pages, and {THP_SETTING} is unreadable: {err}")
    });
    assert!(
        thp_allowed(&setting),
        "this needs transparent huge pages, and {THP_SETTING} is `{}`; \
         set it to `madvise` or `always`",
        setting.trim()
    );
}

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

    /// What is actually mapped: a vector of slack, so a read that runs off the
    /// end of the last chunk still lands inside the mapping.
    fn map_size(len: usize) -> usize {
        Self::size(len + ROW)
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
        let size = Self::map_size(len);
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
        // split into 4K pages and only khugepaged can undo that
        if size >= HUGE {
            require_hugepages();
            let err = unsafe { libc::madvise(ptr, size, libc::MADV_HUGEPAGE) };
            assert_eq!(
                err,
                0,
                "could not ask for huge pages: {}",
                std::io::Error::last_os_error()
            );
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
            libc::munmap(self.ptr as *mut libc::c_void, Self::map_size(self.len));
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

    /// The chunk at position `rank`, which is what [`locate`] works out.
    fn chunk(&self, rank: usize) -> &[AtomicU32] {
        // a rank is of a layout of this table, so its chunk is in the list
        debug_assert!(self.chunk_size * (rank + 1) <= self.list.len());
        unsafe { self.list.get_unchecked(self.chunk_size * rank..self.chunk_size * (rank + 1)) }
    }

    /// The won states, of the ones still in play.
    ///
    /// The slots of a game that is already over hold all ones, and they are
    /// every slot [`PawnCount::states`] leaves out, so they come off as a
    /// count instead of being picked out one by one.
    fn count_ones(&self) -> u64 {
        let all: u64 = self
            .list
            .iter()
            .map(|x| x.load(Ordering::Relaxed).bitand(BLOCK_MASK).count_ones() as u64)
            .sum();
        let over = self.list.len() as u64 - self.counts.states();
        all - over * BLOCK_MASK.count_ones() as u64
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

/// The scratch every chunk is built in, held per thread.
pub struct LocalMem {
    wins: Padded<u32>,
    status: Padded<u32>,
    /// Per king position, `status` and the not won states expanded for the
    /// card mask of the current move group, a vector per row.
    ///
    /// The losses this slot spreads in the first [`ROW`] lanes, a column each,
    /// and the most it could ever spread beyond those in the next, because
    /// [`Spread`] tests both against the target row at once. See
    /// [`Spread::expanded`].
    expanded: [Simd<u32, LANES>; SIDE],
}

/// The moves of the updates in flight, which borrow from the tables.
///
/// That keeps them out of [`LocalMem`], which lives in a thread local and so
/// cannot hold a borrow of anything: these are made per work block instead,
/// which is often enough for the allocation not to matter.
#[derive(Default)]
pub struct Moves<'t> {
    /// The forward moves, of this update or, once its accumulate is done, of
    /// the next one: see [`Update::update_layout`].
    forward: Vec<PreparedMove<'t>>,
    /// The backward moves, kept apart from `forward` so that locating them can
    /// run before the accumulate rather than after it. See
    /// [`Update::prepare_spread`].
    spread: Vec<PreparedMove<'t>>,
    /// The ranking of the layout whose forward moves were located last, which
    /// its backward moves are located with in the next update: see
    /// [`iter::Reranker`].
    reranker: iter::Reranker,
}

impl LocalMem {
    const fn new() -> Self {
        Self {
            wins: Padded([0; 8 * GROUP]),
            status: Padded([0; 8 * GROUP]),
            expanded: [Simd::from_array([0; LANES]); SIDE],
        }
    }
}

/// A move whose target chunk has already been located.
///
/// Locating it means running the multiset [`iter::ranking`], so the address of
/// the (usually DRAM resident) chunk is only known at the very end of a long
/// dependency chain. Resolving every move of a layout up front lets all of
/// those misses be prefetched and overlap with each other.
pub struct PreparedMove<'t> {
    /// Which bit of `TableJob::live` this move owns, possibly shared with
    /// another move. Kept on the move itself so that skipping one during
    /// preparation does not shift the rest.
    bit: u32,
    step: (usize, usize),
    mask: u32,
    new: TeamLayout,
    /// The source columns that read the target column of the same number
    /// rather than the one after, which are the ones below this. A property of
    /// the column, so the same in every row. Only the accumulate reads it.
    split: u32,
    /// The table the target is in, and the start of the target itself once
    /// [`locate`] has worked out where in it: the length is the table's, and
    /// a pointer is one word less to write per move.
    table: &'t Table,
    target: *const AtomicU32,
}

impl<'t> PreparedMove<'t> {
    fn target(&self) -> &'t [AtomicU32] {
        debug_assert!(!self.target.is_null());
        unsafe { std::slice::from_raw_parts(self.target, self.table.chunk_size) }
    }

    /// Point `target` at chunk `rank` of the table, and start the miss on it.
    #[inline(always)]
    fn offset(&mut self, rank: usize) {
        let chunk = self.table.chunk(rank);
        self.target = chunk.as_ptr();
        prefetch(chunk);
    }

    /// The target chunk as plain words.
    ///
    /// The accumulate takes a whole padded row at a time, which no atomic load
    /// can do. The values only ever gain bits and the walk already reads them
    /// relaxed, so a lane is either what a concurrent `fetch_or` left or what
    /// it found, which is what a relaxed load would have given anyway.
    fn src(&self) -> *const u32 {
        self.target().as_ptr() as *const u32
    }
}

/// Point every move at its target chunk, and prefetch it.
///
/// Each `target` leaves as the chunk of `new` in the move's table. `parent` is
/// the layout the moves are from, at `rank` in its own table, and `MOVER` the
/// side whose pieces they move, which is everything ranking a move off the
/// parent needs: see [`iter::Reranker`].
///
/// That also hands out the moves one at a time, with a ranking each between
/// their prefetches. Ranking sixteen at once in vectors was cheaper, but sent
/// the prefetches out in one burst, and they found their lines in L2 a
/// quarter less often (measured at 8 men on Zen 5).
fn locate<const MOVER: usize>(moves: &mut [PreparedMove<'_>], reranker: &mut iter::Reranker, parent: TeamLayout, rank: usize) {
    if !reranker.is_for(parent, rank) {
        reranker.fill(parent, rank);
    }
    for mv in moves {
        let (changed0, changed1) = (mv.new.pieces0 ^ parent.pieces0, mv.new.pieces1 ^ parent.pieces1);
        mv.offset(reranker.rank::<MOVER>(mv.new, changed0, changed1));
    }
}

#[inline(always)]
fn prefetch(slice: &[AtomicU32]) {
    use std::hint::{prefetch_read, Locality};
    let ptr = slice.as_ptr();
    prefetch_read(ptr, Locality::L1);
    // Up to eight men a chunk is at most 64 bytes, so it spans at most two
    // lines. At ten it can take three, and the middle one is left to the
    // demand miss: that costs time, not results.
    prefetch_read(ptr.wrapping_add(slice.len() - 1), Locality::L1);
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

/// What the solve is seeded with, per card: the states won outright, and some
/// of the ones won in two.
///
/// The wins in one are the ones where the side to move can step its king onto
/// the opponent's temple or take the opponent's king. The wins in two are the
/// ones where it gets its king within a step of the opponent's temple and the
/// opponent can not stop it: see [`Self::run`]. They
/// cost nothing but the seed's own work, since all they need is the layout,
/// and they save the passes that would otherwise find them. Then there are the
/// ones where the game is already
/// [`over`](KingPos::over), which count as won with every card. That last is
/// not true of the game, but nothing still in play can tell:
///
/// - The only move from a state in play into such a slot is a king onto the
///   opponent's temple, and that state is seeded as won for every card the
///   move is made with. Reading all ones there is the same as not reading it.
/// - Its mirror is another such slot, so a chunk's game over rows read all
///   ones for their wins. That leaves them nothing lost to spread and nothing
///   they could still spread, so they retire with the rest.
/// - A spread into one finds every bit already there.
///
/// And all ones is something the counts can take back out exactly: see
/// [`Table::count_ones`].
///
/// Every one of these is made of terms of where `pieces1`'s king stands and
/// of where the other king does, so a chunk takes a few per column and a few
/// per row instead of a walk over every king position per card.
pub struct Seeds([SeedCard; 5]);

#[derive(Clone, Copy, Default)]
struct SeedCard {
    /// The squares `pieces0` reaches from the centre with this card, and the
    /// ones `pieces1` does, which is the card turned around.
    to: u32,
    from: u32,
    /// From where `pieces1`'s king steps onto the opponent's temple with it.
    temple: u32,
    /// The configurations `pieces1`, to move, holds the card in, the ones it
    /// is the side card in, and the ones the opponent holds it in.
    mover: u32,
    side: u32,
    opponent: u32,
}

impl Seeds {
    fn new(cards: Cards) -> Self {
        let mut seeds = [SeedCard::default(); 5];
        for (seed, (card, side)) in seeds.iter_mut().zip(zip(cards.iter(), mask_iter())) {
            // the raw mask is the configurations where it is the side card,
            // seen from the side that just moved
            let mover = Block(side).invert().expand().0;
            let side = Block(side).invert().0;
            *seed = SeedCard {
                to: card.bitmap::<false>(),
                from: card.bitmap::<true>(),
                temple: offset_mask(2, card.bitmap::<false>()),
                mover,
                side,
                opponent: BLOCK_MASK & !mover & !side,
            };
        }
        Self(seeds)
    }

    /// The configurations where the opponent can not win right away by
    /// stepping onto our temple, with its king on `king0`.
    ///
    /// This is the half of a win in two that depends on the row: see
    /// [`Self::run`].
    fn quiet(&self, pieces0: u32, king0: usize) -> u32 {
        let mut lost = 0;
        if pieces0 & 1 << 22 == 0 {
            for card in &self.0 {
                if offset_mask(king0, card.to) & 1 << 22 != 0 {
                    lost |= card.opponent;
                }
            }
        }
        BLOCK_MASK & !lost
    }

    /// The configurations where no piece of the opponent can take a king on
    /// `square`.
    fn safe(&self, pieces0: u32, square: usize) -> u32 {
        let mut lost = 0;
        for card in &self.0 {
            if offset_mask(square, card.from) & pieces0 != 0 {
                lost |= card.opponent;
            }
        }
        BLOCK_MASK & !lost
    }

    /// The half of a win in two that depends on the column: `pieces1`'s king
    /// on `king1` gets to where it reaches the opponent's temple next turn,
    /// with a card it will hold then, and the opponent can not take it there.
    ///
    /// Then the opponent has nothing. Every move is a jump, so nothing can be
    /// put in the way, and an opponent piece standing on its own temple is
    /// just taken by the king on the way in. It can only win first, by taking the
    /// king, which this rules out, or by stepping onto our temple, which
    /// [`Self::quiet`] does. The card played goes to the side, and the one
    /// that was there comes into the hand, so next turn's cards are the one
    /// not played and the old side card.
    ///
    /// Either the king walks there, or it is there already and only needs the
    /// side card, and another piece passes the turn with a card of the hand.
    /// Taking a piece on the way only ever takes away from what the opponent
    /// can do, except taking one that stands on our temple, which would let
    /// the opponent's king onto it. A passing move is kept from that, and a
    /// king that walks never gets near our temple: it ends next to the
    /// opponent's.
    fn run(&self, pieces0: u32, pieces1: u32, king1: usize) -> u32 {
        // a piece of our own on the opponent's temple is in the way, and
        // nothing can move it off and have the king arrive in the same turn
        if king1 == 2 || pieces1 & 1 << 2 != 0 {
            return 0;
        }
        let sources = self.0.iter().fold(0, |acc, card| acc | card.temple);
        let mut win = 0;

        for (u, played) in self.0.iter().enumerate() {
            let walks = offset_mask(king1, played.from) & sources & !pieces1;
            for square in BitIter::from(walks) {
                // the cards that take the king on from there next turn
                let mut next = 0;
                for (y, card) in self.0.iter().enumerate() {
                    if y != u && 1 << square & card.temple != 0 {
                        next |= card.mover | card.side;
                    }
                }
                if next != 0 {
                    win |= played.mover & next & self.safe(pieces0, square);
                }
            }
        }

        // it is there already, but only with the side card
        let mut side = 0;
        for card in &self.0 {
            if 1 << king1 & card.temple != 0 {
                side |= card.side;
            }
        }
        if side != 0 {
            // somewhere another piece can go without blocking the opponent's
            // temple, or taking a piece off ours
            let free = !pieces1 & !(1 << 2) & !(pieces0 & 1 << 22) & TABLE_MASK;
            let mut pass = 0;
            for card in &self.0 {
                let others = pieces1 & !(1 << king1);
                if BitIter::from(others).any(|p| offset_mask(p, card.from) & free != 0) {
                    pass |= card.mover;
                }
            }
            win |= side & pass & self.safe(pieces0, king1);
        }
        win
    }

    /// The term of `pieces1`'s king on `king1`: whether it steps onto the
    /// opponent's temple.
    fn temple(&self, pieces1: u32, king1: usize) -> u32 {
        if king1 == 2 {
            return BLOCK_MASK;
        }
        let mut win = 0;
        // a piece of our own on the opponent's temple is in the way
        if pieces1 & 1 << 2 == 0 {
            for card in &self.0 {
                if 1 << king1 & card.temple != 0 {
                    win |= card.mover;
                }
            }
        }
        win
    }

    /// The term of the other king on `king0`: whether `pieces1` takes it.
    fn take(&self, pieces1: u32, king0: usize) -> u32 {
        if king0 == 22 {
            return BLOCK_MASK;
        }
        let mut win = 0;
        for card in &self.0 {
            if offset_mask(king0, card.to) & pieces1 != 0 {
                win |= card.mover;
            }
        }
        win
    }
}

/// `'a` is how long this one update borrows its scratch, and `'t` how long the
/// tables live, which is past it: what is located for the next update, its
/// moves and its mirrored chunk, is carried over to it as plain borrows.
pub struct Update<'a, 't> {
    layout: TeamLayout,
    /// Where `layout` is in its table, which is its index in the enumeration.
    rank: usize,
    immutable: &'a ImmutableUpdate<'t>,
    mem: &'a mut LocalMem,
    moves: &'a mut Moves<'t>,
}

/// The shape of the chunk being written, which every move of it shares.
///
/// A move only ever changes where the walk reads from, so these come off the
/// layout once instead of out of every [`Accum`] and [`Spread`] it builds.
#[derive(Debug, Clone, Copy)]
pub struct Chunk {
    /// Where this side's king can stand, which is on any of its pieces.
    rows: u32,
    /// The same for the other side, and how long a row is.
    cols: u32,
    height: usize,
    width: usize,
}

impl Chunk {
    fn new(layout: TeamLayout) -> Self {
        let TeamLayout {
            pieces0: rows,
            pieces1: cols,
        } = layout;
        let (height, width) = (rows.count_ones() as usize, cols.count_ones() as usize);
        let this = Self {
            rows,
            cols,
            height,
            width,
        };
        // Not a check but a bound for the vectoriser, which it needs to keep
        // the walks over `padded` slots a whole vector at a time: told nothing,
        // it sees a popcount of up to 32 rows, and the expansion in the spread
        // falls back to a pair at a time for +20% instructions on the solve.
        //
        // SAFETY: a side has at most `SIDE` pieces, which `AllTables::build`
        // holds the sizes to
        unsafe { std::hint::assert_unchecked(this.height <= SIDE && this.width <= SIDE) };
        this
    }

    /// How many slots the chunk takes in the local buffers.
    ///
    /// Rounded up to a whole [`GROUP`]: the slots that buys are past the
    /// chunk and nobody reads them, and `wins` is filled with a resolved value
    /// so the walks that do reach them get nothing.
    ///
    /// Rounding folds to a bitmask (`(n + 7) & !7`), which the vectoriser
    /// does not see through, so it is told the count is a multiple of
    /// [`GROUP`]: that is what lets it take the walks over these a whole
    /// vector at a time with no tail.
    fn padded(self) -> usize {
        let padded = (self.height * self.width).next_multiple_of(GROUP);
        // SAFETY: rounded up to a multiple of `GROUP` just now, and a chunk is
        // at most `SIDE` by `SIDE`
        unsafe { std::hint::assert_unchecked(padded % GROUP == 0 && padded <= SLOTS) };
        padded
    }
}

pub struct Spread<'a> {
    /// The chunk being spread out of, which is what `expanded` is indexed by.
    old: Chunk,
    new: TeamLayout,
    step: (usize, usize),
    target: &'a [AtomicU32],
    /// Per king position, what to spread and the most that could ever be
    /// spread here, a vector per row of the chunk.
    ///
    /// The second is not just what it spreads now: a state only ever goes from
    /// unresolved to won or to lost, so the losses it can still gain are
    /// bounded by the states that are not won. That bound shrinks over time
    /// while the target only gains bits, so once the target subsumes it the
    /// chunk is done contributing, for good.
    ///
    /// The two side by side in one vector, the losses in the first [`ROW`]
    /// lanes and the bound in the next: the target row in both halves turns
    /// the two tests into one and against the pair. Halves rather than the two
    /// words of a slot next to each other, because then every operation on the
    /// row is one per half, and a half is one AVX2 register. See
    /// [`Spread::spreadout`].
    expanded: &'a [Simd<u32, LANES>; SIDE],
}

/// What a [`Spread`] did, and whether it could ever do anything again.
pub struct SpreadStatus {
    pub(crate) progress: bool,
    pub(crate) could: bool,
}

impl AllTables {
    pub fn build(size: u32, cards: u16) -> Self {
        Self::build_upto(size, cards, 2 * size)
    }

    /// [`Self::build`], but only the tables of up to `pieces` pieces, both
    /// kings counted.
    ///
    /// Those only ever read tables smaller than themselves, so they come out
    /// exactly as they do in the whole build. The ones past the limit are not
    /// even allocated, and nothing spreads into them, which is what lets the
    /// small tables of a size be solved where the whole of it does not fit.
    pub fn build_upto(size: u32, cards: u16, pieces: u32) -> Self {
        // a side has `size` pieces, so a chunk has at most that many king
        // positions each way, which is what the walks are built for
        assert!(
            size as usize <= SIDE,
            "a side has at most {SIDE} pieces, not {size}"
        );

        let mut mask_lookup = [0; 25];
        let mut directions = 0;
        for (mask, card) in zip(mask_iter(), Cards(cards).iter()) {
            for offset in BitIter::from(card.bitmap::<false>()) {
                mask_lookup[offset] |= mask
            }
            directions |= card.bitmap::<false>();
        }

        let tb = Self {
            size,
            pieces,
            mask_lookup,
            directions,
            seeds: Seeds::new(Cards(cards)),
            list: count_indexer(size)
                .into_iter()
                .map(|counts: PawnCount| {
                    let chunk_size = (counts.count0 + 1) as usize * (counts.count1 + 1) as usize;
                    let num_chunks = counts.total();
                    let built = counts.count0 + counts.count1 + 2 <= pieces;
                    let len = if built { chunk_size * num_chunks } else { 0 };
                    let list = AlignedTable::new_zeroed(len);
                    Table {
                        counts,
                        chunk_size,
                        list,
                    }
                })
                .collect(),
        };

        // Everything a table reads is a piece smaller: a take lands in
        // `take_one`, which is one off `count1`. So a whole piece count only
        // ever reads counts below it and none of its own, and the lot can run
        // together; the mirror a table reads its wins out of is in there with
        // it, which is why those two always had to be.
        //
        // What that buys is the tail. A count on its own runs until its last
        // few layouts stop moving, and those are far too few to keep the
        // machine busy; run the whole piece count at once and the ones still
        // going fill in for the ones that are nearly done.
        let mut schedule: Vec<Vec<TableJob>> = vec![];
        for counts in count_indexer(size) {
            let counts: PawnCount = counts;
            if counts.count0 < counts.count1 || !tb.has(counts) {
                continue;
            }
            let total = (counts.count0 + counts.count1) as usize;
            if schedule.len() <= total {
                schedule.resize_with(total + 1, Vec::new);
            }
            let jobs = &mut schedule[total];
            jobs.push(TableJob::new(&tb, counts));
            if counts.count0 > counts.count1 {
                jobs.push(TableJob::new(&tb, counts.invert()));
            }
        }
        // All of them at once rather than a table at a time: the small ones
        // have nowhere near enough layouts to fill the machine on their own,
        // and taken one by one each is a barrier at the end of it.
        let all: Vec<(&TableJob, usize)> = schedule
            .iter()
            .flatten()
            .flat_map(|job| (0..job.blocks()).map(move |b| (job, b)))
            .collect();
        #[cfg(feature = "parallell")]
        all.par_iter().for_each(|&(job, b)| job.seed_block(b));
        #[cfg(not(feature = "parallell"))]
        all.iter().for_each(|&(job, b)| job.seed_block(b));

        for (total, mut jobs) in schedule.into_iter().enumerate() {
            if jobs.is_empty() {
                continue;
            }
            let tables = jobs.len();
            let mut any_progress = true;
            let mut iters = 0;
            while any_progress {
                // the tables of a count do not read each other, so they are a
                // pass each rather than a pass apiece. Reading a mirror that is
                // still being written only ever leaves a bound wider than it
                // will end up, which costs a pass and never a result
                #[cfg(feature = "parallell")]
                {
                    any_progress = jobs
                        .par_iter_mut()
                        .map(|job| job.next().is_some())
                        .reduce(|| false, |a, b| a | b);
                }
                #[cfg(not(feature = "parallell"))]
                {
                    any_progress = false;
                    for job in &mut jobs {
                        any_progress |= job.next().is_some();
                    }
                }
                iters += 1;
            }

            // the counts are pawns, and both kings are always on the board
            let pieces = total + 2;
            println!("finished {pieces} pieces, {tables} tables, in {iters} iterations");
        }

        tb
    }
}

/// Layouts per work block.
///
/// Each block is one parallel task and one recorded seed, so this trades
/// scheduling overhead against how much of the enumeration has to be replayed
/// to reach the live layouts of a sparse late pass. The blocks are handed out
/// in order (see [`TableJob::next`]), so the threads together cover this many
/// layouts per thread of the table at a time: 1024 was measured 2% slower at 8
/// men, 256 5%, and 16384 the same.
const BLOCK: usize = 4096;

struct TableJob<'a> {
    tb: &'a AllTables,
    /// What the positions in `live` are indices into, so a live one can be
    /// unranked back into a layout.
    counts: PawnCount,
    /// Per layout, one bit per backward move: still worth running. A bit is
    /// cleared once every move on it has had its target subsume everything this
    /// chunk could ever spread there, and a layout is in the work list exactly
    /// while any of its bits is set, so this is the work list and the per move
    /// state in one.
    ///
    /// Usually that is one move per bit, but the count depends on the cards and
    /// the piece count, and 32 is not a bound on it: the default five reach 27
    /// moves at 8 men and 32 at 10, and other sets reach 41 already at 8. So
    /// the move index wraps, and the moves that collide share a bit and retire
    /// together. That loses skips rather than correctness, and only on the
    /// chunks that actually overflow.
    live: Vec<u32>,
    /// Per layout, one bit per forward move: its target can no longer change
    /// anything here, so it is not read. Shared and wrapping like `live`, and
    /// set rather than cleared, so the zeroed allocation starts with every move
    /// read.
    spent: Vec<u32>,
    /// One bit per layout: retired, but its capture predecessors still have to
    /// be written by the `go_up` pass.
    needs_up: Vec<u64>,
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use crate::onitama_simd::Block;

    use super::{mask_iter, AllTables};

    #[test]
    fn build_tb() {
        let tb = AllTables::build(3, 0b11111);
        let wins = tb.count_ones();
        let total = tb.len() * 30;
        println!("{wins} total wins");
        let unresolved = tb.count_unresolved();
        println!("{unresolved} unresolved states");
        println!("{} resolved states", total - unresolved);
        assert_eq!(wins, 831344251);
    }

    /// A card set with more backward moves per chunk than `TableJob::live` has
    /// bits, so some of them share one and can only retire together.
    ///
    /// Cards 0, 3, 4, 5 and 8 span 13 offsets instead of the 8 the default five
    /// span, which takes a chunk to 33 moves at 6 men and 41 at 8, well past
    /// the 27 the default set ever reaches. The counts are the ones `799e9f0`
    /// produces, from before any of the skipping existed, so this pins sharing
    /// a bit to cost a skip and never an answer.
    #[test]
    fn overflowing_card_set() {
        let tb = AllTables::build(3, 0b1_0011_1001);
        assert_eq!(tb.count_ones(), 850996693);
        assert_eq!(tb.count_unresolved(), 300597);
    }

    /// The wins of every table of up to seven pieces, as eight men solves
    /// them. Those are the same tables at ten men, which only adds larger
    /// ones, so a ten men build has to come out with exactly these.
    const SEVEN_PIECES: [((u32, u32), u64); 15] = [
        ((0, 0), 11514),
        ((0, 1), 315358),
        ((0, 2), 3638987),
        ((0, 3), 26229868),
        ((1, 0), 214905),
        ((1, 1), 6210802),
        ((1, 2), 73684072),
        ((1, 3), 514411418),
        ((2, 0), 2133650),
        ((2, 1), 57161872),
        ((2, 2), 687973091),
        ((2, 3), 4756101409),
        ((3, 0), 14050120),
        ((3, 1), 348163949),
        ((3, 2), 3943887060),
    ];

    /// Check random states of every table against the rules, two plies deep.
    ///
    /// Nothing of the solver is used but the lookup of a table entry and the
    /// card masks, which [`card_masks`] pins: the moves, the card exchange
    /// and the game's ends are all done again here, a state at a time. A state
    /// of `pieces1` to move is won exactly when it has a move that wins on the
    /// spot, or one after which every reply either wins on the spot for the
    /// opponent or leads to an entry marked won. That is only a check that the
    /// tables are a solution, not that they are the least one, but anything
    /// read from the wrong slot, chunk or configuration breaks it.
    fn check_states(tb: &AllTables, per_table: usize) {
        use super::{offset_mask, BitIter, Indexer, KingPos, TeamLayout};
        let cards = &tb.seeds.0;

        // per configuration, the cards of the side to move and the side card
        let mut hand = [0u32; 30];
        let mut side = [0usize; 30];
        for (c, card) in cards.iter().enumerate() {
            for k in 0..30 {
                hand[k] |= (card.mover >> k & 1) << c;
                if card.side >> k & 1 != 0 {
                    side[k] = c;
                }
            }
        }
        let config = |hand_: u32, side_: usize| (0..30).find(|&k| hand[k] == hand_ && side[k] == side_).unwrap();

        let won = |layout: TeamLayout, kpos: KingPos, k: usize| {
            let counts = layout.counts();
            let entry = tb.index_count(counts).index(layout)[kpos].load(Ordering::Relaxed);
            entry >> k & 1 != 0
        };

        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |n: usize| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng % n as u64) as usize
        };

        for table in tb.built() {
            let counts = table.counts;
            let mut checked = 0;
            while checked < per_table {
                let layout = counts.nth_layout(next(counts.total()));
                let TeamLayout { pieces0, pieces1 } = layout;
                let pick = |pieces: u32, i: usize| BitIter::from(pieces).nth(i % pieces.count_ones() as usize).unwrap();
                let (king0, king1) = (pick(pieces0, next(8)), pick(pieces1, next(8)));
                let kpos = KingPos {
                    king0: king0 as u32,
                    king1: king1 as u32,
                };
                if kpos.over() {
                    continue;
                }
                let k = next(30);
                let (mine, kept) = (hand[k], side[k]);
                let theirs = 0b11111 & !mine & !(1 << kept);

                let mut expected = false;
                'moves: for c in BitIter::from(mine) {
                    for p in BitIter::from(pieces1) {
                        for q in BitIter::from(offset_mask(p, cards[c].from) & !pieces1) {
                            if q == king0 || p == king1 && q == 2 {
                                expected = true;
                                break 'moves;
                            }
                            let pieces0 = pieces0 & !(1 << q);
                            let pieces1 = pieces1 ^ 1 << p ^ 1 << q;
                            let king1 = if p == king1 { q } else { king1 };
                            // our next hand, which is the opponent's to keep
                            let after = mine & !(1 << c) | 1 << kept;

                            let mut lost = true;
                            'replies: for e in BitIter::from(theirs) {
                                for p0 in BitIter::from(pieces0) {
                                    for q0 in BitIter::from(offset_mask(p0, cards[e].to) & !pieces0) {
                                        if q0 == king1 || p0 == king0 && q0 == 22 {
                                            lost = false;
                                            break 'replies;
                                        }
                                        let layout = TeamLayout {
                                            pieces0: pieces0 ^ 1 << p0 ^ 1 << q0,
                                            pieces1: pieces1 & !(1 << q0),
                                        };
                                        let kpos = KingPos {
                                            king0: if p0 == king0 { q0 } else { king0 } as u32,
                                            king1: king1 as u32,
                                        };
                                        if !won(layout, kpos, config(after, e)) {
                                            lost = false;
                                            break 'replies;
                                        }
                                    }
                                }
                            }
                            if lost {
                                expected = true;
                                break 'moves;
                            }
                        }
                    }
                }
                assert_eq!(won(layout, kpos, k), expected, "{counts:?} {layout:?} {kpos:?} configuration {k}");
                checked += 1;
            }
        }
    }

    /// The check, on tables that are known to be right.
    #[test]
    fn checked_states() {
        check_states(&AllTables::build(3, 0b11111), 20_000);
    }

    /// Ten men, as far as it fits: every table of up to seven pieces, which
    /// takes in the ones with five pieces a side. The ones that eight men has
    /// too have to come out the same, and the rest go through the check.
    #[test]
    #[ignore]
    fn ten_up_to_seven() {
        let tb = AllTables::build_upto(5, 0b11111, 7);
        let wins = tb.table_wins();
        for (counts, expected) in SEVEN_PIECES {
            let got = wins.iter().find(|(c, _)| *c == counts).unwrap().1;
            assert_eq!(got, expected, "{counts:?}");
        }
        println!("{wins:?}");
        check_states(&tb, 20_000);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn thp_settings() {
        use super::{thp_allowed, require_hugepages, THP_SETTING};

        assert!(thp_allowed("[always] madvise never\n"));
        assert!(thp_allowed("always [madvise] never\n"));
        assert!(!thp_allowed("always madvise [never]\n"));
        // the machine this is built for has to be one of the first two, so the
        // check the tables run has to pass here
        println!("{THP_SETTING} is {:?}", std::fs::read_to_string(THP_SETTING));
        require_hugepages();
    }

    /// The same set at 8 men, where a chunk reaches 41 moves. Ignored by
    /// default because the extra offsets make it a minute of work; 6 men
    /// already goes past the 32 bits.
    #[test]
    #[ignore]
    fn overflowing_card_set_8() {
        let tb = AllTables::build(4, 0b1_0011_1001);
        assert_eq!(tb.count_ones(), 38181597408);
    }

    /// The card masks [`Seeds`] uses partition every configuration: the side
    /// card is a card in 6 of them, the side to move holds it in 12 and the
    /// opponent in the other 12, and the side to move always holds two.
    /// A bit is settled exactly when every not lost state it would make is
    /// already won.
    #[test]
    fn settled_bits() {
        use std::simd::Simd;
        let mut x = 0x9e37_79b9u32;
        for _ in 0..2000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            // dense wins, so both answers turn up
            let wins = (x | x.rotate_left(7) | x.rotate_left(19)) & super::BLOCK_MASK;
            let settled = super::accum_spread::settled(Simd::splat(wins))[0];
            for p in 0..30 {
                let lands = Block(1 << p).invert().expand().invert().0;
                assert_eq!(settled >> p & 1 != 0, lands & !wins == 0, "wins {wins:#x} bit {p}");
            }
        }
    }

    #[test]
    fn card_masks() {
        use super::{Cards, Seeds, BLOCK_MASK};
        let seeds = Seeds::new(Cards(0b11111));
        let mut held = [0; 30];
        for card in &seeds.0 {
            assert_eq!(card.side.count_ones(), 6);
            assert_eq!(card.mover.count_ones(), 12);
            assert_eq!(card.opponent.count_ones(), 12);
            assert_eq!(card.side | card.mover | card.opponent, BLOCK_MASK);
            assert_eq!(card.side & card.mover, 0);
            for (k, h) in held.iter_mut().enumerate() {
                *h += card.mover >> k & 1;
            }
        }
        assert_eq!(held, [2; 30]);
    }

    #[test]
    fn mask_test() {
        for mask in mask_iter().take(5) {
            assert_eq!(mask, Block(mask).invert().0)
        }
    }
}
