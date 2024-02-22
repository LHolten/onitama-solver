# Optimizations
This is a fast endgame bitbase generator for the boardgame "Onitama".
A bitbase is a tablebase that only stores a single bit as the evaluation of each state.

Below is a list of some of the optimizations:

## Locality
Generating a bitbase requires a lot of random access to check the evaluation of all possible moves.
Random access is slow because the full bitbase does not fit in cache.

To make bitbase generation faster we can group states that have similar access patterns to be evaluated after one another.
Even better, we can evaluate states with similar access patterns at the same time!
The generator does not need SIMD instructions, because a u32 can already fit 32 state evaluations.

The generator groups 30 states with the same piece layout together into a single u32, allowing them to be processed at the same time.
Up to 25 of these layouts that only differ in king postitions are then grouped together to improve locality even further.

Updating a group of 25 * 30 states will only use states from at most 40 other groups.
This means that these groups will be in cache and only create a cache miss once!

## Multi-threading
An easy way to make generation faster is to use more threads.
However, this means that it is now necessary to synchronise updates to states evalutions.

Updating state evaluations atomically is costly, because it requires to synchronise the memory between cores.
However, we can first do a relaxed read and skip the costly update if it would be a noop!
This does not make the algorithm less correct, because all updates are idempotent.

## Other optimisations
The program uses one of the two unused bits in each u32 to store if the evaluations in the u32 are completely finished.
This allows skipping future updates of this u32 and speeds up generation.
