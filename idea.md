work on exactly two fields at a time, because there is one potential symmetry
1 bit, "symmetric"
we have 3*3 cases
2 * 3 symmetric -> 3 or 6 pointers
3 * assymmetric -> 3 pointers


store the number of non-losing moves,
every time we backup we just subtract from this number

# L0la algo
- loop over all states, check if all of the next states are marked as a win
  - mark the current state as a win
  - mark all previous states as 

# how to compress
8 pointers, high chance of some duplicates
u = len(set(list))
s = len(ptr)

- store the 8 pointers
  - 8 * s
- store pairs of (index_mask, pointer)
  - u * (8 + s)
- RLE does not make sense
breakeven point
1 / u = 1 / s + 1 / 8

# sat step
sat if state is a win/loss
so function with state as input
if we now want to do a step
do transformation on next state and or together, this can be done symbolicaly

# combination with tb
first go through dd, then look up in tb
tb can be much smaller because most states are a fast win
dd is used to index the tb, making indexing easier

# dd can use distance to next piece as choice
this makes dd traversal much faster, but nodes are bigger

# store variable used in parent and choose variable from remaining
tree's can not be shared between depths
pointer is a node plus var from 0..node_depth
or
pointer is a node plus permutation!

# use approximate dd as nnue
nnue is a sparse matrix, take one vector of values for each (pawn, king) tuple.
combine the vectors somehow and then reduce the result vector with nonlinear function.