pub trait Proj<T>: Copy {
    type Output;
    fn proj_ref(self, board: &T) -> &Self::Output;
    fn proj_mut(self, board: &mut T) -> &mut Self::Output;
}

#[derive(Clone, Copy)]
pub struct Projector<X, Y> {
    by_ref: X,
    by_mut: Y,
}

impl<X, Y> Projector<X, Y> {
    pub fn new(by_ref: X, by_mut: Y) -> Self {
        Self { by_ref, by_mut }
    }
}

impl<T, O, X, Y> Proj<T> for Projector<X, Y>
where
    X: Fn(&T) -> &O + Copy,
    Y: Fn(&mut T) -> &mut O + Copy,
{
    type Output = O;

    fn proj_ref(self, board: &T) -> &O {
        (self.by_ref)(board)
    }

    fn proj_mut(self, board: &mut T) -> &mut O {
        (self.by_mut)(board)
    }
}

#[macro_export]
macro_rules! proj {
    (|$var:ident: $t:ty| $proj:expr) => {{
        let by_ref: for<'a> fn(&'a $t) -> &'a _ = |$var| &$proj;
        let by_mut: for<'a> fn(&'a mut $t) -> &'a mut _ = |$var| &mut $proj;
        proj::Projector::new(by_ref, by_mut)
    }};
}

pub trait Mask<A>: Copy {
    type Output: Copy;
    fn get_mask(self, board: &A) -> Self::Output;
}

impl<F: Fn(&A) -> O + Copy, A, O: Copy> Mask<A> for F {
    type Output = O;

    fn get_mask(self, board: &A) -> Self::Output {
        (self)(board)
    }
}

macro_rules! mask_impl {
    ($($t:ty)*) => {$(
        impl<A> Mask<A> for $t {
            type Output = $t;

            fn get_mask(self, _: &A) -> Self::Output {
                self
            }
        }
    )*}
}

mask_impl! { u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize }
