use std::{
    borrow::Borrow,
    collections::HashSet,
    hash::Hash,
    ops::{BitAnd, Not},
};

#[derive(Clone, Debug)]
pub struct Anf<T> {
    terms: HashSet<T>,
}

impl<T> Default for Anf<T> {
    fn default() -> Self {
        Self {
            terms: Default::default(),
        }
    }
}

impl<T: Eq + Hash + Clone> Anf<T> {
    fn xor_term(&mut self, term: impl Borrow<T>) {
        if !self.terms.remove(term.borrow()) {
            assert!(self.terms.insert(term.borrow().clone()))
        }
    }
}

impl<'a, T: Eq + Hash + Clone> BitAnd for &'a Anf<T>
where
    &'a T: BitAnd<Output = Option<T>>,
{
    type Output = Anf<T>;

    fn bitand(self, rhs: &'a Anf<T>) -> Self::Output {
        let mut result = Anf::default();
        for x in &self.terms {
            for y in &rhs.terms {
                if let Some(term) = x & y {
                    result.xor_term(term)
                }
            }
        }
        result
    }
}

impl<'a, T: Eq + Hash + Clone + Default> Not for Anf<T> {
    type Output = Anf<T>;

    fn not(mut self) -> Self::Output {
        self.xor_term(&Default::default());
        self
    }
}

impl<T: Eq + Hash + Clone> Anf<T> {
    pub fn map(&self, f: impl Fn(&T) -> Option<[T; 2]>) -> Self {
        let mut result = Anf::default();
        for x in &self.terms {
            if let Some([a, b]) = f(x) {
                result.xor_term(a);
                result.xor_term(b);
            }
        }
        result
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }
}
