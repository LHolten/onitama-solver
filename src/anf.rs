use std::{
    collections::HashSet,
    ops::{BitAnd, Not},
};

#[derive(Default, Clone)]
struct Anf {
    terms: HashSet<u32>,
}

const MAX_TERM_SIZE: u32 = 5;
// term has an optional king location part
// and for each square optional one of three states

impl Anf {
    fn xor_term(&mut self, term: u32) {
        if term.count_ones() > MAX_TERM_SIZE {
            return;
        }
        if !self.terms.remove(&term) {
            assert!(self.terms.insert(term))
        }
    }

    pub fn forget_vars(&self, var: u32) -> Self {
        let mut result = Self::default();
        for &x in &self.terms {
            result.xor_term(x & !var)
        }
        result
    }
}

impl BitAnd for &Anf {
    type Output = Anf;

    fn bitand(self, rhs: &Anf) -> Self::Output {
        let mut result = Anf::default();
        for &x in &self.terms {
            for &y in &rhs.terms {
                let term = x | y;
                result.xor_term(term)
            }
        }
        result
    }
}

impl Not for &Anf {
    type Output = Anf;

    fn not(self) -> Self::Output {
        let mut result = self.clone();
        result.xor_term(0);
        result
    }
}
