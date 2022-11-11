use std::{
    borrow::Borrow,
    collections::HashSet,
    ops::{BitAnd, Not},
};

use crate::board::Board;

#[derive(Clone, Debug, Default)]
pub struct Anf {
    terms: HashSet<Board>,
}

impl Anf {
    fn xor_term(&mut self, term: impl Borrow<Board>) {
        let Some(term) = term.borrow().clone().check() else {
            return;
        };
        if !self.terms.remove(term.borrow()) {
            assert!(self.terms.insert(term))
        }
    }
}

impl<'a> BitAnd for &'a Anf {
    type Output = Anf;

    fn bitand(self, rhs: &'a Anf) -> Self::Output {
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

impl<'a> Not for Anf {
    type Output = Anf;

    fn not(mut self) -> Self::Output {
        self.xor_term(&Default::default());
        self
    }
}

impl Anf {
    pub fn map(&self, f: impl Fn(&Board) -> Option<[Board; 2]>) -> Self {
        let mut result = Anf::default();
        for x in &self.terms {
            for new in f(x).into_iter().flatten() {
                result.xor_term(new)
            }
        }
        result
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn eval(&self, board: &Board) -> bool {
        let mut res = false;
        for x in &self.terms {
            res ^= x.includes(board)
        }
        res
    }

    pub fn count(&self, list: &[Board]) -> usize {
        list.iter().filter(|b| self.eval(b)).count()
    }
}
