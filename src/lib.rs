mod anf;

use anf::Anf;

mod board;

type Table = Anf<board::Board>;

impl Table {
    pub fn losses(&self) -> Self {
        // self is the wins for player 1.
        let wins = self;

        // check if player 2 can only get wins for player 1
        let mut loss = !Table::default();

        for from in 0..5 {
            if from != 0 {
                let (from, to) = (1 << from, 1 << from >> 1);
                loss = &loss & &!wins.map(|b| b.backward(from, to))
            }
        }

        loss
    }
}

#[cfg(test)]
mod tests {
    use crate::Table;

    #[test]
    fn do_steps() {
        let mut anf = Table::default();
        for _ in 0..2 {
            anf = anf.losses();
            println!("term count: {}", anf.len());
            println!("table: {:#?}", anf);
        }
    }
}
