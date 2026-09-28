use std::{env::args, sync::atomic::Ordering, time::Instant};

use onitama_solver::onitama_simd::AllTables;

pub fn main() {
    let size = args().nth(1).expect("expected one arg: num pieces");
    let size = match size.parse::<u8>().expect("expected integer") {
        2 => 1,
        4 => 2,
        6 => 3,
        8 => 4,
        10 => 5,
        _ => panic!("that size is not supported"),
    };

    // off by default: it is a statistic, and it costs a full rebuild of every
    // chunk, which is more than the solve it reports on
    let count = args().any(|a| a == "--unresolved");

    let before = Instant::now();
    let tb = AllTables::build(size, 0b11111);
    let time = before.elapsed();

    let wins = tb.count_ones();
    let total = tb.len() * 30;
    println!("{wins} total wins");
    println!("{} wins in 1", tb.win_in1);
    println!("{} not win in 1", total - tb.win_in1);
    match count.then(|| tb.count_unresolved()) {
        Some(unresolved) => {
            println!("{unresolved} unresolved states");
            println!("{} resolved, not win in 1", total - tb.win_in1 - unresolved);
        }
        None => println!("unresolved states not counted, pass --unresolved"),
    }
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
    println!("took {:.3} seconds", time.as_secs_f32());

    match size {
        2 => assert_eq!(wins, 6752579),
        3 => assert_eq!(wins, 831344251),
        4 => assert_eq!(wins, 37560295296),
        _ => {}
    };
}
