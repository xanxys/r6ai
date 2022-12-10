mod az;
mod baseline;
mod r6;

use std::time::{SystemTime, UNIX_EPOCH};

use crate::{
    az::{az_init, train_az, az_get_action},
    r6::Board, baseline::mcts_get_action,
};
use dfdx::prelude::{SaveToNpz, LoadFromNpz};
use r6::{Action, Cell};
use rand::{rngs::SmallRng, SeedableRng};

// reward value is +1 for black win, -1 for white win, 0.5 for draw.

fn win_rate<Fn1, Fn2>(rng: &mut SmallRng, p1: Fn1, p2: Fn2, num: usize, budget_usec: u64) -> f32
where
    Fn1: Fn(&mut SmallRng, &Board, u64) -> Action,
    Fn2: Fn(&mut SmallRng, &Board, u64) -> Action,
{
    let mut num_win_p1 = 0;
    let mut num_draw = 0;
    let mut num_win_p2 = 0;

    for i in 0..num {
        let p1_is_black = i % 2 == 0;
        let mut b = Board::new();
        for _i in 0..100 {
            if let Some(res) = b.is_terminal() {
                match res {
                    Cell::Empty => num_draw += 1,
                    Cell::Black => {
                        if p1_is_black {
                            num_win_p1 += 1;
                        } else {
                            num_win_p2 += 1;
                        }
                    }
                    Cell::White => {
                        if p1_is_black {
                            num_win_p2 += 1;
                        } else {
                            num_win_p1 += 1;
                        }
                    }
                }
                break;
            }

            if b.side_black == p1_is_black {
                b.apply(&p1(rng, &b, budget_usec));
            } else {
                b.apply(&p2(rng, &b, budget_usec));
            }
        }
    }
    let win_rate = (num_win_p1 as f32 + num_draw as f32 * 0.5) / num as f32;
    println!(
        "p1_win_rate={:.1}% (p1/draw/p2={}/{}/{})",
        win_rate * 100.0,
        num_win_p1,
        num_draw,
        num_win_p2
    );
    win_rate
}

fn play_single<Fn1, Fn2>(rng: &mut SmallRng, p_black: Fn1, p_white: Fn2, budget_usec: u64)
where
    Fn1: Fn(&mut SmallRng, &Board, u64) -> Action,
    Fn2: Fn(&mut SmallRng, &Board, u64) -> Action,
{
    let mut b = Board::new();

    for i in 0..100 {
        println!("turn={}\n{}", i, b);
        if let Some(res) = b.is_terminal() {
            println!("winner: {:?}", res);
            break;
        }

        let branching_factor = b.legal_actions().len();

        let action = if b.side_black {
            p_black(rng, &b, budget_usec)
        } else {
            p_white(rng, &b, budget_usec)
        };
        println!("chosen: {:?} / bf={}", action, branching_factor);
        b.apply(&action);
    }
}

fn compare() {
    const NN_PATH: &str = "./az_pv.npz";

    let mut rng = SmallRng::seed_from_u64(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64,
    );

    let mcts_1ms = |rng: &mut SmallRng, b: &Board, _: u64| mcts_get_action(rng, b, 1000);
    let mcts_10ms = |rng: &mut SmallRng, b: &Board, _: u64| mcts_get_action(rng, b, 10_000);

    let mut pv = az_init(&mut rng);
    match pv.load(NN_PATH) {
        Ok(_) => println!("Loaded NN from {}", NN_PATH),
        Err(e) => println!("Failed to load NN: {}", e),
    }

    for i in 0..10 {
        println!("------- AZ Training Session {}", i);
        let pv_new = train_az(&mut rng, &pv, 1000, 100);

        let az_1ms = |rng: &mut SmallRng, b: &Board, _: u64| az_get_action(rng, b, 1000, &pv);
        let post_az_1ms = |rng: &mut SmallRng, b: &Board, _: u64| az_get_action(rng, b, 1000, &pv_new);
    
        println!("AZ(old) vs MCTS:");
        win_rate(&mut rng, az_1ms, mcts_1ms, 10, 10000);
        println!("AZ(new) vs MCTS:");
        win_rate(&mut rng, post_az_1ms, mcts_1ms, 10, 10000);
        println!("AZ(new) vs AZ(old):");
        let az_comp = win_rate(&mut rng, post_az_1ms, az_1ms, 10, 10000);

        if az_comp > 0.5 {
            pv = pv_new;
        }

        match pv.save(NN_PATH) {
            Ok(_) => println!("Saved NN to {}", NN_PATH),
            Err(e) => println!("Failed to save NN: {}", e),
        }
    }
}

fn main() {
    //win_rate(&mut rng, random_get_action, mcts_get_action, 1, 10000);
    //play_single(&mut rng, mcts_get_action, random_get_action, 1000);

    firestorm::clear();
    compare();
    firestorm::save("./flames/").unwrap();
}
