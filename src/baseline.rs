use std::{collections::HashMap, time::Instant};

use crate::r6::Board;
use crate::r6::{Action, Cell};
use rand::{rngs::SmallRng, Rng};

pub fn random_get_action(rng: &mut SmallRng, b: &Board, budget_usec: u64) -> Action {
    firestorm::profile_fn!(random_get_action);
    let actions = b.legal_actions();
    actions[rng.gen_range(0..actions.len())]
}

pub fn mcts_get_action(rng: &mut SmallRng, b: &Board, budget_usec: u64) -> Action {
    firestorm::profile_fn!(mcts_get_action);

    let t0 = Instant::now();

    const EPS: f32 = 0.1;
    const MIN_OBS_TO_EXPAND: i32 = 10;

    struct MCTSNode {
        parent: Option<Board>,
        children: Vec<(Action, Board)>,
        num_obs: i32,
        tot_b_reward: f32,
    }
    let mut dag: HashMap<Board, MCTSNode> = HashMap::new();
    fn record_reward(dag: &mut HashMap<Board, MCTSNode>, b: &Board, r_b: f32) {
        firestorm::profile_fn!(record_reward);

        let mut b = b.clone();
        loop {
            match dag.get_mut(&b) {
                Some(node) => {
                    node.num_obs += 1;
                    node.tot_b_reward += r_b;
                    match &node.parent {
                        Some(p) => b = p.clone(),
                        None => break,
                    }
                }
                None => {
                    panic!("Tried to record reward for non-existent node");
                }
            }
        }
    }
    fn insert_empty(dag: &mut HashMap<Board, MCTSNode>, b: &Board, parent: Option<Board>) {
        firestorm::profile_fn!(insert_empty);

        let node = MCTSNode {
            parent: parent.clone(),
            children: vec![],
            num_obs: 0,
            tot_b_reward: 0.0,
        };
        if dag.insert(b.clone(), node).is_some() {
            // this can very rarely happen when self-loop of BoardState exists, because of Pass action.
            // TODO: is ignoring OK?
        }
    }
    fn expand(rng: &mut SmallRng, dag: &mut HashMap<Board, MCTSNode>, b: &Board) {
        firestorm::profile_fn!(expand);

        let actions = b.legal_actions();
        for a in actions {
            let mut b2 = b.clone();
            b2.apply(&a);

            insert_empty(dag, &b2, Some(b.clone()));
            dag.get_mut(b).unwrap().children.push((a, b2));
        }

        // Evaluate all children at least once.
        let children_bs: Vec<Board> = dag
            .get(b)
            .unwrap()
            .children
            .iter()
            .map(|(_, b)| b.clone())
            .collect();
        for b in children_bs.iter() {
            record_reward(dag, b, playout_b_reward(rng, b));
        }
    }
    fn pick_best(dag: &HashMap<Board, MCTSNode>, b: &Board) -> (Action, Board) {
        firestorm::profile_fn!(pick_best);

        dag.get(b)
            .unwrap()
            .children
            .iter()
            .max_by(|(_, b1), (_, b2)| {
                let n1 = dag.get(&b1).unwrap();
                let n2 = dag.get(&b2).unwrap();
                let r1 = n1.tot_b_reward / n1.num_obs as f32;
                let r2 = n2.tot_b_reward / n2.num_obs as f32;

                if b.side_black {
                    r1.partial_cmp(&r2).unwrap() // black's turn -> maximize black reward
                } else {
                    r2.partial_cmp(&r1).unwrap() // white's turn -> minimize black reward
                }
            })
            .unwrap()
            .clone()
    }
    fn select(rng: &mut SmallRng, dag: &mut HashMap<Board, MCTSNode>, b0: &Board) -> Vec<Action> {
        let mut node = dag.get_mut(b0).expect("Invalid MCTS scan");
        if node.children.is_empty() {
            // not yet expanded node
            if node.num_obs < MIN_OBS_TO_EXPAND {
                return vec![];
            } else {
                // ok to expand now
                expand(rng, dag, b0);
                node = dag.get_mut(b0).unwrap();
            }
        }
        if node.children.is_empty() {
            panic!("MCTS #children must be >0");
        }

        if rng.gen::<f32>() < EPS {
            // explore
            let (a, b) = node.children[rng.gen_range(0..node.children.len())].clone();
            let mut ac_path = select(rng, dag, &b);
            ac_path.push(a);
            return ac_path;
        } else {
            // exploit
            let (a, b) = pick_best(dag, b0);
            let mut ac_path = select(rng, dag, &b);
            ac_path.push(a);
            return ac_path;
        }
    }

    insert_empty(&mut dag, b, None);
    expand(rng, &mut dag, b);
    let mut num_iter = 0;
    loop {
        if t0.elapsed().as_micros() as u64 >= budget_usec {
            break;
        }

        let next_action = select(rng, &mut dag, b).last().unwrap().clone();
        let mut b2 = b.clone();
        b2.apply(&next_action);
        record_reward(&mut dag, &b2, playout_b_reward(rng, &b2));
        num_iter += 1;
    }

    let (a, _) = pick_best(&dag, b);
    // println!(
    //     "MCTS: #nodes={} / #obs={} / #iter={}",
    //     dag.len(),
    //     dag.get(b).unwrap().num_obs,
    //     num_iter
    // );
    a
}

fn playout_b_reward(rng: &mut SmallRng, b: &Board) -> f32 {
    firestorm::profile_fn!(playout_b_reward);

    let mut curr_b = b.clone();
    for _i in 0..35 {
        if let Some(res) = curr_b.is_terminal() {
            match res {
                Cell::Black => return 1.0,
                Cell::Empty => return 0.5,
                Cell::White => return 0.0,
            }
        }
        curr_b.apply(&random_get_action(rng, &curr_b, 1000));
    }
    return 0.5;
}
