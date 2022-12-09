mod r6;

use std::{
    collections::HashMap,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use crate::r6::Board;
use dfdx::{
    prelude::{
        cross_entropy_with_logits_loss, mse_loss, Adam, AdamConfig, HasArrayData, Linear, Module,
        Optimizer, ReLU, ResetParams, Sigmoid, SplitInto,
    },
    tensor::{Tensor1D, TensorCreator},
};
use r6::{to_ix, Action, Cell, PutAction, B_SIZE};
use rand::{rngs::SmallRng, Rng, SeedableRng};

fn random_get_action(rng: &mut SmallRng, b: &Board, budget_usec: u64) -> Action {
    firestorm::profile_fn!(random_get_action);
    let actions = b.legal_actions();
    actions[rng.gen_range(0..actions.len())]
}

fn mcts_get_action(rng: &mut SmallRng, b: &Board, budget_usec: u64) -> Action {
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

const STATE_SIZE: usize = 3 * 36 + 2; // cell (E, B, W) x 6^2 + curr_side (B, W)
const ACTION_SIZE: usize = 36 + 1; // put & pass

type PVNetwork = (
    (Linear<STATE_SIZE, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    // selector: logits for action (put: 36, pass: 1)
    // value: 0~1 expected final value
    SplitInto<(Linear<32, ACTION_SIZE>, (Linear<32, 1>, Sigmoid))>,
);

struct AZSample {
    state: Board,
    final_policy: Vec<(Action, f32)>,
    reward_b: f32,
}

fn encode_board(b: &Board) -> Tensor1D<STATE_SIZE> {
    let mut s: Tensor1D<STATE_SIZE> = TensorCreator::zeros();
    let mut sd = s.mut_data();

    // one-hot encoding of cells
    for y in 0..B_SIZE {
        for x in 0..B_SIZE {
            let v_ofs = 3 * to_ix(x, y);
            match b.cells[to_ix(x, y)] {
                Cell::Empty => {
                    sd[v_ofs + 0] = 1.0;
                }
                Cell::Black => {
                    sd[v_ofs + 1] = 1.0;
                }
                Cell::White => {
                    sd[v_ofs + 2] = 1.0;
                }
            }
        }
    }

    // one-hot encoding of side
    if b.side_black {
        sd[3 * 36 + 0] = 1.0;
    } else {
        sd[3 * 36 + 1] = 1.0;
    }

    s
}

fn encode_action_policy(pol: &[(Action, f32)]) -> Tensor1D<ACTION_SIZE> {
    let mut ap: Tensor1D<ACTION_SIZE> = TensorCreator::zeros();
    let mut apd = ap.mut_data();

    for (action, prob) in pol {
        match action {
            Action::Put(PutAction { x, y, .. }) => {
                apd[to_ix(*x, *y)] = *prob;
            }
            Action::Pass => {
                apd[36] = *prob;
            }
        }
    }
    ap
}

fn encode_reward(r: f32) -> Tensor1D<1> {
    let mut t: Tensor1D<1> = TensorCreator::zeros();
    t.mut_data()[0] = r;
    t
}

fn decode_action(b: &Board, p: &Tensor1D<ACTION_SIZE>) -> Vec<(Action, f32)> {
    let probs = p.clone().log_softmax();
    let mut actions = Vec::with_capacity(ACTION_SIZE);
    for y in 0..B_SIZE {
        for x in 0..B_SIZE {
            actions.push((
                Action::Put(PutAction {
                    x: x,
                    y: y,
                    c: if b.side_black {
                        Cell::Black
                    } else {
                        Cell::White
                    },
                }),
                probs.data()[to_ix(x, y)],
            ));
        }
    }
    actions.push((Action::Pass, probs.data()[36]));
    actions
}

fn az_get_action(
    rng: &mut SmallRng,
    b: &Board,
    budget_usec: u64,
    pv: &PVNetwork,
    sample: &mut AZSample,
) -> Action {
    firestorm::profile_fn!(az_get_action);

    let t0 = Instant::now();

    const W_NET: f32 = 0.3;
    const MIN_OBS_TO_EXPAND: i32 = 10;

    struct MCTSNode {
        parent: Option<Board>,
        children: Vec<(Action, Board)>,
        num_obs: i32,
        tot_b_reward: f32,
        net_prob: f32,
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
            net_prob: -1.0,
        };
        if dag.insert(b.clone(), node).is_some() {
            // this can very rarely happen when self-loop of BoardState exists, because of Pass action.
            // TODO: is ignoring OK?
        }
    }
    fn expand(rng: &mut SmallRng, dag: &mut HashMap<Board, MCTSNode>, b: &Board, pv: &PVNetwork) {
        firestorm::profile_fn!(expand);

        // Evaluate PV network.
        let in_s = encode_board(b);
        let (out_p, _) = pv.forward(in_s);
        let actions = decode_action(b, &out_p);

        let legal_actions = b.legal_actions();
        let p_sum_legal: f32 = actions
            .iter()
            .filter(|(a, _)| legal_actions.contains(a))
            .map(|(_, p)| p)
            .sum();

        for (a, a_prob) in actions {
            if !legal_actions.contains(&a) {
                continue;
            }

            let mut b2 = b.clone();
            b2.apply(&a);

            insert_empty(dag, &b2, Some(b.clone()));
            dag.get_mut(&b2).unwrap().net_prob = a_prob / p_sum_legal;
            dag.get_mut(b).unwrap().children.push((a, b2));
        }
    }
    fn selection_score(num_obs: i32, tot_b_reward: f32, net_prob: f32, num_obs_parent: i32) -> f32 {
        let e_b_reward = if num_obs == 0 {
            0.5
        } else {
            tot_b_reward / num_obs as f32
        };
        e_b_reward + W_NET * net_prob * (num_obs_parent as f32).sqrt() / (1.0 + num_obs as f32)
    }
    fn pick_best(dag: &HashMap<Board, MCTSNode>, b: &Board) -> (Action, Board) {
        firestorm::profile_fn!(pick_best);

        let num_obs = dag.get(b).unwrap().num_obs;
        dag.get(b)
            .unwrap()
            .children
            .iter()
            .max_by(|(_, b1), (_, b2)| {
                let n1 = dag.get(&b1).unwrap();
                let n2 = dag.get(&b2).unwrap();
                let s1 = selection_score(n1.num_obs, n1.tot_b_reward, n1.net_prob, num_obs);
                let s2 = selection_score(n2.num_obs, n2.tot_b_reward, n2.net_prob, num_obs);

                if b.side_black {
                    s1.partial_cmp(&s2).unwrap() // black's turn -> maximize black reward
                } else {
                    s2.partial_cmp(&s1).unwrap() // white's turn -> minimize black reward
                }
            })
            .unwrap()
            .clone()
    }
    fn select(
        rng: &mut SmallRng,
        dag: &mut HashMap<Board, MCTSNode>,
        b0: &Board,
        pv: &PVNetwork,
    ) -> Vec<Action> {
        let mut node = dag.get_mut(b0).expect("Invalid MCTS scan");
        if node.children.is_empty() {
            // not yet expanded node
            if node.num_obs < MIN_OBS_TO_EXPAND {
                return vec![];
            } else {
                // ok to expand now
                expand(rng, dag, b0, pv);
                node = dag.get_mut(b0).unwrap();
            }
        }
        if node.children.is_empty() {
            panic!("MCTS #children must be >0");
        }

        let (a, b) = pick_best(dag, b0);
        let mut ac_path = select(rng, dag, &b, pv);
        ac_path.push(a);
        return ac_path;
    }

    insert_empty(&mut dag, b, None);
    expand(rng, &mut dag, b, pv);
    let mut num_iter = 0;
    loop {
        if t0.elapsed().as_micros() as u64 >= budget_usec {
            sample.state = b.clone();

            let mut pol = vec![];
            let num_obs = dag.get(&b).unwrap().num_obs;
            for (a, ch_b) in dag.get(&b).unwrap().children.iter() {
                let ch_num_obs = dag.get(&ch_b).unwrap().num_obs;
                pol.push((a.clone(), ch_num_obs as f32 / num_obs as f32));
            }
            sample.final_policy = pol;
            break;
        }

        let next_action = select(rng, &mut dag, b, pv).last().unwrap().clone();
        let mut b2 = b.clone();
        b2.apply(&next_action);
        let (_, v) = pv.forward(encode_board(&b2));
        record_reward(&mut dag, &b2, v.data()[0]);
        num_iter += 1;
    }

    let (a, _) = pick_best(&dag, b);
    // println!(
    //     "AZ: #nodes={} / #obs={} / #iter={}",
    //     dag.len(),
    //     dag.get(b).unwrap().num_obs,
    //     num_iter
    // );
    a
}

fn playout_b_reward(rng: &mut SmallRng, b: &Board) -> f32 {
    firestorm::profile_fn!(playout_b_reward);

    let mut curr_b = b.clone();
    for i in 0..35 {
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
        for i in 0..100 {
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

fn collect_az_samples(
    rng: &mut SmallRng,
    num: usize,
    budget_usec: u64,
    pv: &PVNetwork,
) -> Vec<AZSample> {
    let mut samples = vec![];
    for i in 0..num {
        let mut b = Board::new();

        let mut battle_samples: Vec<AZSample> = vec![];
        for i in 0..100 {
            if let Some(res) = b.is_terminal() {
                let reward_b = match res {
                    Cell::Empty => 0.5,
                    Cell::Black => 1.0,
                    Cell::White => 0.0,
                };
                for s in battle_samples.iter_mut() {
                    s.reward_b = reward_b;
                }
                samples.extend(battle_samples);
                break;
            }

            let mut sample = AZSample {
                state: b.clone(),
                final_policy: vec![],
                reward_b: 0.5,
            };
            b.apply(&az_get_action(rng, &b, budget_usec, pv, &mut sample));
            battle_samples.push(sample);
        }
    }
    samples
}

fn train_az(rng: &mut SmallRng, pv: &PVNetwork, budget_usec: u64, num_battles: usize) -> PVNetwork {
    let samples = collect_az_samples(rng, num_battles, budget_usec, &pv);
    println!("{} AZ samples collected", samples.len());

    let mut opti = Adam::new(AdamConfig {
        lr: 1e-2,
        weight_decay: None,
        betas: [0.9, 0.999],
        eps: 1e-10,
    });

    let mut pv_training: PVNetwork = pv.clone();

    let start = Instant::now();
    let mut last_updated = Instant::now();
    for _i_epoch in 0..100 {
        // TODO: batching?
        for s in &samples {
            let s_in: Tensor1D<STATE_SIZE> = encode_board(&s.state);
            let p_true: Tensor1D<ACTION_SIZE> = encode_action_policy(&s.final_policy);
            let v_true: Tensor1D<1> = encode_reward(s.reward_b);
            let (p_pred, v_pred) = pv_training.forward(s_in.trace());

            let loss =
                mse_loss(v_pred, v_true) + cross_entropy_with_logits_loss(p_pred.traced(), p_true);
            let loss_v = *loss.data();
            let gradients = loss.backward();
            opti.update(&mut pv_training, gradients)
                .expect("Unused params");

            if last_updated.elapsed().as_secs_f32() > 1.0 {
                println!(
                    "loss={:#.3} @ epoch={}, t={:#.1}",
                    loss_v,
                    _i_epoch,
                    start.elapsed().as_secs_f32()
                );
                last_updated = Instant::now();
            }
        }
    }
    pv_training
}

fn compare() {
    let mut rng = SmallRng::seed_from_u64(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64,
    );

    let mut pv = PVNetwork::default();
    pv.reset_params(&mut rng);

    let mcts_1ms = |rng: &mut SmallRng, b: &Board, _: u64| mcts_get_action(rng, b, 1000);
    let mcts_10ms = |rng: &mut SmallRng, b: &Board, _: u64| mcts_get_action(rng, b, 10_000);
    let az_1ms = |rng: &mut SmallRng, b: &Board, _: u64| {
        let mut dummy_sample = AZSample {
            state: Board::new(),
            final_policy: vec![],
            reward_b: 0.5,
        };
        az_get_action(rng, b, 1000, &pv, &mut dummy_sample)
    };
    win_rate(&mut rng, az_1ms, mcts_1ms, 5, 10000);

    pv = train_az(&mut rng, &pv, 1000, 100);
    let post_az_1ms = |rng: &mut SmallRng, b: &Board, _: u64| {
        let mut dummy_sample = AZSample {
            state: Board::new(),
            final_policy: vec![],
            reward_b: 0.5,
        };
        az_get_action(rng, b, 1000, &pv, &mut dummy_sample)
    };

    win_rate(&mut rng, post_az_1ms, mcts_1ms, 5, 10000);
}

fn main() {
    //win_rate(&mut rng, random_get_action, mcts_get_action, 1, 10000);
    //play_single(&mut rng, mcts_get_action, random_get_action, 1000);

    if firestorm::enabled() {
        firestorm::bench("./flames/", compare).unwrap();
    }
}
