mod r6;

use std::time::{SystemTime, UNIX_EPOCH};

use crate::r6::Board;
use rand::{rngs::SmallRng, Rng, SeedableRng};

fn main() {
    let mut rng =
        SmallRng::seed_from_u64(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64);

    let mut b = Board::new();

    for i in 0..100 {
        println!("turn={}\n{}", i, b);
        if let Some(res) = b.is_terminal() {
            println!("winner: {:?}", res);
            break;
        }

        let actions = b.legal_actions();
        let action = actions[rng.gen_range(0..actions.len())];
        println!("chosen: {:?} / bf={}", action, actions.len());

        b.apply(&action);
    }
}

// use std::time::Instant;

// use dfdx::{
//     gradients::GradientTape,
//     prelude::{
//         cross_entropy_with_logits_loss, mse_loss, Adam, AdamConfig, HasArrayData, Linear, Module,
//         Momentum, NoneTape, Optimizer, OwnedTape, ReLU, ResetParams, Sgd, SgdConfig, Sigmoid,
//         SplitInto,
//     },
//     tensor::{PutTape, Tensor0D, Tensor1D, Tensor2D, TensorCreator},
//     tensor_ops::{mul, SelectTo},
// };
// use rand::{Rng, SeedableRng};
// use screeps_arena::{Part, Terrain, Direction};
// use silica::{
//     area::{Grid, V2},
//     microbattle::{CreepState, MicroBattle, PlayerAction, MeleeAction, RangedAction, CreepAction},
// };

// const BOARD_SIZE: usize = 10;
// const BOARD_STATE_SIZE: usize = BOARD_SIZE * BOARD_SIZE * 1;
// const PGV_OUTPUT_SIZE: usize = PL_SELECTOR_SIZE + 1; // p_g + v
// const PL_SELECTOR_SIZE: usize = BOARD_SIZE * BOARD_SIZE;
// const PL_INPUT_SIZE: usize = BOARD_STATE_SIZE + PL_SELECTOR_SIZE;
// const PL_MELEE_SIZE: usize = BOARD_SIZE * BOARD_SIZE * 2 + 1; // attack, heal, none
// const PL_RANGED_SIZE: usize = BOARD_SIZE * BOARD_SIZE * 2 + 2; // ranged attack, ranged heal, ranged mass attack, none
// const PL_MOVE_SIZE: usize = 9; // 8 directions + none
// const PL_OUTPUT_SIZE: usize = PL_MELEE_SIZE + PL_RANGED_SIZE + PL_MOVE_SIZE;

// // value network
// // input: board state
// // output: value
// type VNetwork = (
//     (Linear<BOARD_STATE_SIZE, 32>, ReLU),
//     (Linear<32, 32>, ReLU),
//     // selector: logits for probability distribution (policy) of exploring unit in the position.
//     // value: 0~1 expected final value
//     (Linear<32, 1>, Sigmoid)
// );

// // unit policy network
// // input: board state + unit selector
// // output: unit policy
// type UPNetwork = (
//     (Linear<PL_INPUT_SIZE, 32>, ReLU),
//     (Linear<32, 32>, ReLU),
//     SplitInto<(
//         Linear<32, PL_MELEE_SIZE>,
//         Linear<32, PL_RANGED_SIZE>,
//         Linear<32, PL_MOVE_SIZE>,
//     )>,
// );

// fn gen_random_mb<R: Rng + ?Sized>(rng: &mut R) -> MicroBattle {
//     let terrain = Grid::new(10, 10, Terrain::Plain);
//     let mut mb = MicroBattle::new(terrain);

//     let cs1 = CreepState {
//         fatigue: 0,
//         hp: 200,
//         pos: V2::new(0 as i8, 0),
//     };
//     let cs2 = CreepState {
//         fatigue: 0,
//         hp: 200,
//         pos: V2::new(9 as i8, 9),
//     };
//     mb.insert_a(cs1, &vec![Part::RangedAttack, Part::Move]);
//     mb.insert_b(cs2, &vec![Part::RangedAttack, Part::Move]);

//     mb
// }

// fn extract_current_state(mb: &MicroBattle) -> Tensor1D<BOARD_STATE_SIZE> {
//     let mut state: Tensor1D<BOARD_STATE_SIZE> = Tensor1D::zeros();

//     for (i, a) in mb.state.entity_a.iter().enumerate() {
//         state.mut_data()[p_to_ix(a.pos)] = 1.0;
//     }
//     for (i, b) in mb.state.entity_b.iter().enumerate() {
//         state.mut_data()[p_to_ix(b.pos)] = -1.0;
//     }

//     state
// }

// fn p_to_ix(p: V2<i8>) -> usize {
//     let x = p.x.clamp(0, BOARD_SIZE as i8 - 1);
//     let y = p.y.clamp(0, BOARD_SIZE as i8 - 1);
//     (y * BOARD_SIZE as i8 + x) as usize
// }

// fn ix_to_p(ix: usize) -> V2<i8> {
//     let x = ix % BOARD_SIZE;
//     let y = ix / BOARD_SIZE;
//     V2::new(x as i8, y as i8)
// }

// fn create_up_input(mb: &MicroBattle, p: V2<i8>) -> Tensor1D<PL_INPUT_SIZE> {
//     let mut output: Tensor1D<PL_INPUT_SIZE> = Tensor1D::zeros();
//     output.mut_data()[0..BOARD_STATE_SIZE].copy_from_slice(extract_current_state(mb).data());
//     output.mut_data()[BOARD_STATE_SIZE..][p_to_ix(p)] = 1.0;
//     output
// }

// fn create_sp_training() -> Tensor1D<PL_SELECTOR_SIZE> {
//     let mut output: Tensor1D<PL_SELECTOR_SIZE> = Tensor1D::zeros();
//     output.mut_data()[0] = 1.0;
//     output
// }

// fn create_val_training() -> Tensor1D<1> {
//     let mut output: Tensor1D<1> = Tensor1D::zeros();
//     output
// }

// fn create_up_training() -> (
//     Tensor1D<PL_MELEE_SIZE>,
//     Tensor1D<PL_RANGED_SIZE>,
//     Tensor1D<PL_MOVE_SIZE>,
// ) {
//     let mut out_mel: Tensor1D<PL_MELEE_SIZE> = Tensor1D::zeros();
//     let mut out_rng: Tensor1D<PL_RANGED_SIZE> = Tensor1D::zeros();
//     let mut out_mov: Tensor1D<PL_MOVE_SIZE> = Tensor1D::zeros();

//     (out_mel, out_rng, out_mov)
// }

// struct AZNetwork {
//     spv_net: VNetwork,
//     up_net: UPNetwork,
// }

// fn sample_from_prob<R: Rng + ?Sized>(rng: &mut R, probs: &[f32]) -> usize {
//     // [0.1, 0.9] -> [0.1, 1.0]
//     let mut p_accum = 0.0;
//     let mut ps_accum = Vec::with_capacity(probs.len());
//     for p in probs {
//         p_accum += p;
//         ps_accum.push(p_accum);
//     }

//     let z = rng.gen::<f32>() * p_accum;
//     for (i, sum_p) in ps_accum.iter().enumerate() {
//         if z < *sum_p {
//             return i;
//         }
//     }
//     return probs.len() - 1;
// }

// fn ix_to_melee(mb: &MicroBattle, ix: usize) -> Option<Option<MeleeAction>> {
//     None
// }

// fn ix_to_ranged(mb: &MicroBattle, ix: usize) -> Option<Option<RangedAction>> {
//     None
// }

// fn ix_to_move(mb: &MicroBattle, ix: usize) -> Option<Option<Direction>> {
//     None
// }

// fn sample_action<R: Rng + ?Sized>(rng: &mut R, mb: &MicroBattle, net: &AZNetwork) -> PlayerAction {
//     let mut actions = vec![];
//     // TODO: initialize with random legal move

//     for (i, ea) in mb.state.entity_a.iter().enumerate() {
//         let sp_in = create_up_input(mb, ea.pos);
//         let (p_mel, p_rng, p_mov) = net.up_net.forward(sp_in);

//         let ac_mel = ix_to_melee(mb, sample_from_prob(rng, p_mel.log_softmax().data()));
//         let ac_rng = ix_to_ranged(mb, sample_from_prob(rng, p_rng.log_softmax().data()));
//         let ac_mov = ix_to_move(mb, sample_from_prob(rng, p_mov.log_softmax().data()));

//         match (ac_mel, ac_rng, ac_mov) {
//             (Some(mel), Some(rng), Some(mov)) => {
//                 let ac = CreepAction {
//                     melee: mel,
//                     ranged: rng,
//                     move_dir: mov,
//                 };
//                 // if is_legal(mb, ac) -> overwrite
//                 //
//             }
//             _ => {}
//         }
//     }
//     PlayerAction { creeps: actions }
// }

// fn mcts_sample<R: Rng + ?Sized>(rng: &mut R, mb: &MicroBattle, net: &AZNetwork) -> PlayerAction {

//     let mut actions = vec![];
// }

// fn main() {
//     let mut rng = rand::rngs::SmallRng::seed_from_u64(12345);

//     let mut mb = gen_random_mb(&mut rng);
//     println!("{}", mb);

//     for i in 0..100 {
//         println!("==== tick {}", i);

//         let ac_a = mb.get_action();
//         mb.flip();
//         let ac_b = mb.get_action();
//         mb.flip();
//         println!("a: {:?}\n b: {:?}", ac_a, ac_b);

//         mb.apply_action(&ac_a, &ac_b);
//         println!("{}", mb);
//     }

//     // let state: Tensor2D<64, STATE_SIZE> = Tensor2D::randn(&mut rng);
//     // let action: [usize; 64] = [(); 64].map(|_| rng.gen_range(0..ACTION_SIZE));
//     // let reward: Tensor1D<64> = Tensor1D::randn(&mut rng);
//     // let done: Tensor1D<64> = Tensor1D::zeros();
//     // let next_state: Tensor2D<64, STATE_SIZE> = Tensor2D::randn(&mut rng);

//     // initiliaze model - all weights are 0s
//     let mut v_net: VNetwork = Default::default();
//     v_net.reset_params(&mut rng);
//     let target_spv_net: VNetwork = v_net.clone();

//     let mut up_net: UPNetwork = Default::default();
//     up_net.reset_params(&mut rng);
//     let target_up_net: UPNetwork = up_net.clone();

//     let mut opti_spv = Adam::new(AdamConfig {
//         lr: 1e-2,
//         weight_decay: None,
//         betas: [0.9, 0.999],
//         eps: 1e-10,
//     });
//     let mut opti_up = Adam::new(AdamConfig {
//         lr: 1e-2,
//         weight_decay: None,
//         betas: [0.9, 0.999],
//         eps: 1e-10,
//     });

//     let spv_in = extract_current_state(&mb);
//     let up_in = create_up_input(&mb, V2::new(0, 0));

//     let sp_true = create_sp_training();
//     let val_true = create_val_training();
//     let (mel_true, rng_true, mv_true) = create_up_training();

//     // run through training data
//     for _i_epoch in 0..15 {
//         let start = Instant::now();

//         // Train V network
//         let val_pred = target_spv_net.forward(spv_in.trace());
//         let v_loss: Tensor0D<OwnedTape> = mse_loss(val_pred, val_true.clone());
//         let v_loss_v = *v_loss.data();
//         {
//             let gradients = v_loss.backward();
//             opti_spv
//                 .update(&mut v_net, gradients)
//                 .expect("Unused params");
//         }

//         // Train PL network
//         let (mel_pred, rng_pred, mv_pred) = target_up_net.forward(up_in.trace());
//         let up_loss = cross_entropy_with_logits_loss(mv_pred, mv_true.clone())
//             + cross_entropy_with_logits_loss(rng_pred.traced(), rng_true.clone())
//             + cross_entropy_with_logits_loss(mel_pred.traced(), mel_true.clone());
//         let up_loss_v = *up_loss.data();
//         {
//             let gradients = up_loss.backward();
//             opti_up
//                 .update(&mut up_net, gradients)
//                 .expect("Unused params");
//         }

//         println!(
//             "V loss={:#.3} / UP loss={:#.3} in {:?}",
//             v_loss_v,
//             up_loss_v,
//             start.elapsed()
//         );
//     }
// }
