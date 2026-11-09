# JEPA World Model from Scratch

Build an action-conditioned Joint Embedding Predictive Architecture (JEPA) world model in pure PyTorch. An encoder and EMA target learn collapse-resistant latents of a 2D room via VICReg, a predictor rolls dynamics in embedding space, and acting is random-shooting MPC toward goal embeddings. No decoder, no pixels, no learned policy.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** init_env_state
- [ ] **2.** apply_action
- [ ] **3.** render_observation
- [ ] **4.** env_reset
- [ ] **5.** env_step
- [ ] **6.** collect_random_transitions
- [ ] **7.** build_transition_dataset
- [ ] **8.** init_encoder_params
- [ ] **9.** encoder_forward
- [ ] **10.** init_target_encoder
- [ ] **11.** ema_update
- [ ] **12.** encode_batch
- [ ] **13.** init_predictor_params
- [ ] **14.** embed_action
- [ ] **15.** predictor_forward
- [ ] **16.** predict_next_embedding
- [ ] **17.** prediction_loss
- [ ] **18.** variance_loss
- [ ] **19.** covariance_loss
- [ ] **20.** vicreg_regularizer
- [ ] **21.** jepa_loss
- [ ] **22.** collapse_metric
- [ ] **23.** jepa_training_step
- [ ] **24.** train_jepa
- [ ] **25.** rollout_latent_dynamics
- [ ] **26.** multi_step_prediction_error
- [ ] **27.** init_linear_probe
- [ ] **28.** train_linear_probe
- [ ] **29.** probe_state_recovery
- [ ] **30.** encode_goal
- [ ] **31.** latent_cost
- [ ] **32.** sample_action_sequences
- [ ] **33.** score_action_sequences
- [ ] **34.** select_best_plan
- [ ] **35.** mpc_step
- [ ] **36.** run_mpc_episode
- [ ] **37.** evaluate_planner
- [ ] **38.** jepa_world_model_experiment

---

Built on Deep-ML.
