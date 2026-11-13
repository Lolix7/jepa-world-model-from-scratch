"""
JEPA World Model from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_env_state
def init_env_state(room_size: int = 8, seed: int | None = None) -> torch.Tensor:
    # TODO: Initialize a random agent position state inside a square 2D room...
    if seed != None:
        torch.manual_seed(seed)
    return torch.randint(0,room_size,(2,)).float()

# Step 2 - apply_action
def apply_action(state: torch.Tensor, action: int, room_size: int = 8) -> torch.Tensor:
    # TODO: Apply a discrete action to the agent state with wall clamping.
    s = state.clone()
    if action == 0:
        s[1] -= 1
    elif action == 1:
        s[1] += 1
    elif action == 2:
        s[0] -= 1
    elif action == 3:
        s[0] += 1
    s[0] = min(room_size-1, max(0, s[0]) )
    s[1] = min(room_size-1, max(0, s[1]) )
    return s

# Step 3 - render_observation
def render_observation(state: torch.Tensor, room_size: int = 8) -> torch.Tensor:
    # TODO: Render a (1, room_size, room_size) float32 obs with agent pixel 1.0...
    box = torch.zeros((1, room_size,room_size))
    box[0,int(state[1]),int(state[0])] = 1.0
    return box

# Step 4 - env_reset
def env_reset(room_size: int = 8, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: Reset the environment by sampling a new state and its observation.
    state = init_env_state(room_size=room_size, seed = seed)
    return state, render_observation(state, room_size=room_size)

# Step 5 - env_step
def env_step(state: torch.Tensor, action: int, room_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: Advance the env by one action; return (next_state, next_observation).
    s = apply_action(state, action, room_size=room_size)
    obs = render_observation(s, room_size=room_size) 
    return s, obs

# Step 6 - collect_random_transitions
def collect_random_transitions(num_transitions: int, room_size: int = 8, seed: int = 0) -> dict:
    # TODO: collect a dataset of (obs, action, next_obs, state, next_state) transitions...
    if seed!=None:
        torch.manual_seed(seed)
    state, obs = env_reset(room_size=room_size, seed=seed)
    o = []
    s = []
    ns = []
    no = []
    a = []
    
    for _ in range(num_transitions):
        action = torch.randint(0,4, ()).item()
        next_state, next_obs = env_step(state, action, room_size=room_size)

        o.append(obs)
        no.append(next_obs)
        ns.append(next_state)
        s.append(state)
        a.append(action)
        state = next_state
        obs = next_obs
    return dict(
        actions=torch.tensor(a, dtype=torch.long),
        next_observations=torch.stack(no) if no else torch.empty((0, 1, room_size, room_size)),
        next_states=torch.stack(ns) if ns else torch.empty((0, 2)),
        observations=torch.stack(o) if o else torch.empty((0, 1, room_size, room_size)),
        states=torch.stack(s) if s else torch.empty((0, 2))
    )

# Step 7 - build_transition_dataset
def build_transition_dataset(num_transitions: int = 512, room_size: int = 8, seed: int = 0) -> dict:
    # TODO: Build a JEPA training-ready transition dataset by collecting random transitions...
    return collect_random_transitions(num_transitions,room_size,seed)

# Step 8 - init_encoder_params
def init_encoder_params(obs_channels: int = 1, room_size: int = 8, latent_dim: int = 32, seed: int = 0) -> dict:
    # TODO: Initialize parameters of a small CNN encoder that maps pixel observations to latent embeddings.
    torch.manual_seed(seed)
    spatial_size = (room_size + 1) // 2
    fc_in = 32 * spatial_size * spatial_size
    params = {
        "conv1_w": (torch.randn(16, obs_channels, 3, 3) * 0.1).requires_grad_(),
        "conv1_b": torch.zeros(16, requires_grad=True),

        "conv2_w": (torch.randn(32, 16, 3, 3) * 0.1).requires_grad_(),
        "conv2_b": torch.zeros(32, requires_grad=True),

        "fc_w": (torch.randn(latent_dim, fc_in) * 0.1).requires_grad_(),
        "fc_b": torch.zeros(latent_dim, requires_grad=True)
    }

    return params

# Step 9 - encoder_forward
def encoder_forward(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    # TODO: Run a small two-layer CNN that maps pixel observations to latent embeddings.
    x = F.conv2d(
        obs, encoder_params["conv1_w"], encoder_params["conv1_b"],
        stride=1, padding=1
    )
    x = F.relu(x)

    x = F.conv2d(
        x, encoder_params["conv2_w"], encoder_params["conv2_b"],
        stride=2, padding=1
    )
    x = F.relu(x)

    x = x.flatten(start_dim=1)

    x = F.linear(x, encoder_params["fc_w"], encoder_params["fc_b"])

    return x

# Step 10 - init_target_encoder
def init_target_encoder(encoder_params: dict) -> dict:
    # TODO: Create the EMA target encoder by deep-copying the online encoder params.
    return {
        key: value.detach().clone()
        for key, value in encoder_params.items()
    }

# Step 11 - ema_update
def ema_update(target_params: dict, encoder_params: dict, tau: float = 0.99) -> dict:
    # TODO: Refresh target encoder params via EMA of the online encoder.
    return {
        key: target_params[key]*tau + (1-tau)*value
        for key, value in encoder_params.items()
    }

# Step 12 - encode_batch
def encode_batch(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    # TODO: Batch-encode observations into latent embeddings using encoder params.
    return encoder_forward(obs, encoder_params)

# Step 13 - init_predictor_params
def init_predictor_params(latent_dim: int = 32, action_dim: int = 4, hidden_dim: int = 64, seed: int = 0) -> dict:
    # TODO: Initialize and return a dict of MLP predictor parameters.
    torch.manual_seed(seed)
    params = {
        'action_embed_w' : torch.randn((action_dim, latent_dim)).requires_grad_(True)*0.02,
        'fc1_w' : torch.randn((hidden_dim, 2*latent_dim)).requires_grad_(True)*0.02,
        'fc1_b' : torch.zeros(hidden_dim).requires_grad_(True),
        'fc2_w' : torch.randn((latent_dim, hidden_dim)).requires_grad_(True)*0.02,
        'fc2_b' : torch.zeros(latent_dim).requires_grad_(True),
    }
    return params

# Step 14 - embed_action
def embed_action(actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # TODO: Embed discrete actions into continuous vectors via a learned action embedding table.
    return predictor_params['action_embed_w'][actions]

# Step 15 - predictor_forward
def predictor_forward(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # TODO: Implement predictor_forward, the forward pass of the action-conditioned dynamics predictor.
    embed = embed_action(actions, predictor_params)
    conc = torch.cat([embeddings, embed], dim=1)
    h1 = torch.nn.functional.linear(conc, predictor_params['fc1_w'], predictor_params['fc1_b'])
    z = torch.nn.functional.relu(h1)
    h2 = torch.nn.functional.linear(z, predictor_params['fc2_w'], predictor_params['fc2_b'])
    return h2

# Step 16 - predict_next_embedding
def predict_next_embedding(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # TODO: Predict the next latent embedding given current embeddings and actions...
    return predictor_forward(embeddings,actions, predictor_params)

# Step 17 - prediction_loss
def prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # TODO: Compute the JEPA prediction loss as mean squared error...
    return torch.mean((predicted-target)**2)

# Step 18 - variance_loss
def variance_loss(embeddings: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    # TODO: Compute VICReg variance hinge loss encouraging each dim std >= gamma.
    std = torch.sqrt(embeddings.var(dim=0) + eps)
    return torch.mean(torch.relu(gamma-std))

# Step 19 - covariance_loss
def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    # TODO: Implement `covariance_loss` to compute the VICReg covariance regularization from embeddings.
    z = embeddings - torch.mean(embeddings, dim=0)
    C = 1/(embeddings.shape[0]-1) * z.T @ z
    l = C.pow(2).sum()
    l -= C.diagonal().pow(2).sum()
    l /= C.shape[1]
    return l

# Step 20 - vicreg_regularizer
def vicreg_regularizer(embeddings: torch.Tensor, var_weight: float = 1.0, cov_weight: float = 0.04, gamma: float = 1.0) -> torch.Tensor:
    # TODO: Combine variance and covariance terms into the VICReg regularizer...
    return var_weight * variance_loss(embeddings, gamma) + cov_weight * covariance_loss(embeddings)

# Step 21 - jepa_loss
def jepa_loss(predicted: torch.Tensor, target: torch.Tensor, online_embeddings: torch.Tensor, pred_weight: float = 1.0, var_weight: float = 1.0, cov_weight: float = 0.04) -> torch.Tensor:
    # TODO: Compose the full JEPA objective from prediction and VICReg terms...
    return pred_weight*prediction_loss(predicted, target) + vicreg_regularizer(online_embeddings, var_weight=var_weight, cov_weight=cov_weight)

# Step 22 - collapse_metric
def collapse_metric(embeddings: torch.Tensor) -> torch.Tensor:
    # TODO: Measure collapse as the mean of per-dimension batch stds
    std = embeddings.std(dim=0)
    return torch.mean(std)

# Step 23 - jepa_training_step
import torch.nn.functional as F
def jepa_training_step(batch: dict, encoder_params: dict, target_params: dict, predictor_params: dict, lr: float = 1e-3, tau: float = 0.99) -> tuple[dict, dict, dict, float, float]:
    # TODO: Perform one full JEPA gradient update on a batch of transitions
    predictor_params = {
        k: v.detach().requires_grad_(True)
        for k, v in predictor_params.items()
    }
    encoder_params = {
        k: v.detach().requires_grad_(True)
        for k, v in encoder_params.items()
    }
    online = encode_batch(batch['observations'], encoder_params)
    with torch.no_grad():
        target = encode_batch(batch['next_observations'], target_params)
    predicted = predict_next_embedding(online, batch['actions'], predictor_params)
    for p in list(encoder_params.values()) + list(predictor_params.values()):
        p.grad = None
    loss = jepa_loss(predicted, target, online)
    loss.backward()

    new_encode_params = {}
    for key,value in encoder_params.items():
        new_encode_params[key] = (value - lr * value.grad).detach().requires_grad_()
    
    new_predict_params = {}
    for key,value in predictor_params.items():
        
        new_predict_params[key] = (value - lr * value.grad).detach().requires_grad_()
    
    new_target_params = ema_update(target_params, new_encode_params, tau )


    loss_value = loss.item()
    col_val = collapse_metric(online.detach()).item()

    return (new_encode_params, new_target_params, new_predict_params, loss_value, col_val)

# Step 24 - train_jepa
def train_jepa(dataset: dict, encoder_params: dict, target_params: dict, predictor_params: dict, num_steps: int = 50, batch_size: int = 32, lr: float = 1e-3, tau: float = 0.99, seed: int = 0) -> tuple[dict, dict, dict, list]:
    # TODO: Train JEPA for num_steps with collapse monitoring, return params + history
    torch.manual_seed(seed)
    N = dataset['observations'].shape[0]
    history = []
    loss_value, collapse_value=0,0
    for _ in range(num_steps):
        indices = torch.randint(N, (batch_size,))
        batch = {'observations':dataset['observations'][indices], 'actions':dataset['actions'][indices], 'next_observations': dataset['next_observations'][indices]}
        encoder_params, target_params, predictor_params, loss_value, collapse_value = jepa_training_step(batch, encoder_params, target_params, predictor_params, lr=lr, tau=tau)
        history.append({'loss':loss_value, 'collapse':collapse_value})
    return encoder_params,target_params,predictor_params, history

# Step 25 - rollout_latent_dynamics
def rollout_latent_dynamics(initial_embedding: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # TODO: Roll out multi-step latent dynamics via the action-conditioned predictor...
    batched = initial_embedding.dim() == 2

    current = initial_embedding.unsqueeze(0) if not batched else initial_embedding

    if actions.dim() == 1:
        actions = actions.unsqueeze(0).expand(current.shape[0], -1)

    init = [current]

    for t in range(actions.shape[1]):
        current = predict_next_embedding(current, actions[:, t], predictor_params)
        init.append(current)

    result = torch.stack(init, dim=0)

    return result if batched else result.squeeze(1)

# Step 26 - multi_step_prediction_error
def multi_step_prediction_error(dataset: dict, encoder_params: dict, target_params: dict, predictor_params: dict, horizon: int = 5, num_samples: int = 32) -> float:
    # TODO: Evaluate multi-step latent prediction accuracy via mean MSE over samples
    obs = dataset['observations']
    action = dataset['actions']
    next_obs = dataset['next_observations']
    N = obs.shape[0]
    indices = list(range(min(num_samples, N - horizon)))

    with torch.no_grad():

        # Encode initial observations
        initial_obs = obs[indices]
        initial_embeddings = encode_batch(initial_obs, encoder_params)

        # Gather action sequences: (B, T)
        actions = torch.stack([
            action[i:i+horizon] for i in indices
        ])

        # Predict future latent states: (T+1, B, D)
        rollout = rollout_latent_dynamics(
            initial_embeddings,
            actions,
            predictor_params
        )

        # Remove initial embedding: (T, B, D)
        pred = rollout[1:]

        # Gather true future observations: (B, T, C, H, W)
        future_obs = torch.stack([
            next_obs[i:i+horizon] for i in indices
        ])

        B, T = future_obs.shape[:2]

        # Flatten batch and time for encoder
        future_obs = future_obs.reshape(
            B * T, *future_obs.shape[2:]
        )

        # Encode with EMA target encoder
        true_embeddings = encode_batch(future_obs, target_params)

        # Restore shape: (T, B, D)
        true_embeddings = true_embeddings.reshape(
            B, T, -1
        ).transpose(0, 1)

        # Mean squared error
        error = ((pred - true_embeddings) ** 2).mean()

    return error.item()

# Step 27 - init_linear_probe
def init_linear_probe(latent_dim: int = 32, state_dim: int = 2, seed: int = 0) -> dict:
    # TODO: Initialize a linear probe that maps latent embeddings to true agent state (x, y).
    torch.manual_seed(seed)
    return {
        'w': torch.randn(state_dim, latent_dim )*0.01,
        'b': torch.zeros(state_dim)
    }

# Step 28 - train_linear_probe
def train_linear_probe(embeddings: torch.Tensor, states: torch.Tensor, probe_params: dict, num_steps: int = 100, lr: float = 1e-2) -> dict:
    # TODO: Train the linear probe via MSE regression from frozen embeddings to true agent states.
    w = probe_params['w'].clone().detach().requires_grad_()
    b = probe_params['b'].clone().detach().requires_grad_()

    X = embeddings.detach()
    y = states.detach()
    for _ in range(num_steps):
        pred = X @ w + b
        loss = (pred-y).pow(2).mean()   

        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    return {'w': w.detach().clone(), 'b': b.detach().clone()}

# Step 29 - probe_state_recovery
def probe_state_recovery(dataset: dict, encoder_params: dict, probe_params: dict | None = None, num_probe_steps: int = 100) -> dict:
    obs = dataset['observations']
    states = dataset['states'].float()

    with torch.no_grad():
        X = encode_batch(obs, encoder_params)

    if probe_params is None:
        torch.manual_seed(0)
        probe_params = init_linear_probe(latent_dim=X.shape[-1],
                                         state_dim=states.shape[-1], seed=0)
    # (state_dim, latent_dim) -> (latent_dim, state_dim) pour train_linear_probe
    p = {'w': probe_params['w'].t().contiguous(), 'b': probe_params['b']}
    out = train_linear_probe(X, states, p, num_steps=num_probe_steps)

    # on remet w au format de init_linear_probe : (state_dim, latent_dim)
    w = out['w'].t().contiguous()
    b = out['b']
    trained = {'w': w, 'b': b}

    with torch.no_grad():
        pred = X @ out['w'] + b          # même calcul que train_linear_probe
        mse = torch.mean((pred - states) ** 2).item()
        mae = torch.mean(torch.abs(pred - states)).item()

    return {'mse': float(mse), 'mean_abs_error': float(mae), 'probe_params': trained}

# Step 30 - encode_goal
def encode_goal(goal_state: torch.Tensor, encoder_params: dict, room_size: int = 8) -> torch.Tensor:
    # TODO: Convert a desired agent position into a latent goal embedding.
    box = render_observation(goal_state,room_size).unsqueeze(0)
    
    x = encoder_forward(box, encoder_params)
    
    return x.squeeze(0)

# Step 31 - latent_cost
def latent_cost(latents, goal_embedding):
    # TODO: Compute squared L2 distance from each latent embedding to a goal embedding...
    mse = (latents-goal_embedding).pow(2).sum(dim=-1)
    return mse

# Step 32 - sample_action_sequences
def sample_action_sequences(n_sequences, horizon, n_actions):
    # TODO: Sample random discrete action sequences for random-shooting MPC...
    return torch.randint(0, n_actions, (n_sequences, horizon))

# Step 33 - score_action_sequences
def score_action_sequences(start_embedding, action_sequences, goal_embedding, predictor_params):
    # TODO: Score each action sequence by cumulative latent distance to the goal...
    N, H = action_sequences.shape[0], action_sequences.shape[1]

    # (D,) -> (N, D) : même point de départ pour chaque séquence
    z = start_embedding.unsqueeze(0).expand(N, -1)

    scores = torch.zeros(N, dtype=z.dtype, device=z.device)
    for t in range(H):
        z = predict_next_embedding(z, action_sequences[:, t], predictor_params)  # (N, D)
        scores = scores + latent_cost(z, goal_embedding)                         # (N,)

    return scores

# Step 34 - select_best_plan
def select_best_plan(action_sequences, costs):
    # TODO: Return the action sequence with the lowest cost from the candidates.
    i = torch.argmin(costs)
    return action_sequences[i]

# Step 35 - mpc_step
def mpc_step(start_embedding, goal_embedding, predictor_params, n_sequences, horizon, n_actions):
    # TODO: Perform one random-shooting MPC step; return first action of best plan
    actions = sample_action_sequences(n_sequences, horizon, n_actions)
    scores = score_action_sequences(start_embedding, actions, goal_embedding,predictor_params)
    return int(select_best_plan(actions, scores)[0])

# Step 36 - run_mpc_episode (not yet solved)
# TODO: implement

# Step 37 - evaluate_planner (not yet solved)
# TODO: implement

# Step 38 - jepa_world_model_experiment (not yet solved)
# TODO: implement

