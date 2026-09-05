"""
NextLat from Scratch: Next-Latent Prediction in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - grid_step
def grid_step(pos: tuple, action: int, G: int) -> tuple:
    # TODO: Apply one action to a (row, col) position on a G x G grid.
    row, col = pos

    # Actions: 0=up, 1=down, 2=left, 3=right
    if action == 0:
        new_pos = (row - 1, col)
    elif action == 1:
        new_pos = (row + 1, col)
    elif action == 2:
        new_pos = (row, col - 1)
    elif action == 3:
        new_pos = (row, col + 1)
    else:
        raise ValueError("action must be one of 0, 1, 2, or 3")

    new_row, new_col = new_pos

    # Move is legal only if the new position remains inside the grid.
    if 0 <= new_row < G and 0 <= new_col < G:
        return new_pos, True

    # Illegal move: keep the original position unchanged.
    return pos, False

# Step 2 - legal_actions
def legal_actions(pos: tuple, G: int) -> list:
    # TODO: Return the sorted list of legal action ids from pos.
    row, col = pos

    actions = []

    # 0 = up
    if row > 0:
        actions.append(0)

    # 1 = down
    if row < G - 1:
        actions.append(1)

    # 2 = left
    if col > 0:
        actions.append(2)

    # 3 = right
    if col < G - 1:
        actions.append(3)

    return actions

# Step 3 - random_walk_to_goal
def random_walk_to_goal(start: tuple, goal: tuple, G: int, max_len: int, rng) -> list:
    # TODO: Random legal moves from start until goal is reached or max_len moves.
    pos = start
    moves = []

    while pos != goal and len(moves) < max_len:
        actions = legal_actions(pos, G)
        action = int(rng.choice(actions))
        moves.append(action)
        pos, _ = grid_step(pos, action, G)

    return moves

# Step 4 - encode_sequence
def encode_sequence(start: tuple, goal: tuple, moves: list, G: int, T: int) -> tuple:
    # TODO: Build [start_cell, goal_cell, moves..., EOS, pad...] as a (T,) long tensor plus a bool mask.
    EOS = 4 + G * G

    start_cell = 4 + start[0] * G + start[1]
    goal_cell = 4 + goal[0] * G + goal[1]

    # Reserve one position for EOS.
    num_moves = max(0, T - 3)
    selected_moves = moves[:num_moves]

    sequence = [start_cell, goal_cell]
    sequence.extend(selected_moves)
    sequence.append(EOS)

    tokens = torch.full((T,), EOS, dtype=torch.long)
    mask = torch.zeros((T,), dtype=torch.bool)

    length = min(len(sequence), T)
    tokens[:length] = torch.tensor(sequence[:length], dtype=torch.long)
    mask[:length] = True

    return tokens, mask

# Step 5 - make_dataset
def make_dataset(n: int, G: int, T: int, seed: int = 0) -> dict:
    # TODO: Generate n encoded walks plus the true cell index after every token.
    rng = np.random.default_rng(seed)

    tokens_list = []
    mask_list = []
    states_list = []

    for _ in range(n):
        # Draw start and goal cells uniformly at random.
        start = (
            int(rng.integers(0, G)),
            int(rng.integers(0, G)),
        )
        goal = (
            int(rng.integers(0, G)),
            int(rng.integers(0, G)),
        )

        # Generate a random legal walk toward the goal.
        moves = random_walk_to_goal(
            start,
            goal,
            G=G,
            max_len=T - 3,
            rng=rng,
        )

        # Encode the sequence.
        tokens, mask = encode_sequence(
            start,
            goal,
            moves,
            G=G,
            T=T,
        )

        # Track the true walker position after consuming each token.
        states = []
        pos = start

        for token in tokens.tolist():
            if 0 <= token <= 3:
                # Action token: apply the action to the current position.
                pos, _ = grid_step(pos, token, G)

            # Cell tokens, EOS, and padding leave the position unchanged.
            states.append(pos[0] * G + pos[1])

        tokens_list.append(tokens)
        mask_list.append(mask)
        states_list.append(torch.tensor(states, dtype=torch.long))

    return {
        "tokens": torch.stack(tokens_list),
        "mask": torch.stack(mask_list),
        "states": torch.stack(states_list),
        "G": G,
    }

# Step 6 - get_batch
def get_batch(dataset: dict, batch_size: int, step: int) -> dict:
    # TODO: Cyclic row slice; return shifted x/y views with aligned mask and states.
    n = dataset["tokens"].shape[0]

    # Deterministic cyclic row indices for this batch.
    indices = [
        (step * batch_size + i) % n
        for i in range(batch_size)
    ]

    tokens = dataset["tokens"][indices]
    mask = dataset["mask"][indices]
    states = dataset["states"][indices]

    return {
        "x": tokens[:, :-1],
        "y": tokens[:, 1:],
        "mask": mask[:, 1:],
        "states": states[:, :-1],
    }

# Step 7 - causal_mask
def causal_mask(T: int):
    # TODO: Return a (T, T) bool tensor, True where key index <= query index.
    return torch.tril(torch.ones((T, T), dtype=torch.bool))

# Step 8 - init_gpt_params
def init_gpt_params(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    max_len: int,
    seed: int = 0,
) -> dict:
    # TODO: Seed, then allocate every GPT tensor in the documented order
    # (std 0.02 matrices, ones/zeros LN, zero biases).
    torch.manual_seed(seed)

    params = {}

    def randn(shape):
        return (torch.randn(*shape, dtype=torch.float32) * 0.02).requires_grad_()

    def zeros(shape):
        return torch.zeros(*shape, dtype=torch.float32, requires_grad=True)

    def ones(shape):
        return torch.ones(*shape, dtype=torch.float32, requires_grad=True)

    # Token and positional embeddings.
    params["wte"] = randn((vocab_size, d_model))
    params["wpe"] = randn((max_len, d_model))

    # Transformer layers.
    for l in range(n_layers):
        params[f"ln1_w{l}"] = ones((d_model,))
        params[f"ln1_b{l}"] = zeros((d_model,))

        params[f"qkv_w{l}"] = randn((d_model, 3 * d_model))
        params[f"qkv_b{l}"] = zeros((3 * d_model,))

        params[f"proj_w{l}"] = randn((d_model, d_model))
        params[f"proj_b{l}"] = zeros((d_model,))

        params[f"ln2_w{l}"] = ones((d_model,))
        params[f"ln2_b{l}"] = zeros((d_model,))

        params[f"fc_w{l}"] = randn((d_model, 4 * d_model))
        params[f"fc_b{l}"] = zeros((4 * d_model,))

        params[f"fc2_w{l}"] = randn((4 * d_model, d_model))
        params[f"fc2_b{l}"] = zeros((d_model,))

    # Final LayerNorm and output head.
    params["lnf_w"] = ones((d_model,))
    params["lnf_b"] = zeros((d_model,))

    params["head_w"] = randn((d_model, vocab_size))
    params["head_b"] = zeros((vocab_size,))

    return params

# Step 9 - attention_block
def attention_block(x, params: dict, layer: int, n_heads: int):
    # TODO: x + Proj(causal multi-head attention(LayerNorm(x))) for the given layer.
    B, T, d = x.shape
    head_dim = d // n_heads

    # Pre-LayerNorm.
    ln_w = params[f"ln1_w{layer}"]
    ln_b = params[f"ln1_b{layer}"]

    z = torch.nn.functional.layer_norm(
        x,
        normalized_shape=(d,),
        weight=ln_w,
        bias=ln_b,
        eps=1e-5,
    )

    # Compute Q, K, V.
    qkv = z @ params[f"qkv_w{layer}"] + params[f"qkv_b{layer}"]
    q, k, v = torch.chunk(qkv, 3, dim=-1)

    # Reshape to (B, n_heads, T, head_dim).
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)

    # Scaled dot-product attention.
    scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)

    # Mask future positions.
    mask = causal_mask(T).to(device=x.device)
    scores = scores.masked_fill(~mask, float("-inf"))

    # Normalize over key positions.
    attn = torch.softmax(scores, dim=-1)

    # Apply attention weights to values.
    context = attn @ v

    # Merge heads back to (B, T, d).
    context = context.transpose(1, 2).contiguous().view(B, T, d)

    # Output projection and residual connection.
    output = context @ params[f"proj_w{layer}"] + params[f"proj_b{layer}"]

    return x + output

# Step 10 - mlp_block
def mlp_block(x, params: dict, layer: int):
    # TODO: x + FC2(gelu_tanh(FC1(LayerNorm(x)))) for the given layer.
    d = x.shape[-1]

    # Pre-LayerNorm.
    z = torch.nn.functional.layer_norm(
        x,
        normalized_shape=(d,),
        weight=params[f"ln2_w{layer}"],
        bias=params[f"ln2_b{layer}"],
        eps=1e-5,
    )

    # Feed-forward network with tanh-approximated GELU.
    h = z @ params[f"fc_w{layer}"] + params[f"fc_b{layer}"]
    h = torch.nn.functional.gelu(h, approximate="tanh")

    out = h @ params[f"fc2_w{layer}"] + params[f"fc2_b{layer}"]

    # Residual connection.
    return x + out

# Step 11 - gpt_hidden_states
def gpt_hidden_states(tokens, params: dict, n_heads: int):
    # TODO: Embeddings -> n pre-LN blocks -> final LayerNorm; return (B, T, d) hidden states.
    B, T = tokens.shape
    d = params["wte"].shape[1]

    # Token + positional embeddings.
    x = params["wte"][tokens] + params["wpe"][:T].unsqueeze(0)

    # Infer the number of transformer layers from ln1_w{layer} keys.
    n_layers = sum(
        1 for key in params
        if key.startswith("ln1_w") and key[len("ln1_w"):].isdigit()
    )

    # Apply attention and MLP blocks for every layer.
    for layer in range(n_layers):
        x = attention_block(x, params, layer=layer, n_heads=n_heads)
        x = mlp_block(x, params, layer=layer)

    # Final LayerNorm.
    x = torch.nn.functional.layer_norm(
        x,
        normalized_shape=(d,),
        weight=params["lnf_w"],
        bias=params["lnf_b"],
        eps=1e-5,
    )

    return x

# Step 12 - output_head
def output_head(h, params: dict):
    # TODO: Linear map from hidden states to logits over the vocabulary.
    return h @ params["head_w"] + params["head_b"]

# Step 13 - next_token_loss
def next_token_loss(logits, targets, mask):
    # TODO: Masked mean cross-entropy from logits; 0.0 tensor if the mask is empty.
    B, T, V = logits.shape

    # Compute per-position cross-entropy from logits.
    losses = torch.nn.functional.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        reduction="none",
    ).reshape(B, T)

    # Keep only valid/real target positions.
    masked_losses = losses[mask]

    # Return a scalar zero tensor when no positions are selected.
    if masked_losses.numel() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    return masked_losses.mean()

# Step 14 - init_dynamics_params
def init_dynamics_params(d_model: int, hidden: int, seed: int = 0) -> dict:
    # TODO: Seed, then allocate W1,b1,W2,b2,W3,b3 for a 3-layer MLP reading 2*d_model inputs.
    torch.manual_seed(seed)

    params = {}

    params["W1"] = (torch.randn(2 * d_model, hidden, dtype=torch.float32) * 0.02).requires_grad_()
    params["b1"] = torch.zeros(hidden, dtype=torch.float32, requires_grad=True)

    params["W2"] = (torch.randn(hidden, hidden, dtype=torch.float32) * 0.02).requires_grad_()
    params["b2"] = torch.zeros(hidden, dtype=torch.float32, requires_grad=True)

    params["W3"] = (torch.randn(hidden, d_model, dtype=torch.float32) * 0.02).requires_grad_()
    params["b3"] = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    return params

# Step 15 - latent_transition
def latent_transition(h, x_emb, dyn: dict):
    # TODO: LayerNorm(concat(h, x_emb)) -> 3-layer GELU MLP -> delta; return delta + h.
    z = torch.cat([h, x_emb], dim=-1)

    # LayerNorm over the concatenated feature dimension, with no
    # learned scale or shift.
    z = torch.nn.functional.layer_norm(
        z,
        normalized_shape=(z.shape[-1],),
        weight=None,
        bias=None,
        eps=1e-5,
    )

    # 3-layer MLP with tanh-approximated GELU activations.
    a1 = z @ dyn["W1"] + dyn["b1"]
    a1 = torch.nn.functional.gelu(a1, approximate="tanh")

    a2 = a1 @ dyn["W2"] + dyn["b2"]
    a2 = torch.nn.functional.gelu(a2, approximate="tanh")

    delta = a2 @ dyn["W3"] + dyn["b3"]

    # Residual next-hidden prediction.
    return h + delta

# Step 16 - rollout_latents
def rollout_latents(h, x, params: dict, dyn: dict, d_steps: int) -> list:
    # TODO: Recursive d-step rollout from h[:, :T-d_steps];
    # step i consumes wte[x[:, i:T-d_steps+i]].
    B, T, d = h.shape
    L = T - d_steps

    # Initial latent states are the true hidden states at positions
    # that have enough room for a d_steps rollout.
    h_hat = h[:, :L]

    predictions = []

    for i in range(1, d_steps + 1):
        # Token embeddings corresponding to the i-th future positions.
        emb = params["wte"][x[:, i:L + i]]

        # Feed the previous prediction forward recursively.
        h_hat = latent_transition(h_hat, emb, dyn)
        predictions.append(h_hat)

    return predictions

# Step 17 - next_hidden_loss
def next_hidden_loss(h, h_hats: list, mask, beta: float = 1.0):
    # TODO: Smooth L1 between each rolled-out latent and the DETACHED true hidden state,
    # masked mean, averaged over steps.

    if not h_hats:
        return torch.tensor(0.0, device=h.device, dtype=h.dtype)

    d_steps = len(h_hats)
    T = h.shape[1]
    L = T - d_steps

    step_losses = []

    for i, h_hat in enumerate(h_hats, start=1):
        # Target hidden states are stop-gradient.
        target = h[:, i:L + i].detach()

        # Aligned validity mask.
        step_mask = mask[:, i:L + i]

        # Smooth L1 loss per feature, then average over features.
        loss = torch.nn.functional.smooth_l1_loss(
            h_hat,
            target,
            reduction="none",
            beta=beta,
        ).mean(dim=-1)

        # Average only over masked positions.
        if step_mask.any():
            loss = loss[step_mask].mean()
        else:
            loss = torch.tensor(0.0, device=h.device, dtype=h.dtype)

        step_losses.append(loss)

    # Average across rollout steps.
    return torch.stack(step_losses).mean()

# Step 18 - kl_alignment_loss
def kl_alignment_loss(h, h_hats: list, mask, params: dict):
    # TODO: Forward KL(true || predicted) in token space through a DETACHED
    # output head; masked mean; mean over steps.

    if not h_hats:
        return torch.tensor(0.0, device=h.device, dtype=h.dtype)

    d_steps = len(h_hats)
    T = h.shape[1]
    L = T - d_steps

    # Freeze the output head by detaching its parameters.
    frozen = {
        "head_w": params["head_w"].detach(),
        "head_b": params["head_b"].detach(),
    }

    step_losses = []

    for i, h_hat in enumerate(h_hats, start=1):
        start = i
        end = L + i

        # Real latent distribution, with the latent itself detached.
        logits_true = output_head(h[:, start:end].detach(), frozen)

        # Predicted latent distribution.
        logits_pred = output_head(h_hat, frozen)

        # KL(true || predicted):
        # sum_v p_true(v) * [log p_true(v) - log p_pred(v)]
        log_probs_true = torch.nn.functional.log_softmax(logits_true, dim=-1)
        log_probs_pred = torch.nn.functional.log_softmax(logits_pred, dim=-1)
        probs_true = log_probs_true.exp()

        kl = (probs_true * (log_probs_true - log_probs_pred)).sum(dim=-1)

        # Align the mask with the current prediction horizon.
        step_mask = mask[:, start:end]

        if step_mask.any():
            step_losses.append(kl[step_mask].mean())
        else:
            step_losses.append(
                torch.tensor(0.0, device=h.device, dtype=h.dtype)
            )

    # Mean KL across rollout steps.
    return torch.stack(step_losses).mean()

# Step 19 - nextlat_loss
def nextlat_loss(
    batch: dict,
    params: dict,
    dyn: dict,
    n_heads: int,
    d_steps: int,
    lam_h: float,
    lam_kl: float,
    beta: float = 1.0,
) -> dict:
    # TODO: next-token CE + lam_h * next-hidden Smooth L1 + lam_kl * frozen-head KL,
    # sharing one rollout.

    # Run the GPT on the input tokens and obtain the real hidden states.
    h = gpt_hidden_states(batch["x"], params, n_heads)

    # Next-token prediction loss.
    logits = output_head(h, params)
    next_token = next_token_loss(
        logits,
        batch["y"],
        batch["mask"],
    )

    if d_steps > 0:
        # EOS is the final vocabulary token.
        eos = params["head_b"].shape[0] - 1

        # Valid input-token positions for latent rollout.
        mask_x = batch["x"] != eos

        # Compute the shared latent rollout once.
        h_hats = rollout_latents(
            h,
            batch["x"],
            params,
            dyn,
            d_steps,
        )

        # Next-hidden-state loss and frozen-head KL alignment.
        next_h = next_hidden_loss(
            h,
            h_hats,
            mask_x,
            beta=beta,
        )

        kl = kl_alignment_loss(
            h,
            h_hats,
            mask_x,
            params,
        )
    else:
        next_h = torch.tensor(0.0, device=h.device, dtype=h.dtype)
        kl = torch.tensor(0.0, device=h.device, dtype=h.dtype)

    # Full NextLat objective.
    total = next_token + lam_h * next_h + lam_kl * kl

    return {
        "total": total,
        "next_token": next_token,
        "next_h": next_h,
        "kl": kl,
    }

# Step 20 - train_step
def train_step(
    batch: dict,
    params: dict,
    dyn: dict,
    opt,
    n_heads: int,
    d_steps: int,
    lam_h: float,
    lam_kl: float,
    beta: float = 1.0,
) -> dict:
    # TODO: zero_grad -> nextlat_loss -> backward on 'total' -> step;
    # return the four losses as floats.

    opt.zero_grad()

    losses = nextlat_loss(
        batch,
        params,
        dyn,
        n_heads,
        d_steps,
        lam_h,
        lam_kl,
        beta,
    )

    losses["total"].backward()
    opt.step()

    return {
        "total": losses["total"].item(),
        "next_token": losses["next_token"].item(),
        "next_h": losses["next_h"].item(),
        "kl": losses["kl"].item(),
    }

# Step 21 - train_model
def train_model(dataset: dict, cfg: dict, seed: int = 0) -> tuple:
    # TODO: Init GPT + dynamics params, one Adam over both, loop train_step
    # over cyclic batches; return (params, dyn, history).

    G = dataset["G"]
    T = dataset["tokens"].shape[1]

    vocab_size = 4 + G * G + 1
    max_len = T

    # Initialize GPT parameters.
    params = init_gpt_params(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        max_len=max_len,
        seed=seed,
    )

    # Initialize latent dynamics parameters.
    dyn = init_dynamics_params(
        cfg["d_model"],
        cfg["hidden"],
        seed=seed,
    )

    # One optimizer over both GPT and dynamics parameters.
    opt = torch.optim.Adam(
        list(params.values()) + list(dyn.values()),
        lr=cfg["lr"],
    )

    history = []

    for step in range(cfg["steps"]):
        batch = get_batch(
            dataset,
            cfg["batch_size"],
            step,
        )

        losses = train_step(
            batch,
            params,
            dyn,
            opt,
            n_heads=cfg["n_heads"],
            d_steps=cfg["d_steps"],
            lam_h=cfg["lam_h"],
            lam_kl=cfg["lam_kl"],
            beta=cfg["beta"],
        )

        history.append(losses)

    return params, dyn, history

# Step 22 - greedy_decode
def greedy_decode(params: dict, n_heads: int, prefix: list, n_tokens: int) -> list:
    # TODO: Repeatedly run the transformer on the full sequence and append
    # the argmax of the last position.
    sequence = list(prefix)
    generated = []

    with torch.no_grad():
        for _ in range(n_tokens):
            tokens = torch.tensor(
                [sequence],
                dtype=torch.long,
                device=params["wte"].device,
            )

            h = gpt_hidden_states(tokens, params, n_heads)
            logits = output_head(h[:, -1, :], params)

            next_token = int(torch.argmax(logits, dim=-1).item())

            sequence.append(next_token)
            generated.append(next_token)

    return generated

# Step 23 - effective_rank
def effective_rank(H, tol: float = 1e-12) -> float:
    # TODO: exp of the Shannon entropy of the normalized singular values above tol.
    singular_values = torch.linalg.svdvals(H)

    # Keep only singular values strictly greater than tol.
    singular_values = singular_values[singular_values > tol]

    if singular_values.numel() == 0:
        return 0.0

    # Normalize singular values into a probability distribution.
    probs = singular_values / singular_values.sum()

    # Shannon entropy using the natural logarithm.
    entropy = -(probs * torch.log(probs)).sum()

    return float(torch.exp(entropy).item())

# Step 24 - eval_hidden_states
def eval_hidden_states(dataset: dict, params: dict, n_heads: int, n_rows: int):
    # TODO: Hidden states at real input positions of the first n_rows sequences,
    # as an (N, d) matrix.

    x = dataset["tokens"][:n_rows, :-1]
    mask_x = dataset["mask"][:n_rows, :-1]

    with torch.no_grad():
        h = gpt_hidden_states(x, params, n_heads)

    return h[mask_x].detach()

# Step 25 - valid_move_rate
def valid_move_rate(dataset: dict, params: dict, n_heads: int, n_rows: int) -> float:
    # TODO: Fraction of argmax predictions (positions t>=1, real targets)
    # that are legal under the true state.

    G = dataset["G"]
    eos = 4 + G * G

    x = dataset["tokens"][:n_rows, :-1]
    y_mask = dataset["mask"][:n_rows, 1:]
    states = dataset["states"][:n_rows, :-1]

    with torch.no_grad():
        h = gpt_hidden_states(x, params, n_heads)
        logits = output_head(h, params)
        predictions = torch.argmax(logits, dim=-1)

    total = 0
    legal = 0

    # Position 0 predicts the goal token, so only positions t >= 1 are scored.
    B, T = x.shape

    for b in range(B):
        goal_cell = int(dataset["tokens"][b, 1].item()) - 4

        for t in range(1, T):
            if not bool(y_mask[b, t]):
                continue

            pos_idx = int(states[b, t].item())
            pos = (pos_idx // G, pos_idx % G)
            pred = int(predictions[b, t].item())

            if pos_idx == goal_cell:
                is_legal = pred == eos
            else:
                is_legal = pred in legal_actions(pos, G)

            total += 1
            if is_legal:
                legal += 1

    if total == 0:
        return 0.0

    return legal / total

# Step 26 - sequence_compression
def sequence_compression(
    dataset: dict,
    params: dict,
    n_heads: int,
    n_tokens: int,
    max_pairs: int,
) -> float:
    # TODO: Pair distinct prefixes with equal (state, goal);
    # fraction whose greedy continuations are identical.

    tokens = dataset["tokens"]
    mask = dataset["mask"]
    states = dataset["states"]
    T = tokens.shape[1]

    # For each (state, goal) key, keep the first two distinct prefixes.
    prefixes_by_key = {}

    for i in range(tokens.shape[0]):
        for t in range(2, T):
            if not bool(mask[i, t]):
                continue

            # Only positions containing an action token are eligible.
            if int(tokens[i, t].item()) >= 4:
                continue

            # Need enough room for the current prefix, the next token,
            # and n_tokens generated tokens.
            if t + 1 + n_tokens > T:
                continue

            prefix = tokens[i, :t + 1].tolist()
            key = (
                int(states[i, t].item()),
                int(tokens[i, 1].item()),
            )

            entries = prefixes_by_key.setdefault(key, [])

            # Keep only distinct prefixes.
            if prefix not in entries and len(entries) < 2:
                entries.append(prefix)

    # Preserve the first-seen key order from the dataset scan.
    pairs = [
        entries
        for entries in prefixes_by_key.values()
        if len(entries) == 2
    ][:max_pairs]

    if not pairs:
        return 0.0

    matches = 0

    for prefix_a, prefix_b in pairs:
        continuation_a = greedy_decode(
            params,
            n_heads=n_heads,
            prefix=prefix_a,
            n_tokens=n_tokens,
        )
        continuation_b = greedy_decode(
            params,
            n_heads=n_heads,
            prefix=prefix_b,
            n_tokens=n_tokens,
        )

        if continuation_a == continuation_b:
            matches += 1

    return matches / len(pairs)

# Step 27 - detour_robustness
def detour_robustness(
    params: dict,
    n_heads: int,
    G: int,
    max_steps: int,
    n_trials: int,
    detour_prob: float = 0.75,
    seed: int = 0,
) -> float:
    # TODO: Episodes with random legal detours; success = only legal model moves and ends on the goal.
    rng = np.random.default_rng(seed)

    successes = 0

    for _ in range(n_trials):
        # Draw start and goal independently and uniformly.
        start = tuple(rng.integers(0, G, size=2).tolist())
        goal = tuple(rng.integers(0, G, size=2).tolist())

        # Sequence contains the start and goal cell tokens.
        start_cell = 4 + start[0] * G + start[1]
        goal_cell = 4 + goal[0] * G + goal[1]
        seq = [start_cell, goal_cell]

        pos = start
        episode_success = True

        for _ in range(max_steps):
            # Reaching the goal ends the episode successfully.
            if pos == goal:
                break

            legal = legal_actions(pos, G)

            if rng.random() < detour_prob:
                # Random legal detour.
                action = int(rng.choice(legal))
            else:
                # Use the transformer's greedy next-token prediction.
                action = greedy_decode(
                    params,
                    n_heads,
                    seq,
                    1,
                )[0]

                # The model's prediction must be a legal action.
                if action not in legal:
                    episode_success = False
                    break

            # Apply the action. This should remain legal by construction
            # for both the random and model branches.
            new_pos, is_legal = grid_step(pos, action, G)

            if not is_legal:
                episode_success = False
                break

            pos = new_pos
            seq.append(action)

        # Success requires ending on the goal and never making an illegal
        # model move.
        if episode_success and pos == goal:
            successes += 1

    if n_trials == 0:
        return 0.0

    return successes / n_trials

# Step 28 - world_model_report
def world_model_report(
    dataset: dict,
    params: dict,
    n_heads: int,
    n_rows: int,
    n_tokens: int,
    max_pairs: int,
    n_trials: int,
    seed: int = 0,
) -> dict:
    # TODO: Dict of the four world-model metrics, each rounded to 4 decimals.
    G = dataset["G"]
    T = dataset["tokens"].shape[1]

    hidden_states = eval_hidden_states(
        dataset,
        params,
        n_heads,
        n_rows,
    )

    return {
        "valid_move_rate": round(
            valid_move_rate(dataset, params, n_heads, n_rows),
            4,
        ),
        "effective_rank": round(
            effective_rank(hidden_states),
            4,
        ),
        "sequence_compression": round(
            sequence_compression(
                dataset,
                params,
                n_heads,
                n_tokens,
                max_pairs,
            ),
            4,
        ),
        "detour_robustness": round(
            detour_robustness(
                params,
                n_heads,
                G,
                max_steps=T - 3,
                n_trials=n_trials,
                detour_prob=0.75,
                seed=seed,
            ),
            4,
        ),
    }

# Step 29 - draft_from_latent
def draft_from_latent(h_last, dyn: dict, params: dict, max_draft: int) -> tuple:
    # TODO: next_token from h_last, then max_draft tokens by alternating
    # output_head and latent_transition.
    drafts = []

    with torch.no_grad():
        # The transformer's own next-token prediction is free.
        logits = output_head(h_last, params)
        next_token = int(torch.argmax(logits, dim=-1).item())

        # Advance the latent state using that predicted token.
        h = latent_transition(
            h_last,
            params["wte"][next_token],
            dyn,
        )

        # Generate speculative draft tokens from the learned dynamics.
        for _ in range(max_draft):
            logits = output_head(h, params)
            tok = int(torch.argmax(logits, dim=-1).item())
            drafts.append(tok)

            h = latent_transition(
                h,
                params["wte"][tok],
                dyn,
            )

    return next_token, drafts

# Step 30 - verify_draft
def verify_draft(
    params: dict,
    n_heads: int,
    prefix: list,
    next_token: int,
    drafts: list,
) -> tuple:
    # TODO: One forward pass over prefix + [next_token] + drafts;
    # accept the longest matching draft prefix; return (n_accepted, correction).

    seq = list(prefix) + [next_token] + list(drafts)

    tokens = torch.tensor(
        [seq],
        dtype=torch.long,
        device=params["wte"].device,
    )

    with torch.no_grad():
        h = gpt_hidden_states(tokens, params, n_heads)
        logits = output_head(h[0], params)
        preds = torch.argmax(logits, dim=-1)

    n_accepted = 0

    for j, draft in enumerate(drafts):
        # Prediction at position len(prefix) + j is the token
        # following the corresponding position in the sequence.
        if int(preds[len(prefix) + j].item()) != draft:
            break
        n_accepted += 1

    # The transformer's own correction after the accepted prefix.
    correction = int(preds[len(prefix) + n_accepted].item())

    return n_accepted, correction

# Step 31 - self_speculative_generate
def self_speculative_generate(
    params: dict,
    dyn: dict,
    n_heads: int,
    prefix: list,
    n_tokens: int,
    max_draft: int,
) -> dict:
    # TODO: Loop draft -> verify -> extend until n_tokens generated;
    # return tokens, cycles, accepted per cycle.

    seq = list(prefix)
    generated = []
    accepted = []
    cycles = 0

    max_len = params["wpe"].shape[0]

    while len(generated) < n_tokens:
        # Compute the transformer's true hidden state for the current
        # sequence and use its last position to start speculation.
        tokens = torch.tensor(
            [seq],
            dtype=torch.long,
            device=params["wte"].device,
        )

        with torch.no_grad():
            h = gpt_hidden_states(tokens, params, n_heads)
            h_last = h[0, -1]

        # Choose a variable draft length based on remaining context space.
        k = max(0, min(max_draft, max_len - len(seq) - 2))

        next_token, drafts = draft_from_latent(
            h_last,
            dyn,
            params,
            k,
        )

        n_accepted, correction = verify_draft(
            params,
            n_heads,
            seq,
            next_token,
            drafts,
        )

        # Tokens produced during this cycle:
        # the transformer's own next token, all accepted drafts,
        # and the correction token.
        cycle_tokens = [next_token] + drafts[:n_accepted] + [correction]

        seq.extend(cycle_tokens)
        generated.extend(cycle_tokens)

        accepted.append(n_accepted)
        cycles += 1

    return {
        "tokens": generated[:n_tokens],
        "cycles": cycles,
        "accepted": accepted,
    }

# Step 32 - speculative_stats
def speculative_stats(result: dict, n_tokens: int) -> dict:
    # TODO: cycles, mean accepted drafts per cycle, and speedup = n_tokens / cycles.
    cycles = result["cycles"]
    accepted = result["accepted"]

    if accepted:
        mean_accepted = sum(accepted) / len(accepted)
    else:
        mean_accepted = 0.0

    if cycles == 0:
        speedup = 0.0
    else:
        speedup = n_tokens / cycles

    return {
        "cycles": cycles,
        "mean_accepted": round(mean_accepted, 4),
        "speedup": round(speedup, 4),
    }

