# NextLat from Scratch: Next-Latent Prediction in PyTorch

Build Next-Latent Prediction (NextLat, arXiv:2511.05963) end to end in functional PyTorch: a grid world whose true belief state is (position, goal), a tiny causal GPT, a residual-delta latent dynamics model trained with a stop-gradient Smooth L1 next-hidden loss and a frozen-head KL term, then the world-model metrics from the paper (effective latent rank, sequence compression, detour robustness) and variable-length self-speculative decoding driven by the learned latent dynamics.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** grid_step
- [x] **2.** legal_actions
- [x] **3.** random_walk_to_goal
- [x] **4.** encode_sequence
- [x] **5.** make_dataset
- [x] **6.** get_batch
- [x] **7.** causal_mask
- [x] **8.** init_gpt_params
- [x] **9.** attention_block
- [x] **10.** mlp_block
- [x] **11.** gpt_hidden_states
- [x] **12.** output_head
- [x] **13.** next_token_loss
- [x] **14.** init_dynamics_params
- [x] **15.** latent_transition
- [x] **16.** rollout_latents
- [x] **17.** next_hidden_loss
- [x] **18.** kl_alignment_loss
- [x] **19.** nextlat_loss
- [x] **20.** train_step
- [x] **21.** train_model
- [x] **22.** greedy_decode
- [x] **23.** effective_rank
- [x] **24.** eval_hidden_states
- [x] **25.** valid_move_rate
- [x] **26.** sequence_compression
- [x] **27.** detour_robustness
- [x] **28.** world_model_report
- [x] **29.** draft_from_latent
- [x] **30.** verify_draft
- [x] **31.** self_speculative_generate
- [x] **32.** speculative_stats

## Results

```
grid 4x4, T=14, vocab=21, train sequences=1024
GPT      next-token loss: 3.038 -> 1.459
NextLat  next-token loss: 3.038 -> 1.634 | next-hidden 0.6609 -> 0.0052 | KL 0.0102 -> 0.0048

metric                   GPT      NextLat
valid_move_rate          0.8841   0.7249   (better = up)
effective_rank           7.4881   2.9002   (better = down)
sequence_compression     0.2333   0.1667   (better = up)
detour_robustness        0.4250   0.2750   (better = up)
(lower effective rank = more compact latent state; the true world has only 16 positions x 16 goals)

self-speculative decoding (6 tokens, draft up to 4):
  mean accepted drafts per cycle: 2.31
  speedup in transformer passes:  3.00x
  identical to greedy decoding:   True

note: at this toy scale two effects are robust across seeds - NextLat's lower effective rank (a more compact latent state) and its usable drafts (GPT's untrained dynamics accept almost none). The other metrics are noisy on 60 pairs / 40 episodes; rerun with more training steps, sequences and trials before reading anything into small gaps.
```
