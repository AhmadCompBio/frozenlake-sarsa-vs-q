# SARSA vs Q-learning on stochastic FrozenLake

**COEN874 (Reinforcement Learning) — Term Project, Individual Submission**

---

## Research question

How does SARSA compare with Q-learning in terms of **learning speed**, **final
success rate**, and **stability** on 4×4 FrozenLake when the **slip
probability** varies from 0 to 2/3?

## Hypotheses (pre-registered before data collection)

- **H1 (low slip).** At slip ∈ {0.0, 0.1} both methods converge to the DP-optimal
  success rate, with no statistically significant difference in final greedy success.
- **H2 (learning speed / online safety).** At slip = 2/3 SARSA achieves a higher
  area under the learning curve than Q-learning, because its on-policy update
  evaluates the ε-greedy behaviour policy and thus learns to avoid holes, whereas
  Q-learning's `max` target is systematically optimistic while the policy is
  still exploratory.
- **H3 (interaction).** The SARSA − Q-learning difference in AUC increases
  monotonically with slip.

## Environment

- **Benchmark:** Gymnasium `FrozenLake-v1`, 4×4 map (`SFFF / FHFH / FFFH / HFFG`), with the
  slip probability exposed as a continuous parameter (`SlipFrozenLake-4x4`).
- **Equivalence:** bit-identical to `gymnasium.make("FrozenLake-v1", is_slippery=...)`
  at `slip = 0.0` and `slip = 2/3`, verified by
  `tests/test_env.py::test_matches_gymnasium` (1e-12 tolerance).
- **MDP:** `S = 16`, `A = {left, down, right, up}`, `P(intended) = 1 − p`,
  `P(each perpendicular) = p/2`, wall collisions self-loop, `R = +1` on entering
  the goal, `γ = 0.99` (agents) / `γ = 1.0` (DP reference), terminate on
  goal or hole, truncate at 200 steps.
- **Reference values:** value iteration gives `V*(s₀) = 1.000000` (slip = 0) and
  `V*(s₀) = 14/17 ≈ 0.823529` (slip = 2/3). The uniform-random baseline is
  ≈ 0.013940 at every slip.

## Algorithms

- **SARSA** (Rummery & Niranjan, 1994): `Q(s,a) ← Q(s,a) + α[r + γ Q(s′,a′) − Q(s,a)]`
  with `a′` drawn from the ε-greedy behaviour policy.
- **Q-learning** (Watkins & Dayan, 1992): `Q(s,a) ← Q(s,a) + α[r + γ max_a′ Q(s′,a′) − Q(s,a)]`.

Both agents share an **identical ε-greedy behaviour policy** (ε: 1.0 → 0.05, linear over
60 % of the step budget). The comparison isolates the update target.

## Experimental design

| Dimension | Setting |
|---|---|
| Slip levels (sensitivity factor) | 0.0, 0.1, 0.2, 2/3 (4 levels) |
| Algorithms | SARSA, Q-learning |
| Training budget | 30 000 env steps per run (identical for both) |
| Tuning budget | 15 000 env steps, α grid {0.02, 0.05, 0.1, 0.2} |
| Tuning seeds (held out) | 0, 1, 2 |
| Reported seeds | 100–109 (10 per configuration) |
| Evaluation | greedy (exploration off), separate env instance, 50 eval episodes periodic, 200 final |
| Threshold | 0.9 × V*(s₀) per slip |

One α per (slip, algorithm) is selected by mean AUC on the **tuning** seeds; the
**reported** seeds (100–109) are disjoint from the tuning seeds. Every selected α is the
argmax on its own tuning surface, so the comparison is fair.

## Results (summary)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** | **Supported** | slip 0.0: 1.000 vs 1.000, p_holm = 1.000; slip 0.1: 0.811 vs 0.794, δ = 0.22, p_holm = 1.000 |
| **H2** | **Refuted** | slip 2/3: AUC 0.434 (SARSA) vs 0.604 (Q-learning), Cliff's δ = **−1.00** (complete separation), p_holm = **0.001** |
| **H3** | **Partially supported** | the SARSA−Q AUC gap widens with slip (0.026 → 0.012 → 0.062 → **0.170**) but **in Q-learning's favour**, not SARSA's |

**Interpretation.** Q-learning learns substantially faster than SARSA at high slip,
and the gap grows with stochasticity. The classic "SARSA is safer" result on
CliffWalking does not transfer to FrozenLake, because FrozenLake's danger is
isotropic and scattered (5 of 16 cells), not a single corridor — "avoid the risky
region" is not a strategy when every path passes close to a hole. SARSA's on-policy
target is more conservative, which slows value propagation; Q-learning's off-policy
`max` target is more aggressive and benefits the greedy-evaluation AUC, while
SARSA's final success rate is actually marginally higher (0.682 vs 0.624, n.s.).
This is the classic on-policy / off-policy trade-off, in the opposite direction
from CliffWalking.

## Reproduce

Requires Python 3.13 and the pinned dependencies in `requirements-lock.txt`.

```bash
# Install
pip install -r requirements-lock.txt

# Tests (14 checks including Gymnasium equivalence and DP reference values)
pytest -q

# Smoke test (~2 minutes): 2 slips × 2 seeds, writes results/smoke/
python -m scripts.run_all --smoke

# Full study (~30 minutes): 4 slips × 2 methods × 10 seeds × 30 k steps
python -m scripts.run_all --out results/

# Regenerate individual figures or tables from saved raw results (no retraining)
python -m scripts.make_figures --only learning_curves
python -m scripts.make_figures --only final_distribution
python -m scripts.make_figures --only sensitivity
python -m scripts.make_figures --only policy_value
python -m scripts.make_tables  --only mdp
python -m scripts.make_tables  --only hyperparams
python -m scripts.make_tables  --only main
python -m scripts.make_tables  --only stats
