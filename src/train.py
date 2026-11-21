from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.agents import AGENTS
from src.fl_env import SlipFrozenLake


def evaluate_greedy(agent, env, n_episodes: int, seed: int) -> float:
    """Exploration-free evaluation on a separate environment instance."""
    wins = 0.0
    for i in range(n_episodes):
        s, _ = env.reset(seed=seed + i)
        done, r = False, 0.0
        while not done:
            s, r, term, trunc, _ = env.step(int(np.argmax(agent.Q[s])))
            done = term or trunc
        wins += float(r > 0.0)
    return wins / n_episodes


def run_training(*, slip: float, algo: str, alpha: float, seed: int, cfg: Dict,
                 stage: str = "train") -> Dict:
    block = cfg[stage]
    budget, eval_every = int(block["budget"]), int(block["eval_every"])
    n_eval = int(block["eval_episodes"])
    max_steps = int(cfg["env"]["max_steps"])
    desc = cfg["env"]["desc"]

    env = SlipFrozenLake(slip=slip, desc=desc, max_steps=max_steps)
    eval_env = SlipFrozenLake(slip=slip, desc=desc, max_steps=max_steps)   # separate instance

    acfg = cfg["agent"]
    agent = AGENTS[algo](
        n_states=env.n_states, n_actions=env.n_actions,
        alpha=alpha, gamma=float(acfg["gamma"]),
        eps_start=float(acfg["eps_start"]), eps_end=float(acfg["eps_end"]),
        eps_decay_steps=max(1, int(round(acfg["eps_decay_frac"] * budget))),
        seed=seed,
    )

    s, _ = env.reset(seed=seed)
    a = agent.act(s, agent.epsilon(0))
    ep_ret, ep_len, episodes = 0.0, 0, []

    eval_steps = [0]
    eval_success = [evaluate_greedy(agent, eval_env, n_eval, seed=10_000)]
    next_eval, steps = eval_every, 0

    while steps < budget:
        s2, r, term, trunc, _ = env.step(a)
        steps += 1
        eps = agent.epsilon(steps)
        a2 = agent.act(s2, eps) if not term else None
        agent.update(s, a, r, s2, term, a2)
        ep_ret += r
        ep_len += 1

        if term or trunc:
            episodes.append(dict(episode=len(episodes), step=steps, ret=ep_ret,
                                 length=ep_len, success=float(r > 0.0),
                                 truncated=bool(trunc)))
            s, _ = env.reset()
            a = agent.act(s, agent.epsilon(steps))
            ep_ret, ep_len = 0.0, 0
        else:
            s, a = s2, a2

        if steps >= next_eval:
            eval_steps.append(steps)
            eval_success.append(
                evaluate_greedy(agent, eval_env, n_eval, seed=10_000 + steps))
            next_eval += eval_every

    final_success = evaluate_greedy(
        agent, eval_env, int(cfg["evaluation"]["final_eval_episodes"]), seed=50_000)

    return dict(slip=float(slip), algo=algo, seed=int(seed), alpha=float(alpha),
                eval_steps=np.asarray(eval_steps, dtype=float),
                eval_success=np.asarray(eval_success, dtype=float),
                final_success=float(final_success),
                episodes=pd.DataFrame(episodes),
                Q=agent.Q.copy())
