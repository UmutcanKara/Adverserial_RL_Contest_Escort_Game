import numpy as np

def sample_genome(bounds: np.ndarray) -> np.ndarray:
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return lo + (hi - lo) * np.random.rand(bounds.shape[0]).astype(np.float32)

def crossover(a: np.ndarray, b: np.ndarray, p: float = 0.5) -> np.ndarray:
    mask = (np.random.rand(a.shape[0]) < p)
    child = a.copy()
    child[mask] = b[mask]
    return child

def mutate(x: np.ndarray, bounds: np.ndarray, sigma: float = 0.15, p: float = 0.2) -> np.ndarray:
    y = x.copy()
    for i in range(len(y)):
        if np.random.rand() < p:
            span = bounds[i, 1] - bounds[i, 0]
            y[i] += np.random.randn() * sigma * span
    return np.clip(y, bounds[:, 0], bounds[:, 1])

def evaluate_pair(env, train_fn, wA: np.ndarray, wB: np.ndarray,
                  train_steps: int = 50_000, eval_episodes: int = 10) -> float:
    # Train
    env.set_both_team_weights(wA, wB)
    modelA, modelB = train_fn(env, total_env_steps=train_steps)


    diffs = []
    for ep in range(eval_episodes):
        obs, infos = env.reset(seed=ep)
        done = False
        while not done:
            actions = {}
            for a in env.agents:
                actions[a] = env.action_space(a).sample()  # replace with your act_det(model..)
            obs, rewards, terminated, truncated, infos = env.step(actions)
            done = any(terminated.values()) or any(truncated.values()) or len(env.agents) == 0

        any_agent = next(iter(infos.keys()))
        progA = float(infos[any_agent]["payload_progress_A"])
        progB = float(infos[any_agent]["payload_progress_B"])
        diffs.append(progA - progB)

    return float(np.mean(diffs))

def evolve_weights(env, train_fn,
                   pop_size=12, generations=8, elite_k=6,
                   train_steps=50_000, eval_episodes=10, seed=0):
    np.random.seed(seed)

    bounds = env.cfg.weight_bounds

    # population: list of (wA, wB)
    pop = [(sample_genome(bounds), sample_genome(bounds)) for _ in range(pop_size)]

    best = None
    best_fit = -1e9

    for g in range(generations):
        scored = []
        for (wA, wB) in pop:
            fit = evaluate_pair(env, train_fn, wA, wB, train_steps=train_steps, eval_episodes=eval_episodes)
            scored.append((fit, wA, wB))

        scored.sort(key=lambda x: x[0], reverse=True)
        elites = scored[:elite_k]

        if elites[0][0] > best_fit:
            best_fit = elites[0][0]
            best = (elites[0][1].copy(), elites[0][2].copy())

        # next gen
        next_pop = [(e[1].copy(), e[2].copy()) for e in elites]
        while len(next_pop) < pop_size:
            _, p1A, p1B = elites[np.random.randint(elite_k)]
            _, p2A, p2B = elites[np.random.randint(elite_k)]

            cA = mutate(crossover(p1A, p2A), bounds)
            cB = mutate(crossover(p1B, p2B), bounds)
            next_pop.append((cA, cB))

        pop = next_pop
        print(f"[GEN {g}] best_fit={best_fit:.4f}")

    return best, best_fit