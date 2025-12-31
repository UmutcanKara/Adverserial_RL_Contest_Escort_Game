import os, csv
import numpy as np
from time import time

from capture_escort import CaptureEscortEnv, CaptureEscortConfig
from PPO import train_two_team_ppo
from game import play_and_log


def make_population(pop_size: int, bounds: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=np.float32)
    D = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    pop = np.empty((pop_size, 2 * D), dtype=np.float32)
    for i in range(pop_size):
        wA = lo + (hi - lo) * rng.random(D, dtype=np.float32)
        wB = lo + (hi - lo) * rng.random(D, dtype=np.float32)
        pop[i] = np.concatenate([wA, wB]).astype(np.float32)
    return pop


def tournament_select(pop: np.ndarray, fit: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(pop), size=k, replace=False)
    return pop[idx[np.argmax(fit[idx])]].copy()


def uniform_crossover(p1: np.ndarray, p2: np.ndarray, p: float, rng: np.random.Generator):
    mask = rng.random(p1.shape[0]) < p
    c1 = np.where(mask, p1, p2).astype(np.float32)
    c2 = np.where(mask, p2, p1).astype(np.float32)
    return c1, c2


def mutate(x: np.ndarray, low: np.ndarray, high: np.ndarray, sigma_frac: float, p: float, rng: np.random.Generator):
    span = (high - low)
    sigma = span * sigma_frac
    m = rng.random(x.shape[0]) < p
    x = x + m.astype(np.float32) * (rng.standard_normal(x.shape[0]).astype(np.float32) * sigma)
    return np.clip(x, low, high).astype(np.float32)


def evaluate_genome(conf: CaptureEscortConfig,
                    genome: np.ndarray,
                    train_steps: int,
                    rollout_steps: int,
                    eval_episodes: int,
                    seed: int,
                    run_name: str):
    from logger import CSVLogger

    D = conf.weight_bounds.shape[0]
    wA, wB = genome[:D], genome[D:]

    env = CaptureEscortEnv(conf)
    logger = CSVLogger(out_dir="logs", run_name=run_name)
    env.logger = logger
    env.set_both_team_weights(wA, wB)

    modelA, modelB = train_two_team_ppo(
        env,
        total_env_steps=int(train_steps),
        rollout_steps=int(rollout_steps),
        seed=int(seed),
    )

    _, results = play_and_log(env, modelA, modelB, episodes=int(eval_episodes), max_steps=conf.max_steps)

    winsA = sum(1 for r in results if r.get("winner") == "A")
    winrateA = winsA / max(1, len(results))
    avgA = float(np.mean([r.get("final_progA", 0.0) for r in results])) if results else 0.0
    avgB = float(np.mean([r.get("final_progB", 0.0) for r in results])) if results else 0.0
    fitness = float(winrateA + 0.25 * (avgA - avgB))

    logger.close()
    return fitness, winrateA, avgA, avgB, logger.step_path, logger.ep_path


if __name__ == "__main__":
    conf = CaptureEscortConfig()
    os.makedirs("logs", exist_ok=True)
    out_csv = f"logs/ga_runs_{int(time())}.csv"

    POP_SIZE = 30
    N_GENERATIONS = 10
    ELITE = 4
    TOURN_K = 5
    CROSS_P = 0.5
    MUT_P = 0.2
    MUT_SIGMA_FRAC = 0.08
    TOP_K_STAGE2 = 3

    STAGE1_TRAIN = 15_000
    STAGE1_EVAL = 5
    STAGE2_TRAIN = 60_000
    STAGE2_EVAL = 20

    ROLLOUT_STEPS = 512
    SEED = 0

    rng = np.random.default_rng(SEED)

    bounds = np.asarray(conf.weight_bounds, dtype=np.float32)
    D = bounds.shape[0]
    low = np.concatenate([bounds[:, 0], bounds[:, 0]]).astype(np.float32)
    high = np.concatenate([bounds[:, 1], bounds[:, 1]]).astype(np.float32)

    pop = make_population(POP_SIZE, bounds, seed=SEED)

    all_rows = []
    best_overall = None
    best_fit = -1e9

    for gen in range(N_GENERATIONS):
        # Stage 1: evaluate all
        fit = np.zeros(POP_SIZE, dtype=np.float32)
        meta = [None] * POP_SIZE

        for i in range(POP_SIZE):
            run_name = f"g{gen}_s1_i{i:02d}_seed{SEED + gen*1000 + i}"
            f1, wr1, pA1, pB1, step_csv, ep_csv = evaluate_genome(
                conf, pop[i],
                STAGE1_TRAIN, ROLLOUT_STEPS, STAGE1_EVAL,
                seed=SEED + gen*1000 + i,
                run_name=run_name
            )
            fit[i] = f1
            meta[i] = (wr1, pA1, pB1, step_csv, ep_csv) # type: ignore

        # elites
        elite_idx = np.argsort(fit)[-ELITE:][::-1]
        gen_best_idx = int(elite_idx[0])

        # Stage 2: confirm top K
        top_idx = np.argsort(fit)[-min(TOP_K_STAGE2, POP_SIZE):][::-1]
        for rank, idx in enumerate(top_idx):
            run_name = f"g{gen}_s2_rank{rank}_i{idx:02d}_seed{SEED + gen*1000 + 500 + idx}"
            f2, wr2, pA2, pB2, step2_csv, ep2_csv = evaluate_genome(
                conf, pop[idx],
                STAGE2_TRAIN, ROLLOUT_STEPS, STAGE2_EVAL,
                seed=SEED + gen*1000 + 500 + idx,
                run_name=run_name
            )

            wA = pop[idx][:D].tolist()
            wB = pop[idx][D:].tolist()
            wr1, pA1, pB1, step1_csv, ep1_csv = meta[idx]

            row = {
                "gen": gen, "idx": int(idx), "rank": rank,
                "fit_stage1": float(fit[idx]), "wrA_stage1": float(wr1), "pA_stage1": float(pA1), "pB_stage1": float(pB1),
                "fit_stage2": float(f2), "wrA_stage2": float(wr2), "pA_stage2": float(pA2), "pB_stage2": float(pB2),
                "wA": wA, "wB": wB,
                "stage1_steps_csv": step1_csv, "stage1_eps_csv": ep1_csv,
                "stage2_steps_csv": step2_csv, "stage2_eps_csv": ep2_csv,
            }
            all_rows.append(row)

            if f2 > best_fit:
                best_fit = float(f2)
                best_overall = (gen, int(idx), pop[idx].copy())

        print(f"[GEN {gen}] best_stage1={float(fit[gen_best_idx]):.3f} best_overall={best_fit:.3f}")

        # Next generation 
        new_pop = [pop[i].copy() for i in elite_idx]  # elitism

        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(pop, fit, TOURN_K, rng)
            p2 = tournament_select(pop, fit, TOURN_K, rng)

            c1, c2 = uniform_crossover(p1, p2, CROSS_P, rng)
            c1 = mutate(c1, low, high, MUT_SIGMA_FRAC, MUT_P, rng)
            c2 = mutate(c2, low, high, MUT_SIGMA_FRAC, MUT_P, rng)

            new_pop.append(c1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(c2)

        pop = np.stack(new_pop, axis=0).astype(np.float32)

    # save summary
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    gen, idx, genome = best_overall # type: ignore
    print("\nBEST OVERALL:")
    print("gen:", gen, "idx:", idx, "fitness:", best_fit)
    print("wA:", genome[:D].tolist())
    print("wB:", genome[D:].tolist())
    print("Saved:", out_csv)