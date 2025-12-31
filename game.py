from capture_escort import CaptureEscortEnv, CaptureEscortConfig
from PPO import train_two_team_ppo 
from logger import CSVLogger
import numpy as np
from EA_weights import evolve_weights
import torch


@torch.no_grad()
def act_det(model, obs_np, device="cpu"):
    ot = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    logits, v = model(ot)
    a = torch.argmax(logits, dim=-1)  # deterministic discrete action
    return int(a.item())

def play_and_log(env, modelA, modelB, episodes=5, max_steps=500, device="cpu"):
    teamA = [a for a in env.possible_agents if a.startswith("teamA_")]
    teamB = [a for a in env.possible_agents if a.startswith("teamB_")]

    # data points for plotting
    # list of dicts: each row is one timestep
    log = []
    ep_results = []

    for ep in range(episodes):
        obs, infos = env.reset(seed=ep)

        for t in range(max_steps):
            actions = {}
            for a in teamA:
                actions[a] = act_det(modelA, obs[a], device=device)
            for a in teamB:
                actions[a] = act_det(modelB, obs[a], device=device)

            obs, rewards, terminated, truncated, infos = env.step(actions)

            any_agent = next(iter(infos.keys()))
            progA = float(infos[any_agent]["payload_progress_A"])
            progB = float(infos[any_agent]["payload_progress_B"])
            mean_rA = float(np.mean([rewards[a] for a in teamA]))
            mean_rB = float(np.mean([rewards[a] for a in teamB]))

            log.append({
                "episode": ep,
                "t": t,
                "progA": progA,
                "progB": progB,
                "mean_reward_A": mean_rA,
                "mean_reward_B": mean_rB,
            })

            done = any(terminated.values()) or any(truncated.values()) or len(env.agents) == 0
            if done:
                winner = "A" if progA > progB else ("B" if progB > progA else "Draw")
                ep_results.append({"episode": ep, "steps": t + 1, "winner": winner, "final_progA": progA, "final_progB": progB})
                break

    return log, ep_results


# if __name__ == "__main__":
#     conf = CaptureEscortConfig()
#     env = CaptureEscortEnv(conf)

#     logger = CSVLogger(out_dir="logs", run_name="ce_discrete_train")
#     env.logger = logger

#     # 1) TRAIN
#     model_a, model_b = train_two_team_ppo(
#         env,
#         total_env_steps=100_000,
#         rollout_steps=1024,
#         gamma=0.99,
#         lam=0.95,
#         clip_eps=0.2,
#         device="cpu",
#     )
#     logger.close()

#     logger = CSVLogger(out_dir="logs", run_name="ce_discrete")
#     env.logger = logger
#     env.reset_logger()
    
#     # 2) PLAY + LOG DATA
#     log, results = play_and_log(env, model_a, model_b, episodes=10, max_steps=conf.max_steps, device="cpu")
#     logger.close()


