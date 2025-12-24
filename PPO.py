import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

n_actions = 5  # discrete actions: up, down, left, right, no-op

# ---------- Model ---------- #
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_act: int = n_actions):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        self.v  = nn.Linear(128, 1)
        self.logits = nn.Linear(128, n_act)  # for discrete actions
        # self.mu = nn.Linear(128, act_dim)
        # self.log_std = nn.Parameter(torch.zeros(act_dim)) TODO: Modify for continuous actions

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        v = self.v(h).squeeze(-1)
        logits = self.logits(h)
        # mu = self.mu(h)
        # std = torch.exp(self.log_std).expand_as(mu)
        return logits, v


# def logprob_gaussian(a, mu, std):
#     var = std**2
#     return (-0.5 * (((a - mu)**2) / (var + 1e-8) + 2*torch.log(std + 1e-8) + np.log(2*np.pi))).sum(-1)


# def entropy_gaussian(std):
#     return (0.5 + 0.5*np.log(2*np.pi) + torch.log(std + 1e-8)).sum(-1)

@torch.no_grad()
def sample_action(model, obs, device="cpu"):
    ot = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    logits, v = model(ot)
    dist = Categorical(logits=logits)
    a = dist.sample()
    logp = dist.log_prob(a)
    return int(a.item()), float(logp.item()), float(v.item())

# def act_deterministic(model, obs, device="cpu"):

def gae(rews, vals, dones, gamma=0.99, lam=0.95):
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nt = 1.0 - dones[t]
        nextv = vals[t+1] if t+1 < len(vals) else 0.0
        delta = rews[t] + gamma * nextv * nt - vals[t]
        last = delta + gamma * lam * nt * last
        adv[t] = last
    ret = adv + vals[:T]
    return adv, ret


class Buffer:
    def __init__(self):
        self.obs, self.act, self.logp, self.rew, self.done, self.val = [], [], [], [], [], []

    def add(self, obs, act, logp, rew, done, val):
        self.obs.append(obs)
        self.act.append(act)
        self.logp.append(logp)
        self.rew.append(rew)
        self.done.append(done)
        self.val.append(val)

    def tensors(self, device):
        obs = torch.tensor(np.asarray(self.obs), dtype=torch.float32, device=device)
        act = torch.tensor(np.asarray(self.act), dtype=torch.int64, device=device)
        logp = torch.tensor(np.asarray(self.logp), dtype=torch.float32, device=device)
        rew = np.asarray(self.rew, dtype=np.float32)
        done = np.asarray(self.done, dtype=np.float32)
        val = np.asarray(self.val, dtype=np.float32)
        return obs, act, logp, rew, done, val


# ---------- PPO update ----------
def ppo_update(model, opt, buf: Buffer,
               gamma=0.99, lam=0.95, clip_eps=0.2,
               vf_coef=0.5, ent_coef=0.01,
               epochs=10, mb_size=256, device="cpu"):

    obs_t, act_t, logp_old_t, rew_np, done_np, val_np = buf.tensors(device)
    adv_np, ret_np = gae(rew_np, val_np, done_np, gamma=gamma, lam=lam)

    print("Advantage mean:", adv_np.mean(), "std:", adv_np.std())

    adv_t = torch.tensor(adv_np, dtype=torch.float32, device=device)
    ret_t = torch.tensor(ret_np, dtype=torch.float32, device=device)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = obs_t.shape[0]
    idx = np.arange(n)

    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, n, mb_size):
            mb = idx[s:s+mb_size]
            o = obs_t[mb]
            a = act_t[mb]
            logp_old = logp_old_t[mb]
            adv = adv_t[mb]
            ret = ret_t[mb]

            logits, v = model(o)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(a)
            ent = dist.entropy().mean()

            # mu, std, v = model(o)
            # logp = logprob_gaussian(a, mu, std)
            
            ratio = torch.exp(logp - logp_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (ret - v).pow(2).mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * ent

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()


# ---------- Train both teams ----------
def train_two_team_ppo(env,
                       total_env_steps=200_000,
                       rollout_steps=1024,
                       gamma=0.99, lam=0.95,
                       clip_eps=0.2,
                       vf_coef=0.5, ent_coef=0.01,
                       lr=3e-4,
                       update_epochs=10,
                       minibatch_size=256,
                       device="cpu",
                       seed=0):

    torch.manual_seed(seed)
    np.random.seed(seed)

    # infer dims from one agent
    sample_agent = env.possible_agents[0]
    obs_dim = env.observation_spaces[sample_agent].shape[0]
    n_act = env.action_spaces[sample_agent].n

    teamA_agents = [a for a in env.possible_agents if a.startswith("teamA_")]
    teamB_agents = [a for a in env.possible_agents if a.startswith("teamB_")]

    modelA = ActorCritic(obs_dim, n_act).to(device)
    modelB = ActorCritic(obs_dim, n_act).to(device)
    optA = optim.Adam(modelA.parameters(), lr=lr)
    optB = optim.Adam(modelB.parameters(), lr=lr)

    obs, _ = env.reset(seed=seed)
    steps = 0

    while steps < total_env_steps:
        bufA, bufB = Buffer(), Buffer()

        for _ in range(rollout_steps):
            actions = {}
            idxA, idxB = {}, {}

            # --- Team A actions ---
            for a in teamA_agents:
                act_int, logp, v = sample_action(modelA, obs[a], device=device)
                actions[a] = act_int

                idxA[a] = len(bufA.rew)
                bufA.add(obs[a], act_int, logp, 0.0, 0.0, v)

            # --- Team B actions ---
            for a in teamB_agents:
                act_int, logp, v = sample_action(modelB, obs[a], device=device)
                actions[a] = act_int

                idxB[a] = len(bufB.rew)
                bufB.add(obs[a], act_int, logp, 0.0, 0.0, v)
            
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            env.render()

            done_any = any(terminated.values()) or any(truncated.values()) or len(env.agents) == 0

            # fill in rewards/dones for last added entries
            for a in teamA_agents:
                k = idxA[a]
                bufA.rew[k] = float(rewards[a])
                bufA.done[k] = 1.0 if done_any else 0.0

            for a in teamB_agents:
                k = idxB[a]
                bufB.rew[k] = float(rewards[a])
                bufB.done[k] = 1.0 if done_any else 0.0

            obs = next_obs
            steps += len(env.possible_agents)

            if done_any:
                obs, _ = env.reset()

        # Update both teams from their own buffers
        ppo_update(modelA, optA, bufA, gamma=gamma, lam=lam, clip_eps=clip_eps,
                   vf_coef=vf_coef, ent_coef=ent_coef,
                   epochs=update_epochs, mb_size=minibatch_size, device=device)

        ppo_update(modelB, optB, bufB, gamma=gamma, lam=lam, clip_eps=clip_eps,
                   vf_coef=vf_coef, ent_coef=ent_coef,
                   epochs=update_epochs, mb_size=minibatch_size, device=device)

    return modelA, modelB
