from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv    


@dataclass
class CaptureEscortConfig:
    width: float = 10.0
    height: float = 6.0
    max_steps: int = 50000

    escort_radius: float = 1.2
    contest_radius: float = 0.8
    goal_radius: float = 0.5

    agent_max_speed: float = 0.3
    payload_max_speed: float = 0.2

    w_min: float = -10.0
    w_max: float = 10.0

    w = np.array([
        5.0, # payload progress
        2.0, # escorting
        1.2, # contesting
        0.8, # teamwork
        0.35 # control penalty
        ], dtype=np.float32)  # Default reward weights

    act_dir = np.array([
        [-1.0, 0.0],  # Left
        [1.0, 0.0],   # Right
        [0.0, -1.0],  # Down
        [0.0, 1.0],   # Up
        [0.0, 0.0],   # No-op TODO Make Attack action
    ], dtype=np.float32)  # Placeholder for action direction limits


class CaptureEscortEnv(ParallelEnv):
    """
    Docstring for CaptureEscortEnv

    3v3 Capture & Escort environment.
    Each team tries to escort their payload to their designated goal while preventing the opposing team from doing the same.

    Agents:
        - 3 Escort Agents (Team A)
        - 3 Contest Agents (Team B)
    
    Payload Mechanics:
        - Each team has a payload that starts near their side of the map.
        - The payload moves towards the team's goal when escorted by teammates and not contested by opponents.
        - If contested, the payload's movement is slowed or reversed based on the number of escorts vs contesters.
    
    Reward Structure:
        - Payload Progress: Rewarded for moving the payload towards the goal.
        - Escorting: Rewarded for being near own payload.
        - Contesting: Rewarded for being near opponent's payload.
        - Teamwork: Rewarded for maintaining proximity to teammates.
        - Control Penalty: Penalized jitter and unnecessary actions to encourage smooth movements.
    
    Termination:
        - The episode ends when one team's payload reaches the goal or when the maximum number of steps is reached.

    


    
    """

    metadata = {
        'name': 'capture_escort_v0',
        'render_modes': ['human'], # add other render modes if implemented
        'is_parallelizable': True, 
        }
    
    def __init__(self, 
                 config: Optional[CaptureEscortConfig] = None, 
                 #reward_weights:Optional[np.ndarray] = None, 
                 seed: Optional[int] = 0) -> None:
        
        self.cfg = config or CaptureEscortConfig()
        self.rng = np.random.default_rng(seed)

        # Required by pettingZoo API
        self.possible_agents = [f'teamA_{i}' for i in range(3)] + [f'teamB_{i}' for i in range(3)]
        self.agents: List[str] = []

        # Reward weights for payload progress, escorting, contesting, teamwork, and control penalty
        #if reward_weights is None:
        #    reward_weights = np.array([5.0, 2.0, 1.0, -1.0, 0.01], dtype=np.float32)  # Example weights
        reward_weights = self.cfg.w
        
        self.w = np.clip(reward_weights.astype(np.float32), self.cfg.w_min, self.cfg.w_max)

        self.__init__action_spaces()
        self._dirs = self.cfg.act_dir

        obs_shape = 26  # Placeholder for actual observation shape
        self.observation_spaces = {a: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), seed=0, dtype=np.float32) for a in self.possible_agents  }

        # State
        self._step_count = 0
        self._agent_pos = {}
        self._agent_vel = {}
        self._payload_pos = {'A': np.zeros(2, dtype=np.float32), 'B': np.zeros(2, dtype=np.float32)}
        self._payload_progress = {'A': 0.0, 'B': 0.0}
        self._last_payload_progress = {'A': 0.0, 'B': 0.0}

        # Zones (Left -> Right for team A, Right -> Left for team B)
        self._goal_A = np.array([self.cfg.width * 0.9, self.cfg.height * 0.6], dtype=np.float32)
        self._goal_B = np.array([self.cfg.width * 0.1, self.cfg.height * 0.4], dtype=np.float32)

        self.render_mode = 'human'  # Default

    def action_space(self, agent) -> spaces.Space:
        return super().action_space(agent)
    
    def __init__action_spaces(self) -> None:
        self.action_spaces = {a: spaces.Discrete(5) for a in self.possible_agents}

    
    # PettingZoo required API 
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.agents = list(self.possible_agents)
        self._step_count = 0

        # Spawn team A on left side and team B on right side
        for i in range(3):
            self._agent_pos[f'teamA_{i}'] = np.array([self.cfg.width * 0.15, self.cfg.height * (0.25 + 0.25 * i)], dtype=np.float32)
            self._agent_vel[f'teamA_{i}'] = np.zeros(2, dtype=np.float32)

            self._agent_pos[f'teamB_{i}'] = np.array([self.cfg.width * 0.85, self.cfg.height * (0.25 + 0.25 * i)], dtype=np.float32)
            self._agent_vel[f'teamB_{i}'] = np.zeros(2, dtype=np.float32)

        # Payload spawn near own side
        self._payload_pos['A'] = np.array([self.cfg.width * 0.2, self.cfg.height * 0.5], dtype=np.float32)
        self._payload_pos['B'] = np.array([self.cfg.width * 0.8, self.cfg.height * 0.5], dtype=np.float32)

        self._payload_progress = {'A': 0.0, 'B': 0.0}
        self._last_payload_progress = {'A': 0.0, 'B': 0.0}

        observations = {a: self.observe(a) for a in self.agents}
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:
        self._step_count += 1

        # Move agents
        for ag, act in actions.items():
            if ag not in self.agents:
                print(f"Warning: Action received for inactive agent {ag}. Ignoring.")
                continue

            vel = self._dirs[act] * self.cfg.agent_max_speed
            self._agent_vel[ag] = vel
            self._agent_pos[ag] = self._clip_to_bounds(self._agent_pos[ag] + vel)

            # act = np.asarray(act, dtype=np.float32)
            # act = np.clip(act, -1.0, 1.0)

            # vel = act * self.cfg.agent_max_speed
            # self._agent_vel[ag] = vel
            # self._agent_pos[ag] = self._clip_to_bounds(self._agent_pos[ag] + vel) 
            # TODO : Replace with continuous movement logic if time left

        # Update payload positions
        self._last_payload_progress['A'] = self._payload_progress['A']
        self._last_payload_progress['B'] = self._payload_progress['B']

        self._update_payload(team = 'A')   
        self._update_payload(team = 'B')

        # Compute termination
        terminated = {a: False for a in self.agents}
        truncated = {a: self._step_count >= self.cfg.max_steps for a in self.agents}

        # Check for payload delivery
        winA = self._in_goal(self._payload_pos['A'], team='A')
        winB = self._in_goal(self._payload_pos['B'], team='B')

        if winA or winB:
            for a in self.agents:
                terminated[a] = True
        
        # Compute rewards
        rewards = self._compute_rewards(winA, winB)

        # Gather observations and infos
        observations = {a: self.observe(a) for a in self.agents}
        infos = {a: {"payload_progress_A": self._payload_progress["A"], "payload_progress_B": self._payload_progress["B"]} for a in self.agents}

        # Clear dead agents
        if all(terminated.values()) or all(truncated.values()):
            self.agents = []

        return observations, rewards, terminated, truncated, infos
    
    def observe(self, agent: str) -> np.ndarray:
        # Build fix sized observation vector. 
        # Placeholder implementation TODO: Replace with actual observation logic

        pos = self._agent_pos[agent]
        vel = self._agent_vel[agent]

        is_A = agent.startswith('teamA')
        team = 'A' if is_A else 'B'
        opp = 'B' if is_A else 'A'

        # Teammates and opponents
        tm = [self._agent_pos[a] for a in self.agents if a.startswith(f'team{team}') and a != agent]
        opps = [self._agent_pos[a] for a in self.agents if a.startswith(f'team{opp}')]

        tm_rel_pos = np.concatenate([p - pos for p in tm]) if tm else np.array([], dtype=np.float32)
        op_rel_pos = np.concatenate([p - pos for p in opps]) if opps else np.array([], dtype=np.float32)

        own_payload_rel_pos = self._payload_pos[team] - pos
        opp_payload_rel_pos = self._payload_pos[opp] - pos

        own_goal_rel_pos = (self._goal_A if is_A else self._goal_B) - pos
        opp_goal_rel_pos = (self._goal_B if is_A else self._goal_A) - pos

        # Progress
        own_prog = np.array([self._payload_progress[team]], dtype=np.float32)
        opp_prog = np.array([self._payload_progress[opp]], dtype=np.float32)

        obs = np.concatenate(
            [pos, vel, tm_rel_pos, op_rel_pos, own_payload_rel_pos, opp_payload_rel_pos, own_goal_rel_pos, opp_goal_rel_pos, own_prog, opp_prog],
            axis=0
        ).astype(np.float32)

        # Shape check   
        target_shape = self.observation_spaces[agent].shape[0] # type: ignore
        if obs.shape[0] < target_shape:
            padding = np.zeros(target_shape - obs.shape[0], dtype=np.float32)
            obs = np.concatenate([obs, padding], axis=0)
        else:
            assert obs.shape[0] == target_shape


        return obs
    
    def render(self) -> None:
        # Minimal Render
        print(f"Step: {self._step_count} | Payload A Pos: {self._payload_pos['A']} Prog: {self._payload_progress['A']:.2f} | Payload B Pos: {self._payload_pos['B']} Prog: {self._payload_progress['B']:.2f}")
        print(f' Agent Positions:\nTeam A: {[self._agent_pos[a] for a in self.agents if a.startswith("teamA_")]}\nTeam B: {[self._agent_pos[a] for a in self.agents if a.startswith("teamB_")]}')
    def close(self) -> None:
        return
    
    # Game Logic Methods
    def _clip_to_bounds(self, pos: np.ndarray) -> np.ndarray:
        pos[0] = np.clip(pos[0], 0.0, self.cfg.width)
        pos[1] = np.clip(pos[1], 0.0, self.cfg.height)
        return pos
    
    def _in_goal(self, payload_pos: np.ndarray, team: str) -> bool:
        goal_pos = self._goal_A if team == 'A' else self._goal_B
        return np.linalg.norm(payload_pos - goal_pos) < self.cfg.goal_radius  # type: ignore
    
    def _update_payload(self, team: str) -> None:
        opp = 'B' if team == 'A' else 'A'
        payload_pos = self._payload_pos[team]
        goal = self._goal_A if team == 'A' else self._goal_B

        # Find escorts end contesters
        escorts = [self._agent_pos[a] for a in self.agents if a.startswith(f'team{team}') and np.linalg.norm(self._agent_pos[a] - payload_pos) < self.cfg.escort_radius]
        contesters = [self._agent_pos[a] for a in self.agents if a.startswith(f'team{opp}') and np.linalg.norm(self._agent_pos[a] - payload_pos) < self.cfg.contest_radius]

        if len(escorts) > 0 and len(contesters) == 0:
            direction = (goal - payload_pos)
            direction /= np.linalg.norm(direction) + 1e-8
            step = direction * self.cfg.payload_max_speed

            self._payload_pos[team] = self._clip_to_bounds(payload_pos + step) 

        # elif len(escorts) > len(contesters) and len(contesters) != 0:
        #     direction = (goal - payload_pos)
        #     direction /= np.linalg.norm(direction) + 1e-8
        #     step = direction * (self.cfg.payload_max_speed) #TODO: modify speed based on escort/contester ratio
        #     pass

        # elif len(contesters) > len(escorts):
        #     direction = (payload_pos - goal)
        #     direction /= np.linalg.norm(direction) + 1e-8
        #     step = direction * (self.cfg.payload_max_speed / 2)

        #     self._payload_pos[team] = self._clip_to_bounds(payload_pos + step)

        # Update normalized progress

        start_x = self.cfg.width * 0.2 if team == 'A' else self.cfg.width * 0.8
        end_x = self._goal_A[0] if team == 'A' else self._goal_B[0]
        denom = (end_x - start_x) if abs(end_x - start_x) > 1e-6 else 1.0

        self._payload_progress[team] = float((self._payload_pos[team][0] - start_x) / denom) if team == "A" else float((start_x - self._payload_pos[team][0]) / (start_x - end_x))
        self._payload_progress[team] = np.clip(self._payload_progress[team], 0.0, 1.0)

    def _compute_rewards(self, winA: bool, winB: bool) -> Dict[str, float]:
        # Unpack weights
        w_p, w_e, w_c, w_t, w_ctrl = self.w

        # Team win reward
        w_wA = 1.0 if winA and not winB else (-1.0 if winB and not winA else 0.0)
        w_wB = -w_wA

        # Progress delta
        dA = self._payload_progress['A'] - self._last_payload_progress['A']
        dB = self._payload_progress['B'] - self._last_payload_progress['B']

        rewards = {}
        for a in self.agents:
            isA = a.startswith('teamA')
            team = 'A' if isA else 'B'
            opp = 'B' if isA else 'A'

            # Escort and contest counts
            payload = self._payload_pos[team]
            opp_payload = self._payload_pos[opp]

            R_p = dA if isA else dB
            R_p *= 2

            dist_payload = np.linalg.norm(self._agent_pos[a] - payload)
            R_e = 1.0 - np.clip(dist_payload / 3.0, 0.0, 1.0)

            dist_opp_payload = np.linalg.norm(self._agent_pos[a] - opp_payload)
            R_b = 0.8 - np.clip(dist_opp_payload / 3.0, 0.0, 1.0)

            # Cohesion: mean distance to teammates
            mates = [m for m in self.agents if m.startswith("teamA" if isA else "teamB") and m != a]
            mean_mate_dist = np.mean([np.linalg.norm(self._agent_pos[m] - self._agent_pos[a]) for m in mates]) if mates else 0.0
            R_d = float(mean_mate_dist)

            # Control cost
            R_c = 0
            if (dist_payload > self.cfg.escort_radius) or (dist_opp_payload > self.cfg.contest_radius):
                next_pos = self._clip_to_bounds(self._agent_pos[a] + self._agent_vel[a])
                if np.linalg.norm(next_pos - payload) < dist_payload :
                    R_c = (np.linalg.norm(self._agent_vel[a]) / self.cfg.agent_max_speed) /3
                elif np.linalg.norm(next_pos - opp_payload) < dist_opp_payload:
                    R_c = (np.linalg.norm(self._agent_vel[a]) / self.cfg.agent_max_speed) /3
                else :
                    R_c = np.linalg.norm(self._agent_vel[a]) / self.cfg.agent_max_speed
           


            # Bonus for currently escorting
            is_escort = dist_payload < self.cfg.escort_radius
            R_push = (R_p > 0.0) * (1.0 if is_escort else 0.4)

            shaped = w_p * R_p * R_push + w_e * R_e + w_c * R_b + w_t * R_d - w_ctrl * R_c 
            terminal = w_wA if isA else w_wB

            rewards[a] = float(shaped + terminal)

        return rewards
    
