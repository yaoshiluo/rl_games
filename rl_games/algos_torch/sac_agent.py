from rl_games.algos_torch import torch_ext

from rl_games.common import vecenv
from rl_games.common import schedulers
from rl_games.common import experience
from rl_games.common.a2c_common import print_statistics

from rl_games.interfaces.base_algorithm import  BaseAlgorithm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from rl_games.algos_torch import  model_builder
from torch import optim
import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np
import time
import os


class SACAgent(BaseAlgorithm):

    def __init__(self, base_name, params):

        self.config = config = params['config']
        print(config)

        # TODO: Get obs shape and self.network
        self.load_networks(params)
        self.base_init(base_name, config)

        # ======== 新增：finetune 标志 ========
        # 来自 train.py 里写进 config 的 is_finetune / discard_replay_buffer
        self.is_finetune = self.config.get("is_finetune", False)
        self.discard_replay_buffer = self.config.get("discard_replay_buffer", False)
        # ===================================

        self.num_warmup_steps = config["num_warmup_steps"]
        self.gamma = config["gamma"]
        self.critic_tau = float(config["critic_tau"])
        self.batch_size = config["batch_size"]
        self.init_alpha = config["init_alpha"]
        self.learnable_temperature = config["learnable_temperature"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.num_steps_per_episode = config.get("num_steps_per_episode", 1)
        self.normalize_input = config.get("normalize_input", False)
        print("[DEBUG] SACAgent normalize_input =", self.normalize_input)

        # TODO: double-check! To use bootstrap instead?
        self.max_env_steps = config.get("max_env_steps", 1000) # temporary, in future we will use other approach

        print(self.batch_size, self.num_actors, self.num_agents)

        self.num_frames_per_epoch = self.num_actors * self.num_steps_per_episode

        self.log_alpha = torch.tensor(np.log(self.init_alpha)).float().to(self._device)
        self.log_alpha.requires_grad = True
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        net_config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'normalize_input': self.normalize_input,
        }
        self.model = self.network.build(net_config)
        self.model.to(self._device)

        # ======== Warmstart actor from BC checkpoint (optional) ========
        bc_path = self.config.get("warmstart_bc_ckpt", "")
        # 如果你是 resume（load_checkpoint=True），通常不希望被 BC 覆盖
        if bc_path and not self.config.get("load_checkpoint", False):
            bc_path = os.path.abspath(bc_path)
            print(f"[Warmstart] Loading BC actor from: {bc_path}")

            bc_ckpt = torch.load(bc_path, map_location=self._device)

            # 你已验证 STRICT-LOAD COMPATIBLE=True，所以 strict=True
            self.model.sac_network.actor.load_state_dict(bc_ckpt["actor"], strict=True)

            # RMS（若 normalize_input=True 且 BC 保存了 RMS）
            if "running_mean_std" in bc_ckpt and hasattr(self.model, "running_mean_std"):
                self.model.running_mean_std.load_state_dict(bc_ckpt["running_mean_std"], strict=True)

            print("[Warmstart] BC actor (+RMS) loaded successfully.")
        # =============================================================

        print("Number of Agents", self.num_actors, "Batch Size", self.batch_size)

        self.actor_optimizer = torch.optim.Adam(self.model.sac_network.actor.parameters(),
                                                lr=float(self.config['actor_lr']),
                                                betas=self.config.get("actor_betas", [0.9, 0.999]))

        self.critic_optimizer = torch.optim.Adam(self.model.sac_network.critic.parameters(),
                                                 lr=float(self.config["critic_lr"]),
                                                 betas=self.config.get("critic_betas", [0.9, 0.999]))

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=float(self.config["alpha_lr"]),
                                                    betas=self.config.get("alphas_betas", [0.9, 0.999]))

        self.replay_buffer = experience.VectorizedReplayBuffer(self.env_info['observation_space'].shape,
        self.env_info['action_space'].shape,
        self.replay_buffer_size,
        self._device)
        self.target_entropy_coef = config.get("target_entropy_coef", 1.0)
        self.target_entropy = self.target_entropy_coef * -self.env_info['action_space'].shape[0]
        print("Target entropy", self.target_entropy)

        self.algo_observer = config['features']['observer']

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)

    def base_init(self, base_name, config):
        self.env_config = config.get('env_config', {})
        self.num_actors = config.get('num_actors', 1)
        self.env_name = config['env_name']
        print("Env name:", self.env_name)

        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()

        self._device = config.get('device', 'cuda:0')

        #temporary for Isaac gym compatibility
        self.ppo_device = self._device
        print('Env info:')
        print(self.env_info)

        self.rewards_shaper = config['reward_shaper']
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        #self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.c_loss = nn.MSELoss()
        # self.c2_loss = nn.SmoothL1Loss()

        self.save_best_after = config.get('save_best_after', 500)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name

        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)

        self.save_freq = config.get('save_frequency', 0)

        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.num_agents = self.env_info.get('agents', 1)
        self.obs_shape = self.observation_space.shape

        self.games_to_track = self.config.get('games_to_track', 100)
        self.game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.obs = None

        self.min_alpha = torch.tensor(np.log(1)).float().to(self._device)

        self.frame = 0
        self.epoch_num = 0
        self.update_time = 0
        self.last_mean_rewards = -1000000000
        self.play_time = 0

        # TODO: put it into the separate class
        pbt_str = ''
        self.population_based_training = config.get('population_based_training', False)
        if self.population_based_training:
            # in PBT, make sure experiment name contains a unique id of the policy within a population
            pbt_str = f'_pbt_{config["pbt_idx"]:02d}'
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.writer = SummaryWriter('runs/' + config['name'] + datetime.now().strftime("_%d-%H-%M-%S"))
        print("Run Directory:", config['name'] + datetime.now().strftime("_%d-%H-%M-%S"))

        self.is_tensor_obses = False
        self.is_rnn = False
        self.last_rnn_indices = None
        self.last_state_indices = None

    def init_tensors(self):
        if self.observation_space.dtype == np.uint8:
            torch_dtype = torch.uint8
        else:
            torch_dtype = torch.float32
        batch_size = self.num_agents * self.num_actors

        self.current_rewards = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.long, device=self._device)

        self.dones = torch.zeros((batch_size,), dtype=torch.uint8, device=self._device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def device(self):
        return self._device

    def get_weights(self):
        print("Get weights for saving checkpoint")

        state = {}

        # 1) 给 rl-games Player / 通用使用：整套 model 的 state_dict
        if hasattr(self.model, "state_dict"):
            state["model"] = self.model.state_dict()

        # 2) 给我们自己 + 以后调试使用：拆开的 actor / critic / critic_target
        if hasattr(self.model, "sac_network"):
            sac_net = self.model.sac_network
            state["actor"] = sac_net.actor.state_dict()
            state["critic"] = sac_net.critic.state_dict()
            state["critic_target"] = sac_net.critic_target.state_dict()

        # 3) 观测归一化参数（RunningMeanStd），如果开启了 normalize_input
        if self.normalize_input and hasattr(self.model, "running_mean_std"):
            state["running_mean_std"] = self.model.running_mean_std.state_dict()

        # 4) replay buffer（为了继续训练不从空 buffer 开始）
        if hasattr(self, "replay_buffer") and self.replay_buffer is not None:
            state["replay_buffer"] = self.replay_buffer.get_state()

        return state

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def set_weights(self, weights):

        # if "log_alpha" in weights and hasattr(self, "log_alpha"):
        #     with torch.no_grad():
        #         self.log_alpha.data.copy_(weights["log_alpha"].to(self._device))

        # 1) 先恢复整套 model（给 Player 用的）
        if "model" in weights and hasattr(self.model, "load_state_dict"):
            self.model.load_state_dict(weights["model"])

        # 2) 再细粒度覆盖 actor / critic（可以保证和老格式兼容）
        if hasattr(self.model, "sac_network"):
            sac_net = self.model.sac_network
            if "actor" in weights:
                sac_net.actor.load_state_dict(weights["actor"])
            if "critic" in weights:
                sac_net.critic.load_state_dict(weights["critic"])
            if "critic_target" in weights:
                sac_net.critic_target.load_state_dict(weights["critic_target"])

        # 3) 观测归一化
        if self.normalize_input and "running_mean_std" in weights \
        and hasattr(self.model, "running_mean_std"):
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"])

        # 4) replay buffer
        # if "replay_buffer" in weights and hasattr(self, "replay_buffer"):
        #     try:
        #         self.replay_buffer.set_state(weights["replay_buffer"])
        #     except Exception as e:
        #         print("[WARN] Failed to restore replay_buffer:", e)


    def get_full_state_weights(self):
        print("Loading full weights")
        state = self.get_weights()

        state['epoch'] = self.epoch_num
        state['frame'] = self.frame
        state['actor_optimizer'] = self.actor_optimizer.state_dict()
        state['critic_optimizer'] = self.critic_optimizer.state_dict()
        state['log_alpha_optimizer'] = self.log_alpha_optimizer.state_dict() 
        # state['log_alpha'] = self.log_alpha.detach().to('cpu')

        if hasattr(self.replay_buffer, "get_state"):
            state['replay_buffer'] = self.replay_buffer.get_state()       

        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        self.set_weights(weights)

        if set_epoch:
            self.epoch_num = weights['epoch']
            self.frame = weights['frame']

        self.actor_optimizer.load_state_dict(weights['actor_optimizer'])
        self.critic_optimizer.load_state_dict(weights['critic_optimizer'])
        self.log_alpha_optimizer.load_state_dict(weights['log_alpha_optimizer'])

        # 恢复 Replay Buffer
        # if 'replay_buffer' in weights and hasattr(self.replay_buffer, "set_state"):
        #     self.replay_buffer.set_state(weights['replay_buffer'])

        # ====== 1) FINETUNE: 可选丢弃 replay buffer ======
        if 'replay_buffer' in weights and hasattr(self.replay_buffer, "set_state"):
            if self.discard_replay_buffer:
                print("[INFO][SACAgent] Finetune mode: NOT restoring replay_buffer (start fresh).")
                # 不调用 set_state，相当于保持当前（初始化后的）空 buffer
            else:
                self.replay_buffer.set_state(weights['replay_buffer'])

        # ====== 2) FINETUNE: 覆盖优化器学习率 ======
        # if self.is_finetune:
        #     new_actor_lr = float(self.config["actor_lr"])
        #     new_critic_lr = float(self.config["critic_lr"])
        #     for g in self.actor_optimizer.param_groups:
        #         g["lr"] = new_actor_lr
        #     for g in self.critic_optimizer.param_groups:
        #         g["lr"] = new_critic_lr
        #     print(f"[INFO][SACAgent] Finetune mode: override optimizer lr: "
        #           f"actor_lr={new_actor_lr}, critic_lr={new_critic_lr}")

        # 恢复 obs 归一化状态
        # if self.normalize_input and 'running_mean_std' in weights:
        #     self.model.running_mean_std.load_state_dict(weights['running_mean_std'])

        self.last_mean_rewards = weights.get('last_mean_rewards', -1000000000)

        if self.vec_env is not None:
            env_state = weights.get('env_state', None)
            self.vec_env.set_env_state(env_state)

    def restore(self, fn, set_epoch=True):
        print("SAC restore")
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def get_param(self, param_name):
        pass

    def set_param(self, param_name, param_value):
        pass

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def set_eval(self):
        self.model.eval()

    def set_train(self):
        self.model.train()

    def update_critic(self, obs, action, reward, next_obs, not_done, step):
        with torch.no_grad():
            dist = self.model.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

            target_Q1, target_Q2 = self.model.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha * log_prob

            target_Q = reward + (not_done * self.gamma * target_V)
            target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.model.critic(obs, action)

        critic1_loss = self.c_loss(current_Q1, target_Q)
        critic2_loss = self.c_loss(current_Q2, target_Q)
        critic_loss = critic1_loss + critic2_loss 

        # ===== DEBUG: Q statistics =====
        if step % 100 == 0:   
            with torch.no_grad():
                q1_mean = current_Q1.mean().item()
                q2_mean = current_Q2.mean().item()
                target_q_mean = target_Q.mean().item()
                q_gap = (current_Q1 - current_Q2).abs().mean().item()

            print(
                f"[Q DEBUG][step {step}] "
                f"Q1={q1_mean:.3f} "
                f"Q2={q2_mean:.3f} "
                f"TargetQ={target_q_mean:.3f} "
                f"|Q1-Q2|={q_gap:.3f}"
            )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.detach(), critic1_loss.detach(), critic2_loss.detach()

    def update_actor_and_alpha(self, obs, step):
        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = False

        dist = self.model.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = -log_prob.mean() #dist.entropy().sum(-1, keepdim=True).mean()
        actor_Q1, actor_Q2 = self.model.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (torch.max(self.alpha.detach(), self.min_alpha) * log_prob - actor_Q)
        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = True

        if self.learnable_temperature:
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            self.log_alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        else:
            alpha_loss = None

        return actor_loss.detach(), entropy.detach(), self.alpha.detach(), alpha_loss # TODO: maybe not self.alpha

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1.0 - tau) * target_param.data)

    def update(self, step):
        obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)
        not_done = ~done

        obs = self.preproc_obs(obs)
        next_obs = self.preproc_obs(next_obs)
        critic_loss, critic1_loss, critic2_loss = self.update_critic(obs, action, reward, next_obs, not_done, step)

        actor_loss, entropy, alpha, alpha_loss = self.update_actor_and_alpha(obs, step)

        actor_loss_info = actor_loss, entropy, alpha, alpha_loss
        self.soft_update_params(self.model.sac_network.critic, self.model.sac_network.critic_target,
                                     self.critic_tau)
        return actor_loss_info, critic1_loss, critic2_loss

    # def preproc_obs(self, obs):
    #     if isinstance(obs, dict):
    #         obs = obs['obs']
    #     obs = self.model.norm_obs(obs)

    #     return obs

    def preproc_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs["obs"]

        obs_before = obs.clone()
        obs = self.model.norm_obs(obs)

        if not hasattr(self, "_obs_debug_step"):
            self._obs_debug_step = 0
            self._obs_debug_enabled = True

        self._obs_debug_step += 1

        PRINT_INTERVAL = 200      
        MAX_PRINT = 10            

        if self._obs_debug_enabled and (self._obs_debug_step % PRINT_INTERVAL == 0):
            diff = (obs - obs_before).abs().max().item()
            print("\n===== SAC Obs Normalization Debug =====")
            print(f"Step #{self._obs_debug_step}")
            print("normalize_input =", getattr(self.model, "normalize_input", None))
            print("obs_before[:8] =", obs_before[0, :8].cpu().numpy())
            print("obs_after[:8]  =", obs[0, :8].cpu().numpy())
            print("max_abs_diff   =", diff)
            print("Has RMS?       =", hasattr(self.model, "running_mean_std"))
            print("=====================================\n")

            # 打印次数达到 MAX_PRINT 后关闭
            if (self._obs_debug_step // PRINT_INTERVAL) >= MAX_PRINT:
                self._obs_debug_enabled = False
                print("[DEBUG] disable obs normalization debug print (MAX_PRINT reached)\n")

        return obs



    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(self.observation_space.dtype != np.int8)
            if self.observation_space.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self._device)
            else:
                obs = torch.FloatTensor(obs).to(self._device)

        return obs

    # TODO: move to common utils
    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:    
            upd_obs = {'obs' : upd_obs}

        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)

        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()

        return actions

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions) # (obs_space) -> (n, obs_space)

        if self.is_tensor_obses:
            return self.obs_to_tensors(obs), rewards.to(self._device), dones.to(self._device), infos
        else:
            return torch.from_numpy(obs).to(self._device), torch.from_numpy(rewards).to(self._device), torch.from_numpy(dones).to(self._device), infos

    def env_reset(self):
        with torch.no_grad():
            obs = self.vec_env.reset()

        obs = self.obs_to_tensors(obs)

        return obs

    def act(self, obs, action_dim, sample=False):
        obs = self.preproc_obs(obs)
        dist = self.model.actor(obs)

        actions = dist.sample() if sample else dist.mean
        actions = actions.clamp(*self.action_range)
        assert actions.ndim == 2

        return actions

    def extract_actor_stats(self, actor_losses, entropies, alphas, alpha_losses, actor_loss_info):
        actor_loss, entropy, alpha, alpha_loss = actor_loss_info

        actor_losses.append(actor_loss)
        entropies.append(entropy)
        if alpha_losses is not None:
            alphas.append(alpha)
            alpha_losses.append(alpha_loss)

    def clear_stats(self):
        self.game_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -1000000000
        self.algo_observer.after_clear_stats()

    def play_steps(self, random_exploration = False):
        total_time_start = time.time()
        total_update_time = 0
        total_time = 0
        step_time = 0.0
        actor_losses = []
        entropies = []
        alphas = []
        alpha_losses = []
        critic1_losses = []
        critic2_losses = []

        obs = self.obs
        if isinstance(obs, dict):
            # obs = self.obs['obs']
            obs = obs['obs']

        next_obs_processed = obs.clone()

        for s in range(self.num_steps_per_episode):
            self.set_eval()
            if random_exploration:
                action = torch.rand((self.num_actors, *self.env_info["action_space"].shape), device=self._device) * 2.0 - 1.0
            else:
                with torch.no_grad():
                    action = self.act(obs.float(), self.env_info["action_space"].shape, sample=True)

            step_start = time.time()

            with torch.no_grad():
                next_obs, rewards, dones, infos = self.env_step(action)
            step_end = time.time()

            self.current_rewards += rewards
            self.current_lengths += 1

            total_time += (step_end - step_start)
            step_time += (step_end - step_start)

            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])

            not_dones = 1.0 - dones.float()

            self.algo_observer.process_infos(infos, done_indices)

            no_timeouts = self.current_lengths != self.max_env_steps
            dones = dones * no_timeouts

            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones

            # if isinstance(next_obs, dict):    
            #     next_obs_processed = next_obs['obs']

            # self.obs = next_obs.clone()
            if isinstance(next_obs, dict):
                next_obs_processed = next_obs['obs']
                self.obs = next_obs_processed.clone()
            else:
                next_obs_processed = next_obs
                self.obs = next_obs.clone()


            rewards = self.rewards_shaper(rewards)

            self.replay_buffer.add(obs, action, torch.unsqueeze(rewards, 1), next_obs_processed, torch.unsqueeze(dones, 1))

            # if isinstance(obs, dict):
            #     obs = self.obs['obs']
            obs = next_obs_processed


            if not random_exploration:
                self.set_train()
                update_time_start = time.time()
                actor_loss_info, critic1_loss, critic2_loss = self.update(self.epoch_num)
                update_time_end = time.time()
                update_time = update_time_end - update_time_start

                self.extract_actor_stats(actor_losses, entropies, alphas, alpha_losses, actor_loss_info)
                critic1_losses.append(critic1_loss)
                critic2_losses.append(critic2_loss)
            else:
                update_time = 0

            total_update_time += update_time

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        play_time = total_time - total_update_time

        return step_time, play_time, total_update_time, total_time, actor_losses, entropies, alphas, alpha_losses, critic1_losses, critic2_losses

    def train_epoch(self):
        random_exploration = self.epoch_num < self.num_warmup_steps
        return self.play_steps(random_exploration)

    def train(self):
        self.init_tensors()
        self.algo_observer.after_init(self)
        total_time = 0
        # rep_count = 0

        self.obs = self.env_reset()
        #添加自己的buffer
        # self.maybe_prefill_replay_buffer()

        while True:
            self.epoch_num += 1
            step_time, play_time, update_time, epoch_total_time, actor_losses, entropies, alphas, alpha_losses, critic1_losses, critic2_losses = self.train_epoch()

            total_time += epoch_total_time

            curr_frames = self.num_frames_per_epoch
            self.frame += curr_frames

            fps_step = curr_frames / step_time
            fps_step_inference = curr_frames / play_time
            fps_total = curr_frames / epoch_total_time

            print_statistics(self.print_stats, curr_frames, step_time, play_time, epoch_total_time, 
                self.epoch_num, self.max_epochs, self.frame, self.max_frames)

            self.writer.add_scalar('performance/step_inference_rl_update_fps', fps_total, self.frame)
            self.writer.add_scalar('performance/step_inference_fps', fps_step_inference, self.frame)
            self.writer.add_scalar('performance/step_fps', fps_step, self.frame)
            self.writer.add_scalar('performance/rl_update_time', update_time, self.frame)
            self.writer.add_scalar('performance/step_inference_time', play_time, self.frame)
            self.writer.add_scalar('performance/step_time', step_time, self.frame)

            if self.epoch_num >= self.num_warmup_steps:
                self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(actor_losses).item(), self.frame)
                self.writer.add_scalar('losses/c1_loss', torch_ext.mean_list(critic1_losses).item(), self.frame)
                self.writer.add_scalar('losses/c2_loss', torch_ext.mean_list(critic2_losses).item(), self.frame)
                self.writer.add_scalar('losses/entropy', torch_ext.mean_list(entropies).item(), self.frame)

                if alpha_losses[0] is not None:
                    self.writer.add_scalar('losses/alpha_loss', torch_ext.mean_list(alpha_losses).item(), self.frame)
                self.writer.add_scalar('info/alpha', torch_ext.mean_list(alphas).item(), self.frame)

            self.writer.add_scalar('info/epochs', self.epoch_num, self.frame)
            self.algo_observer.after_print_stats(self.frame, self.epoch_num, total_time)

            if self.game_rewards.current_size > 0:
                mean_rewards = self.game_rewards.get_mean()
                mean_lengths = self.game_lengths.get_mean()

                self.writer.add_scalar('rewards/step', mean_rewards, self.frame)
                self.writer.add_scalar('rewards/time', mean_rewards, total_time)
                self.writer.add_scalar('episode_lengths/step', mean_lengths, self.frame)
                self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)
                checkpoint_name = self.config['name'] + '_ep_' + str(self.epoch_num) + '_rew_' + str(mean_rewards)

                should_exit = False

                if self.save_freq > 0:
                    if self.epoch_num % self.save_freq == 0:
                        self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                if mean_rewards > self.last_mean_rewards and self.epoch_num >= self.save_best_after:
                    print('saving next best rewards: ', mean_rewards)
                    self.last_mean_rewards = mean_rewards
                    self.save(os.path.join(self.nn_dir, self.config['name']))
                    if self.last_mean_rewards > self.config.get('score_to_win', float('inf')):
                        print('Maximum reward achieved. Network won!')
                        self.save(os.path.join(self.nn_dir, checkpoint_name))
                        should_exit = True

                if self.epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(self.epoch_num) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0

                if should_exit:
                    return self.last_mean_rewards, self.epoch_num

    def _check_offline_dataset(self, obs, actions, rewards, next_obs, dones):
        # ---- shape checks ----
        assert obs.ndim == 2, f"obs must be (T, obs_dim), got {obs.shape}"
        assert next_obs.ndim == 2, f"next_obs must be (T, obs_dim), got {next_obs.shape}"
        assert actions.ndim == 2, f"actions must be (T, act_dim), got {actions.shape}"
        T = obs.shape[0]
        assert next_obs.shape[0] == T, "next_obs length mismatch"
        assert actions.shape[0] == T, "actions length mismatch"
        assert rewards.shape[0] == T, "rewards length mismatch"
        assert dones.shape[0] == T, "dones length mismatch"

        # ---- dim checks against env_info ----
        obs_dim = self.env_info["observation_space"].shape[0]
        act_dim = self.env_info["action_space"].shape[0]
        assert obs.shape[1] == obs_dim, f"obs_dim mismatch: data {obs.shape[1]} vs env {obs_dim}"
        assert next_obs.shape[1] == obs_dim, f"next_obs_dim mismatch: data {next_obs.shape[1]} vs env {obs_dim}"
        assert actions.shape[1] == act_dim, f"act_dim mismatch: data {actions.shape[1]} vs env {act_dim}"

        # ---- finite checks ----
        assert np.isfinite(obs).all(), "obs contains NaN/Inf"
        assert np.isfinite(next_obs).all(), "next_obs contains NaN/Inf"
        assert np.isfinite(actions).all(), "actions contains NaN/Inf"
        assert np.isfinite(rewards).all(), "rewards contains NaN/Inf"
        assert np.isfinite(dones).all(), "dones contains NaN/Inf"

        # ---- dones sanity ----
        # allow {0,1} (float/bool/uint8)
        u = np.unique(dones)
        # dones may be float; tolerate small numerical noise
        if not ((u.min() >= -1e-6) and (u.max() <= 1.0 + 1e-6)):
            raise ValueError(f"dones values out of range [0,1]: min={u.min()}, max={u.max()}")

        # ---- quick stats ----
        print("[OFFLINE DATA] T =", T)
        print("[OFFLINE DATA] rewards: mean =", float(rewards.mean()), "std =", float(rewards.std()),
            "min =", float(rewards.min()), "max =", float(rewards.max()))
        print("[OFFLINE DATA] dones:   mean =", float(dones.mean()),
            "(fraction done if 0/1)")

    def load_replay_from_npz(self, npz_path: str, limit: int = -1, clear_existing: bool = False):
        """
        Load offline dataset transitions into SAC replay buffer.

        IMPORTANT for your sim->real finetune:
        - This function DOES NOT update RunningMeanStd (RMS). Keep sim RMS and let it adapt online.
        """
        data = np.load(npz_path)

        obs = data["obs"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        rewards = data["rewards"].astype(np.float32)
        next_obs = data["next_obs"].astype(np.float32)
        dones = data["dones"]

        # shape normalize to (T,1) for rewards/dones
        if rewards.ndim == 1:
            rewards = rewards[:, None]
        if dones.ndim == 1:
            dones = dones[:, None]

        # convert dones to float {0,1}
        if dones.dtype != np.float32:
            dones = dones.astype(np.float32)

        T = obs.shape[0]
        if limit > 0:
            T = min(T, limit)
            obs = obs[:T]
            actions = actions[:T]
            rewards = rewards[:T]
            next_obs = next_obs[:T]
            dones = dones[:T]

        # validate dataset (strongly recommended)
        self._check_offline_dataset(obs, actions, rewards, next_obs, dones)

        # optionally clear buffer first
        if clear_existing:
            # VectorizedReplayBuffer in rl_games often supports reset/clear; if not, re-create it
            if hasattr(self.replay_buffer, "reset"):
                self.replay_buffer.reset()
                print("[INFO] replay_buffer.reset() called")
            else:
                # safest fallback: re-create buffer with same params
                from rl_games.common import experience
                self.replay_buffer = experience.VectorizedReplayBuffer(
                    self.env_info['observation_space'].shape,
                    self.env_info['action_space'].shape,
                    self.replay_buffer_size,
                    self._device
                )
                print("[INFO] replay_buffer re-created (no reset() method found)")

        # to torch on correct device
        obs_t  = torch.from_numpy(obs).to(self._device)
        act_t  = torch.from_numpy(actions).to(self._device)
        rew_t  = torch.from_numpy(rewards).to(self._device)
        nxt_t  = torch.from_numpy(next_obs).to(self._device)
        done_t = torch.from_numpy(dones).to(self._device)

        # feed into replay buffer in chunks
        chunk = 4096
        for i in range(0, T, chunk):
            j = min(T, i + chunk)
            self.replay_buffer.add(
                obs_t[i:j],
                act_t[i:j],
                rew_t[i:j],
                nxt_t[i:j],
                done_t[i:j],
            )

        print(f"[INFO] Loaded {T} offline transitions into replay buffer from: {npz_path}")

    def maybe_prefill_replay_buffer(self):
        """
        Call once at the beginning of training (after env_info/replay_buffer are created).
        Controlled via config keys:
        - offline_buffer_path
        - offline_buffer_limit
        - offline_buffer_clear_existing
        """
        path = self.config.get("offline_buffer_path", "")
        if not path:
            return

        limit = int(self.config.get("offline_buffer_limit", -1))
        clear_existing = bool(self.config.get("offline_buffer_clear_existing", False))
        self.load_replay_from_npz(path, limit=limit, clear_existing=clear_existing)

        # For real finetune: do NOT random explore warmup if offline data exists
        if int(self.config.get("num_warmup_steps", 0)) > 0:
            print("[INFO] Offline buffer provided: set num_warmup_steps=0 recommended for real finetune.")
