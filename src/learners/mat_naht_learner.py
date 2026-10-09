# code heavily adapted from original ePymarl implementation, based on the MAPPO implementation.
import os
import numpy as np
from components.episode_buffer import EpisodeBatch
import torch as th
import torch.nn.functional as F
from torch.optim import Adam
import math

from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_registry

from components.mat_naht.mat_naht import MATEncoder, MATDecoder

from utils.rl_utils import compute_per_agent_terminated_mask, get_noop_act
from utils.mappo_util import huber_loss

class MATNAHTLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        self.mac = mac

        obs_dim = scheme["obs"]["vshape"]
        act_dim = args.n_actions
        hidden_dim = getattr(args, "hidden_dim", 128)
        history_len = getattr(args, "history_len", 5)
        n_heads = getattr(args, "n_heads", 4)
        n_layers = getattr(args, "n_layers", 2)
        dropout = getattr(args, "dropout", 0.1)

        self.encoder = MATEncoder(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            history_len=history_len
        )

        self.decoder = MATDecoder(
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )

        self.mac.set_modules(encoder=self.encoder, decoder=self.decoder)

        self.critic = critic_registry[args.critic_type](scheme, args, shared_encoder=self.encoder)
        self.critic.set_encoder(self.encoder)

        ed_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.encoder_decoder_optimiser = Adam(
            params=ed_params,
            lr=getattr(args, "ed_lr", args.lr),
            eps=self.args.optim_eps
        )

        agent_core = self.mac.trained_agent.policy.core
        actor_only_params = list(self.decoder.parameters())
        if hasattr(agent_core, "value_head"):
            actor_only_params += list(agent_core.value_head.parameters())
        
        self.agent_params = actor_only_params

        self.agent_optimiser = Adam(
            params=self.agent_params,
            lr=args.lr,
            eps=self.args.optim_eps
        )

        critic_only_params = list(self.critic.v_head.parameters())
        if hasattr(self.critic, "v_out") and self.critic.v_out is not None:
            critic_only_params += list(self.critic.v_out.parameters())

        self.critic_params = critic_only_params

        self.critic_optimiser = Adam(
            params=self.critic_params,
            lr=getattr(args, "critic_lr", args.lr),
            eps=self.args.optim_eps
        )

        self.log_stats_t = 0
        self.log_extra_stats_t = 0
        device = "cuda" if args.use_cuda else "cpu"

        if getattr(self.args, "standardise_rewards", False):
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

        self.value_normalizer = getattr(self.critic, "value_normalizer", None)
        self.noop_acts = get_noop_act(args, device)

        self.extra_stats_path = os.path.join(args.local_results_path, "data", args.expt_logname)
        os.makedirs(self.extra_stats_path, exist_ok=True)


    def compute_mask(self, batch, max_t):
        if self.args.mask_type == "team":
            terminated = batch["terminated"][:, :max_t].float()
            mask = batch["filled"][:, :max_t].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            mask = mask.repeat(1, 1, self.n_agents) 
        elif self.args.mask_type == "agent":
            mask = compute_per_agent_terminated_mask(terminated_batch=batch['terminated'][:, :max_t],
                                                        avail_actions_batch= batch['avail_actions'][:, :max_t], 
                                                        n_rollout_threads=batch.batch_size,
                                                        no_op_tensor_list=self.noop_acts,
                                                        ep_limit=max_t, device=self.args.device
                                                        )
        else: 
            raise Exception("ERROR: mask_type {} not implemented".format(self.args.mask_type))
        # mask shape: (bs, ep_len, n_agents)
        return mask

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        # batch[terminated] has shape (bs, ep_len + 1, 1)
        max_t = batch['terminated'].shape[1] - 1
        actor_mask = self.compute_mask(batch, max_t=max_t)
        critic_mask = actor_mask.detach().clone()
        ed_mask = actor_mask.detach().clone()
        if self.args.open_train_or_eval:
            # in batch[trainable_agents], 1 means trainable, 0 means untrainable
            if self.args.ed_model_uncontrolled_only:
                ed_mask = ed_mask * (~batch["trainable_agents"][:, :max_t].squeeze(-1))
            else: 
                ed_mask = ed_mask * batch["trainable_agents"][:, :max_t].squeeze(-1)
            if self.args.trainable_agents_mask_actor:
                actor_mask = actor_mask * batch["trainable_agents"][:, :max_t].squeeze(-1)
            if self.args.trainable_agents_mask_critic:
                critic_mask = critic_mask * batch["trainable_agents"][:, :max_t].squeeze(-1)
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
        ##################################
        # Compute old values, targets, and advantages
        old_values, old_critic_hidden_states = self.critic_forward_nograd(self.critic, batch)
        old_values = old_values[:, :-1].squeeze(3)
        old_critic_hidden_states = old_critic_hidden_states

        target_returns = self.compute_target_returns(self.critic, 
                                                        batch, old_critic_hidden_states, 
                                                        rewards).clone().detach()
        
        if self.value_normalizer:
            self.value_normalizer.update(target_returns)
            denorm_old_values = self.value_normalizer.denormalize(old_values)
        else:
            denorm_old_values = old_values
        advantages = (target_returns - denorm_old_values)

        if self.args.use_adv_std:
            advantages_copy = advantages.clone().detach()
            std_advantages, mean_advantages = th.std_mean(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        ##################################
        # Compute old log probs
        old_log_prob_taken, _ = self.actor_forward_all(self.mac, batch, actions)
        old_log_prob_taken = old_log_prob_taken.clone().detach()

        #################################
        actor_train_stats = {
            "actor_loss": [],
            "actor_grad_norm": [],
            "entropy": [],
            "log_pi_taken": [],
            "ratio": [],
        }
        critic_train_stats = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "curr_taken_v": [],
        }
        encoder_decoder_train_stats = {
            "ed_obs_loss": [],
            "ed_act_loss": [],
            "ed_loss": [],
            "ed_entropy": [],
            "ed_grad_norm": [],
        }

        # perform encoder-decoder update
        if t_env <= self.args.ed_pretrain_ts:
            print(f"Training ED only. t_env is {t_env}")
            for _ in range(self.args.ed_pretrain_epochs):
                encoder_decoder_train_stats = self.encoder_decoder_update(batch=batch,
                                                                        mask=ed_mask,
                                                                        encoder_decoder_train_stats=encoder_decoder_train_stats,
                                                                        t_env=t_env
                                                                        )
        else: 
            print(f"Switching from updating ED only to updating all. t_env is {t_env}")

            # perform PPO update
            for _ in range(self.args.epochs):
                # generate randomized minibatches across batch dim only (preserve temporal structure)
                mb_rand = th.randperm(batch.batch_size).numpy() 
                mb_size = batch.batch_size // self.args.n_minibatch
                sampler = [np.array(mb_rand[i * mb_size:(i + 1) * mb_size]) for i in range(self.args.n_minibatch)]
                old_ed_hidden, old_critic_hidden = old_critic_hidden_states

                for indices in sampler:             
                    log_pi_taken, entropy = self.actor_forward_all(self.mac, batch[indices], actions[indices])
                    curr_v = self.critic_forward_all(self.critic, 
                                                    batch[indices], 
                                                    hidden_states=(old_ed_hidden[indices], old_critic_hidden[indices]))
                    curr_v = curr_v[:, :-1].squeeze(3)
                    ##################################
                    actor_train_stats = self.actor_update(log_pi_taken=log_pi_taken, 
                                                        entropy=entropy, 
                                                        advantages=advantages[indices],
                                                        old_log_prob_taken=old_log_prob_taken[indices], 
                                                        mask=actor_mask[indices],
                                                        actor_train_stats=actor_train_stats,
                                                        )
                    critic_train_stats = self.critic_update(curr_values=curr_v,
                                                            old_values=old_values[indices], 
                                                            target_returns=target_returns[indices],
                                                            mask=critic_mask[indices], 
                                                            critic_train_stats=critic_train_stats
                                                            )
                    check_encoder_gradient_isolation(self)
                    ###################################
                    encoder_decoder_train_stats = self.encoder_decoder_update(batch=batch[indices],
                                                                                mask=ed_mask[indices],
                                                                                encoder_decoder_train_stats=encoder_decoder_train_stats,
                                                                                t_env=t_env
                                                                                )

        # logging
        if t_env - self.log_stats_t >= self.args.learner_log_interval or self.log_stats_t == 0:
            ts_logged = len(critic_train_stats["critic_loss"]) if len(critic_train_stats["critic_loss"]) > 0 else 1
            for key in critic_train_stats:
                self.logger.log_stat(key, sum(critic_train_stats[key]) / ts_logged, t_env)
            ts_logged = len(actor_train_stats["actor_loss"]) if len(actor_train_stats["actor_loss"]) > 0 else 1
            for key in actor_train_stats:
                self.logger.log_stat(key, sum(actor_train_stats[key]) / ts_logged, t_env)
            ts_logged = len(encoder_decoder_train_stats["ed_loss"]) if len(encoder_decoder_train_stats["ed_loss"]) > 0 else 1
            for key in encoder_decoder_train_stats:
                self.logger.log_stat(key, sum(encoder_decoder_train_stats[key]) / ts_logged, t_env)

            self.logger.log_stat("old_values_mean", old_values.mean().item(), t_env)
            self.logger.log_stat("advantage_mean", advantages.mean().item(), t_env)
            self.logger.log_stat("advantage_std", advantages.std().item(), t_env)
            self.log_stats_t = t_env

    def reshape_batches(self, batch_list, bs, max_t):
        '''Reshape data from (bs, ep_len, n_agents, feat_size) to (bs * ep_len, n_agents, feat_size)'''
        reshaped = []
        for batch in batch_list:
            reshaped.append(batch.reshape(bs * max_t, -1))
        return tuple(reshaped)

    def encoder_decoder_update(self, batch, mask, encoder_decoder_train_stats, t_env):
        device = batch.device
        bs = batch.batch_size
        history_len = getattr(self.args, "history_len", 5)
        
        obs = batch["obs"][:, :-1]
        actions_onehot = batch["actions_onehot"][:, :-1]
        actions = batch["actions"][:, :-1]
        comm_mask = batch["comm_mask"][:, :-1] if "comm_mask" in batch.scheme else None
        
        pad_o = th.zeros(bs, history_len - 1, self.n_agents, obs.shape[-1], device=device)
        pad_a = th.zeros(bs, history_len - 1, self.n_agents, actions_onehot.shape[-1], device=device)
        
        padded_obs = th.cat([pad_o, obs], dim=1)
        padded_act = th.cat([pad_a, actions_onehot], dim=1)
        
        hist_obs_all = padded_obs.unfold(dimension=1, size=history_len, step=1).permute(0, 1, 4, 2, 3)
        hist_act_all = padded_act.unfold(dimension=1, size=history_len, step=1).permute(0, 1, 4, 2, 3)
        mask_expanded = mask.expand(-1, -1, self.n_agents) if mask.dim() == 2 else mask
        
        core_model = self.mac.agent if hasattr(self.mac, "agent") else self.mac.trained_agent.policy.core

        for _ in range(getattr(self.args, "ed_epochs", 1)):
            mb_rand = th.randperm(bs, device=device)
            n_ed_mb = getattr(self.args, "n_ed_minibatch", 1)
            mb_size = bs // n_ed_mb
            sampler = [mb_rand[i * mb_size:(i + 1) * mb_size] for i in range(n_ed_mb)]
            
            for indices in sampler:
                hist_obs_mb = hist_obs_all[indices].reshape(-1, self.n_agents, history_len, obs.shape[-1])
                hist_act_mb = hist_act_all[indices].reshape(-1, self.n_agents, history_len, actions_onehot.shape[-1])
                
                actions_mb = actions[indices].reshape(-1, self.n_agents, 1)
                act_onehot_mb = actions_onehot[indices].reshape(-1, self.n_agents, self.n_actions)
                mask_mb = mask_expanded[indices].reshape(-1, self.n_agents)
                
                comm_mask_flat = comm_mask[indices].reshape(-1, self.n_agents, self.n_agents) if comm_mask is not None else None

                embeddings = core_model.encoder(hist_obs_mb, hist_act_mb, comm_mask=comm_mask_flat)
                decoded_act_logits = core_model.decoder(
                    history_rep=embeddings, 
                    prev_actions=act_onehot_mb, 
                    comm_mask=comm_mask_flat
                )
                
                log_probs = F.log_softmax(decoded_act_logits, dim=-1)
                action_log_pi = th.gather(log_probs, dim=-1, index=actions_mb).squeeze(-1)
                
                act_loss = (-action_log_pi * mask_mb).sum() / (mask_mb.sum() + 1e-8)

                probs = F.softmax(decoded_act_logits, dim=-1)
                dist_entropy = -(probs * log_probs).sum(dim=-1)
                entropy_loss = (dist_entropy * mask_mb).sum() / (mask_mb.sum() + 1e-8)

                ed_entropy_coef = getattr(self.args, "ed_entropy_coef", 0.01)
                loss = act_loss - ed_entropy_coef * entropy_loss

                opt = self.encoder_decoder_optimiser if self.encoder_decoder_optimiser is not None else self.agent_optimiser
                params_to_clip = (
                    list(self.encoder.parameters()) + list(self.decoder.parameters()) 
                    if self.encoder is not None and self.decoder is not None 
                    else self.agent_params
                )

                opt.zero_grad()
                loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(params_to_clip, self.args.grad_norm_clip)
                opt.step()

                encoder_decoder_train_stats["ed_act_loss"].append(act_loss.item())
                encoder_decoder_train_stats["ed_loss"].append(loss.item())
                encoder_decoder_train_stats["ed_entropy"].append(entropy_loss.item())
                encoder_decoder_train_stats["ed_grad_norm"].append(grad_norm.item())

        return encoder_decoder_train_stats
    
    def actor_update(self, log_pi_taken, entropy, 
                     advantages, old_log_prob_taken, 
                     mask, actor_train_stats):

        valid_mask = mask.bool()
        if valid_mask.sum() > 1:
            adv_mean = advantages[valid_mask].mean()
            adv_std = advantages[valid_mask].std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std
        
        advantages = th.clamp(advantages, -2.0, 2.0)

        log_diff = th.clamp(log_pi_taken - old_log_prob_taken, min=-1.0, max=1.0)
        ratios = th.exp(log_diff)

        surr1 = ratios * advantages  
        surr2 = th.clamp(ratios, 1.0 - self.args.eps_clip, 1.0 + self.args.eps_clip) * advantages
        
        policy_surr = th.min(surr1, surr2)
        mask_sum = mask.sum() + 1e-8
        
        policy_loss = -(policy_surr * mask).sum() / mask_sum
        mean_entropy = (entropy * mask).sum() / mask_sum
        
        # استفاده از ضریب کانفیگ شما
        entropy_coef = getattr(self.args, "entropy_coef", 0.01)
        
        # مکانیزم ایمنی جدید: اگر میانگین آنتروپی از 1.0 کمتر شد، ضریب آنتروپی به طور موقت افزایش می‌یابد تا جلوی فاجعه گرفته شود
        if mean_entropy.item() < 1.0:
            entropy_coef = entropy_coef * 5.0

        actor_loss = policy_loss - (entropy_coef * mean_entropy)

        self.agent_optimiser.zero_grad()
        actor_loss.backward()
        
        # ==================== DETAILED GRADIENT & ADVANTAGE DIAGNOSTICS ====================
        with th.no_grad():
            print("\n" + "="*60)
            print("[ACTOR UPDATE DIAGNOSTICS]")
            print(f"Advantage -> Mean: {advantages.mean().item():.8f} | Std: {advantages.std().item():.8f} | Max: {advantages.max().item():.8f}")
            print(f"Ratios    -> Mean: {ratios.mean().item():.6f} | Min: {ratios.min().item():.6f} | Max: {ratios.max().item():.6f}")
            print(f"Policy Loss Term  : {(policy_loss * mask).sum().item() / mask.sum().item():.8f}")
            
            # چک کردن گرادیان لایه خروجی Actor (Action Head)
            core_model = self.mac.agent if hasattr(self.mac, "agent") else self.mac.trained_agent.policy.core
            if hasattr(core_model.decoder, "action_head"):
                act_grad = core_model.decoder.action_head.weight.grad
                if act_grad is not None:
                    print(f"Action Head Grad Norm: {act_grad.norm().item():.8f} | Grad Std: {act_grad.std().item():.8f}")
            print("="*60 + "\n")
        # ===================================================================================
        
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        actor_train_stats["actor_loss"].append(actor_loss.item())
        actor_train_stats.setdefault("policy_loss", []).append(policy_loss.item())
        actor_train_stats["actor_grad_norm"].append(grad_norm.item())
        actor_train_stats["entropy"].append(mean_entropy.item())
        actor_train_stats["log_pi_taken"].append(log_pi_taken.mean().item())
        actor_train_stats["ratio"].append((ratios * mask).sum().item() / mask_sum.item())
        
        return actor_train_stats

    @th.no_grad()
    def critic_forward_nograd(self, critic, batch): 
        v = []
        h_e, h = critic.init_hidden()
        critic_hidden_states = []
        ed_hidden_states = []
        for t in range(batch.max_seq_length):
            ed_hidden_states.append(h_e)
            critic_hidden_states.append(h)
            v_t, h_e, h = critic.forward(batch, (h_e, h.detach()), t=t)
            v.append(v_t)
        v = th.cat(v, dim=1)  # Concat over time
        ed_hidden_states = th.cat(ed_hidden_states, dim=1)
        critic_hidden_states = th.cat(critic_hidden_states, dim=1)
        return v, (ed_hidden_states, critic_hidden_states)
    
    def critic_forward_all(self, critic, batch, hidden_states):
        '''Compute values using given hidden states, 
        for all timesteps in a single forward pass.'''
        v, _, _ = critic.forward(batch, hidden_states, t=None)
        return v
    
    def actor_forward_all(self, mac, batch, actions):
        device = batch.device
        bs = batch.batch_size
        max_t = batch.max_seq_length - 1
        history_len = getattr(self.args, "history_len", 5)

        obs = batch["obs"][:, :-1]  # [bs, max_t, n_agents, obs_dim]
        actions_onehot = batch["actions_onehot"][:, :-1]
        avail_actions = batch["avail_actions"][:, :-1] if "avail_actions" in batch.scheme else None
        comm_mask = batch["comm_mask"][:, :-1] if "comm_mask" in batch.scheme else None

        pad_o = th.zeros(bs, history_len - 1, self.n_agents, obs.shape[-1], device=device)
        pad_a = th.zeros(bs, history_len - 1, self.n_agents, actions_onehot.shape[-1], device=device)

        padded_obs = th.cat([pad_o, obs], dim=1)
        padded_act = th.cat([pad_a, actions_onehot], dim=1)

        # اضافه شدن .contiguous() جهت حفظ پیوستگی حافظه GPU در اجراهای بلندمدت
        hist_obs_all = padded_obs.unfold(dimension=1, size=history_len, step=1).permute(0, 1, 4, 2, 3).contiguous()
        hist_act_all = padded_act.unfold(dimension=1, size=history_len, step=1).permute(0, 1, 4, 2, 3).contiguous()

        hist_obs_flat = hist_obs_all.reshape(-1, self.n_agents, history_len, obs.shape[-1])
        hist_act_flat = hist_act_all.reshape(-1, self.n_agents, history_len, actions_onehot.shape[-1])
        actions_flat = actions.reshape(-1, self.n_agents)
        
        avail_actions_flat = avail_actions.reshape(-1, self.n_agents, self.n_actions) if avail_actions is not None else None
        comm_mask_flat = comm_mask.reshape(-1, self.n_agents, self.n_agents) if comm_mask is not None else None

        core_model = mac.agent if hasattr(mac, "agent") else mac.trained_agent.policy.core
        action_log_probs, values, dist_entropy = core_model.evaluate_actions(
            history_obs=hist_obs_flat,
            history_act=hist_act_flat,
            actions=actions_flat,
            comm_mask=comm_mask_flat,
            avail_actions=avail_actions_flat
        )

        # تغییر شکل خروجی‌ها به [bs, max_t, n_agents]
        log_pi_taken = action_log_probs.reshape(bs, max_t, self.n_agents)
        entropy = dist_entropy.reshape(bs, max_t, self.n_agents)

        return log_pi_taken, entropy
    
    def compute_target_returns(self, critic, batch, old_critic_hidden_states, rewards):
        '''
        old_critic_hidden_states: the hidden states computed by the critic 
        before any updates (i.e. at time data was gathered)
        '''
        # TODO: remove extra critic call 
        with th.no_grad(): 
            target_vals = self.critic_forward_all(critic, batch, hidden_states=old_critic_hidden_states)
            target_vals = target_vals.squeeze(3)

        target_mask = self.compute_mask(batch, max_t=batch['terminated'].shape[1])
        if self.value_normalizer is not None:
            target_vals = self.value_normalizer.denormalize(target_vals)

        if self.args.use_gae:
            target_returns = self.gae_target(rewards, target_mask, target_vals)
        else:
            target_returns = self.nstep_returns(rewards, target_mask, target_vals, self.args.q_nstep)
        
        return target_returns # shape (batch_size, episode_length, n_agents)

    def critic_update(self, curr_values, old_values,
                        target_returns, mask, critic_train_stats):
        if self.value_normalizer is not None:
            target_returns = self.value_normalizer.normalize(target_returns)            

        mb, max_t = curr_values.shape[:2]
        rand = th.randperm(mb * max_t).numpy() 
        curr_values = curr_values.reshape(mb * max_t, -1)[rand]
        old_values = old_values.reshape(mb * max_t, -1)[rand]
        target_returns = target_returns.reshape(mb * max_t, -1)[rand]
        mask = mask.reshape(mb * max_t, -1)[rand]
        
        clipped_v = old_values + (curr_values - old_values).clamp(-self.args.eps_clip, 
                                                                    self.args.eps_clip)

        td_error = (target_returns - curr_values)
        clipped_td_error = (target_returns - clipped_v)

        if self.args.use_huber_loss:
            loss = huber_loss(td_error, self.args.huber_delta)
            clipped_loss = huber_loss(clipped_td_error, self.args.huber_delta)
        else:
            loss = (td_error ** 2) / 2.0
            clipped_loss = (clipped_td_error ** 2) / 2.0

        if self.args.clip_value_loss:
            loss = th.max(loss, clipped_loss)
        
        loss = (loss * mask).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        (loss * 0.5).backward()
        
        # تضمین قطع گرادیان روی پارامترهای اختصاصی کریتیک
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        critic_train_stats["critic_loss"].append(loss.item())
        critic_train_stats["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        critic_train_stats["td_error_abs"].append(((td_error * mask).abs().sum().item() / mask_elems))
        critic_train_stats["curr_taken_v"].append((curr_values * mask).sum().item() / mask_elems)
        critic_train_stats["target_mean"].append((target_returns * mask).sum().item() / mask_elems)

        return critic_train_stats

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def gae_target(self, rewards, mask, values):
        # Returns is computed as the GAE +V_pi(s_t), which is equivalent to the TD lambda return
        # V_pi(s_t) = Q_pi(s_t, pi(s_t)) = A^GAE_pi(s_t, pi(s_t)) + V_pi(s_t)
        returns = []
        T = rewards.shape[1]
        gae = 0
        for step in reversed(range(T)):
            delta = rewards[:, step] + self.args.gamma * values[:, step + 1] * mask[:, step + 1] - \
                    values[:, step]
            gae = delta + self.args.gamma * self.args.gae_lambda * mask[:, step + 1] * gae
            returns.append((gae + values[:, step]).unsqueeze(1)) # list of len 52
        returns = th.flip(th.cat(returns, axis=1), dims=[1])
        return returns[:, :T]

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        if self.encoder is not None:
            self.encoder.cuda()
        if self.decoder is not None:
            self.decoder.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), f"{path}/critic.th")
        if self.encoder is not None:
            th.save(self.encoder.state_dict(), f"{path}/encoder.th")
        if self.decoder is not None:
            th.save(self.decoder.state_dict(), f"{path}/decoder.th")

        th.save(self.agent_optimiser.state_dict(), f"{path}/agent_opt.th")
        th.save(self.critic_optimiser.state_dict(), f"{path}/critic_opt.th")
        if self.encoder_decoder_optimiser is not None:
            th.save(self.encoder_decoder_optimiser.state_dict(), f"{path}/encoder_decoder_opt.th")

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load(f"{path}/critic.th", map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(th.load(f"{path}/agent_opt.th", map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load(f"{path}/critic_opt.th", map_location=lambda storage, loc: storage))

        if self.encoder is not None and os.path.exists(f"{path}/encoder.th"):
            self.encoder.load_state_dict(th.load(f"{path}/encoder.th", map_location=lambda storage, loc: storage))
        if self.decoder is not None and os.path.exists(f"{path}/decoder.th"):
            self.decoder.load_state_dict(th.load(f"{path}/decoder.th", map_location=lambda storage, loc: storage))
        if self.encoder_decoder_optimiser is not None and os.path.exists(f"{path}/encoder_decoder_opt.th"):
            self.encoder_decoder_optimiser.load_state_dict(th.load(f"{path}/encoder_decoder_opt.th", map_location=lambda storage, loc: storage))
            
            
def check_encoder_gradient_isolation(learner):
    """
    تست صحت قطع کامل گرادیان از Critic و Actor به Encoder
    """
    agent_param_ids = {id(p) for p in learner.agent_params}
    critic_param_ids = {id(p) for p in learner.critic_params}
    encoder_param_ids = {id(p) for p in learner.encoder.parameters()}
    
    overlap_critic = critic_param_ids.intersection(encoder_param_ids)
    overlap_actor = agent_param_ids.intersection(encoder_param_ids)
    
    assert len(overlap_critic) == 0, f"[ERROR] {len(overlap_critic)} Encoder parameters found in Critic Optimizer!"
    assert len(overlap_actor) == 0, f"[ERROR] {len(overlap_actor)} Encoder parameters found in Actor Optimizer!"

    for name, param in learner.encoder.named_parameters():
        if param.grad is not None:
            assert not th.isnan(param.grad).any(), f"[ERROR] NaN gradient detected in Encoder layer: {name}"
            assert not th.isinf(param.grad).any(), f"[ERROR] Inf gradient detected in Encoder layer: {name}"

    print("[SUCCESS] Gradient isolation test passed: Neither Actor nor Critic updates Encoder parameters directly.")