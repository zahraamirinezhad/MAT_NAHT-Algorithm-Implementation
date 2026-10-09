import torch as th
import torch.nn as nn
import torch.nn.functional as F
from components.mat_naht.mat_naht import MAT_NAHT, build_communication_mask, prepare_history_buffer

class MATNAHTAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MATNAHTAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.obs_dim = args.obs_shape if hasattr(args, "obs_shape") else input_shape
        self.act_dim = args.n_actions
        self.hidden_dim = getattr(args, "hidden_dim", 128)
        self.history_len = getattr(args, "history_len", 5)

        self.core = MAT_NAHT(
            args=args,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim
        )

    def forward(self, ep_batch, t=None, hidden_state=None, controlled_indices=None):
        bs = ep_batch.batch_size
        max_t = ep_batch.max_seq_length
        device = ep_batch.device
        
        obs = ep_batch["obs"] if t is None else ep_batch["obs"][:, t]
        neighbor_mask = ep_batch["neighbor_mask"] if "neighbor_mask" in ep_batch.scheme else None
        comm_mask = build_communication_mask(
            obs=obs,
            n_agents=self.n_agents,
            neighbor_mask=neighbor_mask
        )
        
        if t is not None:
            obs_hist, act_hist = prepare_history_buffer(
                batch=ep_batch,
                t=t,
                history_len=self.history_len,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim
            )
            if controlled_indices is not None:
                ctrl_mask = th.zeros(bs, self.n_agents, 1, 1, device=device)
                ctrl_mask[:, controlled_indices] = 1.0
                obs_hist = obs_hist * ctrl_mask
                act_hist = act_hist * ctrl_mask
            
            # نمونه‌برداری گام‌به‌گام اتورگرسیو در زمان Rollout برای حفظ یکنواختی توزیع
            with th.no_grad():
                history_rep = self.core.encoder(obs_hist, act_hist, comm_mask=comm_mask)
                sampled_actions = th.zeros(bs, self.n_agents, self.act_dim, device=device)
                logits_collector = th.zeros(bs, self.n_agents, self.act_dim, device=device)

                for agent_idx in range(self.n_agents):
                    agent_logits = self.core.decoder(history_rep, prev_actions=sampled_actions, comm_mask=comm_mask)
                    curr_logits = agent_logits[:, agent_idx, :]
                    logits_collector[:, agent_idx, :] = curr_logits
                    
                    probs = F.softmax(curr_logits, dim=-1)
                    act_idx = th.multinomial(probs, num_samples=1).squeeze(-1)
                    sampled_actions[:, agent_idx] = F.one_hot(act_idx, num_classes=self.act_dim).float()

            q_out = logits_collector.unsqueeze(1)  # [bs, 1, n_agents, act_dim]
            h_out = th.zeros(bs, 1, self.n_agents, self.hidden_dim, device=device)

        else:
            obs_all = ep_batch["obs"]
            act_all = ep_batch["actions_onehot"]
            last_act_all = th.cat([
                th.zeros_like(act_all[:, [0]]),
                act_all[:, :-1]
            ], dim=1)
            if controlled_indices is not None:
                ctrl_mask = th.zeros(bs, 1, self.n_agents, 1, device=device)
                ctrl_mask[:, :, controlled_indices] = 1.0
                obs_all = obs_all * ctrl_mask
                last_act_all = last_act_all * ctrl_mask
                
            cur_obs_dim = obs_all.shape[-1]
            cur_act_dim = last_act_all.shape[-1]
            
            padded_obs = th.cat([th.zeros(bs, self.history_len - 1, self.n_agents, cur_obs_dim, device=device), obs_all], dim=1)
            padded_act = th.cat([th.zeros(bs, self.history_len - 1, self.n_agents, cur_act_dim, device=device), last_act_all], dim=1)
            hist_obs_list, hist_act_list = [], []
            for step in range(max_t):
                h_o = padded_obs[:, step:step + self.history_len].transpose(1, 2)
                h_a = padded_act[:, step:step + self.history_len].transpose(1, 2)
                hist_obs_list.append(h_o)
                hist_act_list.append(h_a)
            batch_obs_hist = th.stack(hist_obs_list, dim=1).view(bs * max_t, self.n_agents, self.history_len, cur_obs_dim)
            batch_act_hist = th.stack(hist_act_list, dim=1).view(bs * max_t, self.n_agents, self.history_len, cur_act_dim)
            if comm_mask is not None and comm_mask.dim() == 3:
                comm_mask = comm_mask.unsqueeze(1).expand(-1, max_t, -1, -1).reshape(-1, self.n_agents, self.n_agents)
            flat_prev_actions = act_all.view(bs * max_t, self.n_agents, self.act_dim)
            history_rep = self.core.encoder(batch_obs_hist, batch_act_hist, comm_mask=comm_mask)
            action_logits = self.core.decoder(history_rep, prev_actions=flat_prev_actions, comm_mask=comm_mask)
            q_out = action_logits.view(bs, max_t, self.n_agents, self.act_dim)
            h_out = th.zeros(bs, max_t, self.n_agents, self.hidden_dim, device=device)

        return q_out, h_out
    
    
    def init_hidden(self, batch_size):
        return th.zeros(batch_size, 1, self.n_agents, self.hidden_dim, device=self.args.device)

    def get_out_shapes(self, curr_t, batch_size, max_t):
        if curr_t is None:
            ts = slice(None)
            out_batch_dim = (batch_size, max_t, self.n_agents)
            hidden_batch_dim = (batch_size, max_t, self.n_agents, self.hidden_dim)
        else: 
            ts = slice(curr_t, curr_t + 1)
            out_batch_dim = (batch_size, 1, self.n_agents)
            hidden_batch_dim = (batch_size, 1, self.n_agents, self.hidden_dim)
        return ts, out_batch_dim, hidden_batch_dim, hidden_batch_dim

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        inputs.append(batch["obs"][:, ts])
        if self.args.obs_last_action:
            if t is None:
                last_act_all = th.cat([
                    th.zeros_like(batch["actions_onehot"][:, [0]]), 
                    batch["actions_onehot"][:, slice(0, max_t-1)]
                ], dim=1)
                inputs.append(last_act_all) 
            else:
                last_act = th.zeros_like(batch["actions_onehot"][:, ts]) if t == 0 else batch["actions_onehot"][:, slice(t-1, t)]
                inputs.append(last_act)

        if self.args.obs_agent_id:
            agent_id_onehot = th.eye(self.n_agents, device=batch.device)
            agent_id_onehot = agent_id_onehot.expand(bs, max_t, self.n_agents, -1) if t is None \
                else agent_id_onehot.expand(bs, 1, self.n_agents, -1)
            inputs.append(agent_id_onehot)
        inputs = th.cat(inputs, dim=-1)
        return inputs
    
    def set_modules(self, encoder=None, decoder=None, **kwargs):
        if encoder is not None:
            self.core.encoder = encoder
        if decoder is not None:
            self.core.decoder = decoder

    def load(self, path):
        state_dict_all = th.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict_all)
