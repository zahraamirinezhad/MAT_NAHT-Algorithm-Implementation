import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MaskedMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, attn_mask=None):
        bs = q.size(0)

        Q = self.q_linear(q).view(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(k).view(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(v).view(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            
            fill_value = -1e9 if scores.dtype == torch.float32 else -1e4
            scores = scores.masked_fill(attn_mask == 0, fill_value)

        attn_weights = F.softmax(scores, dim=-1)
        
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(context)
        return output

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attn = MaskedMultiHeadAttention(d_model, n_heads)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.self_attn = MaskedMultiHeadAttention(d_model, n_heads)
        self.cross_attn = MaskedMultiHeadAttention(d_model, n_heads)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, self_mask=None, cross_mask=None):
        attn_out1 = self.self_attn(x, x, x, attn_mask=self_mask)
        x = self.norm1(x + self.dropout(attn_out1))

        attn_out2 = self.cross_attn(x, memory, memory, attn_mask=cross_mask)
        x = self.norm2(x + self.dropout(attn_out2))

        ff_out = self.linear2(F.relu(self.linear1(x)))
        x = self.norm3(x + self.dropout(ff_out))
        return x

class MATEncoder(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, n_layers=2, n_heads=4, history_len=5):
        super(MATEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.history_len = history_len

        self.feat_embed = nn.Linear(obs_dim + act_dim, hidden_dim)
        nn.init.orthogonal_(self.feat_embed.weight, gain=1.0)
        nn.init.constant_(self.feat_embed.bias, 0.0)

        self.pos_embed = nn.Parameter(torch.zeros(1, 1, history_len, hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.01)
        
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, n_heads, hidden_dim * 2)
            for _ in range(n_layers)
        ])

    def forward(self, history_obs, history_act, comm_mask=None):
        bs, n_agents, seq_len, _ = history_obs.shape

        x = torch.cat([history_obs, history_act], dim=-1) # [bs, n_agents, seq_len, total_dim]
        
        if x.shape[-1] != self.feat_embed.in_features:
            self.feat_embed = nn.Linear(x.shape[-1], self.hidden_dim).to(x.device)
            nn.init.orthogonal_(self.feat_embed.weight, gain=1.0)
            nn.init.constant_(self.feat_embed.bias, 0.0)

        x = self.feat_embed(x) # [bs, n_agents, seq_len, hidden_dim]
        x = x + self.pos_embed[:, :, :seq_len, :]

        x = x.view(bs, n_agents * seq_len, self.hidden_dim)

        full_mask = None
        if comm_mask is not None:
            full_mask = comm_mask.unsqueeze(2).unsqueeze(4).repeat(1, 1, seq_len, 1, seq_len)
            full_mask = full_mask.permute(0, 1, 3, 2, 4).reshape(bs, n_agents * seq_len, n_agents * seq_len)
            full_mask = full_mask.unsqueeze(1)

        for layer in self.layers:
            x = layer(x, mask=full_mask)

        x = x.view(bs, n_agents, seq_len, self.hidden_dim)
        history_rep = x[:, :, -1, :]

        return history_rep

class MATDecoder(nn.Module):
    def __init__(self, act_dim, hidden_dim, n_layers=2, n_heads=4, dropout=0.1):
        super(MATDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim

        self.act_embed = nn.Linear(act_dim, hidden_dim)
        nn.init.orthogonal_(self.act_embed.weight, gain=0.01)
        nn.init.constant_(self.act_embed.bias, 0.0)
        
        self.embed_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.start_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.start_token, std=0.01)

        self.layers = nn.ModuleList([
            TransformerDecoderBlock(hidden_dim, n_heads, hidden_dim * 2, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.action_head = nn.Linear(hidden_dim, act_dim)
        nn.init.orthogonal_(self.action_head.weight, gain=0.001)
        nn.init.constant_(self.action_head.bias, 0.0)

    def forward(self, history_rep, prev_actions=None, comm_mask=None):
        bs, n_agents, _ = history_rep.shape

        if prev_actions is None:
            decoder_input = self.start_token.expand(bs, n_agents, -1)
        else:
            if prev_actions.dim() == 2:
                prev_actions = F.one_hot(prev_actions.long(), num_classes=self.act_dim).float()
            
            act_emb = self.act_embed(prev_actions) * 0.01
            shifted_acts = torch.cat([self.start_token.expand(bs, 1, -1), act_emb[:, :-1, :]], dim=1)
            decoder_input = self.dropout(self.embed_norm(shifted_acts))

        causal_mask = torch.tril(torch.ones((n_agents, n_agents), device=history_rep.device)).unsqueeze(0).expand(bs, -1, -1)

        if comm_mask is not None:
            eye = torch.eye(n_agents, device=history_rep.device).unsqueeze(0)
            valid_comm_mask = (comm_mask.bool() | eye.bool()).float()

            decoder_self_mask = (causal_mask.bool() & valid_comm_mask.bool()).float().unsqueeze(1)
            cross_mask = valid_comm_mask.unsqueeze(1)
        else:
            decoder_self_mask = causal_mask.unsqueeze(1)
            cross_mask = None

        x = decoder_input
        for layer in self.layers:
            x = layer(x, memory=history_rep, self_mask=decoder_self_mask, cross_mask=cross_mask)

        raw_logits = self.action_head(x)
        
        action_logits = raw_logits / 6.0
        
        return action_logits

class MATCentralizedValue(nn.Module):
    def __init__(self, hidden_dim):
        super(MATCentralizedValue, self).__init__()
        self.v_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        nn.init.orthogonal_(self.v_head[-1].weight, gain=1.0)
        nn.init.constant_(self.v_head[-1].bias, 0.0)

    def forward(self, history_rep, comm_mask=None):
        bs, n_agents, hidden_dim = history_rep.shape

        if comm_mask is not None:
            mask_expanded = comm_mask.unsqueeze(-1) # [bs, n_agents, n_agents, 1]
            rep_expanded = history_rep.unsqueeze(1).expand(-1, n_agents, -1, -1) # [bs, n_agents, n_agents, hidden_dim]
            
            masked_reps = rep_expanded * mask_expanded
            neighbor_counts = comm_mask.sum(dim=-1, keepdim=True).clamp(min=1.0) # [bs, n_agents, 1]
            local_context = masked_reps.sum(dim=2) / neighbor_counts # [bs, n_agents, hidden_dim]
            
            v_values = self.v_head(local_context + history_rep).squeeze(-1) # [bs, n_agents]
        else:
            v_values = self.v_head(history_rep).squeeze(-1) # [bs, n_agents]

        return v_values
    
class MAT_NAHT(nn.Module):
    def __init__(self, args, obs_dim, act_dim, encoder=None, decoder=None):
        super(MAT_NAHT, self).__init__()
        self.args = args
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = getattr(args, "hidden_dim", 64)
        self.history_len = getattr(args, "history_len", 5)
        self.n_heads = getattr(args, "n_heads", 4)
        self.n_layers = getattr(args, "n_layers", 2)
        self.dropout = getattr(args, "dropout", 0.1)

        self.encoder = encoder if encoder is not None else MATEncoder(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            history_len=self.history_len
        )

        self.decoder = decoder if decoder is not None else MATDecoder(
            act_dim=act_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout
        )

        self.value_head = MATCentralizedValue(hidden_dim=self.hidden_dim)

    def forward(self, history_obs, history_act, comm_mask=None, avail_actions=None):
        history_rep = self.encoder(history_obs, history_act, comm_mask=comm_mask)
        values = self.value_head(history_rep, comm_mask=comm_mask)
        action_logits = self.decoder(history_rep, prev_actions=None, comm_mask=comm_mask)

        if avail_actions is not None:
            fill_value = -1e9 if action_logits.dtype == torch.float32 else -1e4
            action_logits = action_logits.masked_fill(avail_actions == 0, fill_value)

        return action_logits, values

    def sample_actions_autoregressively(self, history_obs, history_act, comm_mask=None, avail_actions=None, deterministic=False):
        self.eval()
        with torch.no_grad():
            bs, n_controlled, _, _ = history_obs.shape
            device = history_obs.device

            history_rep = self.encoder(history_obs, history_act, comm_mask=comm_mask)

            sampled_actions = torch.zeros(bs, n_controlled, self.act_dim, device=device)
            action_indices = torch.zeros(bs, n_controlled, dtype=torch.long, device=device)

            for agent_idx in range(n_controlled):
                logits = self.decoder(history_rep, prev_actions=sampled_actions, comm_mask=comm_mask)
                current_agent_logits = logits[:, agent_idx, :]

                if avail_actions is not None:
                    current_agent_logits = current_agent_logits.masked_fill(avail_actions[:, agent_idx] == 0, -1e4)

                probs = F.softmax(current_agent_logits, dim=-1)

                if deterministic:
                    act_idx = torch.argmax(probs, dim=-1)
                else:
                    act_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)

                action_indices[:, agent_idx] = act_idx
                sampled_actions[:, agent_idx] = F.one_hot(act_idx, num_classes=self.act_dim).float()

        return action_indices, sampled_actions
    
    def evaluate_actions(self, history_obs, history_act, actions, comm_mask=None, avail_actions=None):
        bs, n_controlled, _, _ = history_obs.shape

        with torch.no_grad():
            history_rep = self.encoder(history_obs, history_act, comm_mask=comm_mask)
        history_rep = history_rep.detach()

        values = self.value_head(history_rep, comm_mask=comm_mask)

        if actions.dim() == 2:
            actions_onehot = F.one_hot(actions.long(), num_classes=self.act_dim).float()
            action_targets = actions.long()
        else:
            actions_onehot = actions
            action_targets = torch.argmax(actions, dim=-1)

        actions_onehot = actions_onehot.detach()
        
        action_logits = self.decoder(history_rep, prev_actions=actions_onehot, comm_mask=comm_mask)

        # ========================== DEBUG PRINT BLOCK ==========================
        with torch.no_grad():
            print("\n" + "="*50)
            print("[DEBUG MAT_NAHT DIAGNOSTICS]")
            print(f"Logits Mean: {action_logits.mean().item():.4f} | Std: {action_logits.std().item():.4f} | Min: {action_logits.min().item():.4f} | Max: {action_logits.max().item():.4f}")
            
            if avail_actions is not None:
                invalid_cnt = (avail_actions == 0).sum().item()
                total_cnt = avail_actions.numel()
                print(f"Avail Actions Masked Ratio: {invalid_cnt / total_cnt * 100:.2f}% ({invalid_cnt}/{total_cnt})")
                
            probs_pre_mask = F.softmax(action_logits, dim=-1)
            entropy_pre_mask = -(probs_pre_mask * torch.log(probs_pre_mask + 1e-8)).sum(dim=-1).mean().item()
            print(f"Entropy PRE-masking: {entropy_pre_mask:.4f}")
        # =======================================================================

        if avail_actions is not None:
            action_logits = action_logits.masked_fill(avail_actions == 0, -1e4)

        scaled_logits = action_logits / 1.5

        log_probs_all = F.log_softmax(scaled_logits, dim=-1)
        probs_all = F.softmax(scaled_logits, dim=-1)

        action_log_probs = log_probs_all.gather(-1, action_targets.unsqueeze(-1)).squeeze(-1)
        dist_entropy = -(probs_all * log_probs_all).sum(dim=-1)
        
        # ========================== DEBUG PRINT BLOCK ==========================
        with torch.no_grad():
            print(f"Entropy POST-masking: {dist_entropy.mean().item():.4f}")
            print("="*50 + "\n")
        # =======================================================================
        
        return action_log_probs, values, dist_entropy
    
def build_communication_mask(obs, n_agents, neighbor_mask=None, availability_prob=0.7, distance_threshold=None, symmetric=False, occlude_prob=0.0, t=None):
    if neighbor_mask is not None:
        return neighbor_mask

    if obs.dim() == 3: #[bs, n_agents, obs_dim]
        obs_seq = obs.unsqueeze(1)
    else:
        obs_seq = obs

    vis_mask = extract_visibility_mask(
        obs=obs_seq,
        n_agents=n_agents,
        availability_prob=availability_prob,
        distance_threshold=distance_threshold,
        symmetric=symmetric,
        occlude_prob=occlude_prob
    )

    if obs.dim() == 3:
        return vis_mask.squeeze(1) # [bs, n_agents, n_agents]
    
    return vis_mask # [bs, max_t, n_agents, n_agents]


def prepare_history_buffer(batch, t, history_len, obs_dim, act_dim):
    bs, seq_total, n_agents, _ = batch["obs"].shape
    device = batch["obs"].device

    start_t = max(0, t - history_len + 1)
    current_len = t - start_t + 1
    pad_len = history_len - current_len

    obs_slice = batch["obs"][:, start_t:t+1].transpose(1, 2) # [bs, n_agents, current_len, real_obs_dim]
    act_slice = batch["actions_onehot"][:, start_t:t+1].transpose(1, 2) # [bs, n_agents, current_len, real_act_dim]

    real_obs_dim = obs_slice.shape[-1]
    real_act_dim = act_slice.shape[-1]

    if pad_len > 0:
        obs_hist = torch.zeros(bs, n_agents, history_len, real_obs_dim, device=device, dtype=obs_slice.dtype)
        act_hist = torch.zeros(bs, n_agents, history_len, real_act_dim, device=device, dtype=act_slice.dtype)
        
        obs_hist[:, :, pad_len:] = obs_slice
        act_hist[:, :, pad_len:] = act_slice
    else:
        obs_hist = obs_slice
        act_hist = act_slice

    return obs_hist, act_hist

def extract_visibility_mask(obs, n_agents, availability_prob=0.7, distance_threshold=None, symmetric=False, occlude_prob=0.0, verbose=False, t=None, **kwargs):
    bs, max_t = obs.shape[0], obs.shape[1]
    
    base_mask = torch.ones(bs, max_t, n_agents, n_agents, device=obs.device)

    if distance_threshold is not None and obs.shape[-1] >= 2:
        pos = obs[..., :2]
        pos_i = pos.unsqueeze(-2)
        pos_j = pos.unsqueeze(-3)
        dist = torch.norm(pos_i - pos_j, dim=-1)
        spatial_mask = (dist <= distance_threshold).float()
        base_mask = base_mask * spatial_mask

    rand_draw = torch.rand(bs, max_t, n_agents, n_agents, device=obs.device)
    random_gate = (rand_draw < availability_prob).float()
    
    if symmetric:
        random_gate = random_gate * random_gate.transpose(-1, -2)

    eye = torch.eye(n_agents, device=obs.device).unsqueeze(0).unsqueeze(0)
    random_gate = random_gate * (1.0 - eye) + eye

    visibility_mask = base_mask * random_gate

    if occlude_prob > 0.0:
        occlusion_draw = (torch.rand(bs, max_t, n_agents, n_agents, device=obs.device) >= occlude_prob).float()
        occlusion_draw = occlusion_draw * (1.0 - eye) + eye
        visibility_mask = visibility_mask * occlusion_draw

    return visibility_mask