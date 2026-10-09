import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.popart import PopArt
from components.mat_naht.mat_naht import MATEncoder, MATCentralizedValue

class MATNAHTCritic(nn.Module):
    def __init__(self, scheme, args, shared_encoder=None):
        super(MATNAHTCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        
        self.hidden_dim = getattr(args, "hidden_dim", 128)
        self.history_len = getattr(args, "history_len", 5)
        self.n_heads = getattr(args, "n_heads", 4)
        self.n_layers = getattr(args, "n_layers", 2)
        
        self.output_type = "v"
        self.obs_dim = scheme["obs"]["vshape"]
        self.act_dim = args.n_actions

        self.encoder = shared_encoder

        self.v_head = MATCentralizedValue(hidden_dim=self.hidden_dim)

        use_popart = getattr(args, "use_popart", False)
        device = getattr(args, "device", "cpu")
        
        if use_popart:
            self.v_out = PopArt(self.hidden_dim, 1, norm_axes=3, device=device)
        else:
            self.v_out = nn.Linear(self.hidden_dim, 1, device=device)
        
        self.value_normalizer = self.v_out if use_popart else None

    def set_encoder(self, encoder):
        self.encoder = encoder

    def init_hidden(self):
        device = getattr(self.args, "device", "cpu")
        dummy_hist_obs = torch.zeros(self.args.batch_size, self.n_agents, self.history_len, self.obs_dim, device=device)
        dummy_hist_act = torch.zeros(self.args.batch_size, self.n_agents, self.history_len, self.act_dim, device=device)
        return dummy_hist_obs, dummy_hist_act

    def forward(self, batch, hidden_state=None, t=None, comm_mask=None, controlled_indices=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1

        # ۱. ساخت ورودی‌های تاریخچه
        history_obs, history_act = self._build_history_inputs(batch, t=t)

        # ۲. ماسک کردن اطلاعات عامل‌های غیرکنترل‌شده
        if controlled_indices is not None:
            ctrl_mask = torch.zeros(bs * max_t, self.n_agents, 1, 1, device=batch.device)
            ctrl_mask[:, controlled_indices] = 1.0
            history_obs_masked = history_obs * ctrl_mask
            history_act_masked = history_act * ctrl_mask
        else:
            history_obs_masked = history_obs
            history_act_masked = history_act

        # ۳. گسترش ماسک ارتباطات
        if comm_mask is not None and comm_mask.dim() == 3 and t is None:
            comm_mask_expanded = comm_mask.unsqueeze(1).expand(-1, max_t, -1, -1).reshape(-1, self.n_agents, self.n_agents)
        else:
            comm_mask_expanded = comm_mask

        # ۴. محاسبه خروجی انکودر با no_grad کامل و detach صریح برای قطع گرادیان به انکودر مشترک
        with torch.no_grad():
            history_rep = self.encoder(history_obs_masked.detach(), history_act_masked.detach(), comm_mask=comm_mask_expanded)
        
        history_rep = history_rep.detach().clone()

        # ۵. محاسبه Value صرفاً توسط v_head
        v_values = self.v_head(history_rep, comm_mask=comm_mask_expanded) # [bs * max_t, n_agents]
        values = v_values.unsqueeze(-1)  # shape: [bs * max_t, n_agents, 1]

        if t is None:
            values = values.view(bs, max_t, self.n_agents, 1)
        else:
            values = values.view(bs, 1, self.n_agents, 1)

        h_e = history_obs_masked[:, :, -self.history_len:]
        h = history_act_masked[:, :, -self.history_len:]

        return values, h_e, h

    def _build_history_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        device = batch.device

        obs_all = batch["obs"]
        
        # بررسی ایمن وجود actions_onehot در scheme
        if "actions_onehot" in batch.scheme:
            act_all = batch["actions_onehot"]
        elif "actions" in batch.scheme:
            # ساخت one-hot از روی اکشن‌های عددی در صورت عدم وجود actions_onehot
            actions_raw = batch["actions"]
            act_all = torch.nn.functional.one_hot(actions_raw.squeeze(-1).long(), num_classes=self.act_dim).float()
        else:
            act_all = torch.zeros(bs, batch.max_seq_length, self.n_agents, self.act_dim, device=device)

        last_act_all = torch.cat([
            torch.zeros_like(act_all[:, [0]]),
            act_all[:, :-1]
        ], dim=1)

        if t is not None:
            start_idx = max(0, t - self.history_len + 1)
            pad_len = self.history_len - (t - start_idx + 1)

            slice_obs = obs_all[:, start_idx:t+1]
            slice_act = last_act_all[:, start_idx:t+1]

            if pad_len > 0:
                pad_obs = torch.zeros(bs, pad_len, self.n_agents, self.obs_dim, device=device)
                pad_act = torch.zeros(bs, pad_len, self.n_agents, self.act_dim, device=device)
                slice_obs = torch.cat([pad_obs, slice_obs], dim=1)
                slice_act = torch.cat([pad_act, slice_act], dim=1)

            hist_obs = slice_obs.transpose(1, 2)
            hist_act = slice_act.transpose(1, 2)
        else:
            padded_obs = torch.cat([torch.zeros(bs, self.history_len - 1, self.n_agents, self.obs_dim, device=device), obs_all], dim=1)
            padded_act = torch.cat([torch.zeros(bs, self.history_len - 1, self.n_agents, self.act_dim, device=device), last_act_all], dim=1)

            hist_obs_list, hist_act_list = [], []
            for step in range(max_t):
                hist_o = padded_obs[:, step:step + self.history_len]
                hist_a = padded_act[:, step:step + self.history_len]
                hist_obs_list.append(hist_o.transpose(1, 2))
                hist_act_list.append(hist_a.transpose(1, 2))

            hist_obs = torch.stack(hist_obs_list, dim=1).view(bs * max_t, self.n_agents, self.history_len, self.obs_dim)
            hist_act = torch.stack(hist_act_list, dim=1).view(bs * max_t, self.n_agents, self.history_len, self.act_dim)

        return hist_obs, hist_act