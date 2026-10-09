import json
import torch as th
from types import SimpleNamespace as SN
from components.action_selectors import REGISTRY as action_REGISTRY
from utils.load_utils import find_model_path

from modules.agent_loaders.base_agent_loader import BaseAgentLoader
from modules.agents import *

class MATNAHTEvalAgentLoader(BaseAgentLoader):
    '''
    کلاس اختصاصی جهت بارگذاری و ارزیابی (Eval) عامل‌های MAT-NAHT در محیط‌های Open-Train
    '''
    def __init__(self, 
                 args, scheme, 
                 model_path, 
                 load_step, load_agent_idx,
                 test_mode=True
                ):
        config_path = f"{model_path.replace('models', 'sacred')}/1/config.json"
        self.saved_args = SN(**json.load(open(config_path, "r")))
        
        super().__init__(args, scheme, 
                         n_agents=args.n_agents,
                         obs_last_action=getattr(self.saved_args, "obs_last_action", True), 
                         obs_agent_id=getattr(self.saved_args, "obs_agent_id", True), 
                        )
        
        self.saved_args.n_agents = args.n_agents
        self.n_actions = self.saved_args.n_actions = args.n_actions
        self.agent_output_type = self.saved_args.agent_output_type
        self.action_selector = action_REGISTRY[self.saved_args.action_selector](self.saved_args)
        
        input_shape = self._get_input_shape(scheme, self.saved_args)
        assert load_step in ['best', 'last'] or load_step.isdigit(), "load_step must be 'best', 'last', or a digit"
        
        self.load(input_shape, model_path, load_step=load_step, load_agent_idx=load_agent_idx)

        self.batch_size_run = self.args.batch_size_run
        self.device = self.args.device
        try:
            self.policy.cuda(self.device)
        except Exception as e:
            print("Device configuration note:", e)
            
        self.test_mode = test_mode

    def predict(self, ep_batch, agent_idx, t_ep, t_env, bs, test_mode=None):
        if test_mode is None:
            test_mode = self.test_mode

        ts = slice(None) if t_ep is None else slice(t_ep, t_ep + 1)
        ret_t = ep_batch.max_seq_length if t_ep is None else 1
        agent_slice = slice(agent_idx, agent_idx + 1)
        ep_batch_sliced = ep_batch[bs]
        avail_actions = ep_batch["avail_actions"][bs, ts, agent_slice]

        if self.saved_args.agent == "mat_naht_agent":
            if t_ep is None:
                h_in = self.policy.init_hidden(ep_batch.batch_size)
                q_outs, h_out = self.policy(ep_batch, t=None, hidden_state=h_in)
                agent_outs = q_outs[bs][:, :, agent_slice]
            else:
                if t_ep == 0:
                    self.hidden_state = self.policy.init_hidden(ep_batch.batch_size)

                q_outs, h_out = self.policy(
                    ep_batch, t=t_ep, hidden_state=self.hidden_state
                )
                self.hidden_state = h_out
                agent_outs = q_outs[bs, :][:, :, agent_slice]

        else:
            inputs = self._build_inputs(ep_batch_sliced, t=t_ep)
            inputs = inputs[:, :, agent_slice]

            if t_ep is None:
                h_in = self.policy.init_hidden(ep_batch.batch_size)
                q_outs, h_out = self.policy(inputs, h_in)
                agent_outs = q_outs
            else:
                if t_ep == 0:
                    self.hidden_state = self.policy.init_hidden(
                        ep_batch_sliced.batch_size
                    )

                q_outs, h_out = self.policy(
                    inputs, self.hidden_state[:, :, agent_slice]
                )
                self.hidden_state[:, :, agent_slice] = h_out
                agent_outs = q_outs

        ret_shape = (ep_batch_sliced.batch_size, ret_t, 1, -1)
        agent_outs = agent_outs.reshape(*ret_shape)

        if self.agent_output_type == "pi_logits":
            if getattr(self.saved_args, "mask_before_softmax", True):
                agent_outs[avail_actions == 0] = -1e10

        chosen_actions = self.action_selector.select_action(
            agent_outs, avail_actions, t_env, test_mode=test_mode
        )
        chosen_actions = chosen_actions.reshape(*ret_shape)

        hidden_state = th.zeros(
            ep_batch_sliced.batch_size, 
            ret_t, 
            1, 
            self.args.hidden_dim, 
            device=self.args.device
        )

        return agent_outs, chosen_actions, hidden_state

    def load(self, input_shape, path, load_step, load_agent_idx):
        model_path, _ = find_model_path(path, load_step=load_step, logger=None)

        # هندل کردن ساختار MATNAHTAgent
        if self.saved_args.agent == "mat_naht_agent":
            self.use_param_sharing = True
            self.policy = MATNAHTAgent(input_shape, self.saved_args)
            self.policy.load_state_dict(
                th.load(
                    f"{model_path}/agent.th",
                    map_location=lambda storage, loc: storage,
                )
            )

        elif self.saved_args.agent == "rnn":
            self.use_param_sharing = True
            self.policy = RNNAgent(input_shape, self.saved_args)
            self.policy.load_state_dict(
                th.load(
                    f"{model_path}/agent.th",
                    map_location=lambda storage, loc: storage,
                )
            )

        # ۲. اضافه کردن پشتیبانی از rnn_norm
        elif self.saved_args.agent == "rnn_norm":
            self.use_param_sharing = True
            self.policy = RNNNormAgent(input_shape, self.saved_args)
            self.policy.load_state_dict(
                th.load(
                    f"{model_path}/agent.th",
                    map_location=lambda storage, loc: storage,
                )
            )

        else:
            raise NotImplementedError(
                f"Agent type {self.saved_args.agent} not supported in"
                " MATNAHTEvalAgentLoader"
            )