import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY
from modules.agent_loaders.base_agent_loader import BaseAgentLoader

class MATNAHTTrainAgentLoader(BaseAgentLoader):
    def __init__(self, args, scheme, model_path):
        super().__init__(
            args, scheme,
            n_agents=args.n_agents,
            obs_last_action=args.obs_last_action,
            obs_agent_id=args.obs_agent_id,
        )
        
        input_shape = self._get_input_shape(scheme, args)
        self._build_agent(input_shape=input_shape)

        if model_path != "":
            self.policy.load(model_path)

        self.agent_output_type = self.args.agent_output_type
        self.action_selector = action_REGISTRY[self.args.action_selector](self.args)

    def predict(self, ep_batch, agent_idx_list, t_ep, t_env, bs, test_mode):
        ts = slice(None) if t_ep is None else slice(t_ep, t_ep + 1)
        ret_t = ep_batch.max_seq_length if t_ep is None else 1
        ep_batch_sliced = ep_batch[bs]
        avail_actions = ep_batch["avail_actions"][bs, ts][:, :, agent_idx_list]

        if t_ep is None:  # Forward pass during training
            h = self.policy.init_hidden(ep_batch.batch_size)
            q_outs, h_out = self.policy(ep_batch, t=None, hidden_state=h)
            agent_outs = q_outs[bs][:, :, agent_idx_list]
            self.hidden_state = h_out[bs]
        else:  # Forward pass during runner rollouts
            if t_ep == 0:
                self.hidden_state = self.policy.init_hidden(ep_batch.batch_size)
            
            q_outs, h_out = self.policy(ep_batch, t=t_ep, hidden_state=self.hidden_state)
            self.hidden_state = h_out
            agent_outs = q_outs[bs, :][:, :, agent_idx_list]

        hidden_state = th.zeros(
            ep_batch_sliced.batch_size, 
            ret_t, 
            len(agent_idx_list), 
            self.args.hidden_dim,
            device=self.args.device
        )

        ret_shape = (ep_batch_sliced.batch_size, ret_t, len(agent_idx_list), -1)
        agent_outs = agent_outs.reshape(*ret_shape)

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                agent_outs[avail_actions == 0] = -1e10

        chosen_actions = self.action_selector.select_action(
            agent_outs, avail_actions, t_env, test_mode=test_mode
        )
        chosen_actions = chosen_actions.reshape(*ret_shape)
        hidden_state = hidden_state.reshape(ep_batch_sliced.batch_size, ret_t, len(agent_idx_list), self.args.hidden_dim)

        return agent_outs, chosen_actions, hidden_state
    
    def set_modules____(
    self,
    encoder,
    history=None,
    history_embedding=None,
    history_positional_embedding=None,
    mask_builder=None):

        self.policy.set_modules(
            encoder=encoder,
            history=history,
            history_embedding=history_embedding,
            history_positional_embedding=history_positional_embedding,
            mask_builder=mask_builder,
        )
        
    def set_modules(
        self,
        encoder=None,
        decoder=None,
        history=None,
        history_embedding=None,
        history_positional_embedding=None,
        mask_builder=None):

        if hasattr(self.policy, "set_modules"):
            self.policy.set_modules(
                encoder=encoder,
                decoder=decoder,
                history=history,
                history_embedding=history_embedding,
                history_positional_embedding=history_positional_embedding,
                mask_builder=mask_builder,
            )

    def _build_agent(self, input_shape):
        '''Initialize agent'''
        self.policy = agent_REGISTRY[self.args.agent](input_shape, self.args)        