import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from crowd_nav.policy.graph_model import RGL,GAT_RL

def build_transition_fn(name, config, nonlin=nn.Tanh(), kernel_size=None):
    embedding_dim = config.gcn.X_dim
    num_actions = config.action_space.speed_samples * config.action_space.rotation_samples + 1
    transition_fun1 = nn.Linear(embedding_dim, embedding_dim)
    transition_fun2 = Parameter(torch.Tensor(embedding_dim, embedding_dim, num_actions).type(dtype))
    return transition_fun1, nn.init.xavier_normal(transition_fun2)

class MLPRewardFn(nn.Module):
    def __init__(self, config):
        super(MLPRewardFn, self).__init__()
        embedding_dim = config.gcn.X_dim
        num_actions = config.action_space.speed_samples * config.action_space.rotation_samples + 1
        self.mlp = nn.Sequential(
            nn_init(nn.Linear(embed_dim, 64), w_scale=np.sqrt(2)),
            nn.ReLU(inplace=True),
            nn_init(nn.Linear(64, num_actions), w_scale=0.01)
        )

    def forward(self, x):
        x = x.view(-1, self.embedding_dim)
        return self.mlp(x).view(-1, self.num_actions)

class TreeQNNetwork(nn.Module):
    def __init__(self,
                 config,
                 use_actor_critic=False,
                 transition_nonlin="tanh",
                 predict_rewards=True,
                 gamma=0.99,
                 td_lambda=0.8,
                 value_aggregation="softmax",
                 output_tree=False):
        super(TreeQNNetwork, self).__init__()
        self.num_actions =config.action_space.speed_samples * config.action_space.rotation_samples + 1
        self.embedding_dim = config.gcn.X_dim
        self.use_actor_critic = use_actor_critic
        self.obs_scale = 1.0
        self.eps_threshold = 0
        self.predict_rewards = predict_rewards
        self.gamma = gamma
        self.output_tree = output_tree
        self.td_lambda = td_lambda
        self.value_aggregation = value_aggregation
        self.tree_depth = config.model_predictive_rl.planning_depth

        self.graph_model = GAT_RL(config, self.robot_state_dim, self.human_state_dim)

        # construct the value function
         self.value_fn = nn_init(nn.Linear(config.gcn.X_dim, 1), w_scale=.01)

        # construct the transition function
        transition_fun_name = "relu"
        self.transition_fun_name = transition_fun_name
        if transition_nonlin == "tanh":
            self.transition_nonlin = nn.Tanh()
        elif transition_nonlin == "relu":
            self.transition_nonlin = nn.ReLU()
        else:
            raise ValueError
        self.transition_fun = build_transition_fn(transition_fun_name, config, nonlin=self.transition_nonlin)

        # construct the reward function
        self.tree_reward_fun = MLPRewardFn(config)

    def forward(self, ob, volatile=False):
        """
        :param ob: [batch_size x nodes x feature_dims]
        :return: [batch_size x num_actions], -- Q-values
                 [batch_size x 1], -- V = max_a(Q)
                 [batch_size x num_actions x embedding_dim], -- embeddings after first transition
                 [batch_size x num_actions] -- rewards after first transition
        """

        st = self.graph_model(ob, volatile=volatile)

        Q, tree_result = self.planning(st)

        if self.use_actor_critic:
            V = self.ac_value_fn(st).squeeze()
        else:
            V = torch.max(Q, 1)[0]

        return Q, V, tree_result
    # eps_greedy exploration
    def step(self, ob):
        Q, V, _ = self.forward(ob, volatile=True)
        a = self.sample(Q)
        return a, V

    def value(self, ob):
        _, V, _ = self.forward(ob, volatile=True)
        return V

    def sample(self, Q):
        if self.use_actor_critic:
            pi = F.softmax(Q, dim=-1)
            a = torch.multinomial(pi, 1).squeeze()
            return a.data.cpu().numpy()
        else:
        self.eps_threshold = 0.1
            sample = random.random()
            if sample > self.eps_threshold:
                # return the index of the optimal action
                return Q.data.max(1)[1].cpu().numpy()
            else:
                return np.random.randint(0, self.num_actions)

    def tree_planning(self, x, return_intermediate_values=True):
        """
        :param x: [batch_size x embedding_dim]
        :return:
            dict tree_result:
            - "embeddings":
                list of length tree_depth, [batch_size * num_actions^depth x embedding_dim] state
                representations after tree planning
            - "values":
                list of length tree_depth, [batch_size * num_actions^depth x 1] values predicted
                from each embedding
            - "rewards":
                list of length tree_depth, [batch_size * num_actions^depth x 1] rewards predicted
                from each transition
        """

        tree_result = {
            "embeddings": [x],
            "values": []
        }
        if self.predict_rewards:
            tree_result["rewards"] = []

        if return_intermediate_values:
            tree_result["values"].append(self.value_fn(x))

        for i in range(self.tree_depth):
            if self.predict_rewards:
                r = self.tree_reward_fun(x)
                tree_result["rewards"].append(r.view(-1, 1))

            x = self.tree_transitioning(x)

            x = x.view(-1, self.embedding_dim)

            tree_result["embeddings"].append(x)

            if return_intermediate_values or i == self.tree_depth - 1:
                tree_result["values"].append(self.value_fn(x))

        return tree_result

    def tree_transitioning(self, x):
        """
        :param x: [? x embedding_dim]
        :return: [? x num_actions x embedding_dim]
        """

        x1 = self.transition_nonlin(self.transition_fun1(x))
        x2 = x + x1
        x2 = x2.unsqueeze(1)
        x3 = self.transition_nonlin(np.einsum("ij,jab->iba", x, self.transition_fun2))
        x2 = x2.expand_as(x3)
        next_state = x2 + x3

#         if self.residual_transition:
#             next_state = x.unsqueeze(1).expand_as(next_state) + next_state
#
#         if self.normalise_state:
#             next_state = next_state / next_state.pow(2).sum(-1, keepdim=True).sqrt()

        return next_state

    def planning(self, x):
        """
        :param x: [batch_size x embedding_dim] state representations
        :return:
            - [batch_size x embedding_dim x num_actions] state-action values
            - [batch_size x num_actions x embedding_dim] state representations after planning one step
              used for regularizing/grounding the transition model
        """
        batch_size = x.size(0)
        if self.tree_depth > 0:
            tree_result = self.tree_planning(x)
        else:
            raise NotImplementedError

        q_values = self.tree_backup(tree_result, batch_size)

        return q_values, tree_result

    def tree_backup(self, tree_result, batch_size):
        backup_values = tree_result["values"][-1]
        for i in range(1, self.tree_depth + 1):
            one_step_backup = tree_result["rewards"][-i] + self.gamma*backup_values

            if i < self.tree_depth:
                one_step_backup = one_step_backup.view(batch_size, -1, self.num_actions)

                if self.value_aggregation == "max":
                    max_backup = one_step_backup.max(2)[0]
                elif self.value_aggregation == "logsumexp":
                    max_backup = logsumexp(one_step_backup, 2)
                elif self.value_aggregation == "softmax":
                    max_backup = (one_step_backup * F.softmax(one_step_backup, dim=2)).sum(dim=2)
                else:
                    raise ValueError("Unknown value aggregation function %s" % self.value_aggregation)

                backup_values = ((1 - self.td_lambda) * tree_result["values"][-i-1] +
                                 (self.td_lambda) * max_backup.view(-1, 1))
            else:
                backup_values = one_step_backup

        backup_values = backup_values.view(batch_size, self.num_actions)

        return backup_values