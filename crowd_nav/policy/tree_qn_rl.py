import logging
import torch
import numpy as np
from numpy.linalg import norm
import itertools
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import tensor_to_joint_state
from crowd_sim.envs.utils.info import  *
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_nav.policy.state_predictor import StatePredictor, LinearStatePredictor_batch
from crowd_nav.policy.graph_model import RGL,GAT_RL
from crowd_nav.policy.value_estimator import DQNNetwork, Noisy_DQNNetwork
from crowd_nav.policy.TreeQN import TreeQNNetwork

class TreeQNRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'TreeQNRL'
        self.trainable = True
        self.multiagent_training = True
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.action_space = None
        self.rotation_constraint = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.robot_state_dim = 9
        self.human_state_dim = 5
        self.v_pref = 1
        self.share_graph_model = None
        self.value_estimator = None
        self.linear_state_predictor = None
        self.state_predictor = None
        self.planning_depth = None
        self.planning_width = None
        self.do_action_clip = None
        self.sparse_search = None
        self.sparse_speed_samples = 2
        self.sparse_rotation_samples = 8
        self.action_group_index = []
        self.traj = None
        self.use_noisy_net = False
        self.count=0
        self.time_step = 0.25

    def configure(self, config, device):
        self.set_common_parameters(config)
        self.planning_depth = config.model_predictive_rl.planning_depth
        self.do_action_clip = config.model_predictive_rl.do_action_clip
        self.planning_width = config.model_predictive_rl.planning_width
        self.share_graph_model = config.model_predictive_rl.share_graph_model
        self.linear_state_predictor = config.model_predictive_rl.linear_state_predictor
        # self.set_device(device)
        self.device = device
        use_actor_critic=False
        transition_nonlin="tanh"
        predict_rewards=True
        gamma=0.99
        td_lambda=0.8
        value_aggregation="softmax"
        output_tree=False
        self.treeqn = TreeQNNetwork(config, use_actor_critic, transition_nonlin, predict_rewards, gamma, td_lambda, value_aggregation, output_tree)
        self.model = [self.treeqn]

        if self.planning_depth > 1 and not self.do_action_clip:
            logging.warning('Performing treeQN planning without action space clipping!')

    def set_common_parameters(self, config):
        self.gamma = config.rl.gamma
        self.kinematics = config.action_space.kinematics
        self.sampling = config.action_space.sampling
        self.speed_samples = config.action_space.speed_samples
        self.rotation_samples = config.action_space.rotation_samples
        self.v_pref = config.action_space.v_pref
        self.rotation_constraint = config.action_space.rotation_constraint

    def set_device(self, device):
        self.device = device
        for model in self.model:
            model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_noisy_net(self, use_noisy_net):
        self.use_noisy_net = use_noisy_net

    def set_time_step(self, time_step):
        self.time_step = time_step

    def get_normalized_gamma(self):
        return pow(self.gamma, self.time_step * self.v_pref)

    def get_model(self):
        return self.treeqn

    def get_traj(self):
        return self.traj

    def get_state_dict(self):
        return {
                'treeqn': self.treeqn.state_dict()
                }

    def load_state_dict(self, state_dict):
            self.treeqn.load_state_dict(state_dict['treeqn'])

    def save_model(self, file):
        torch.save(self.get_state_dict(), file)

    def load_model(self, file):
        checkpoint = torch.load(file)
        self.load_state_dict(checkpoint)


    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        holonomic = True if self.kinematics == 'holonomic' else False
        # speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
        speeds = [(i+1)/self.speed_samples * v_pref for i in range(self.speed_samples)]
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            if self.rotation_constraint == np.pi:
                rotations = np.linspace(-self.rotation_constraint, self.rotation_constraint, self.rotation_samples, endpoint=False)
            else:
                rotations = np.linspace(-self.rotation_constraint, self.rotation_constraint, self.rotation_samples)

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        self.action_group_index.append(0)

        for i, rotation in enumerate(rotations):
            for j, speed in enumerate(speeds):
                action_index = i * self.speed_samples + j + 1
                self.action_group_index.append(action_index)
                if holonomic:
                    action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
                else:
                    action_space.append(ActionRot(speed, rotation))
        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def predict(self, state):
        self.count=self.count+1
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')
        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(self.v_pref)
        state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
        self.last_state = self.transform(state)
        max_action_index = self.treeqn.step(state_tensor)[0]
        max_action = self.action_space[int(max_action_index)]
        return max_action, int(max_action_index)
