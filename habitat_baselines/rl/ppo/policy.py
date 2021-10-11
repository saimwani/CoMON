#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import numpy as np
import torch
import torch.nn as nn

from habitat_baselines.common.utils import CategoricalNet, Flatten
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.simple_cnn import BeliefRefinement, SimpleCNN, MapCNN, CreateDiscreteMessage


class Policy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class PointNavBaselinePolicy(Policy):
    def __init__(
        self,
     
        observation_space,
        action_space,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        comm_type,
        hidden_size=512,
    ):
        super().__init__(
            PointNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                device=device,
                object_category_embedding_size=object_category_embedding_size,
                previous_action_embedding_size=previous_action_embedding_size,
                use_previous_action=use_previous_action,
                comm_type=comm_type,
            ),
            action_space.n,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class PointNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN. Also simulates
    communication between the agents.
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, device, 
        object_category_embedding_size, previous_action_embedding_size, use_previous_action,
        comm_type
    ):
        super().__init__()

        self.messageLength = 8 if comm_type == "structured" else 256
        self.vocabularySize_a1 = 2
        self.vocabularySize_a2 = 2
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[
            self.goal_sensor_uuid
        ].shape[0]
        self._hidden_size = hidden_size
        self.device = device
        self.use_previous_action = use_previous_action

        self.visual_encoder = SimpleCNN(observation_space, 512)
        self.map_encoder = MapCNN(observation_space, 256)

        if self.use_previous_action:
            self.state_encoder = RNNStateEncoder(
                self._hidden_size + previous_action_embedding_size, 
                self._hidden_size,
            )
        else:
            self.state_encoder = RNNStateEncoder(
                (self._hidden_size),
                self._hidden_size,  
            )
        self.goal_embedding = nn.Embedding(9, object_category_embedding_size)
        self.action_embedding = nn.Embedding(4, previous_action_embedding_size)

        self.occupancy_embedding = nn.Embedding(3, 16)
        self.object_embedding = nn.Embedding(9, 16)
        # self.exploration_embedding = nn.Embedding(3, 16)

        self.a1Primary = nn.Linear(512 + object_category_embedding_size, 512)

        if comm_type == "structured":
            self.a1MessageCreate1 = CreateDiscreteMessage(256, self.messageLength, self.vocabularySize_a1)
            self.a2MessageCreate1 = CreateDiscreteMessage(512, self.messageLength, self.vocabularySize_a2)
            self.a1MessageCreate2 = CreateDiscreteMessage(256, self.messageLength, self.vocabularySize_a1)
            self.a2MessageCreate2 = CreateDiscreteMessage(512, self.messageLength, self.vocabularySize_a2)
        else:
            self.a1MessageCreate1 = nn.Linear(256, self.messageLength)
            self.a2MessageCreate1 = nn.Linear(512, self.messageLength)
            self.a1MessageCreate2 = nn.Linear(256, self.messageLength)
            self.a2MessageCreate2 = nn.Linear(512, self.messageLength)
            

        self.beliefRefine11 = BeliefRefinement(256 + self.messageLength, 256)
        self.beliefRefine21 = BeliefRefinement(512 + self.messageLength, 512)

        self.beliefRefine12 = BeliefRefinement(256 + self.messageLength, 256)
        self.beliefRefine22 = BeliefRefinement(512 + self.messageLength, 512)
        self.train()
    


    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = self.get_target_encoding(observations)
        goal_embed = [self.goal_embedding((target_encoding).type(torch.LongTensor).to(self.device)).squeeze(1)]
        bs = target_encoding.shape[0]
        
        perception_embed = self.visual_encoder(observations)
        goalPlusPerception = [perception_embed] + goal_embed
        goalPlusPerception = torch.cat(goalPlusPerception, dim=1)
        goalPlusPerception = self.a1Primary(goalPlusPerception)  

        global_map_embedding = []
        global_map = observations['semMap']
        global_map_embedding.append(self.occupancy_embedding(global_map[:, :, :, 0].type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50 , -1))
        global_map_embedding.append(self.object_embedding(global_map[:, :, :, 1].type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50, -1))
        global_map_embedding = torch.cat(global_map_embedding, dim=3)
        map_embed = self.map_encoder(global_map_embedding)
        
        m11 = self.a1MessageCreate1(map_embed)
        m21 = self.a2MessageCreate1(goalPlusPerception)

        b11 = self.beliefRefine11(map_embed, m21)
        b21 = self.beliefRefine21(goalPlusPerception, m11)

        m12 = self.a1MessageCreate2(b11)
        m22 = self.a2MessageCreate2(b21)

        b12 = self.beliefRefine12(b11, m22)
        b22 = self.beliefRefine22(b21, m12)
        
        x = torch.cat([b22] + [self.action_embedding(prev_actions).squeeze(1)], dim=1 )
        
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states
