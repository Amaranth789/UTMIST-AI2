# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
# 
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
# 
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
# 
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import gdown
from typing import Optional
from environment.agent import Agent
from stable_baselines3 import PPO, A2C # Sample RL Algo imports
from sb3_contrib import RecurrentPPO # Importing an LSTM
import torch 
import gymnasium as gym
from torch.nn import functional as F
from torch import nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# To run the sample TTNN model, you can uncomment the 2 lines below: 
# import ttnn
# from user.my_agent_tt import TTMLPPolicy

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        """
        A 3-layer MLP policy:
        obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
        """
        super(MLPPolicy, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)

    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        returns: [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MLPExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP Base Features Extractor
    '''
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, hidden_dim: int = 64):
        super(MLPExtractor, self).__init__(observation_space, features_dim)
        self.model = MLPPolicy(
            obs_dim=observation_space.shape[0], 
            action_dim=10,
            hidden_dim=hidden_dim,
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)
    
    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, hidden_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim) #NOTE: features_dim = 10 to match action space output
        )


class SubmittedAgent(Agent):
    '''
    Input the **file_path** to your agent here for submission!
    '''
    def __init__(
        self,
        file_path: Optional[str] = None,
    ):
        super().__init__(file_path)

        # To run a TTNN model, you must maintain a pointer to the device and can be done by 
        # uncommmenting the line below to use the device pointer
        # self.mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1,1))

    def _initialize(self) -> None:
        policy_kwargs = MLPExtractor.get_policy_kwargs(features_dim=
64, hidden_dim=64
)
        if self.file_path is None:
            self.model = PPO(
    "MlpPolicy",
    self.env,
    verbose=0,
    learning_rate=0.0001,
    n_steps=4096,
    gamma=0.999,
    ent_coef=0.01,
    policy_kwargs=policy_kwargs
)
            del self.env
        else:
            # 1. 像创建新模型时一样，获取 policy_kwargs
            #
            policy_kwargs = MLPExtractor.get_policy_kwargs(features_dim=64, hidden_dim=64)
            
            # 2. 定义 custom_objects，并把 policy_kwargs *作为键值对* 放进去
            #    这才是警告信息 真正的含义
            custom_objects = {
                # 修复 "Could not deserialize object policy_kwargs"
                "policy_kwargs": policy_kwargs, 
                
                # (以下部分已证明有效，予以保留)
                # 修复 "Could not deserialize object lr_schedule"
                "lr_schedule": lambda _: 0.0003,  
                
                # 修复 "Could not deserialize object clip_range"
                "clip_range": lambda _: 0.2       
            }

            # 3. 在 PPO.load() 中只传入 custom_objects
            #    注意：顶层的 policy_kwargs 参数已被移除
            self.model = PPO.load(
                self.file_path, 
                custom_objects=custom_objects
            )
        # To run the sample TTNN model during inference, you can uncomment the 5 lines below:
        # This assumes that your self.model.policy has the MLPPolicy architecture defined in `train_agent.py` or `my_agent_tt.py`
        # mlp_state_dict = self.model.policy.features_extractor.model.state_dict()
        # self.tt_model = TTMLPPolicy(mlp_state_dict, self.mesh_device)
        # self.model.policy.features_extractor.model = self.tt_model
        # self.model.policy.vf_features_extractor.model = self.tt_model
        # self.model.policy.pi_features_extractor.model = self.tt_model

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            # Place a link to your PUBLIC model data here. This is where we will download it from on the tournament server.
            url = "https://drive.google.com/file/d/1cdhx3xq-pns21IN1UvcgxVCdAEgZ9GGx/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    # If modifying the number of models (or training in general), modify this
    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
