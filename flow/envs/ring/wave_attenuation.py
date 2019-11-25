"""
Environment used to train a stop-and-go dissipating controller.

This is the environment that was used in:

C. Wu, A. Kreidieh, K. Parvate, E. Vinitsky, A. Bayen, "Flow: Architecture and
Benchmarking for Reinforcement Learning in Traffic Control," CoRR, vol.
abs/1710.05465, 2017. [Online]. Available: https://arxiv.org/abs/1710.05465
"""

from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.envs.base import Env
from flow.core import rewards

from gym.spaces.box import Box

from copy import deepcopy
import numpy as np
import random
from scipy.optimize import fsolve

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 1,
    # maximum deceleration of autonomous vehicles
    'max_decel': 1,
    # bounds on the ranges of ring road lengths the autonomous vehicle is
    # trained on
    'ring_length': [220, 270],
}


def v_eq_max_function(v, *args):
    """Return the error between the desired and actual equivalent gap."""
    num_vehicles, length = args

    # maximum gap in the presence of one rl vehicle
    s_eq_max = (length - num_vehicles * 5) / (num_vehicles - 1)

    v0 = 30
    s0 = 2
    tau = 1
    gamma = 4

    error = s_eq_max - (s0 + v * tau) * (1 - (v / v0) ** gamma) ** -0.5

    return error


class WaveAttenuationEnv(Env):
    """Fully observable wave attenuation environment.

    This environment is used to train autonomous vehicles to attenuate the
    formation and propagation of waves in a variable density ring road.

    Required from env_params:

    * max_accel: maximum acceleration of autonomous vehicles
    * max_decel: maximum deceleration of autonomous vehicles
    * ring_length: bounds on the ranges of ring road lengths the autonomous
      vehicle is trained on. If set to None, the environment sticks to the ring
      road specified in the original network definition.

    States
        The state consists of the velocities and absolute position of all
        vehicles in the network. This assumes a constant number of vehicles.

    Actions
        Actions are a list of acceleration for each rl vehicles, bounded by the
        maximum accelerations and decelerations specified in EnvParams.

    Rewards
        The reward function rewards high average speeds from all vehicles in
        the network, and penalizes accelerations by the rl vehicle.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(self.initial_vehicles.num_rl_vehicles, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ["Velocity", "Absolute_pos"]
        return Box(
            low=0,
            high=1,
            shape=(2 * self.initial_vehicles.num_vehicles, ),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        self.k.vehicle.apply_acceleration(
            self.k.vehicle.get_rl_ids(), rl_actions)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.

        # reward average velocity
        eta_2 = 4.
        reward = eta_2 * np.mean(vel) / 20

        # punish accelerations (should lead to reduced stop-and-go waves)
        eta = 4  # 0.25
        mean_actions = np.mean(np.abs(np.array(rl_actions)))
        accel_threshold = 0

        if mean_actions > accel_threshold:
            reward += eta * (accel_threshold - mean_actions)

        return float(reward)

    def get_state(self):
        """See class definition."""
        speed = [self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed()
                 for veh_id in self.k.vehicle.get_ids()]
        pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.network.length()
               for veh_id in self.k.vehicle.get_ids()]

        return np.array(speed + pos)

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)

    def reset(self):
        """See parent class.

        The sumo instance is reset with a new ring length, and a number of
        steps are performed with the rl vehicle acting as a human vehicle.
        """
        # skip if ring length is None
        if self.env_params.additional_params['ring_length'] is None:
            return super().reset()

        # reset the step counter
        self.step_counter = 0

        # update the network
        initial_config = InitialConfig(bunching=50, min_gap=0)
        length = random.randint(
            self.env_params.additional_params['ring_length'][0],
            self.env_params.additional_params['ring_length'][1])
        additional_net_params = {
            'length':
                length,
            'lanes':
                self.net_params.additional_params['lanes'],
            'speed_limit':
                self.net_params.additional_params['speed_limit'],
            'resolution':
                self.net_params.additional_params['resolution']
        }
        net_params = NetParams(additional_params=additional_net_params)

        self.network = self.network.__class__(
            self.network.orig_name, self.network.vehicles,
            net_params, initial_config)
        self.k.vehicle = deepcopy(self.initial_vehicles)
        self.k.vehicle.kernel_api = self.k.kernel_api
        self.k.vehicle.master_kernel = self.k

        # solve for the velocity upper bound of the ring
        v_guess = 4
        v_eq_max = fsolve(v_eq_max_function, np.array(v_guess),
                          args=(len(self.initial_ids), length))[0]

        print('\n-----------------------')
        print('ring length:', net_params.additional_params['length'])
        print('v_max:', v_eq_max)
        print('-----------------------')

        # restart the sumo instance
        self.restart_simulation(
            sim_params=self.sim_params,
            render=self.sim_params.render)

        # perform the generic reset function
        observation = super().reset()

        # reset the timer to zero
        self.time_counter = 0

        return observation

class WaveAttenuationEnvAvgSpeedreward(WaveAttenuationEnv):
    """Fully observable wave attenuation environment.

    overriding reward to be average speed reward since WaveAttenuationEnv's reward seems completely wrong (mostly negative unless vehicle stops)
    """
    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # return a reward of 0 if a collision occurred
        if kwargs["fail"]:
            return 0

        # reward high system-level velocities, but according to their L_2 norm,
        # which is bad, since encourages increase in high-speeds more than in
        # low-speeds and is not the real-reward
        #
        #return rewards.desired_velocity(self, fail=kwargs["fail"])
        veh_ids = self.vehicles.get_ids() 
        vel = np.array(self.vehicles.get_speed(veh_ids))
        num_vehicles = len(veh_ids)

        if any(vel < -100):
            return 0.

        target_vel = self.env_params.additional_params['target_velocity']
        max_reward = target_vel
        print("max_reward " + str(max_reward))

        reward = np.sum(vel) / num_vehicles
        print("reward " + str(reward))
        
        #return reward / max_reward
        return reward

class WaveAttenuationPOEnv(WaveAttenuationEnv):
    """POMDP version of WaveAttenuationEnv.

    Note that this environment only works when there is one autonomous vehicle
    on the network.

    Required from env_params:

    * max_accel: maximum acceleration of autonomous vehicles
    * max_decel: maximum deceleration of autonomous vehicles
    * ring_length: bounds on the ranges of ring road lengths the autonomous
      vehicle is trained on

    States
        The state consists of the speed and headway of the ego vehicle, as well
        as the difference in speed between the ego vehicle and its leader.
        There is no assumption on the number of vehicles in the network.

    Actions
        See parent class

    Rewards
        See parent class

    Termination
        See parent class

    """

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=-float('inf'), high=float('inf'),
                   shape=(3, ), dtype=np.float32)

    def get_state(self):
        """See class definition."""
        rl_id = self.k.vehicle.get_rl_ids()[0]
        lead_id = self.k.vehicle.get_leader(rl_id) or rl_id

        # normalizers
        max_speed = 15.
        if self.env_params.additional_params['ring_length'] is not None:
            max_length = self.env_params.additional_params['ring_length'][1]
        else:
            max_length = self.k.network.length()

        observation = np.array([
            self.k.vehicle.get_speed(rl_id) / max_speed,
            (self.k.vehicle.get_speed(lead_id) -
             self.k.vehicle.get_speed(rl_id)) / max_speed,
            (self.k.vehicle.get_x_by_id(lead_id) -
             self.k.vehicle.get_x_by_id(rl_id)) % self.k.network.length()
            / max_length
        ])

        return observation

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        rl_id = self.k.vehicle.get_rl_ids()[0]
        lead_id = self.k.vehicle.get_leader(rl_id) or rl_id
        self.k.vehicle.set_observed(lead_id)

class WaveAttenuationPOEnvNoisy(WaveAttenuationPOEnv):
    """Adding observation noise to WaveAttenuationPOEnv
    """

    def get_state(self):
        """See class definition."""

        # add gaussian noise with 0.05 standard deviation => 2sigma is 10% error
        observation = super().get_state()
        return observation + np.random.normal(0,0.05,len(observation))

class WaveAttenuationPOEnvSpeedreward(WaveAttenuationPOEnv):
    def compute_reward(self, rl_actions, **kwargs):
#        """See class definition."""
#        # return a reward of 0 if a collision occurred
#        if kwargs["fail"]:
#            return 0
#
#        # reward high system-level velocities, but according to their L_2 norm,
#        # which is bad, since encourages increase in high-speeds more than in
#        # low-speeds and is not the real-reward
#        #
#        #return rewards.desired_velocity(self, fail=kwargs["fail"])
#        veh_ids = self.vehicles.get_ids() 
#        vel = np.array(self.vehicles.get_speed(veh_ids))
#        num_vehicles = len(veh_ids)
#
#        if any(vel < -100):
#            return 0.
#
#        target_vel = self.env_params.additional_params['target_velocity']
#        max_reward = np.array([target_vel])
#        print("max_reward " + str(max_reward))
#
#        reward = np.sum(vel) / num_vehicles
#        print("reward " + str(reward))
#        
#        #return reward / max_reward
#        return reward
        """See class definition."""
        # return a reward of 0 if a collision occurred
        if kwargs["fail"]:
            return 0
        
        # reward high system-level velocities
        return rewards.desired_velocity(self, fail=kwargs["fail"])

class WaveAttenuationPOEnvAvgSpeedreward(WaveAttenuationPOEnv):
    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # return a reward of 0 if a collision occurred
        if kwargs["fail"]:
            return 0

        # reward high system-level velocities, but according to their L_2 norm,
        # which is bad, since encourages increase in high-speeds more than in
        # low-speeds and is not the real-reward
        #
        #return rewards.desired_velocity(self, fail=kwargs["fail"])
        veh_ids = self.vehicles.get_ids() 
        vel = np.array(self.vehicles.get_speed(veh_ids))
        num_vehicles = len(veh_ids)

        if any(vel < -100):
            return 0.

        target_vel = self.env_params.additional_params['target_velocity']
        max_reward = target_vel
        print("max_reward " + str(max_reward))

        reward = np.sum(vel) / num_vehicles
        print("reward " + str(reward))
        
        #return reward / max_reward
        return reward

class WaveAttenuationPOEnvSmallAccelPenalty(WaveAttenuationPOEnv):
    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.vehicles.get_speed(veh_id)
            for veh_id in self.vehicles.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.

        # reward average velocity
        eta_2 = 4.
        reward = eta_2 * np.mean(vel) / 20

        # punish accelerations (should lead to reduced stop-and-go waves)
        eta = 2  # 0.25
        rl_actions = np.array(rl_actions)
        accel_threshold = 0

        if np.mean(np.abs(rl_actions)) > accel_threshold:
            reward += eta * (accel_threshold - np.mean(np.abs(rl_actions)))

        return float(reward)

class WaveAttenuationPOEnvMediumAccelPenalty(WaveAttenuationPOEnv):
    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.vehicles.get_speed(veh_id)
            for veh_id in self.vehicles.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.

        # reward average velocity
        eta_2 = 4.
        reward = eta_2 * np.mean(vel) / 20

        # punish accelerations (should lead to reduced stop-and-go waves)
        eta = 4  # 0.25
        rl_actions = np.array(rl_actions)
        accel_threshold = 0

        if np.mean(np.abs(rl_actions)) > accel_threshold:
            reward += eta * (accel_threshold - np.mean(np.abs(rl_actions)))

        return float(reward)

class WaveAttenuationPORadiusEnv(WaveAttenuationEnv):
    """POMDP version of WaveAttenuationEnv.

    Note that this environment only works when there is one autonomous vehicle
    on the network.

    Required from env_params:

    * max_accel: maximum acceleration of autonomous vehicles
    * max_decel: maximum deceleration of autonomous vehicles
    * ring_length: bounds on the ranges of ring road lengths the autonomous
      vehicle is trained on

    States
        The state consists of the speed and headway of the ego vehicle, as well
        as the difference in speed between the ego vehicle and its leader.
        There is no assumption on the number of vehicles in the network.

    Actions
        See parent class

    Rewards
        See parent class

    Termination
        See parent class

    """
    # TODO toggle comments in the next two lines
    OBSERVATION_RADIUS = 1 # default, like WaveAttenuationMergePOEnv
    #OBSERVATION_RADIUS = obs_radius # factory parameter

    # TODO: REFACTOR? taken from .../flow/envs/merge.py::WaveAttenuationMergePORadiusEnv
    @property
    def observation_space(self):
        """See class definition."""
        # taken from .../flow/envs/merge.py::WaveAttenuationMergePORadiusEnv
        num_attributes_self = 1 # observe just self speed
        num_attributes_others = 2 # observe others' speed and distance
        num_directions = 2 # observe back and front
        self.obs_dimension_per_rl = num_attributes_self + \
          self.OBSERVATION_RADIUS * num_attributes_others * num_directions
        return Box(low=0, high=1, shape=(self.obs_dimension_per_rl *
          self.vehicles.num_rl_vehicles, ), dtype=np.float32)

    # TODO: REFACTOR? taken from .../flow/envs/merge.py::WaveAttenuationMergePORadiusEnv
    def get_state(self):
        """See class definition."""
        self.leader = []
        self.follower = []

        max_speed = 15.
        max_length = self.env_params.additional_params['ring_length'][1]

        observation = []
        #
        # if more than 1 RL vehicle, add a loop here like in merge.py
        rl_id = self.vehicles.get_rl_ids()[0]
        #
        # fill with observations of self
        this_speed = self.vehicles.get_speed(rl_id)
        observation.append(this_speed / max_speed)

        # fill with observations of leading vehicles
        lead_id = rl_id
        for _ in range(self.OBSERVATION_RADIUS):
          lead_id = self.vehicles.get_leader(lead_id)
          if lead_id in ["", None]:
              # in case leader is not visible
              lead_speed = max_speed
              lead_head = max_length
          else:
              self.leader.append(lead_id)
              lead_speed = self.vehicles.get_speed(lead_id)
              # modulo resolves the cycle start-crossing issue
              lead_head = (self.get_x_by_id(lead_id) \
                  - self.get_x_by_id(rl_id) - self.vehicles.get_length(rl_id)) % max_length
          observation.append((lead_speed - this_speed) / max_speed)
          observation.append(lead_head / max_length)

        # fill with observations of following vehicles
        follower_id = rl_id
        for _ in range(self.OBSERVATION_RADIUS):
          follower_id = self.vehicles.get_follower(follower_id)
          if follower_id in ["", None]:
              # in case follower_id is not visible
              follow_speed = 0
              follow_head = max_length
          else:
              self.follower.append(follower_id)
              follow_speed = self.vehicles.get_speed(follower_id)
              # modulo resolves the cycle start-crossing issue
              follow_head = (self.get_x_by_id(rl_id) \
                  - self.get_x_by_id(follower_id) - self.vehicles.get_length(follower_id)) % max_length
          observation.append((this_speed - follow_speed) / max_speed)
          observation.append(follow_head / max_length)

        return observation

    # TODO: REFACTOR? taken from .../flow/envs/merge.py::WaveAttenuationMergePORadiusEnv
    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""

        # specify observed vehicles
        for veh_id in self.leader + self.follower:
            self.vehicles.set_observed(veh_id)

    # TODO: REFACTOR? taken from WaveAttenuationPOEnvAvgSpeedreward
    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # return a reward of 0 if a collision occurred
        if kwargs["fail"]:
            return 0

        # reward high system-level velocities, but according to their L_2 norm,
        # which is bad, since encourages increase in high-speeds more than in
        # low-speeds and is not the real-reward
        #
        #return rewards.desired_velocity(self, fail=kwargs["fail"])
        veh_ids = self.vehicles.get_ids() 
        vel = np.array(self.vehicles.get_speed(veh_ids))
        num_vehicles = len(veh_ids)

        if any(vel < -100):
            return 0.

        target_vel = self.env_params.additional_params['target_velocity']
        max_reward = target_vel
        print("max_reward " + str(max_reward))

        reward = np.sum(vel) / num_vehicles
        print("reward " + str(reward))
        
        #return reward / max_reward
        return reward

class WaveAttenuationPORadius1Env(WaveAttenuationPORadiusEnv):
    OBSERVATION_RADIUS = 1 # default 

class WaveAttenuationPORadius2Env(WaveAttenuationPORadiusEnv):
    OBSERVATION_RADIUS = 2 # default 
