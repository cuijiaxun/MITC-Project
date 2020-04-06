"""
Environments for training vehicles to reduce congestion in a merge.

This environment was used in:
TODO(ak): add paper after it has been published.
"""

from flow.envs.base import Env
from flow.core import rewards

from gym.spaces.box import Box

import numpy as np
import collections

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25,
    # maximum number of controllable vehicles in the network
    "num_rl": 5,
}


class MergePOEnv_noheadway_encourageRLmove(Env):
    """Partially observable merge environment.

    This environment is used to train autonomous vehicles to attenuate the
    formation and propagation of waves in an open merge network.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s
    * num_rl: maximum number of controllable vehicles in the network

    States
        The observation consists of the speeds and bumper-to-bumper headways of
        the vehicles immediately preceding and following autonomous vehicle, as
        well as the ego speed of the autonomous vehicles.

        In order to maintain a fixed observation size, when the number of AVs
        in the network is less than "num_rl", the extra entries are filled in
        with zeros. Conversely, if the number of autonomous vehicles is greater
        than "num_rl", the observations from the additional vehicles are not
        included in the state space.

    Actions
        The action space consists of a vector of bounded accelerations for each
        autonomous vehicle $i$. In order to ensure safety, these actions are
        bounded by failsafes provided by the simulator at every time step.

        In order to account for variability in the number of autonomous
        vehicles, if n_AV < "num_rl" the additional actions provided by the
        agent are not assigned to any vehicle. Moreover, if n_AV > "num_rl",
        the additional vehicles are not provided with actions from the learning
        agent, and instead act as human-driven vehicles as well.

    Rewards
        The reward function encourages proximity of the system-level velocity
        to a desired velocity, while slightly penalizing small time headways
        among autonomous vehicles.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # maximum number of controlled vehicles
        self.num_rl = env_params.additional_params["num_rl"]

        # queue of rl vehicles waiting to be controlled
        self.rl_queue = collections.deque()

        # names of the rl vehicles controlled at any step
        self.rl_veh = []

        # used for visualization: the vehicles behind and after RL vehicles
        # (ie the observed vehicles) will have a different color
        self.leader = []
        self.follower = []

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            shape=(self.num_rl, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=-1, high=1, shape=(5 * self.num_rl, ), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        for i, rl_id in enumerate(self.rl_veh):
            # ignore rl vehicles outside the network
            if rl_id not in self.k.vehicle.get_rl_ids():
                continue
            self.k.vehicle.apply_acceleration(rl_id, rl_actions[i])

    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""
        self.leader = []
        self.follower = []

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        observation = [0 for _ in range(5 * self.num_rl)]
        for i, rl_id in enumerate(self.rl_veh):
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower_id = self.k.vehicle.get_follower(rl_id)

            if lead_id in ["", None]:
                # in case leader is not visible
                lead_speed = max_speed
                lead_head = max_length
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_x_by_id(lead_id) \
                    - self.k.vehicle.get_x_by_id(rl_id) \
                    - self.k.vehicle.get_length(rl_id)

            if follower_id in ["", None]:
                # in case follower_id is not visible
                follow_speed = 0
                follow_head = max_length
            else:
                self.follower.append(follower_id)
                follow_speed = self.k.vehicle.get_speed(follower_id)
                follow_head = self.k.vehicle.get_headway(follower_id)
            #FIXME do not clip!!!

            observation[5 * i + 0] = np.clip(this_speed / max_speed,-1,1)
            observation[5 * i + 1] = np.clip((lead_speed - this_speed) / max_speed,-1,1)
            observation[5 * i + 2] = np.clip(lead_head / max_length,-1,1)
            observation[5 * i + 3] = np.clip((this_speed - follow_speed) / max_speed,-1,1)
            observation[5 * i + 4] = np.clip(follow_head / max_length,-1,1)
            for j in range(5):
                if observation[5*i+j] < -1 or observation[5*i+j] > 1:
                #if np.random.random() < 0.01 or observation[5*i+2] < -1:
                  print(
                    "ERROR OBSERVATION OUT OF RANGE " + 
                    " rl_veh " + str(self.rl_veh) +
                    " rl_ids " + str(self.k.vehicle.get_rl_ids()) +
                    " lead_id " + str(lead_id) +
                    " rl_id " + str(rl_id) +
                    " follower_id " + str(follower_id) +
                    " this_speed " + str(this_speed) +
                    " max_speed " + str(max_speed) +
                    " lead_speed " + str(lead_speed) +
                    " lead_head " + str(lead_head) +
                    " max_length " + str(max_length) +
                    " follow_speed " + str(follow_speed) +
                    " follow_head " + str(follow_head) +
                    " lead_x " + str(self.k.vehicle.get_x_by_id(lead_id)) +
                    " self_length " + str(self.k.vehicle.get_length(rl_id)) +
                    " rl_x " + str(self.k.vehicle.get_x_by_id(rl_id)) +
                    " follower_x " + str(self.k.vehicle.get_x_by_id(follower_id))
                    )
                  #while True:
                  #  pass

        return observation

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            # return a reward of 0 if a collision occurred
            if kwargs["fail"]:
                return 0

            # reward high system-level velocities
            cost1 = rewards.desired_velocity(self, fail=kwargs["fail"])

            # encourage rl to move
            cost2 = rewards.rl_forward_progress(self, gain=1)/self.k.network.max_speed()
            # weights for cost1, cost2, and cost3, respectively
            eta1, eta2 = 0.50, 0.50

            return max(eta1 * cost1 + eta2 * cost2, 0)

    def additional_command(self):
        """See parent class.

        This method performs to auxiliary tasks:

        * Define which vehicles are observed for visualization purposes.
        * Maintains the "rl_veh" and "rl_queue" variables to ensure the RL
          vehicles that are represented in the state space does not change
          until one of the vehicles in the state space leaves the network.
          Then, the next vehicle in the queue is added to the state space and
          provided with actions from the policy.
        """
        #print("additional_command()")
        rl_ids = self.k.vehicle.get_rl_ids()
        #print("rl_ids " + str(rl_ids))
        # add rl vehicles that just entered the network into the rl queue
        #print("rl_queue before " + str(self.rl_queue))
        for veh_id in rl_ids:
            if veh_id not in list(self.rl_queue) + self.rl_veh:
                self.rl_queue.append(veh_id)
        #print("rl_queue after " + str(self.rl_queue))

        # remove rl vehicles that exited the network
        #print("rl_queue before " + str(self.rl_queue))
        for veh_id in list(self.rl_queue):
            if veh_id not in rl_ids:
                self.rl_queue.remove(veh_id)
        #print("rl_queue after " + str(self.rl_queue))
        #print("rl_veh before " + str(self.rl_veh))
        for veh_id in self.rl_veh:
            if veh_id not in rl_ids:
                self.rl_veh.remove(veh_id)
        #print("rl_veh after " + str(self.rl_veh))

        # fil up rl_veh until they are enough controlled vehicles
        while len(self.rl_queue) > 0 and len(self.rl_veh) < self.num_rl:
            rl_id = self.rl_queue.popleft()
            self.rl_veh.append(rl_id)

        # specify observed vehicles
        for veh_id in self.leader + self.follower:
            self.k.vehicle.set_observed(veh_id)

    def reset(self):
        """See parent class.

        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        # initializing like in constructor
        self.rl_queue = collections.deque()
        self.rl_veh = []
        self.leader = []
        self.follower = []
        return super().reset()


#def factoryMergePORadiusEnv(obs_radius):
class MergePORadiusEnv(MergePOEnv_noheadway_encourageRLmove):
    """Partially observable merge environment.

    This environment is similar to WaveAttenuationMergePOEnv, except that
    each RL car sees OBSERVATION_RADIUS cars infront and in the back of it instead of 1.
    """

    # TODO toggle comments in the next two lines
    OBSERVATION_RADIUS = 1 # default, like WaveAttenuationMergePOEnv
    #OBSERVATION_RADIUS = obs_radius # factory parameter

    @property
    def observation_space(self):
        """See class definition."""
        num_attributes_self = 1 # observe just self speed
        num_attributes_others = 2 # observe others' speed and distance
        num_directions = 2 # observe back and front
        self.obs_dimension_per_rl = num_attributes_self + \
          self.OBSERVATION_RADIUS * num_attributes_others * num_directions
        return Box(low=-1, high=1, shape=(self.obs_dimension_per_rl *
          self.num_rl, ), dtype=np.float32)

    def get_state(self, rl_id=None, **kwargs):
        """Generalization of the WaveAttenuationMergePOEnv code to radius > 1."""
        self.leader = []
        self.follower = []
        
        # normalizing constants
        max_speed = self.scenario.max_speed
        max_length = self.scenario.length

        observation = []
        for i, rl_id in enumerate(self.rl_veh):
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
                  lead_head = self.get_x_by_id(lead_id) \
                      - self.get_x_by_id(rl_id) - self.vehicles.get_length(rl_id)
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
                  follow_head = self.vehicles.get_headway(follower_id)
              observation.append((this_speed - follow_speed) / max_speed)
              observation.append(follow_head / max_length)

        # if doesn't see enough RL vehicles, pad with 0s, as was done originally
        obs_dimension = self.obs_dimension_per_rl * self.num_rl
        if len(observation) < obs_dimension:
          observation += [0] * (obs_dimension - len(observation)) 
         
        return observation


# alias for MergePORadiusEnv classes with different observation
# radii. An alternative software solution would be to send a parameter through
# ADDITIONAL_ENV_PARAMS, but that would require updating ADDITIONAL_ENV_PARAMS
# above and in every calling location, breaking backward compatibility
class MergePORadius2Env(MergePORadiusEnv):
#
#MergePORadius2Env = factoryMergePORadiusEnv(2)
#MergePORadius4Env = factoryMergePORadiusEnv(4)
#MergePORadius6Env = factoryMergePORadiusEnv(6)
#MergePORadius7Env = factoryMergePORadiusEnv(7)
    """Partially observable merge environment.

    This environment is just an alias for specific OBSERVATION_RADIUS.
    """
    OBSERVATION_RADIUS = 2 
class MergePORadius4Env(MergePORadiusEnv):
    """Partially observable merge environment.

    This environment is just an alias for specific OBSERVATION_RADIUS.
    """
    OBSERVATION_RADIUS = 4 
class MergePORadius6Env(MergePORadiusEnv):
    """Partially observable merge environment.

    This environment is just an alias for specific OBSERVATION_RADIUS.
    """
    OBSERVATION_RADIUS = 6 
class MergePORadius7Env(MergePORadiusEnv):
    """Partially observable merge environment.

    This environment is just an alias for specific OBSERVATION_RADIUS.
    """
    OBSERVATION_RADIUS = 7 

# Actually can do a factory 
#def WaveAttenuationMergePORadiusEnvFactory(obs_radius):
#  cls = WaveAttenuationMergePORadiusEnv()
#
#def silly(n):
#    class Silly(object):
#            buh = ' '.join(n * ['hello'])
#                return Silly
#
#Silly1 = silly(1)
#Silly2 = silly(2)
#a = Silly1()
#print(a.buh)
#b = Silly2()
#print(b.buh)
#
#will print
#
#hello
#hello hello
#
#
