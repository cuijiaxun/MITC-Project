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
    # maximum number of vehicles arrived
    #"max_num_vehicles":-1,
}


class MergePOEnv(Env):
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

        In order to account for variability in the number of autonomself.k.vehicle.get_speed(self.k.vehicle.get_ids())ous
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
    
    def setup_initial_state(self):
        """Store information on the initial state of vehicles in the network.

        This information is to be used upon reset. This method also adds this
        information to the self.vehicles class and starts a subscription with
        sumo to collect state information each step.
        """
        # determine whether to shuffle the vehicles
        if self.initial_config.shuffle:
            random.shuffle(self.initial_ids)
        # generate starting position for vehicles in the network
        start_pos, start_lanes = self.k.network.generate_starting_positions(
            initial_config=self.initial_config,
            num_vehicles=len(self.initial_ids))
        # save the initial state. This is used in the _reset function
        #print(self.initial_ids)
        if isinstance(self.initial_config.edges_distribution,dict):
            additional_params = self.env_params.additional_params
            main_human = additional_params['main_human']
            main_rl = additional_params['main_rl']
            merge_human = additional_params['merge_human']
            penetration_rate = main_rl /(main_rl+main_human)
            gap = int(1/penetration_rate) - 1
            count_human = 0
            #print("pene_rate",penetration_rate)
            human_ids,rl_ids =[],[]
            for i,veh_id in enumerate(self.initial_ids):
                if self.k.vehicle.get_type(veh_id) == 'human':
                    human_ids.append(veh_id)
                if self.k.vehicle.get_type(veh_id) == 'rl':
                    rl_ids.append(veh_id)

            for i in range(len(start_pos)):
                if start_pos[i][0] in ['inflow_highway','left']:
                    if count_human == gap and len(rl_ids)>0:
                        count_human = 0
                        veh_id = rl_ids.pop(0)
                    else:
                        count_human+=1
                        veh_id = human_ids.pop(0)
                elif start_pos[i][0] in ['inflow_merge','bottom']:
                    veh_id = human_ids.pop(0)
                else:
                    continue
                type_id = self.k.vehicle.get_type(veh_id)
                pos = start_pos[i][1]
                lane = start_lanes[i]
                speed = self.k.vehicle.get_initial_speed(veh_id)
                edge = start_pos[i][0]
                #print(veh_id,speed)
                self.initial_state[veh_id] = (type_id, edge, lane, pos, speed)
            #print(self.initial_state)

        else:
            for i, veh_id in enumerate(self.initial_ids):
                type_id = self.k.vehicle.get_type(veh_id)
                pos = start_pos[i][1]
                lane = start_lanes[i]
                speed = self.k.vehicle.get_initial_speed(veh_id)
                edge = start_pos[i][0]

                self.initial_state[veh_id] = (type_id, edge, lane, pos, speed)

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

            # penalize small time headways
            cost2 = 0
            t_min = 1  # smallest acceptable time headway
            for rl_id in self.rl_veh:
                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id not in ["", None] \
                        and self.k.vehicle.get_speed(rl_id) > 0:
                    t_headway = max(
                        self.k.vehicle.get_headway(rl_id) /
                        self.k.vehicle.get_speed(rl_id), 0)
                    cost2 += min((t_headway - t_min) / t_min, 0)

            # weights for cost1, cost2, and cost3, respectively
            eta1, eta2 = 1.00, 0.10

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
        #print("rl_queue before " + str(self.rl_qu 
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

class MergePOEnvScaleInflow(MergePOEnv):
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

            # penalize small time headways
            cost2 = 0
            t_min = 1  # smallest acceptable time headway
            for rl_id in self.rl_veh:
                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id not in ["", None] \
                        and self.k.vehicle.get_speed(rl_id) > 0:
                    t_headway = max(
                        self.k.vehicle.get_headway(rl_id) /
                        self.k.vehicle.get_speed(rl_id), 0)
                    cost2 += min((t_headway - t_min) / t_min, 0)
            if "max_inflow" in self.env_params.additional_params.keys():
                #print("max_inflow specified ")
                max_inflow = self.env_params.additional_params["max_inflow"]
            else:
                max_inflow = 2200
            InflowScale = rewards.optimize_inflow(self, max_flow=max_inflow,timespan=500)
            # weights for cost1, cost2, and cost3, respectively
            eta1, eta2 = 1.00, 0.00
            reward = max(eta1 * cost1 + eta2 * cost2, 0) * InflowScale
            return reward

class MergePOEnvDeparted(MergePOEnv):
    def additional_command(self):
            if 'max_num_vehicles' not in self.env_params.additional_params:
                super().additional_command()
            else:
                rl_ids = self.k.vehicle.get_rl_ids()
                # add rl vehicles that just entered the network into the rl queue
                if self.k.vehicle.get_num_departed() >= self.env_params.additional_params["max_num_vehicles"]:
                    for veh_id in rl_ids:
                        edge = self.k.vehicle.get_edge(veh_id) 
                        if veh_id not in list(self.rl_queue)+self.rl_veh:
                            self.rl_queue.append(veh_id)
                # remove rl vehicles that exited the network 
                for veh_id in list(self.rl_queue):
                    if veh_id not in rl_ids:
                        self.rl_queue.remove(veh_id)
                for veh_id in self.rl_veh:
                    if veh_id not in rl_ids:
                        self.rl_veh.remove(veh_id)
                # fil up rl_veh until they are enough controlled vehicles
                while len(self.rl_queue) > 0 and len(self.rl_veh) < self.num_rl:
                    rl_id = self.rl_queue.popleft()
                    self.rl_veh.append(rl_id)
                # specify observed vehicles
                for veh_id in self.leader + self.follower:
                    self.k.vehicle.set_observed(veh_id)

class MergePOEnvAvgVel(MergePOEnv):
    def compute_reward(self, rl_actions, **kwargs):
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            reward = rewards.average_velocity(self)
            return reward/30

class MergePOEnvNegativeAvgVel(MergePOEnv):
     def compute_reward(self, rl_actions, **kwargs):
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            reward = rewards.average_velocity(self)
            return reward/30-1
   
class MergePOEnvNegativeEstimateAvgVel(MergePOEnv):
     def compute_reward(self, rl_actions, **kwargs):
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            instant_speed = np.array(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
            instant_speed = instant_speed + 1e-6
            instant_speed_inverse = 1.0/instant_speed
            est_avg_speed =  1.0/(np.mean(instant_speed_inverse))
            return est_avg_speed/30-1

class MergePOEnvIgnore(MergePOEnv):
    def additional_command(self):
            if 'ignore_edges' not in self.env_params.additional_params:
                super().additional_command()
            else:
                rl_ids = self.k.vehicle.get_rl_ids()
                # add rl vehicles that just entered the network into the rl queue
                for veh_id in rl_ids:
                    edge = self.k.vehicle.get_edge(veh_id) 
                    if (veh_id not in list(self.rl_queue)+self.rl_veh)\
                            and (edge not in self.env_params.additional_params['ignore_edges']):
                        self.rl_queue.append(veh_id)
                # remove rl vehicles that exited the network 
                for veh_id in list(self.rl_queue):
                    if veh_id not in rl_ids:
                        self.rl_queue.remove(veh_id)
                for veh_id in self.rl_veh:
                    if veh_id not in rl_ids:
                        self.rl_veh.remove(veh_id)
                # fil up rl_veh until they are enough controlled vehicles
                while len(self.rl_queue) > 0 and len(self.rl_veh) < self.num_rl:
                    rl_id = self.rl_queue.popleft()
                    self.rl_veh.append(rl_id)
                # specify observed vehicles
                for veh_id in self.leader + self.follower:
                    self.k.vehicle.set_observed(veh_id)

class MergePOEnvIgnoreAvgVel(MergePOEnvIgnore):
    def compute_reward(self, rl_actions, **kwargs):
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            reward = rewards.average_velocity(self)
            return reward/30

class MergePOEnvAvgVelEnExit(MergePOEnv):
    def compute_reward(self, rl_actions, **kwargs):
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            if len(self.k.vehicle.get_ids()) == 0:
                return 1.0
            reward = rewards.average_velocity(self)
            return reward/30


class MergePOEnvIgnoreAvgVelDistance(MergePOEnvIgnoreAvgVel):
    @property
    def observation_space(self):
        return Box(low=float('-inf'), high=float('inf'), shape=(6 * self.num_rl, ), dtype=np.float32)


    def get_state(self):
        state = super().get_state()
        observation = [0 for _ in range(6*self.num_rl)]
        for i,rl_id in enumerate(self.rl_veh):
            veh_pos = self.k.vehicle.get_position(rl_id)
            veh_x = self.k.vehicle.get_x_by_id(rl_id)
            center = self.network.specify_nodes(self.network.net_params)[2]
            center_x = center['x']+1000
            distance = (center_x - veh_x)/2000
            #print(rl_id,veh_pos,veh_x,center_x,distance)
            edge = self.k.vehicle.get_edge(rl_id)
            num_vehicles = len(self.k.vehicle.get_ids_by_edge(edge))
            length = self.k.network.edge_length(edge)
            vehicle_length = self.k.vehicle.get_length(rl_id)
            observation[6 * i + 0] = state[5*i+0]
            observation[6 * i + 1] = state[5*i+1]
            observation[6 * i + 2] = state[5*i+2]
            observation[6 * i + 3] = state[5*i+3]
            observation[6 * i + 4] = state[5*i+4]
            observation[6 * i + 5] = distance
        return observation

class MergePOEnvIgnoreAvgVelDistanceMergeInfo(MergePOEnvIgnoreAvgVel):
    @property
    def observation_space(self):
        return Box(low=-1, high=1, shape=(7 * self.num_rl, ), dtype=np.float32)


    def get_state(self):
        state = super().get_state()
        observation = [0 for _ in range(7*self.num_rl)]
        merge_vehs = self.k.vehicle.get_ids_by_edge("bottom")
        merge_dists = [self.k.vehicle.get_position(veh) for veh in merge_vehs]
        merge_distance = 1
        if len(merge_dists)>0:
            merge_distance = 0.5-max(merge_dists)/2000
        #print(merge_vehs, merge_dists)
        #print(merge_distance)
        
        for i,rl_id in enumerate(self.rl_veh):
            veh_pos = self.k.vehicle.get_position(rl_id)
            veh_x = self.k.vehicle.get_x_by_id(rl_id)
            center = self.network.specify_nodes(self.network.net_params)[2]
            center_x = center['x']+1000
            distance = (center_x - veh_x)/2000
            #print(rl_id,veh_pos,veh_x,center_x,distance)
            edge = self.k.vehicle.get_edge(rl_id)
            num_vehicles = len(self.k.vehicle.get_ids_by_edge(edge))
            length = self.k.network.edge_length(edge)
            vehicle_length = self.k.vehicle.get_length(rl_id)
            observation[7 * i + 0] = state[5*i+0]
            observation[7 * i + 1] = state[5*i+1]
            observation[7 * i + 2] = state[5*i+2]
            observation[7 * i + 3] = state[5*i+3]
            observation[7 * i + 4] = state[5*i+4]
            observation[7 * i + 5] = np.clip(distance,-1,1)
            observation[7 * i + 6] = np.clip(merge_distance,-1,1)
        return observation
    
class MergePOEnvScaleInflowIgnore(MergePOEnv):
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

            # penalize small time headways
            cost2 = 0
            t_min = 1  # smallest acceptable time headway
            for rl_id in self.rl_veh:
                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id not in ["", None] \
                        and self.k.vehicle.get_speed(rl_id) > 0:
                    t_headway = max(
                        self.k.vehicle.get_headway(rl_id) /
                        self.k.vehicle.get_speed(rl_id), 0)
                    cost2 += min((t_headway - t_min) / t_min, 0)
            if "max_inflow" in self.env_params.additional_params.keys():
                #print("max_inflow specified ")
                max_inflow = self.env_params.additional_params["max_inflow"]
            else:
                max_inflow = 2200
            InflowScale = rewards.optimize_inflow(self, max_flow=max_inflow,timespan=500)
            # weights for cost1, cost2, and cost3, respectively
            eta1, eta2 = 1.00, 0.00
            reward = max(eta1 * cost1 + eta2 * cost2, 0) * InflowScale
            return reward

    def additional_command(self):
            if 'ignore_edges' not in self.env_params.additional_params:
                super().additional_command()
            else:
                rl_ids = self.k.vehicle.get_rl_ids()
                # add rl vehicles that just entered the network into the rl queue
                for veh_id in rl_ids:
                    edge = self.k.vehicle.get_edge(veh_id) 
                    if (veh_id not in list(self.rl_queue)+self.rl_veh)\
                            and (edge not in self.env_params.additional_params['ignore_edges']):
                        self.rl_queue.append(veh_id)
                # remove rl vehicles that exited the network 
                for veh_id in list(self.rl_queue):
                    if veh_id not in rl_ids:
                        self.rl_queue.remove(veh_id)
                for veh_id in self.rl_veh:
                    if veh_id not in rl_ids:
                        self.rl_veh.remove(veh_id)
                # fil up rl_veh until they are enough controlled vehicles
                while len(self.rl_queue) > 0 and len(self.rl_veh) < self.num_rl:
                    rl_id = self.rl_queue.popleft()
                    self.rl_veh.append(rl_id)
                # specify observed vehicles
                for veh_id in self.leader + self.follower:
                    self.k.vehicle.set_observed(veh_id)

class MergePOEnvMinDelay(MergePOEnv):
    def compute_reward(self, rl_actions, **kwargs):
        reward = rewards.min_delay(self)
        return reward
'''
class MergePOEnvAvgVel(MergePOEnv):
    def compute_reward(self, rl_actions, **kwargs):
        reward = rewards.average_velocity(self)
        return reward/30
'''

class MergePOEnvSparseRewardDelay(MergePOEnv):
    def compute_reward(self, rl_actions, **kwargs):
        if 'max_num_vehicles' in self.env_params.additional_params.keys():
            max_num_vehicles = self.env_params.additional_params['max_num_vehicles']
            if max_num_vehicles > 0:
                if self.k.vehicle.get_num_arrived() >= max_num_vehicles:
                    return - self.time_counter
        
        if self.time_counter >= self.env_params.warmup_steps + self.env_params.horizon :
            return - self.time_counter

        return 0

class MergePOEnvPunishDelay(MergePOEnv):
    def compute_reward(self, rl_actions, **kwargs):
        return -0.1

class MergePOEnvIncludePotential(MergePOEnv):
    def compute_reward(self, rl_actions, **kwargs):
        if 'max_num_vehicles' in self.env_params.additional_params.keys():
            max_num_vehicles = self.env_params.additional_params['max_num_vehicles']
            if max_num_vehicles > 0:
                num_arrived = self.k.vehicle.get_num_arrived()
                num_remain = max_num_vehicles - num_arrived
                vel = self.k.vehicle.get_speed(self.k.vehicle.get_ids())
                vel_sum = np.sum(vel)# + num_remain * 30
                reward = vel_sum/(num_remain + 1e-6)
        else:
            reward = rewards.average_velocity(self)
        return reward

class MergePOEnvGuidedPunishDelay(MergePOEnv):
    def compute_reward(self, rl_actions, **kwargs):
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            # return a reward of 0 if a collision occurred
            if kwargs["fail"]:
                return 0

            # reward high system-level velocities
            cost1 = rewards.desired_velocity(self, fail=kwargs["fail"])

            # penalize small time headways
            cost2 = 0
            t_min = 1  # smallest acceptable time headway
            for rl_id in self.rl_veh:
                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id not in ["", None] \
                        and self.k.vehicle.get_speed(rl_id) > 0:
                    t_headway = max(
                        self.k.vehicle.get_headway(rl_id) /
                        self.k.vehicle.get_speed(rl_id), 0)
                    cost2 += min((t_headway - t_min) / t_min, 0)

            # weights for cost1, cost2, and cost3, respectively
            eta1, eta2 = 1.00, 0.00

            return max(eta1 * cost1 + eta2 * cost2, 0) - 1
  


class MergePOEnvEdgePrior(MergePOEnv):
    @property
    def observation_space(self):
        return Box(low=float('-inf'), high=float('inf'), shape=(6 * self.num_rl, ), dtype=np.float32)


    def get_state(self):
        state = super().get_state()
        observation = [0 for _ in range(6*self.num_rl)]
        for i,rl_id in enumerate(self.rl_veh):
            edge = self.k.vehicle.get_edge(rl_id)
            num_vehicles = len(self.k.vehicle.get_ids_by_edge(edge))
            length = self.k.network.edge_length(edge)
            vehicle_length = self.k.vehicle.get_length(rl_id)
            observation[6 * i + 0] = state[5*i+0]
            observation[6 * i + 1] = state[5*i+1]
            observation[6 * i + 2] = state[5*i+2]
            observation[6 * i + 3] = state[5*i+3]
            observation[6 * i + 4] = state[5*i+4]
            observation[6 * i + 5] = num_vehicles*vehicle_length/length
        return observation


#def factoryMergePORadiusEnv(obs_radius):
class MergePORadiusEnv(MergePOEnv):
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
