"""Environment used to train vehicles to improve traffic on a highway."""
import numpy as np
from gym.spaces.box import Box
from flow.core.rewards import desired_velocity, average_velocity
from flow.envs.multiagent.base import MultiEnv
import collections

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 1,
    # maximum deceleration of autonomous vehicles
    'max_decel': 1,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25
}


class MultiAgentHighwayPOEnv(MultiEnv):
    """Partially observable multi-agent environment for an highway with ramps.

    This environment is used to train autonomous vehicles to attenuate the
    formation and propagation of waves in an open highway network.

    The highway can contain an arbitrary number of entrance and exit ramps, and
    is intended to be used with the HighwayRampsNetwork network.

    The policy is shared among the agents, so there can be a non-constant
    number of RL vehicles throughout the simulation.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s

    The following states, actions and rewards are considered for one autonomous
    vehicle only, as they will be computed in the same way for each of them.

    States
        The observation consists of the speeds and bumper-to-bumper headways of
        the vehicles immediately preceding and following autonomous vehicle, as
        well as the speed of the autonomous vehicle.

    Actions
        The action consists of an acceleration, bound according to the
        environment parameters, as well as three values that will be converted
        into probabilities via softmax to decide of a lane change (left, none
        or right).

    Rewards
        The reward function encourages proximity of the system-level velocity
        to a desired velocity specified in the environment parameters, while
        slightly penalizing small time headways among autonomous vehicles.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(-float('inf'), float('inf'), shape=(5,), dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1,),  # (4,),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # in the warmup steps, rl_actions is None
        if rl_actions:
            for rl_id, actions in rl_actions.items():
                accel = actions[0]

                # lane_change_softmax = np.exp(actions[1:4])
                # lane_change_softmax /= np.sum(lane_change_softmax)
                # lane_change_action = np.random.choice([-1, 0, 1],
                #                                       p=lane_change_softmax)

                self.k.vehicle.apply_acceleration(rl_id, accel)
                # self.k.vehicle.apply_lane_change(rl_id, lane_change_action)

    def get_state(self):
        """See class definition."""
        obs = {}

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        for rl_id in self.k.vehicle.get_rl_ids():
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)

            if lead_id in ["", None]:
                # in case leader is not visible
                lead_speed = max_speed
                lead_head = max_length
            else:
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_headway(rl_id)

            if follower in ["", None]:
                # in case follower is not visible
                follow_speed = 0
                follow_head = max_length
            else:
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)
            
            observation = np.array([
                this_speed / max_speed,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_length,
                (this_speed - follow_speed) / max_speed,
                follow_head / max_length
            ])
            '''
            observation = np.array([
                this_speed / max_speed,
                lead_speed / max_speed,
                lead_head / max_length,
                follow_speed / max_speed,
                follow_head / max_length
            ])
            '''

            obs.update({rl_id: observation})

        return obs
    
    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rewards = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            if self.env_params.evaluate:
                # reward is speed of vehicle if we are in evaluation mode
                reward = self.k.vehicle.get_speed(rl_id)
            elif kwargs['fail']:
                # reward is 0 if a collision occurred
                reward = 0
            else:
                # reward high system-level velocities
                cost1 = desired_velocity(self, fail=kwargs['fail'])

                # penalize small time headways
                cost2 = 0
                t_min = 1  # smallest acceptable time headway

                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id not in ["", None] \
                        and self.k.vehicle.get_speed(rl_id) > 0:
                    t_headway = max(
                        self.k.vehicle.get_headway(rl_id) /
                        self.k.vehicle.get_speed(rl_id), 0)
                    cost2 += min((t_headway - t_min) / t_min, 0)

                # weights for cost1, cost2, and cost3, respectively
                eta1, eta2 = 1.00, 0.00

                reward = max(eta1 * cost1 + eta2 * cost2, 0)

            rewards[rl_id] = reward
        return rewards

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes.
        """
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            # leader
            lead_id = self.k.vehicle.get_leader(rl_id)
            if lead_id:
                self.k.vehicle.set_observed(lead_id)
            # follower
            follow_id = self.k.vehicle.get_follower(rl_id)
            if follow_id:
                self.k.vehicle.set_observed(follow_id)

class MultiAgentHighwayPOEnvWindow(MultiAgentHighwayPOEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))
        self.rl_queue = collections.deque()
        self.rl_veh = []
        self.exited_rl_veh = []
        self.exiting_rl_veh = []
        self.leader = []
        self.follower = []

        super().__init__(env_params, sim_params, network, simulator)

    def step(self, rl_actions):
        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # perform acceleration actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_ids()) > 0:
                accel = []
                for veh_id in self.k.vehicle.get_controlled_ids():
                    accel_contr = self.k.vehicle.get_acc_controller(veh_id)
                    action = accel_contr.get_action(self)
                    accel.append(action)
                self.k.vehicle.apply_acceleration(
                    self.k.vehicle.get_controlled_ids(), accel)

            # perform lane change actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_lc_ids()) > 0:
                direction = []
                for veh_id in self.k.vehicle.get_controlled_lc_ids():
                    target_lane = self.k.vehicle.get_lane_changing_controller(
                        veh_id).get_action(self)
                    direction.append(target_lane)
                self.k.vehicle.apply_lane_change(
                    self.k.vehicle.get_controlled_lc_ids(),
                    direction=direction)

            # perform (optionally) routing actions for all vehicle in the
            # network, including rl and sumo-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(veh_id)
                    routing_actions.append(route_contr.choose_route(self))
            self.k.vehicle.choose_routes(routing_ids, routing_actions)

            self.apply_rl_actions(rl_actions)

            self.additional_command()

            # advance the simulation in the simulator by one step
            self.k.simulation.simulation_step()

            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)

            # update the colors of vehicles
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()

            # crash encodes whether the simulator experienced a collision
            crash = self.k.simulation.check_collision()

            # stop collecting new simulation steps if there is a collision
            if crash:
                break

        states = self.get_state()
        done = {key: key in self.k.vehicle.get_arrived_ids()
                for key in states.keys()}
        if crash:
            done['__all__'] = True
        else:
            done['__all__'] = False

        infos = {key: {} for key in states.keys()}

        # compute the reward
        if self.env_params.clip_actions:
            clipped_actions = self.clip_actions(rl_actions)
            reward = self.compute_reward(clipped_actions, fail=crash)
        else:
            reward = self.compute_reward(rl_actions, fail=crash)

        for rl_id in self.exiting_rl_veh: #self.k.vehicle.get_arrived_rl_ids():
            #done[rl_id] = True
            reward[rl_id] = 20 #1 #0
            #states[rl_id] = np.zeros(self.observation_space.shape[0])
            #print("rl_id",rl_id, states)
        for rl_id in self.k.vehicle.get_arrived_rl_ids():
            #print("arrived:",rl_id)
            done[rl_id] = True
            reward[rl_id] = 0
            states[rl_id] = np.zeros(self.observation_space.shape[0])

        return states, reward, done, infos
    
    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # in the warmup steps, rl_actions is None
        if rl_actions:
            for rl_id, actions in rl_actions.items():
                accel = actions[0]
                if rl_id not in self.rl_veh:
                    continue
                # lane_change_softmax = np.exp(actions[1:4])
                # lane_change_softmax /= np.sum(lane_change_softmax)
                # lane_change_action = np.random.choice([-1, 0, 1],
                #                                       p=lane_change_softmax)
                self.k.vehicle.apply_acceleration(rl_id, accel)
                # self.k.vehicle.apply_lane_change(rl_id, lane_change_action)
    def get_state(self):
        """See class definition."""
        obs = {}

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        for rl_id in self.k.vehicle.get_rl_ids():
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)

            if lead_id in ["", None]:
                # in case leader is not visible
                lead_speed = max_speed
                lead_head = max_length
            else:
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_headway(rl_id)

            if follower in ["", None]:
                # in case follower is not visible
                follow_speed = 0
                follow_head = max_length
            else:
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)
            
            observation = np.array([
                this_speed / max_speed,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_length,
                (this_speed - follow_speed) / max_speed,
                follow_head / max_length
            ])
            '''
            observation = np.array([
                this_speed / max_speed,
                lead_speed / max_speed,
                lead_head / max_length,
                follow_speed / max_speed,
                follow_head / max_length
            ])
            '''

            obs.update({rl_id: observation})

        return obs

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes.
        """
        if 'ignore_edges' not in self.env_params.additional_params:
                super().additional_command()
        else:
                rl_ids = self.k.vehicle.get_rl_ids()
                self.exiting_rl_veh = []
                # add rl vehicles that just entered the network into the rl queue
                for veh_id in rl_ids:
                    edge = self.k.vehicle.get_edge(veh_id)
                    if (veh_id not in list(self.rl_queue)+self.rl_veh+self.exited_rl_veh)\
                            and (edge not in self.env_params.additional_params['ignore_edges']):
                        self.rl_queue.append(veh_id)

                    elif veh_id in self.rl_veh and edge in self.env_params.additional_params['ignore_edges']:
                        self.rl_veh.remove(veh_id)
                        self.exited_rl_veh.append(veh_id)
                        self.exiting_rl_veh.append(veh_id)

                # remove rl vehicles that exited the network
                for veh_id in list(self.rl_queue):
                    if veh_id not in rl_ids:
                        self.rl_queue.remove(veh_id)
                for veh_id in self.rl_veh:
                    if veh_id not in rl_ids:
                        self.rl_veh.remove(veh_id)
                # fil up rl_veh until they are enough controlled vehicles
                while len(self.rl_queue) > 0:
                    rl_id = self.rl_queue.popleft()
                    self.rl_veh.append(rl_id)
                # specify observed vehicles
                for rl_id in self.rl_veh:
                    # leader
                    lead_id = self.k.vehicle.get_leader(rl_id)
                    if lead_id:
                        self.k.vehicle.set_observed(lead_id)
                    # follower
                    follow_id = self.k.vehicle.get_follower(rl_id)
                    if follow_id:
                        self.k.vehicle.set_observed(follow_id)
                #print(self.exiting_rl_veh)



    def reset(self):
        self.rl_queue = collections.deque()
        self.rl_veh = []
        self.exited_rl_veh = []
        self.leader = []
        self.follower = []
        self.exiting_rl_id = []
        return super().reset()

class MultiAgentHighwayPOEnvWindowCollaborate(MultiAgentHighwayPOEnvWindow):
    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        rewards = {}
        eta1 = 0.9
        eta2 = 0.1
        reward1 = -0.1
        current_rl_vehs = self.rl_veh
        edges = []
        for veh_id in current_rl_vehs:
            edge = self.k.vehicle.get_edge(veh_id)
            if edge not in edges:
                edges.append(edge)
        interested_vehs = self.k.vehicle.get_ids_by_edge(edges)
        if len(interested_vehs) >0:
            reward2 = np.mean(self.k.vehicle.get_speed(interested_vehs))/300
        else:
            reward2 = 0

        reward  = reward1 * eta1 + reward2 * eta2
        for rl_id in self.k.vehicle.get_rl_ids():
            rewards[rl_id] = reward
        return rewards

class MultiAgentHighwayPOEnvAvgVel(MultiAgentHighwayPOEnv):
  
    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rewards = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            if self.env_params.evaluate:
                # reward is speed of vehicle if we are in evaluation mode
                reward = self.k.vehicle.get_speed(rl_id)
            elif kwargs['fail']:
                # reward is 0 if a collision occurred
                reward = 0
            else:
                # reward high system-level velocities
                cost1 = desired_velocity(self, fail=kwargs['fail'])

                # penalize small time headways
                cost2 = 0
                t_min = 1  # smallest acceptable time headway

                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id not in ["", None] \
                        and self.k.vehicle.get_speed(rl_id) > 0:
                    t_headway = max(
                        self.k.vehicle.get_headway(rl_id) /
                        self.k.vehicle.get_speed(rl_id), 0)
                    cost2 += min((t_headway - t_min) / t_min, 0)

                # weights for cost1, cost2, and cost3, respectively
                eta1, eta2 = 1.00, 0.00

                reward = max(eta1 * cost1 + eta2 * cost2, 0)

            rewards[rl_id] = reward
        return rewards

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes.
        """
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            # leader
            lead_id = self.k.vehicle.get_leader(rl_id)
            if lead_id:
                self.k.vehicle.set_observed(lead_id)
            # follower
            follow_id = self.k.vehicle.get_follower(rl_id)
            if follow_id:
                self.k.vehicle.set_observed(follow_id)


class MultiAgentHighwayPOEnvLocalReward(MultiAgentHighwayPOEnv):
    def _veh_edge_lane(self, edge, lane):
        return [veh for veh in self.k.vehicle.get_ids_by_edge(edge) if self.k.vehicle.get_lane(veh) == lane]

    def _veh_edge_lane_backward_pass(self, edge, lane, junctions):
        veh = []
        for prev_edge, prev_lane in self.k.network.prev_edge(edge, lane):
            if prev_edge in junctions:
                veh += self._veh_edge_lane_backward_pass(prev_edge, prev_lane, junctions)
            veh += self._veh_edge_lane(prev_edge, prev_lane)
        return veh

    def _compute_avgspeed_agent(self, rl_id, **kwargs):
        if kwargs["fail"]:
            return 0 
        edge = self.k.vehicle.get_edge(rl_id)
        lane = self.k.vehicle.get_lane(rl_id)
        pos = self.k.vehicle.get_position(edge)
        edge_veh = self.k.vehicle.get_ids_by_edge(edge)
        junctions = set(self.k.network.get_junction_list())
        neighbours = []
        for veh in edge_veh:
            veh_lane = self.k.vehicle.get_lane(veh)
            veh_pos = self.k.vehicle.get_position(veh)
            if veh_lane == lane and veh_pos <= pos:
                neighbours.append(veh)
        neighbours += self._veh_edge_lane_backward_pass(edge, lane, junctions)
        neighbours = np.array(neighbours)
        if len(neighbours) == 0:
            return 0
        vel = np.array(self.k.vehicle.get_speed(neighbours))
        if any(vel<-100):
            return 0
        return np.sum(vel)/len(vel)

    def _compute_avgspeednormalized_agent(self, rl_id, **kwargs):
        reward = self._compute_avgspeed_agent(rl_id, **kwargs)
        return reward/self.k.network.speed_limit(self.k.vehicle.get_edge(rl_id))

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}
        rewards = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            rewards[rl_id] = self._compute_avgspeednormalized_agent(rl_id, **kwargs)
        return rewards


class MultiAgentHighwayPOEnvDistanceMergeInfo(MultiAgentHighwayPOEnv):
    @property
    def observation_space(self):
        return Box(low=-1, high=1, shape=(7, ), dtype=np.float32)

    def get_state(self):
        """See class definition."""
        obs = {}

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()
        merge_vehs = self.k.vehicle.get_ids_by_edge("bottom")
        merge_dists = [self.k.vehicle.get_position(veh) for veh in merge_vehs]
        merge_distance = 1
        len_bottom = self.k.network.edge_length("bottom")
        position = self.k.network.total_edgestarts_dict["bottom"]
        if len(merge_dists)>0:
            position = max(merge_dists)
            merge_distance = (len_bottom - position)/len_bottom

        for rl_id in self.k.vehicle.get_rl_ids():
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)

            if lead_id in ["", None]:
                # in case leader is not visible
                lead_speed = max_speed
                lead_head = max_length
            else:
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_headway(rl_id)

            if follower in ["", None]:
                # in case follower is not visible
                follow_speed = 0
                follow_head = max_length
            else:
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)
            
            veh_x = self.k.vehicle.get_x_by_id(rl_id)
            edge = self.k.vehicle.get_edge(rl_id)
            length = self.k.network.edge_length(edge)
            center_x = self.k.network.total_edgestarts_dict["center"]
            distance = 1
            if edge in ["inflow_highway","left","center"]:
                distance = (veh_x - center_x)/(center_x)
            else:
                pass #FIXME implement

            observation = np.array([
                this_speed / max_speed,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_length,
                (this_speed - follow_speed) / max_speed,
                follow_head / max_length,
                np.clip(distance,-1,1),
                np.clip(merge_distance,-1,1),

            ])

            obs.update({rl_id: observation})

        return obs

class MultiAgentHighwayPOEnvNewStates(MultiAgentHighwayPOEnv):
    @property
    def observation_space(self):
        #See class definition
        return Box(-float('inf'), float('inf'), shape=(9,), dtype=np.float32)

    def _closest_vehicle(self, edge, lane, base_edge):
        if edge == base_edge: return float('inf'), 0
        if edge == '': return float('inf'), 0
        veh = self.k.vehicle.get_ids_by_edge(edge)
        if len(veh) == 0:
            veh_pos = 0
            veh_id = None
        else:
            veh_ind = np.argmax(self.k.vehicle.get_position(veh))
            veh_id = veh[veh_ind]
            veh_pos = self.k.vehicle.get_position(veh_id)
        veh_dist = self.k.network.edge_length(edge)-veh_pos
        if veh_id:
            veh_speed = self.k.vehicle.get_speed(veh_id)
        else:
            veh_speed = 0
        return veh_dist, veh_speed

    def _merging_vehicle_backward_pass(self, edge, lane, base_edge, junctions):
        try:
            return min(self._merging_vehicle_backward_pass(e, l, base_edge, junctions) if e in junctions else self._closest_vehicle(e, l, base_edge)
                    for e,l in self.k.network.prev_edge(edge, lane))
        except ValueError:
            return float('inf'), 0

    def _merging_vehicle_forward_pass(self, edge, lane, base_edge, junctions):
        try:
            return min(self._merging_vehicle_forward_pass(e, l, base_edge, junctions) if e in junctions else self._merging_vehicle_backward_pass(e, l, base_edge, junctions)
                    for e,l in self.k.network.next_edge(edge, lane))
        except ValueError:
            return float('inf'), 0

    def get_state(self):
        states = super().get_state()
        junctions = set(self.k.network.get_junction_list())
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()
        for rl_id in states:
            edge_id = self.k.vehicle.get_edge(rl_id)
            lane = self.k.vehicle.get_lane(rl_id)
            edge_len = self.k.network.edge_length(edge_id)
            rl_position = self.k.vehicle.get_position(rl_id)
            rl_dist = max(edge_len-rl_position, 0) / max_length
            veh_vel = []
            for veh_id in self.k.vehicle.get_ids_by_edge(edge_id):
                veh_position = self.k.vehicle.get_position(veh_id)
                if veh_position > rl_position:
                    veh_vel.append(self.k.vehicle.get_speed(veh_id))
            if len(veh_vel) > 0:
                veh_vel = np.mean(veh_vel)
            else:
                veh_vel = self.k.network.speed_limit(edge_id)
            veh_vel /= max_speed
            merge_dist, merge_vel = self._merging_vehicle_forward_pass(edge_id, lane, edge_id, junctions)
            merge_dist /= max_length
            merge_vel /= max_speed
            if merge_dist == float('inf'):
                merge_dist = 1
            states[rl_id] = np.array(list(states[rl_id]) + [rl_dist, veh_vel, merge_dist, merge_vel])
        #print(states)
        return states

class MultiAgentHighwayPOEnvNewStatesNegative(MultiAgentHighwayPOEnvNewStates):
    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        rewards = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            self_speed = self.k.vehicle.get_speed(rl_id)
            reward = -0.1
            #prevent RL stop
            if self_speed < 1:
                reward = -0.15
            rewards[rl_id] = reward
        return rewards

class MultiAgentHighwayPOEnvNewStatesZero(MultiAgentHighwayPOEnvNewStates):
    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        rewards = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            rewards[rl_id] = 0
        return rewards

class MultiAgentHighwayPOEnvNewStatesNegativeInflow(MultiAgentHighwayPOEnvNewStates):
    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        rewards = {}
        inflow_reward = self.k.vehicle._num_departed[-1]*0.1
        alpha = 0.5
        for rl_id in self.k.vehicle.get_rl_ids():
            rewards[rl_id] = -0.1 + inflow_reward
        return rewards

class MultiAgentHighwayPOEnvNewStatesCollaborate(MultiAgentHighwayPOEnvNewStates):
    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        rewards = {}
        eta1 = 0.9
        eta2 = 0.1
        reward1 = -0.1
        reward2 = average_velocity(self)/300
        reward  = reward1 * eta1 + reward2 * eta2
        for rl_id in self.k.vehicle.get_rl_ids():
            rewards[rl_id] = reward
        return rewards



class MultiAgentHighwayPOEnvDistanceMergeInfoNegative(MultiAgentHighwayPOEnvDistanceMergeInfo):
    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        rewards = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            reward = -0.1
            rewards[rl_id] = reward
        return rewards

class MultiAgentHighwayPOEnvDistanceMergeInfoCollaborate(MultiAgentHighwayPOEnvDistanceMergeInfo):
    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        rewards = {}
        eta1 = 0.5
        eta2 = 0.5
        reward1 = -0.1
        reward2 = average_velocity(self)/300
        reward  = reward1 * eta1 + reward2 * eta2
        for rl_id in self.k.vehicle.get_rl_ids():
            rewards[rl_id] = reward
        return rewards
                   
class MultiAgentHighwayPOEnvNegative(MultiAgentHighwayPOEnv):
    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        rewards = {}
        for rl_id in self.k.vehicle.get_rl_ids(): 
            reward = -0.1
            rewards[rl_id] = reward
        return rewards


class MultiAgentHighwayPOEnvCollaborate(MultiAgentHighwayPOEnv):
    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}
        rewards = {}
        eta1 = 0.9
        eta2 = 0.1
        reward1 = -0.1
        reward2 = average_velocity(self)/300
        reward  = reward1 * eta1 + reward2 * eta2
        for rl_id in self.k.vehicle.get_rl_ids():
            rewards[rl_id] = reward
        return rewards

class MultiAgentHighwayPOEnvMerge4(MultiAgentHighwayPOEnv):
    @property
    def observation_space(self):
        #See class definition
        return Box(-float('inf'), float('inf'), shape=(9,), dtype=np.float32)

    def _closest_vehicle(self, edge, lane, base_edge):
        if edge == base_edge: return float('inf'), 0
        if edge == '': return float('inf'), 0
        veh = self.k.vehicle.get_ids_by_edge(edge)
        if len(veh) == 0:
            veh_pos = 0
            veh_id = None
        else:
            veh_ind = np.argmax(self.k.vehicle.get_position(veh))
            veh_id = veh[veh_ind]
            veh_pos = self.k.vehicle.get_position(veh_id)
        veh_dist = self.k.network.edge_length(edge)-veh_pos
        if veh_id:
            veh_speed = self.k.vehicle.get_speed(veh_id)
        else:
            veh_speed = 0
        return veh_dist, veh_speed

    def _merging_vehicle_backward_pass(self, edge, lane, base_edge, junctions):
        try:
            return min(self._merging_vehicle_backward_pass(e, l, base_edge, junctions) if e in junctions else self._closest_vehicle(e, l, base_edge)
                    for e,l in self.k.network.prev_edge(edge, lane))
        except ValueError:
            return float('inf'), 0

    def _merging_vehicle_forward_pass(self, edge, lane, base_edge, junctions):
        try:
            return min(self._merging_vehicle_forward_pass(e, l, base_edge, junctions) if e in junctions else self._merging_vehicle_backward_pass(e, l, base_edge, junctions)
                    for e,l in self.k.network.next_edge(edge, lane))
        except ValueError:
            return float('inf'), 0

    def get_state(self):
        states = super().get_state()
        junctions = set(self.k.network.get_junction_list())
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()
        merge_vehs = self.k.vehicle.get_ids_by_edge(["bottom","inflow_merge"])
        #merge_dists = [self.k.vehicle.get_x(veh) for veh in merge_vehs]
        merge_distance = 1
        len_merge = self.k.network.edge_length("bottom") + self.k.network.edge_length("inflow_merge")
        start_position = self.k.network.total_edgestarts_dict["inflow_merge"]
        merge_vel = 0
        if len(merge_vehs)>0:
            for veh in merge_vehs:
                merge_dist = (len_merge - (self.k.vehicle.get_x_by_id(veh) - start_position))/len_merge
                if merge_dist < merge_distance:
                    merge_distance = merge_dist
                    merge_vel = self.k.vehicle.get_speed(veh)/max_speed
                
        
        for rl_id in states:
            edge_id = self.k.vehicle.get_edge(rl_id)
            lane = self.k.vehicle.get_lane(rl_id)
            edge_len = self.k.network.edge_length(edge_id)
            rl_position = self.k.vehicle.get_position(rl_id)
            rl_x = self.k.vehicle.get_x_by_id(rl_id)
            #rl_dist = max(edge_len-rl_position, 0) / max_length
            veh_vel = []
            
            #calculate RL distance to the center junction
            veh_x = self.k.vehicle.get_x_by_id(rl_id)
            edge = self.k.vehicle.get_edge(rl_id)
            length = self.k.network.edge_length(edge)
            center_x = self.k.network.total_edgestarts_dict["center"]
            rl_dist = 1
            if edge in ["inflow_highway","left","center"]:
                rl_dist = (veh_x - center_x)/(center_x)
            else:
                pass #FIXME: not yet implemented
            num_veh_ahead = 0 
            for veh_id in self.k.vehicle.get_ids_by_edge(["left","inflow_highway"]):
                veh_position = self.k.vehicle.get_x_by_id(veh_id)
                if veh_position > rl_x:
                    veh_vel.append(self.k.vehicle.get_speed(veh_id))
                    num_veh_ahead += 1
            if len(veh_vel) > 0:
                veh_vel = np.mean(veh_vel)
            else:
                veh_vel = self.k.network.speed_limit(edge_id)
            veh_vel /= max_speed
            
            if edge in ["center"]:
                states[rl_id] = np.array(list(states[rl_id]) + [rl_dist, veh_vel, 1.0, 0.0])
            else:
                states[rl_id] = np.array(list(states[rl_id]) + [rl_dist, veh_vel, merge_distance, merge_vel])
        #print(states)
        return states

class MultiAgentHighwayPOEnvMerge4Negative(MultiAgentHighwayPOEnvMerge4):
    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        rewards = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            self_speed = self.k.vehicle.get_speed(rl_id)
            reward = -0.1
            rewards[rl_id] = reward
        return rewards

class MultiAgentHighwayPOEnvMerge4Collaborate(MultiAgentHighwayPOEnvMerge4):
    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        rewards = {}
        eta1 = 0.9
        eta2 = 0.1
        reward1 = -0.1
        reward2 = average_velocity(self)/300
        reward  = reward1 * eta1 + reward2 * eta2
        for rl_id in self.k.vehicle.get_rl_ids():
            rewards[rl_id] = reward
        return rewards

class MultiAgentHighwayPOEnvAblationDistance(MultiAgentHighwayPOEnvMerge4):
    @property
    def observation_space(self):
        #See class definition
        return Box(-float('inf'), float('inf'), shape=(8,), dtype=np.float32)

    def get_state(self):
        states = super().get_state()
        for rl_id in states:
            edge = self.k.vehicle.get_edge(rl_id)
            state_rl_id = list(states[rl_id])
            state_rl_id.pop(5)
            states[rl_id] = state_rl_id
        return states

class MultiAgentHighwayPOEnvAblationDistanceCollaborate(MultiAgentHighwayPOEnvAblationDistance):
    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        rewards = {}
        eta1 = 0.9
        eta2 = 0.1
        reward1 = -0.1
        reward2 = average_velocity(self)/300
        reward  = reward1 * eta1 + reward2 * eta2
        for rl_id in self.k.vehicle.get_rl_ids():
            rewards[rl_id] = reward
        return rewards

class MultiAgentHighwayPOEnvAblationConjestion(MultiAgentHighwayPOEnvMerge4):
    @property
    def observation_space(self):
        #See class definition
        return Box(-float('inf'), float('inf'), shape=(8,), dtype=np.float32)

    def get_state(self):
        states = super().get_state()
        for rl_id in states:
            edge = self.k.vehicle.get_edge(rl_id)
            state_rl_id = list(states[rl_id])
            state_rl_id.pop(6)
            states[rl_id] = state_rl_id
        return states

class MultiAgentHighwayPOEnvAblationConjestionCollaborate(MultiAgentHighwayPOEnvAblationConjestion):
    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        rewards = {}
        eta1 = 0.9
        eta2 = 0.1
        reward1 = -0.1
        reward2 = average_velocity(self)/300
        reward  = reward1 * eta1 + reward2 * eta2
        for rl_id in self.k.vehicle.get_rl_ids():
            rewards[rl_id] = reward
        return rewards

class MultiAgentHighwayPOEnvAblationConjestionArrive(MultiAgentHighwayPOEnvAblationConjestion):
    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}
        if(len(self.k.vehicle._num_arrived)>0):
            reward = self.k.vehicle._num_arrived[-1] * 0.05
        else:
            reward = 0

        rewards = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            rewards[rl_id] = reward
        return rewards

class MultiAgentHighwayPOEnvAblationMergeInfo(MultiAgentHighwayPOEnvMerge4):
    @property
    def observation_space(self):
        #See class definition
        return Box(-float('inf'), float('inf'), shape=(7,), dtype=np.float32)

    def get_state(self):
        states = super().get_state()
        for rl_id in states:
            edge = self.k.vehicle.get_edge(rl_id)
            state_rl_id = list(states[rl_id])
            state_rl_id.pop(-1)
            state_rl_id.pop(-1)
            states[rl_id] = state_rl_id
        return states

class MultiAgentHighwayPOEnvAblationMergeInfoCollaborate(MultiAgentHighwayPOEnvAblationMergeInfo):
    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        rewards = {}
        eta1 = 0.9
        eta2 = 0.1
        reward1 = -0.1
        reward2 = average_velocity(self)/300
        reward  = reward1 * eta1 + reward2 * eta2
        for rl_id in self.k.vehicle.get_rl_ids():
            rewards[rl_id] = reward
        return rewards
