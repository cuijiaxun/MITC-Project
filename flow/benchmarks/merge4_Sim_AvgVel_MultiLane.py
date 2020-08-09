"""Benchmark for merge0.

Trains a small percentage of autonomous vehicles to dissipate shockwaves caused
by merges in an open network. The autonomous penetration rate in this example
is 10%.

- **Action Dimension**: (5, )
- **Observation Dimension**: (25, )
- **Horizon**: 750 steps
"""
from flow.envs import MergePOEnvAvgVel
from flow.networks import MergeNetwork
from copy import deepcopy
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.networks.merge import ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams, SumoLaneChangeParams
from flow.controllers import SimCarFollowingController, RLController,IDMController,SimLaneChangeController

# time horizon of a single rollout
HORIZON = 1500
# inflow rate at the highway
FLOW_RATE = 2000
# percent of autonomous vehicles
RL_PENETRATION = 0.1
# num_rl term (see ADDITIONAL_ENV_PARAMs)
NUM_RL = 20

# We consider a highway network with an upstream merging lane producing
# shockwaves
additional_net_params = deepcopy(ADDITIONAL_NET_PARAMS)
additional_net_params["merge_lanes"] = 1
additional_net_params["highway_lanes"] = 2
additional_net_params["pre_merge_length"] = 500

# RL vehicles constitute 5% of the total number of vehicles
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {
        }
    ),
    lane_change_controller=(SimLaneChangeController,{}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=9,
    ),
    lane_change_params=SumoLaneChangeParams(
      #model="SL2015",
      lane_change_mode=1621,
      #lc_pushy=0,
      #lc_assertive=5,
      lc_impatience=1e-8,
      lc_time_to_impatience=1e12
    ), 
    num_vehicles=0)

vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    lane_change_controller=(SimLaneChangeController,{}),

    car_following_params=SumoCarFollowingParams(
        speed_mode=9,
    ),
    lane_change_params=SumoLaneChangeParams(
      #model="SL2015",
      lane_change_mode=1621,
      #lc_pushy=0,
      #lc_assertive=5,
      lc_impatience=1e-8,
      lc_time_to_impatience=1e12
    ), 
    num_vehicles=0)

# Vehicles are introduced from both sides of merge, with RL vehicles entering
# from the highway portion as well
inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="inflow_highway",
    vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE,
    depart_lane=0,#"first",#"free",
    depart_speed=10)
inflow.add(
    veh_type="rl",
    edge="inflow_highway",
    vehs_per_hour=RL_PENETRATION * FLOW_RATE,
    depart_lane=0,#"free",
    depart_speed=10)
inflow.add(
    veh_type="human",
    edge="inflow_merge",
    vehs_per_hour=200,
    depart_lane="first",#"free",
    depart_speed=7.5)

flow_params = dict(
    # name of the experiment
    exp_tag="merge_4_Sim_AvgVel_MultiLane",

    # name of the flow environment the experiment is running on
    env_name=MergePOEnvAvgVel,

    # name of the network class the experiment is running on
    network=MergeNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        restart_instance=True,
        sim_step=0.5,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        sims_per_step=2,
        warmup_steps=0,
        additional_params={
            "max_accel": 9,
            "max_decel": 9,
            "target_velocity": 30,
            "num_rl": NUM_RL,
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params=additional_net_params,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)
