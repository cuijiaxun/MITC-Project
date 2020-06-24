"""Benchmark for merge0.

Trains a small percentage of autonomous vehicles to dissipate shockwaves caused
by merges in an open network. The autonomous penetration rate in this example
is 10%.

- **Action Dimension**: (5, )
- **Observation Dimension**: (25, )
- **Horizon**: 750 steps
"""
from flow.envs import MergePOEnvGuidedPunishDelay
from flow.networks import MergeNetwork
from copy import deepcopy
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.networks.merge import ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams
from flow.controllers import SimCarFollowingController, RLController,IDMController

# time horizon of a single rollout
HORIZON = 6000
# inflow rate at the highway
FLOW_RATE = 2000
MERGE_RATE = 200
# percent of autonomous vehicles
RL_PENETRATION = 0.1
# num_rl term (see ADDITIONAL_ENV_PARAMs)
NUM_RL = 5
VEHICLE_NUMBER = 100
# We consider a highway network with an upstream merging lane producing
# shockwaves
additional_net_params = deepcopy(ADDITIONAL_NET_PARAMS)
additional_net_params["merge_lanes"] = 1
additional_net_params["highway_lanes"] = 1
additional_net_params["pre_merge_length"] = 500

# RL vehicles constitute 5% of the total number of vehicles
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=9,
    ),
    num_vehicles=0)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=9,
    ),
    num_vehicles=0)

# Vehicles are introduced from both sides of merge, with RL vehicles entering
# from the highway portion as well
inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="inflow_highway",
    vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE,
    number = round(FLOW_RATE/(FLOW_RATE+MERGE_RATE)*(1-RL_PENETRATION) * VEHICLE_NUMBER),
    depart_lane="free",
    depart_speed=10)
inflow.add(
    veh_type="rl",
    edge="inflow_highway",
    vehs_per_hour=RL_PENETRATION * FLOW_RATE,
    number = round(FLOW_RATE/(FLOW_RATE+MERGE_RATE)*RL_PENETRATION * VEHICLE_NUMBER),
    depart_lane="free",
    depart_speed=10)
inflow.add(
    veh_type="human",
    edge="inflow_merge",
    vehs_per_hour=MERGE_RATE,
    number = round(MERGE_RATE/(FLOW_RATE+MERGE_RATE)*VEHICLE_NUMBER),
    depart_lane="free",
    depart_speed=7.5)

flow_params = dict(
    # name of the experiment
    exp_tag="merge_4_Sim_Guided_PunishDelay_Number100_accelTrial_noheadway",

    # name of the flow environment the experiment is running on
    env_name=MergePOEnvGuidedPunishDelay,

    # name of the network class the experiment is running on
    network=MergeNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        restart_instance=True,
        sim_step=0.5,
        render=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        sims_per_step=2,
        warmup_steps=0,
        additional_params={
            "max_accel": 2.5,
            "max_decel": 4.5,
            "target_velocity": 30,
            "num_rl": NUM_RL,
            "max_num_vehicles":VEHICLE_NUMBER,
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