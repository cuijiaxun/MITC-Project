"""Benchmark for us_merge_200_6.

Trains a small percentage of autonomous vehicles to dissipate shockwaves caused
by merges in an open network. The autonomous penetration rate in this example
is 6%.

# Action Dimension: (3, )

# Observation Dimension: (15, )

Horizon: 750 steps
"""

from copy import deepcopy
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import SimCarFollowingController, RLController
import os

# time horizon of a single rollout
HORIZON = 750
# inflow rate at the highway
FLOW_RATE = 2000
# percent of autonomous vehicles
RL_PENETRATION = 0.06
# num_rl term (see ADDITIONAL_ENV_PARAMs)
NUM_RL = 3

# We consider a highway network with an upstream merging lane producing
# shockwaves
# is the following OK

# RL vehicles constitute 5% of the total number of vehicles
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
      speed_mode="no_collide",
    ),
    num_vehicles=5)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
      speed_mode="no_collide",
    ),
    num_vehicles=0)

# Vehicles are introduced from both sides of merge, with RL vehicles entering
# from the highway portion as well
inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="inflow_highway",
    vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE,
    departLane="free",
    departSpeed=10)
inflow.add(
    veh_type="rl",
    edge="inflow_highway",
    vehs_per_hour=RL_PENETRATION * FLOW_RATE,
    departLane="free",
    departSpeed=10)
inflow.add(
    veh_type="human",
    edge="inflow_merge",
    vehs_per_hour=200, #500, #200, #100
    departLane="free",
    departSpeed=7.5)

flow_params = dict(
    # name of the experiment
    exp_tag="us_merge_200_6_4",

    # name of the flow environment the experiment is running on
    env_name="WaveAttenuationMergePORadius4Env",

    # name of the scenario class the experiment is running on
    scenario="Scenario",

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
            "max_accel": 1.5,
            "max_decel": 1.5,
            "target_velocity": 20,
            "num_rl": NUM_RL,
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        template= {
          "net" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "us_merge.net.xml"),
          "rou" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "us_merge.rou.xml")
        },
        inflows=inflow,
        no_internal_links=False,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)
