"""Benchmark for merge0.

Trains a small percentage of autonomous vehicles to dissipate shockwaves caused
by merges in an open network. The autonomous penetration rate in this example
is 10%.

- **Action Dimension**: (5, )
- **Observation Dimension**: (25, )
- **Horizon**: 750 steps
"""
import json

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.envs import MergePOEnvPunishDelay
from flow.networks import MergeNetwork
from copy import deepcopy
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.networks.merge import ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams
from flow.controllers import SimCarFollowingController, RLController,IDMController

# time horizon of a single rollout
HORIZON = 2000
# number of rollouts per training iteration
N_ROLLOUTS = 1
# number of parallel workers
N_CPUS = 1
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
    acceleration_controller=(SimCarFollowingController, {}),
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
    exp_tag="merge_4_ALLHUMAN_Sim_Number100_PunishDelay",

    # name of the flow environment the experiment is running on
    env_name=MergePOEnvPunishDelay,

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
def setup_exps():
    """Return the relevant components of an RLlib experiment.

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = N_CPUS
    config["train_batch_size"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [32, 32, 32]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["horizon"] = HORIZON

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, gym_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config

if __name__ == "__main__":
    alg_run, gym_name, config = setup_exps()
    ray.init(num_cpus=N_CPUS + 1,
    object_store_memory=1024*1024*1024)
    trials = run_experiments({
        flow_params["exp_tag"]: {
            "run": alg_run,
            "env": gym_name,
            "config": {
                **config
            },
            "checkpoint_freq": 1,
            "checkpoint_at_end": True,
            "max_failures": 999,
            "stop": {
                "training_iteration": 1,
            },
        }
    })
