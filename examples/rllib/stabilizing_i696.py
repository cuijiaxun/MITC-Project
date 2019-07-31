"""Open merge example.

Trains a a small percentage of rl vehicles to dissipate shockwaves caused by
merges in an open network.
"""
import json
import os

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.scenarios.merge import ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams, SumoLaneChangeParams
from flow.controllers import IDMController, RLController, SimLaneChangeController, ContinuousRouter

# TODO hard coded
scenarios_dir = os.path.join(os.path.expanduser("~/"), 'local', 'flow_2019_07', 'flow', 'scenarios')

# experiment number
# - 0: 10% RL penetration,  5 max controllable vehicles
# - 1: 25% RL penetration, 13 max controllable vehicles
# - 2: 33% RL penetration, 17 max controllable vehicles
EXP_NUM = 0

# time horizon of a single rollout
HORIZON = 600
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 2

# inflow rate at the highway
FLOW_RATE = 2000
# percent of autonomous vehicles
RL_PENETRATION = [0.1, 0.25, 0.33][EXP_NUM]
# num_rl term (see ADDITIONAL_ENV_PARAMs)
NUM_RL = [5, 13, 17][EXP_NUM]

## We consider a highway network with an upstream merging lane producing
# shockwaves
additional_net_params = ADDITIONAL_NET_PARAMS.copy()
#additional_net_params["merge_lanes"] = 1
#additional_net_params["highway_lanes"] = 1
#additional_net_params["pre_merge_length"] = 500

# RL vehicles constitute 5% of the total number of vehicles
# Daniel: adding vehicles and flow from osm.passenger.trips.xml
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
      speed_mode="no_collide",
      min_gap=2.5,
      speed_factor=1,
      speed_dev=0.1
    ),
    lane_change_params=SumoLaneChangeParams(
      model="SL2015",
      lc_speed_gain=1000000,
      lc_pushy=1,
      lc_assertive=20,
      #lc_impatience=1,
    ), 
    num_vehicles=5)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
      speed_mode="no_collide",
      min_gap=2.5,
      speed_factor=1,
      speed_dev=0.1
    ),
    lane_change_params=SumoLaneChangeParams(
      model="SL2015",
      lc_speed_gain=1000000,
      lc_pushy=1,
      lc_assertive=20,
      #lc_impatience=1,
    ), 
    num_vehicles=1)

# Vehicles are introduced from both sides of merge, with RL vehicles entering
# from the highway portion as well
inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="404969345#0", # flow id sw2w1 from xml file
    begin=0,
    end=90000,
    probability=(1 - RL_PENETRATION), #* FLOW_RATE,
    departSpeed="max",
    departLane="free",
    departPos="free",
    )
inflow.add(
    veh_type="rl",
    edge="404969345#0", # flow id sw2w1 from xml file
    begin=0,
    end=90000,
    probability=RL_PENETRATION, #* FLOW_RATE,
    departSpeed="max",
    departLane="free",
    departPos="free",
    )
inflow.add(
    veh_type="human",
    edge="59440544#0", # flow id se2w1 from xml file
    begin=0,
    end=90000,
    probability=(1 - RL_PENETRATION), #* FLOW_RATE,
    departSpeed="max",
    departLane="free",
    departPos="free",
    )
inflow.add(
    veh_type="rl",
    edge="59440544#0", # flow id se2w1 from xml file
    begin=0,
    end=90000,
    probability=RL_PENETRATION, #* FLOW_RATE,
    departSpeed="max",
    departLane="free",
    departPos="free",
    )
inflow.add(
    veh_type="human",
    edge="124433709", # flow id e2w1 from xml file
    begin=0,
    end=90000,
    probability=(1 - RL_PENETRATION), #* FLOW_RATE,
    departSpeed="max",
    departLane="free",
    departPos="free",
    )
inflow.add(
    veh_type="rl",
    edge="124433709", # flow id e2w1 from xml file
    begin=0,
    end=90000,
    probability=RL_PENETRATION, # * FLOW_RATE,
    departSpeed="max",
    departLane="free",
    departPos="free",
    )
inflow.add(
    veh_type="human",
    edge="38726647", # flow id n2w1 from xml file
    begin=0,
    end=90000,
    probability=(1 - RL_PENETRATION), # * FLOW_RATE,
    departSpeed="max",
    departLane="free",
    departPos="free",
    )
inflow.add(
    veh_type="rl",
    edge="38726647", # flow id n2w1 from xml file
    begin=0,
    end=90000,
    probability=RL_PENETRATION, # * FLOW_RATE,
    departSpeed="max",
    departLane="free",
    departPos="free",
    )

flow_params = dict(
    # name of the experiment
    exp_tag="stabilizing_i696",

    # name of the flow environment the experiment is running on
    env_name="WaveAttenuationMergePOEnv",

    # name of the scenario class the experiment is running on
    #scenario="i696Scenario",
    scenario="Scenario",

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.5,            # Daniel updated from osm.sumocfg
        #lateral_resolution=0.25, # Daniel added from osm.sumocfg
        render=False,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        sims_per_step=1, #5,
        warmup_steps=0,
        additional_params={
            "max_accel": 1.5,
            "max_decel": 1.5,
            "target_velocity": 20,
            "num_rl": NUM_RL, # used by WaveAttenuationMergePOEnv e.g. to fix action dimension
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params,
        # TODO this is not moduar, since i696Scenario in another file expects this
        template={
          "net" : os.path.join(scenarios_dir, 'i696', 'osm.net.xml'), # Daniel added to load i696 net from file
          "rou" : [os.path.join(scenarios_dir, 'i696', 'i696.rou.xml')]
        }
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
      # Daniel: distributing only at the beginning of routes specified
      # in the i696Scenario
      edges_distribution=["404969345#0", "59440544#0", "124433709", "38726647"] 
    ),
)


def setup_exps():

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
    ray.init(num_cpus=N_CPUS + 1, redirect_output=False)
    trials = run_experiments({
        flow_params["exp_tag"]: {
            "run": alg_run,
            "env": gym_name,
            "config": {
                **config
            },
            "checkpoint_freq": 1, #20,
            "checkpoint_at_end": True,
            "max_failures": 999,
            "stop": {
                "training_iteration": 200,
            },
        }
    })
