"""Ring road example.

Trains a single autonomous vehicle to stabilize the flow of 21 human-driven
vehicles in a variable length ring road.
"""

import json

import ray
from ray.rllib.agents.agent import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.controllers import RLController, IDMController, ContinuousRouter

# that's the baseline scenario for this learning agent
from examples.sumo import sugiyama_8 #import ADDITIONAL_ENV_PARAMS, ADDITIONAL_NET_PARAMS, SUMO_STEP

# time horizon of a single rollout
HORIZON = 3000
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 4

# We place one autonomous vehicle and 22 human-driven vehicles in the network
vehicles = Vehicles()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=7)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=1)


# import from baseline test
ADDITIONAL_ENV_PARAMS = sugiyama_8.ADDITIONAL_ENV_PARAMS
ADDITIONAL_ENV_PARAMS["ring_length"] = [100, 125]
#
ADDITIONAL_NET_PARAMS = sugiyama_8.ADDITIONAL_NET_PARAMS


flow_params = dict(
    # name of the experiment
    exp_tag="stabilizing_the_ring_8_nonoise_avgspeedreward_step0.5sec_2_obs_radius",

    # name of the flow environment the experiment is running on
    env_name="WaveAttenuationPORadius2Env",

    # name of the scenario class the experiment is running on
    scenario="LoopScenario",

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        sim_step=sugiyama_8.SUMO_STEP,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=750,
        additional_params=ADDITIONAL_ENV_PARAMS
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(additional_params=ADDITIONAL_NET_PARAMS),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)


def setup_exps():

    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = N_CPUS
    config["train_batch_size"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [16, 16]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
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
            "checkpoint_freq": 20,
            "max_failures": 999,
            "stop": {
                "training_iteration": 500,
            },
        }
    })
