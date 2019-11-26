"""Ring road example.

Trains a single autonomous vehicle to stabilize the flow of 21 human-driven
vehicles in a variable length ring road.
"""

import json

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.envs import WaveAttenuationPORadius1EnvAvgSpeedNormalized
from flow.networks import RingNetwork
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.controllers import RLController, IDMController, ContinuousRouter

# that's the baseline scenario for this learning agent
from examples.sumo import sugiyama_8 #import ADDITIONAL_ENV_PARAMS, ADDITIONAL_NET_PARAMS, SUMO_STEP

# time horizon of a single rollout
HORIZON = 3000
# number of rollouts per training iteration
N_ROLLOUTS = 20# 1000
# number of parallel workers
N_CPUS = 8

# We place one autonomous vehicle and 22 human-driven vehicles in the network
vehicles = VehicleParams()
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
    exp_tag="stabilizing_the_ring_8_nonoise_avgspeedrewardnormalized_step0.5sec_1_obs_radius_ppotuning_batch20rolls_with_vf_loss_50000iter",

    # name of the flow environment the experiment is running on
    env_name=WaveAttenuationPORadius1EnvAvgSpeedNormalized,

    # name of the scenario class the experiment is running on
    network=RingNetwork,


    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=sugiyama_8.SUMO_STEP,
        render=True, #False,
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
    config["kl_target"] = 0.02# 0.01# 0.02
    config["num_sgd_iter"] = 10
    config["horizon"] = HORIZON
    config["clip_param"] = 0.15 #default was 0.3, 0.2 was reported to work well, but maybe too large?
    #config["vf_loss_coeff"] = 1#0.2 #default was 1, this is when using multitask, saw that vf_loss drops to ~10 while policy loss is around 0.01, but with 0.01 coeff didn't estimate loss well

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
    ray.init(num_cpus=N_CPUS + 1)
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
                "training_iteration": 50000,
            },
        }
    })
