"""Used as an example of sugiyama experiment.

This example consists of 22 IDM cars on a ring creating shockwaves.
"""

from flow.controllers import IDMController, ContinuousRouter
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, \
    InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.loop import LoopScenario, ADDITIONAL_NET_PARAMS


# CONFIGURATION 
# override
ADDITIONAL_NET_PARAMS["length"] = 100
SUMO_STEP = 0.5
# END CONFIGURATION 

def sugiyama_example(render=None):
    """
    Perform a simulation of vehicles on a ring road.

    Parameters
    ----------
    render : bool, optional
        specifies whether to use sumo's gui during execution

    Returns
    -------
    exp: flow.core.Experiment type
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a ring road.
    """
    sumo_params = SumoParams(sim_step=SUMO_STEP, render=True)

    if render is not None:
        sumo_params.render = render

    vehicles = VehicleParams()
    vehicles.add(
          veh_id="idm",
          acceleration_controller=(IDMController, {
          "noise": 0.2 
          }),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=8 )

    # in sugiyama taken from .../flow/scenarios/loop/loop_accel.py
    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    # in sugiyama taken from .../flow/scenarios/loop.py
    net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)

    initial_config = InitialConfig(bunching=20)

    scenario = LoopScenario(
        name="sugiyama",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    env = AccelEnv(env_params, sumo_params, scenario)

    return Experiment(env)


if __name__ == "__main__":
    # import the experiment variable
    exp = sugiyama_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
