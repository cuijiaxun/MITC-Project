"""Multi-agent highway with ramps example.

Trains a non-constant number of agents, all sharing the same policy, on the
highway with ramps network.
"""
import json
import ray
import os
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray import tune
from ray.tune.registry import register_env
from ray.tune import run_experiments
from flow.networks import Network
from flow.controllers import SimCarFollowingController,IDMController, RLController, SimLaneChangeController, ContinuousRouter

from flow.core.params import EnvParams, NetParams, InitialConfig, InFlows, \
                             VehicleParams, SumoParams, \
                             SumoCarFollowingParams, SumoLaneChangeParams

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

from flow.envs.multiagent import MultiAgentHighwayPOEnvWindowCollaborate
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.networks import MergeNetwork
from flow.networks.merge import ADDITIONAL_NET_PARAMS
from copy import deepcopy

# SET UP PARAMETERS FOR THE SIMULATION

# number of training iterations
N_TRAINING_ITERATIONS = 500
# number of rollouts per training iteration
N_ROLLOUTS = 30 
# number of steps per rollout
HORIZON = 2000
# number of parallel workers
N_CPUS = 10
NUM_RL = 30
# inflow rate on the highway in vehicles per hour
FLOW_RATE = 2000
# inflow rate on each on-ramp in vehicles per hour
MERGE_RATE = 200
# percentage of autonomous vehicles compared to human vehicles on highway
RL_PENETRATION = 0.1


# SET UP PARAMETERS FOR THE NETWORK
additional_net_params = deepcopy(ADDITIONAL_NET_PARAMS)
scenarios_dir = os.path.join(os.path.expanduser("~/"), 'Documents', 'MITC', 'flow', 'scenarios')
scenario_road_data = {"name" : "I696_ONE_LANE",
            "net" : os.path.join(scenarios_dir, 'i696', 'osm.net.i696_onelane.xml'), 
            "rou" : [os.path.join(scenarios_dir, 'i696', 'i696.rou.xml')],
            #"rou" : [os.path.join(scenarios_dir, 'i696', 'i696.rou.i696_onelane_Evenshorter.xml')],
            "edges_distribution" : ["404969345#0", "59440544#0", "124433709", "38726647"] 
            }



# SET UP PARAMETERS FOR THE ENVIRONMENT

additional_env_params = ADDITIONAL_ENV_PARAMS.copy()



# CREATE VEHICLE TYPES AND INFLOWS

vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {
        #"noise": 0.2
    }),
    lane_change_controller=(SimLaneChangeController, {}),
    #routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
      # Define speed mode that will minimize collisions: https://sumo.dlr.de/wiki/TraCI/Change_Vehicle_State#speed_mode_.280xb3.29
      speed_mode=9,#"all_checks", #"all_checks", #no_collide",
      #decel=7.5,  # avoid collisions at emergency stops 
      # desired time-gap from leader
      #tau=2, #7,
      #min_gap=2.5,
      #speed_factor=1,
      #speed_dev=0.1
    ),
    lane_change_params=SumoLaneChangeParams(
      model="SL2015",
      # Define a lane changing mode that will allow lane changes
      # See: https://sumo.dlr.de/wiki/TraCI/Change_Vehicle_State#lane_change_mode_.280xb6.29
      # and: ~/local/flow_2019_07/flow/core/params.py, see LC_MODES = {"aggressive": 0 /*bug, 0 is no lane-changes*/, "no_lat_collide": 512, "strategic": 1621}, where "strategic" is the default behavior
      lane_change_mode=1621,#0b011000000001, # (like default 1621 mode, but no lane changes other than strategic to follow route, # 512, #(collision avoidance and safety gap enforcement) # "strategic", 
      #lc_speed_gain=1000000,
      lc_pushy=0, #0.5, #1,
      lc_assertive=5, #20,
      # the following two replace default values which are not read well by xml parser
      lc_impatience=1e-8,
      lc_time_to_impatience=1e12
    ), 
    num_vehicles=0)

vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    lane_change_controller=(SimLaneChangeController, {}),
    #routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
      # Define speed mode that will minimize collisions: https://sumo.dlr.de/wiki/TraCI/Change_Vehicle_State#speed_mode_.280xb3.29
      speed_mode=9,#"all_checks", #no_collide",
      #decel=7.5,  # avoid collisions at emergency stops 
      # desired time-gap from leader
      #tau=2, #7,
      #min_gap=2.5,
      #speed_factor=1,
      #speed_dev=0.1,
    ),
    lane_change_params=SumoLaneChangeParams(
      model="SL2015",
      # Define a lane changing mode that will allow lane changes
      # See: https://sumo.dlr.de/wiki/TraCI/Change_Vehicle_State#lane_change_mode_.280xb6.29
      # and: ~/local/flow_2019_07/flow/core/params.py, see LC_MODES = {"aggressive": 0 /*bug, 0 is no lane-changes*/, "no_lat_collide": 512, "strategic": 1621}, where "strategic" is the default behavior
      lane_change_mode=1621,#0b011000000001, # (like default 1621 mode, but no lane changes other than strategic to follow route, # 512, #(collision avoidance and safety gap enforcement) # "strategic", 
      #lc_speed_gain=1000000,
      lc_pushy=0, #0.5, #1,
      lc_assertive=5, #20,
      # the following two replace default values which are not read well by xml parser
      lc_impatience=1e-8,
      lc_time_to_impatience=1e12
    ), 
    num_vehicles=0)

# Vehicles are introduced from both sides of merge, with RL vehicles entering
# from the highway portion as well
inflow = InFlows()
'''
inflow.add(
    veh_type="human",
    edge="404969345#0", # flow id sw2w1 from xml file
    begin=10,#0,
    end=90000,
    #probability=(1 - RL_PENETRATION), #* FLOW_RATE,
    vehs_per_hour = MERGE_RATE,#(1 - RL_PENETRATION)*FLOW_RATE,
    departSpeed=7.5,
    departLane="free",
    )

'''
'''
inflow.add(
    veh_type="rl",
    edge="404969345#0", # flow id sw2w1 from xml file
    begin=10,#0,
    end=90000,
    vehs_per_hour = RL_PENETRATION * FLOW_RATE,
    depart_speed="max",
    depart_lane="free",
    )
'''
inflow.add(
    veh_type="human",
    edge="59440544#0", # flow id se2w1 from xml file
    begin=10,#0,
    end=90000,
    vehs_per_hour = (1 - RL_PENETRATION)*FLOW_RATE,
    departSpeed=10,
    departLane="free",
    )

inflow.add(
    veh_type="rl",
    edge="59440544#0", # flow id se2w1 from xml file
    begin=10,#0,
    end=90000,
    #probability=RL_PENETRATION, # * 0.8, #* FLOW_RATE,
    vehs_per_hour = RL_PENETRATION*FLOW_RATE,
    depart_speed=10,
    depart_lane="free",
    )

inflow.add(
    veh_type="human",
    edge="124433709", # flow id e2w1 from xml file
    begin=10,#0,
    end=90000,
    vehs_per_hour = MERGE_RATE, #(1 - RL_PENETRATION)*FLOW_RATE,
    departSpeed=7.5,
    departLane="free",
    )
'''
inflow.add(
    veh_type="rl",
    edge="124433709", # flow id e2w1 from xml file
    begin=10,#0,
    end=90000,
    probability=RL_PENETRATION, # * 0.8, # * FLOW_RATE,
    depart_speed="max",
    depart_lane="free",
    )
'''
'''
inflow.add(
    veh_type="human",
    edge="38726647", # flow id n2w1 from xml file
    begin=10,#0,
    end=90000,
    vehs_per_hour = MERGE_RATE,#(1 - RL_PENETRATION)*FLOW_RATE,
    departSpeed=7.5,
    departLane="free",
    )

'''
'''
inflow.add(
    veh_type="rl",
    edge="38726647", # flow id n2w1 from xml file
    begin=10,#0,
    end=90000,
    probability=RL_PENETRATION, # * 0.8, # * FLOW_RATE,
    depart_speed="max",
    depart_lane="free",
    )
'''


flow_params = dict(
    exp_tag='multiagent_highway_i696_1merge_Window_Collaborate_lrschedule',

    env_name=MultiAgentHighwayPOEnvWindowCollaborate,
    network=MergeNetwork,
    simulator='traci',

    #env=EnvParams(
    #    horizon=HORIZON,
    #    warmup_steps=200,
    #    sims_per_step=1,  # do not put more than one #FIXME why do not put more than one
    #    additional_params=additional_env_params,
    #),

    sim=SumoParams(
        restart_instance=True,
        sim_step=0.5,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        sims_per_step=1, #5,
        warmup_steps=0,
        additional_params={
            "max_accel": 2.6,
            "max_decel": 4.5,
            "target_velocity": 30,
            "num_rl": NUM_RL, # used by WaveAttenuationMergePOEnv e.g. to fix action dimension
            "ignore_edges":[
                            #before window
                            "59440544#0",
                            "59440544#1",
                            ":4308145956_0",
                            ":gneJ29_2",
                            ":62290112_0",
                            ":gneJ26_0",
                            "59440544#1-AddedOffRampEdge",
                            "22723058#0",
                            "22723058#1",
                            "491515539",
                            ":gneJ24_1",
                            "341040160#0",
                            ":4308145961_0",
                            "4308145961",

                            #after window
                            "422314897#1",
                            "489256509",
                            "456874110",
                            ],
            #"max_inflow":FLOW_RATE + 3*MERGE_RATE,
        },
    ),
    net=NetParams(
        inflows=inflow,
        #no_internal_links=False,
        additional_params=additional_net_params,
        template={
          "net" : scenario_road_data["net"],# see above
          "rou" : scenario_road_data["rou"],# see above 
        }
    ),


    veh=vehicles,
    initial=InitialConfig(
      # Distributing only at the beginning of routes
      scenario_road_data["edges_distribution"]
    ),

)


# SET UP EXPERIMENT

def setup_exps(flow_params):
    """Create the relevant components of a multiagent RLlib experiment.

    Parameters
    ----------
    flow_params : dict
        input flow-parameters

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    alg_run = 'PPO'
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config['num_workers'] = N_CPUS
    config['train_batch_size'] = HORIZON * N_ROLLOUTS
    config['sgd_minibatch_size'] = 4096
    #config['simple_optimizer'] = True
    config['gamma'] = 0.998  # discount rate
    config['model'].update({'fcnet_hiddens': [100, 50, 25]})
    #config['lr'] = tune.grid_search([5e-4, 1e-4])
    config['lr_schedule'] = [
            [0, 5e-4],
            [1000000, 1e-4],
            [4000000, 1e-5],
            [8000000, 1e-6]]
    config['horizon'] = HORIZON
    config['clip_actions'] = False
    config['observation_filter'] = 'NoFilter'
    config["use_gae"] = True
    config["lambda"] = 0.95
    config["shuffle_sequences"] = True
    config["vf_clip_param"] = 1e8
    config["num_sgd_iter"] = 10
    #config["kl_target"] = 0.003
    config["kl_coeff"] = 0.01
    config["entropy_coeff"] = 0.001
    config["clip_param"] = 0.2
    config["grad_clip"] = None
    config["use_critic"] = True
    config["vf_share_layers"] = True
    config["vf_loss_coeff"] = 0.5


    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # register as rllib env
    register_env(env_name, create_env)

    # multiagent configuration
    temp_env = create_env()
    policy_graphs = {'av': (PPOTFPolicy,
                            temp_env.observation_space,
                            temp_env.action_space,
                            {})}

    def policy_mapping_fn(_):
        return 'av'

    config.update({
        'multiagent': {
            'policies': policy_graphs,
            'policy_mapping_fn': tune.function(policy_mapping_fn),
            'policies_to_train': ['av']
        }
    })

    return alg_run, env_name, config


# RUN EXPERIMENT

if __name__ == '__main__':
    alg_run, env_name, config = setup_exps(flow_params)
    ray.init(num_cpus=N_CPUS + 1)

    run_experiments({
        flow_params['exp_tag']: {
            'run': alg_run,
            'env': env_name,
            'checkpoint_freq': 5,
            'checkpoint_at_end': True,
            'stop': {
                'training_iteration': N_TRAINING_ITERATIONS
            },
            'config': config,
        },
    })
