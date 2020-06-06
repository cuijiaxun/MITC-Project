"""Open merge example.

Trains a a small percentage of rl vehicles to dissipate shockwaves caused by
merges in an open network.
"""
import json
import os
import random
import numpy as np
from argparse import ArgumentParser
import pickle
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
from flow.core.params import VehicleParams, SumoLaneChangeParams, TrafficLightParams
from flow.controllers import IDMController, RLController, SimLaneChangeController, ContinuousRouter,SimCarFollowingController
from flow.core.experiment import Experiment
from flow.scenarios import i696Scenario
# TODO hard coded
#scenarios_dir = os.path.join(os.path.expanduser("~/"), 'local', 'flow_2019_07', 'flow', 'scenarios')
scenarios_dir = os.path.join(os.path.expanduser("~/"),'Documents/MITC/flow/scenarios/',)
# UNCOMMENT ONE OF THE FOLLOWING 3 VARIATIONS OF I696 SCENARIO 
#one-lane (no lane-changes)
##########################
scenario_road_data = {"name" : "I696_ONE_LANE",
            "net" : os.path.join(scenarios_dir, 'i696', 'osm.net.i696_onelane.xml'), 
            #"rou" : [os.path.join(scenarios_dir, 'i696', 'i696.rou.xml')],
            #"rou" : [os.path.join(scenarios_dir,'i696','i696.rou.i696_onelane_cropped.xml')],
            #"rou" : [os.path.join(scenarios_dir, 'i696', 'i696.rou.i696_onelane_shorter.xml')],
            "rou" : [os.path.join(scenarios_dir, 'i696', 'i696.rou.i696_onelane_Evenshorter.xml')],
            "edges_distribution" : ["404969345#0", "59440544#0", "124433709", "38726647"] 
            }
            
# experiment number
# - 0: 10% RL penetration,  5 max controllable vehicles
# - 1: 25% RL penetration, 13 max controllable vehicles
# - 2: 33% RL penetration, 17 max controllable vehicles

N_CPUS = 2#1#8#2
#random.seed(13)
# inflow rate at the highway
FLOW_RATE = 2000
MERGE_RATE = 150

## We consider a highway network with an upstream merging lane producing
# shockwaves
additional_net_params = ADDITIONAL_NET_PARAMS.copy()

# RL vehicles constitute 5% of the total number of vehicles
# Daniel: adding vehicles and flow from osm.passenger.trips.xml
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {
        #"noise": 0.2,
    }),
    lane_change_controller=(SimLaneChangeController, {}),
    #routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
      # Define speed mode that will minimize collisions: https://sumo.dlr.de/wiki/TraCI/Change_Vehicle_State#speed_mode_.280xb3.29
      speed_mode=9, #"right_of_way",#"right_of_way", #"right_of_way", #"all_checks", #no_collide",
      #decel=7.5,  # avoid collisions at emergency stops 
      # desired time-gap from leader
      #tau=1, #7,
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
      #lc_pushy=0.5, #0.5, #1,
      #lc_assertive=5, #20,
      # the following two replace default values which are not read well by xml parser
      lc_impatience=1e-8,
      lc_time_to_impatience=1e12
    ), 
    num_vehicles=0)

inflow = InFlows()
'''
inflow.add(
    veh_type="human",
    edge="404969345#0", # flow id sw2w1 from xml file
    begin=10,#0,
    end=90000,
    #probability=(1 - RL_PENETRATION), #* FLOW_RATE,
    vehs_per_hour = MERGE_RATE,#(1 - RL_PENETRATION)*FLOW_RATE,
    departSpeed=20,
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
    vehs_per_hour = FLOW_RATE,
    departSpeed=20,
    departLane="free",
    )

'''
inflow.add(
    veh_type="rl",
    edge="59440544#0", # flow id se2w1 from xml file
    begin=10,#0,
    end=90000,
    probability=RL_PENETRATION, # * 0.8, #* FLOW_RATE,
    depart_speed="max",
    depart_lane="free",
    )
'''

inflow.add(
    veh_type="human",
    edge="124433709", # flow id e2w1 from xml file
    begin=10,#0,
    end=90000,
    vehs_per_hour = MERGE_RATE, #(1 - RL_PENETRATION)*FLOW_RATE,
    departSpeed=20,
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
    departSpeed=20,
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

sumo_params=SumoParams(
        sim_step=0.5,            # Daniel updated from osm.sumocfg
        lateral_resolution=0.01, # determines lateral discretization of lanes
        render=True,#True,             # False for training, True for debugging
        restart_instance=True,
    )
from flow.envs.test import TestEnv
env_params=EnvParams(
        sims_per_step=2,
        warmup_steps=200,
        additional_params={
            "max_accel": 9,
            "max_decel": 9,
            "target_velocity": 30,
            "sort_vehicles":True,
            },
    )

net_params=NetParams(
        inflows=inflow,
        #no_internal_links=False,
        additional_params=additional_net_params,
        template={
          "net" : scenario_road_data["net"],# see above
          "rou" : scenario_road_data["rou"],# see above 
        }
    )

initial_config=InitialConfig(
        #scenario_road_data["edges_distribution"]
        edges_distribution=scenario_road_data["edges_distribution"],
    )
if __name__ == "__main__":
    scenario = i696Scenario(name='i696',
                            vehicles=vehicles,
                            net_params=net_params,
                            initial_config=initial_config,)
    #env = AccelEnv(env_params,sumo_params,scenario)
    env = TestEnv(env_params,sumo_params,scenario)
    exp = Experiment(env)
    _ = exp.run(1,400)#,convert_to_csv=True)
    #from IPython import embed
    #embed()
    with open("info.pkl","wb") as f:
        pickle.dump(_,f)

