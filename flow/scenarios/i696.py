"""Contains the merge scenario class."""

#from flow.scenarios.base_scenario import Scenario
from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.core.params import NetParams
from numpy import pi, sin, cos

INFLOW_EDGE_LEN = 100  # length of the inflow edges (needed for resets)
#VEHICLE_LENGTH = 5

ADDITIONAL_NET_PARAMS = {
#    # length of the merge edge
#    "merge_length": 100,
#    # length of the highway leading to the merge
#    "pre_merge_length": 200,
#    # length of the highway past the merge
#    "post_merge_length": 100,
#    # number of lanes in the merge
#    "merge_lanes": 1,
#    # number of lanes in the highway
#    "highway_lanes": 1,
#    # max speed limit of the network
#    "speed_limit": 30,
}

#i696_net_params = NetParams(
#    template='./i696/osm.net.xml',
#    no_internal_links=False
#)


class i696Scenario(Network):

#    def __init__(self,
#                 name,
#                 vehicles,
#                 net_params,
#                 initial_config=InitialConfig(),
#                 traffic_lights=TrafficLightParams()):
#        """Initialize a merge scenario."""
#        for p in ADDITIONAL_NET_PARAMS.keys():
#            if p not in net_params.additional_params:
#                raise KeyError('Network parameter "{}" not supplied'.format(p))
#
#        super().__init__(name, vehicles, initial_config,
#                         traffic_lights, net_params=i696_net_params)

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            # sw2w
            "404969345#0": ["404969345#0", "404969345#1", "456864110", "40788302", "124433730#0", "124433730#1-AddedOnRampEdge", "124433730#1", "124433730#2-AddedOnRampEdge", "124433730#2"],
            # added since from some reason there is rerouting when reaching the last edge
            "124433730#2": ["124433730#2"],
            # se2w
            "59440544#0": ["59440544#0", "59440544#1", "59440544#1-AddedOffRampEdge", "22723058#0", "22723058#1", "491515539", "341040160#0", "341040160#1", "491266613", "422314897#0", "422314897#1", "489256509", "456864110", "40788302", "124433730#0", "124433730#1-AddedOnRampEdge", "124433730#1", "124433730#2-AddedOnRampEdge", "124433730#2"],
            # e2w
            "124433709": ["124433709", "422314897#0", "422314897#1", "489256509", "456864110", "40788302", "124433730#0", "124433730#1-AddedOnRampEdge", "124433730#1", "124433730#2-AddedOnRampEdge", "124433730#2"],
            # n2w
            "38726647": ["38726647", "491111706", "8666737", "40788302", "124433730#0", "124433730#1-AddedOnRampEdge", "124433730#1", "124433730#2-AddedOnRampEdge", "124433730#2"]
        }

        return rts

#<vTypeDistribution id="DEFAULT_VEHTYPE">
#    <vType probability="0.2" id="truck" guiShape="truck" length="12.0" minGap="2.5" speedFactor="1" speedDev="0.1" maxSpeed= "27.78"  laneChangeModel="SL2015" lcSpeedGain="1000000" lcPushy="1" lcAssertive="20"/> 
#    <vType probability="0.8" id="passenger" length="5.0" minGap="2.5" speedFactor="1" speedDev="0.1" laneChangeModel="SL2015" lcSpeedGain="1000000" lcPushy="1" lcAssertive="20"/> 
#</vTypeDistribution>
#<flow id="sw2w1" route="sw2w" begin="0" end="90000" probability="0.8" departSpeed="max" departLane="free" departPos="free"/>
#<flow id="se2w1" route="se2w" begin="0" end="90000" probability="0.8" departSpeed="max" departLane="free" departPos="free"/>
#<flow id="e2w1" route="e2w" begin="0" end="90000" probability="0.8" departSpeed="max" departLane="free" departPos="free"/>
#<flow id="n2w1" route="n2w" begin="0" end="90000" probability="0.8" departSpeed="max" departLane="free" departPos="free"/>

