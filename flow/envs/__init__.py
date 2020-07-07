"""Contains all callable environments in Flow."""
from flow.envs.base import Env
from flow.envs.bay_bridge import BayBridgeEnv
from flow.envs.bottleneck import BottleneckAccelEnv, BottleneckEnv, \
    BottleneckDesiredVelocityEnv
from flow.envs.traffic_light_grid import TrafficLightGridEnv, \
    TrafficLightGridPOEnv, TrafficLightGridTestEnv
from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv, \
    LaneChangeAccelPOEnv
from flow.envs.ring.accel import AccelEnv
from flow.envs.ring.wave_attenuation import WaveAttenuationEnv, \
    WaveAttenuationPOEnv, WaveAttenuationPOEnvNoisy, WaveAttenuationPOEnvSpeedreward, WaveAttenuationPOEnvAvgSpeedreward, WaveAttenuationEnvAvgSpeedreward, WaveAttenuationPORadiusEnv, WaveAttenuationPORadius1Env, WaveAttenuationPORadius2Env, \
      WaveAttenuationPOEnvSmallAccelPenalty, \
      WaveAttenuationPOEnvMediumAccelPenalty, \
      WaveAttenuationPORadiusEnvAvgSpeedNormalized, \
      WaveAttenuationPORadius1EnvAvgSpeedNormalized, \
      WaveAttenuationPORadius2EnvAvgSpeedNormalized  
from flow.envs.merge import MergePOEnv,MergePOEnvEdgePrior,MergePOEnvPunishDelay,MergePOEnvGuidedPunishDelay,MergePOEnvSparseRewardDelay,\
                            MergePOEnvMinDelay, MergePOEnvAvgVel, MergePOEnvIncludePotential, MergePOEnvScaleInflow,MergePOEnvScaleInflowIgnore,\
                            MergePORadius2Env, MergePORadius4Env, MergePORadius7Env,MergePOEnvIgnore,\
                            MergePOEnvIgnoreAvgVel,MergePOEnvIgnoreAvgVelDistance,MergePOEnvIgnoreAvgVelDistanceMergeInfo,\
                            MergePOEnvDeparted
from flow.envs.merge_Ignore import MergePOEnv_Ignore
from flow.envs.test import TestEnv
from flow.envs.merge_no_headway import MergePOEnv_noheadway, MergePOEnvEdgePrior_noheadway
from flow.envs.merge_noheadway_encourageRLmove import MergePOEnv_noheadway_encourageRLmove
from flow.envs.merge_noheadway_encourageRLmove_sumSpeed import MergePOEnv_noheadway_encourageRLmove_sumSpeed
from flow.envs.merge_noheadway_sumSpeed import MergePOEnv_noheadway_sumSpeed
from flow.envs.merge_optOutflow import MergePOEnv_optOutflow
from flow.envs.merge_optInflow import MergePOEnv_optInflow
from flow.envs.merge_InflowScale import MergePOEnv_InflowScale
from flow.envs.merge_minV0R0 import MergePOEnv_minV0R0
# deprecated classes whose names have changed
from flow.envs.bottleneck_env import BottleNeckAccelEnv
from flow.envs.bottleneck_env import DesiredVelocityEnv
from flow.envs.green_wave_env import PO_TrafficLightGridEnv
from flow.envs.green_wave_env import GreenWaveTestEnv


__all__ = [
    'Env',
    'AccelEnv',
    'LaneChangeAccelEnv',
    'LaneChangeAccelPOEnv',
    'TrafficLightGridTestEnv',
    'MergePOEnv',
    'MergePOEnvEdgePrior',
    'MergePOEnvDeparted',
    'MergePOEnvIgnore',
    'MergePOEnvIgnoreAvgVel',
    'MergePOEnvIgnoreAvgVelDistance',
    'MergePOEnvIgnoreAvgVelDistanceMergeInfo',
    'MergePOEnvScaleInflow',
    'MergePOEnvScaleInflowIgnore',
    'MergePOEnvEdgePrior_noheadway',
    'MergePOEnvGuidedPunishDelay',
    'MergePOEnvSparseRewardDelay',
    'MergePOEnv_Ignore',
    'MergePOEnvPunishDelay',
    'MergePOEnvMinDelay',
    'MergePOEnvAvgVel',
    'MergePOEnv_noheadway',
    'MergePOEnvIncludePotential',
    'MergePOEnv_noheadway_encourageRLmove',
    'MergePOEnv_noheadway_encourageRLmove_sumSpeed',
    'MergePOEnv_noheadway_sumSpeed',
    'MergePOEnv_optOutflow',
    'MergePOEnv_optInflow',
    'MergePOEnv_InflowScale',
    'MergePOEnv_minV0R0',
    'BottleneckEnv',
    'BottleneckAccelEnv',
    'WaveAttenuationEnv',
    'WaveAttenuationPOEnv',
    'TrafficLightGridEnv',
    'TrafficLightGridPOEnv',
    'BottleneckDesiredVelocityEnv',
    'TestEnv',
    'BayBridgeEnv',
    'MergePORadius2Env', 
    'MergePORadius4Env', 
    'MergePORadius7Env',
    # deprecated classes
    'BottleNeckAccelEnv',
    'DesiredVelocityEnv',
    'PO_TrafficLightGridEnv',
    'GreenWaveTestEnv',
    'WaveAttenuationPOEnvNoisy', 
    'WaveAttenuationPOEnvSpeedreward',
    'WaveAttenuationPOEnvAvgSpeedreward', 
    'WaveAttenuationEnvAvgSpeedreward',
    'WaveAttenuationPORadiusEnv', 
    'WaveAttenuationPORadius1Env',
    'WaveAttenuationPORadius2Env', 
    'WaveAttenuationPOEnvSmallAccelPenalty',
    'WaveAttenuationPOEnvMediumAccelPenalty',
    'WaveAttenuationPORadiusEnvAvgSpeedNormalized', 
    'WaveAttenuationPORadius1EnvAvgSpeedNormalized',
    'WaveAttenuationPORadius2EnvAvgSpeedNormalized',
]
