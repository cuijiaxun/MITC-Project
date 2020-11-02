"""Visualizer for rllib experiments.

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::

        python ./visualizer_rllib.py /tmp/ray/result_dir 1

parser : ArgumentParser
    Command-line argument parser
"""

import argparse
from datetime import datetime
import gym
import numpy as np
import os
import sys
import time
import pprint
import matplotlib.pyplot as plt
import glob
import ray
import copy
from ray.tune.utils import merge_dicts
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env,get_trainable_cls

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl
from ray.rllib.agents.callbacks import DefaultCallbacks
seed_filename = glob.glob("eval_seeds/*/seeds.pkl")
print(seed_filename)
print("Using ", len(seed_filename), " random seeds")
EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""

# global variables configuring diagnostics
PRINT_TO_SCREEN = False
SUMMARY_PLOTS = False
REALTIME_PLOTS = False

def generateHtmlplots(actions, rewards, states):
    import plotly.graph_objs as go
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

    # time series
    plot([go.Scatter(x=[i for i in range(len(actions))], y=[a[0] for a in actions])], filename="actions-plot-checkpoint_" + args.checkpoint_num)
    plot([go.Scatter(x=[i for i in range(len(rewards))], y=rewards)], filename="rewards-plot-checkpoint_" + args.checkpoint_num)
    plot([go.Scatter(x=[i for i in range(len(states))], y=[s[0] for s in states])], filename="state0-plot_" + args.checkpoint_num)
    plot([go.Scatter(x=[i for i in range(len(states))], y=[s[1] for s in states])], filename="state1-plot_" + args.checkpoint_num)
    plot([go.Scatter(x=[i for i in range(len(states))], y=[s[2] for s in states])], filename="state2-plot_" + args.checkpoint_num)
    # histograms
    plot([go.Histogram(x=[a[0] for a in actions], nbinsx=100)], filename="actions-hist_" + args.checkpoint_num)
    plot([go.Histogram(x=rewards, nbinsx=100)], filename="rewards-hist_" + args.checkpoint_num)
    plot([go.Histogram(x=[s[0] for s in states], nbinsx=100)], filename="state0-hist_" + args.checkpoint_num)
    plot([go.Histogram(x=[s[1] for s in states], nbinsx=100)], filename="state1-hist_" + args.checkpoint_num)
    plot([go.Histogram(x=[s[2] for s in states], nbinsx=100)], filename="state2-hist_" + args.checkpoint_num)
    # 3d scatter of policy
    trace1 = go.Scatter3d(
        x=[s[0] for s in states],
        y=[s[2] for s in states],
        z=[np.clip(a[0], -5, 5) for a in actions],
        mode='markers',
        marker=dict(
            size=0.5,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.1
            ),
            opacity=0.8
        )
    ) 
    data = [trace1]
    layout = go.Layout(
      margin=dict(
          l=0,
          r=0,
          b=0,
          t=0
      ),
      scene = dict(
        xaxis = dict( title='speed' ),
        yaxis = dict( title='distance' ),
        zaxis = dict( title='action' ) # not supported here, but appears in example?
      )
    )
    plot(go.Figure(data=data, layout=layout), filename='speed-dist-2-action')

class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        # save env state when an episode starts
        env = base_env.get_unwrapped()[0]
        state = env.get_state()
        episode.user_data["initial_state"] = state


def visualizer_rllib(args, seed=None):
    """Visualizer for RLlib experiments.

    This function takes args (see function create_parser below for
    more detailed information on what information can be fed to this
    visualizer), and renders the experiment associated with it.
    """
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    config = get_rllib_config(result_dir)

    # check if we have a multiagent environment but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policies', None):
        multiagent = True
        pkl = get_rllib_pkl(result_dir)
        config['multiagent'] = pkl['multiagent']
    else:
        multiagent = False

    config['callbacks'] = MyCallbacks
    # Run on only one cpu for rendering purposes
    config['num_workers'] = 0

    flow_params = get_flow_params(config)
    flow_params['env'].additional_params["use_seeds"]=args.use_seeds
    print(args.use_seeds)
    # hack for old pkl files
    # TODO(ev) remove eventually
    sim_params = flow_params['sim']
    setattr(sim_params, 'num_clients', 1)

    # Determine agent and checkpoint
    config_run = config['env_config']['run'] if 'run' in config['env_config'] \
        else None
    if args.run and config_run:
        if args.run != config_run:
            print('visualizer_rllib.py: error: run argument '
                  + '\'{}\' passed in '.format(args.run)
                  + 'differs from the one stored in params.json '
                  + '\'{}\''.format(config_run))
            sys.exit(1)
    
    # Merge with `evaluation_config`.
    evaluation_config = copy.deepcopy(config.get("evaluation_config", {}))
    config = merge_dicts(config, evaluation_config) 
    
    if args.run:
        agent_cls = get_trainable_cls(args.run)
    elif config_run:
        agent_cls = get_trainable_cls(config_run)
    else:
        print('visualizer_rllib.py: error: could not find flow parameter '
              '\'run\' in params.json, '
              'add argument --run to provide the algorithm or model used '
              'to train the results\n e.g. '
              'python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO')
        sys.exit(1)

    sim_params.restart_instance = True
    dir_path = os.path.dirname(os.path.realpath(__file__))
    emission_path = '{0}/test_time_rollout/'.format(dir_path)
    sim_params.emission_path = emission_path if args.gen_emission else None

    # pick your rendering mode
    if args.render_mode == 'sumo_web3d':
        sim_params.num_clients = 2
        sim_params.render = False
    elif args.render_mode == 'drgb':
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
    elif args.render_mode == 'sumo_gui':
        sim_params.render = True
        print('NOTE: With render mode {}, an extra instance of the SUMO GUI '
              'will display before the GUI for visualizing the result. Click '
              'the green Play arrow to continue.'.format(args.render_mode))
    elif args.render_mode == 'no_render':
        sim_params.render = False
    if args.save_render:
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
        sim_params.save_render = True
    if seed is not None: 
        flow_params["env"].additional_params["use_seeds"] = seed
    else:
        flow_params["env"].additional_params["use_seeds"] = args.use_seeds
    if args.horizon:
        config['horizon'] = args.horizon
        flow_params['env'].horizon = args.horizon

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(params=flow_params, version=0)
    register_env(env_name, create_env)

    # check if the environment is a single or multiagent environment, and
    # get the right address accordingly
    # single_agent_envs = [env for env in dir(flow.envs)
    #                      if not env.startswith('__')]

    # if flow_params['env_name'] in single_agent_envs:
    #     env_loc = 'flow.envs'
    # else:
    #     env_loc = 'flow.envs.multiagent'

    # Start the environment with the gui turned on and a path for the
    # emission file
    env_params = flow_params['env']
    env_params.restart_instance = False
    if args.evaluate:
        env_params.evaluate = True

    # lower the horizon if testing
    if args.horizon:
        config['horizon'] = args.horizon
        env_params.horizon = args.horizon
    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)

    if hasattr(agent, "local_evaluator") and \
            os.environ.get("TEST_FLAG") != 'True':
        env = agent.local_evaluator.env
    else:
        env = gym.make(env_name)

    if multiagent:
        rets = {}
        # map the agent id to its policy
        print(config['multiagent']['policy_mapping_fn'])
        policy_map_fn = config['multiagent']['policy_mapping_fn']#.func

        for key in config['multiagent']['policies'].keys():
            rets[key] = []
    else:
        rets = []

    if config['model']['use_lstm']:
        use_lstm = True
        if multiagent:
            state_init = {}
            # map the agent id to its policy
            policy_map_fn = config['multiagent']['policy_mapping_fn'].func
            size = config['model']['lstm_cell_size']
            for key in config['multiagent']['policies'].keys():
                state_init[key] = [np.zeros(size, np.float32),
                                   np.zeros(size, np.float32)]
        else:
            state_init = [
                np.zeros(config['model']['lstm_cell_size'], np.float32),
                np.zeros(config['model']['lstm_cell_size'], np.float32)
            ]
    else:
        use_lstm = False

    env.restart_simulation(
        sim_params=sim_params, render=sim_params.render)

    # Simulate and collect metrics
    final_outflows = []
    final_inflows = []
    mean_speed = []
    std_speed = []

    if PRINT_TO_SCREEN:
      pp = pprint.PrettyPrinter(indent=2)
      print("config " )
      pp.pprint(config)
      print("flow_params " )
      pp.pprint(flow_params)

    if REALTIME_PLOTS:
      # prepare plots
      # You probably won't need this if you're embedding things in a tkinter plot...
      plt.ion()
      fig = plt.figure()
      axA = fig.add_subplot(331)
      axA.set_title("Actions")
      axR = fig.add_subplot(332)
      axR.set_title("Rewards")
      axS = fig.add_subplot(333)
      axS.set_title("States")
      axS0 = fig.add_subplot(334)
      axS0.set_title("S0")
      axS1= fig.add_subplot(335)
      axS1.set_title("S1")
      axS2 = fig.add_subplot(336)
      axS2.set_title("S2")
      axA_hist = fig.add_subplot(337)
      axA_hist.set_title("Actions")
      axR_hist = fig.add_subplot(338)
      axR_hist.set_title("Rewards")
      axS_hist = fig.add_subplot(339)
      axS_hist.set_title("States")
      axS.set_ylim((-2, 3))
      axA.set_ylim((-5, 5))
      axR.set_ylim((-1, 1))
      initialized_plot = False

    # record for visualization purposes
    actions = []
    rewards = []
    states = []
    times = []
    WARMUP = args.warmup
    for i in range(args.num_rollouts):
        vel = []
        time_to_exit = 0
        state = env.reset()
        if multiagent:
            ret = {key: [0] for key in rets.keys()}
        else:
            ret = 0
        for _ in range(env_params.horizon):
            time_to_exit += 1;
            vehicles = env.unwrapped.k.vehicle
            if np.mean(vehicles.get_speed(vehicles.get_ids()))>0:
                vel.append(np.mean(vehicles.get_speed(vehicles.get_ids())))
            #vel.append(np.mean(vehicles.get_speed(vehicles.get_ids())))
            if multiagent:
                action = {}
                for agent_id in state.keys():
                    if use_lstm:
                        action[agent_id], state_init[agent_id], logits = \
                            agent.compute_action(
                            state[agent_id], state=state_init[agent_id],
                            policy_id=policy_map_fn(agent_id))
                    else:
                        action[agent_id] = agent.compute_action(
                            state[agent_id], policy_id=policy_map_fn(agent_id))
            else:
                action = agent.compute_action(state)
            state, reward, done, _ = env.step(action)

            if SUMMARY_PLOTS:
              # record for visualization purposes
              actions.append(action)
              rewards.append(reward)
              states.append(state)

            if PRINT_TO_SCREEN:
              print("action")
              pp.pprint(action)
              print("reward")
              pp.pprint(reward)
              print("state")
              pp.pprint(state)
              print("after step ")

            if REALTIME_PLOTS:
              # Update plots. 
              if not initialized_plot: # initialize
                lineA, = axA.plot([0] * len(action), 'g^') # Returns a tuple of line objects, thus the comma
                lineR, = axR.plot(0, 'bs') # Returns a tuple of line objects, thus the comma 
                lineS, = axS.plot([0] * len(state), 'r+') # Returns a tuple of line objects, thus the comma 
                initialized_plot = True
              lineA.set_ydata(action)
              lineR.set_ydata(reward)
              lineS.set_ydata(state) 
              fig.canvas.draw()
              fig.canvas.flush_events()    

            if multiagent:
                for actor, rew in reward.items():
                    ret[policy_map_fn(actor)][0] += rew
            else:
                ret += reward
            
            if multiagent and done['__all__']:
                break
            if not multiagent and done:
                break
            if args.use_delay>0:
                if vehicles.get_num_arrived()>=args.use_delay :
                    break
            
        if multiagent:
            for key in rets.keys():
                rets[key].append(ret[key])
        else:
            rets.append(ret)
        outflow = vehicles.get_outflow_rate(5000)
        final_outflows.append(outflow)
        inflow = vehicles.get_inflow_rate(5000)
        final_inflows.append(inflow)
        times.append(time_to_exit)
        if np.all(np.array(final_inflows) > 1e-5):
            throughput_efficiency = [x / y for x, y in
                                     zip(final_outflows, final_inflows)]
        else:
            throughput_efficiency = [0] * len(final_inflows)
        mean_speed.append(np.mean(vel))
        std_speed.append(np.std(vel))
        if multiagent:
            for agent_id, rew in rets.items():
                print('Round {}, Return: {} for agent {}'.format(
                    i, ret, agent_id))
        else:
            print('Round {}, Return: {}'.format(i, ret))

    print('==== Summary of results ====')
    print("Return:")
    if multiagent:
        for agent_id, rew in rets.items():
            print('For agent', agent_id)
            print(rew)
            print('Average, std return: {}, {} for agent {}'.format(
                np.mean(rew), np.std(rew), agent_id))
    else:
        print(rets)
        print('Average, std: {:.2f}, {:.5f}'.format(
            np.mean(rets), np.std(rets)))

    print("\nSpeed, mean (m/s):")
    print(mean_speed)
    print('Average, std: {:.2f}, {:.5f}'.format(np.mean(mean_speed), np.std(
        mean_speed)))
    print("\nSpeed, std (m/s):")
    print(std_speed)
    print('Average, std: {:.2f}, {:.5f}'.format(np.mean(std_speed), np.std(
        std_speed)))

    # Compute arrival rate of vehicles in the last 500 sec of the run
    print("\nOutflows (veh/hr):")
    print(final_outflows)
    print('Average, std: {:.2f}, {:.5f}'.format(np.mean(final_outflows),
                                        np.std(final_outflows)))
    # Compute departure rate of vehicles in the last 500 sec of the run
    print("Inflows (veh/hr):")
    print(final_inflows)
    print('Average, std: {:.2f}, {:.5f}'.format(np.mean(final_inflows),
                                        np.std(final_inflows)))
    # Compute throughput efficiency in the last 500 sec of the
    print("Throughput efficiency (veh/hr):")
    print(throughput_efficiency)
    print('Average, std: {:.2f}, {:.5f}'.format(np.mean(throughput_efficiency),
                                        np.std(throughput_efficiency)))
    print("Time Delay")
    print(times)
    print("Time for certain number of vehicles to exit {:.2f},{:.5f}".format((np.mean(times)),np.std(times)))

    if args.output:
        np.savetxt(args.output, [mean_speed, std_speed,final_inflows, final_outflows,times])
    if SUMMARY_PLOTS:
      generateHtmlplots(actions, rewards, states)

    # terminate the environment
    env.unwrapped.terminate()

    # if prompted, convert the emission file into a csv file
    if args.gen_emission:
        time.sleep(0.1)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        emission_filename = '{0}-emission.xml'.format(env.network.name)

        emission_path = \
            '{0}/test_time_rollout/{1}'.format(dir_path, emission_filename)

        # convert the emission file into a csv file
        emission_to_csv(emission_path)

        # print the location of the emission csv file
        emission_path_csv = emission_path[:-4] + ".csv"
        print("\nGenerated emission file at " + emission_path_csv)

        # delete the .xml version of the emission file
        os.remove(emission_path)

    # if we wanted to save the render, here we create the movie
    if args.save_render:
        dirs = os.listdir(os.path.expanduser('~')+'/flow_rendering')
        # Ignore hidden files
        dirs = [d for d in dirs if d[0] != '.']
        dirs.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d-%H%M%S"))
        recent_dir = dirs[-1]
        # create the movie
        movie_dir = os.path.expanduser('~') + '/flow_rendering/' + recent_dir
        save_dir = os.path.expanduser('~') + '/flow_movies'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        os_cmd = "cd " + movie_dir + " && ffmpeg -i frame_%06d.png"
        os_cmd += " -pix_fmt yuv420p " + dirs[-1] + ".mp4"
        os_cmd += "&& cp " + dirs[-1] + ".mp4 " + save_dir + "/"
        os.system(os_cmd)
    return mean_speed, final_inflows, final_outflows 

def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional input parameters
    parser.add_argument(
        '--run',
        type=str,
        help='The algorithm or model to train. This may refer to '
             'the name of a built-on algorithm (e.g. RLLib\'s DQN '
             'or PPO), or a user-defined trainable function or '
             'class registered in the tune registry. '
             'Required for results trained with flow-0.2.0 and before.')
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--gen_emission',
        action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Specifies whether to use the \'evaluate\' reward '
             'for the environment.')
    parser.add_argument(
        '--render_mode',
        type=str,
        default='sumo_gui',
        help='Pick the render mode. Options include sumo_web3d, '
             'rgbd and sumo_gui')
    parser.add_argument(
        '--save_render',
        action='store_true',
        help='Saves a rendered video to a file. NOTE: Overrides render_mode '
             'with pyglet rendering.')
    parser.add_argument(
        '--horizon',
        type=int,
        help='Specifies the horizon.')
    parser.add_argument(
        '--warmup',
        type=int,
        default=800,)
    parser.add_argument('-o','--output',type=str,help='output file')
    parser.add_argument('--use_delay',type=int,default=-1,help='weather use time delay or not')
    parser.add_argument("-s","--use_seeds",dest = "use_seeds",help="name of pickle file containing seeds", default=None)
    return parser
if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    for i in range(len(seed_filename)):
        seed = seed_filename[i]
        print("Using seed: ", seed)
        ray.init(
            num_cpus=1,
            object_store_memory=1024*1024*1024)
        speed, inflow, outflow = visualizer_rllib(args, seed)
        print("Round ",i, ":", speed, inflow, outflow)
        ray.shutdown()
