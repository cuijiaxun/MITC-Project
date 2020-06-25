"""Runs the environments located in flow/benchmarks.

The environment file can be modified in the imports to change the environment
this runner script is executed on. This file runs the PPO algorithm in rllib
and utilizes the hyper-parameters specified in:
Proximal Policy Optimization Algorithms by Schulman et. al.
"""
import json
import argparse

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

EXAMPLE_USAGE = """
example usage:
    python ppo_runner.py grid0
Here the arguments are:
benchmark_name - name of the benchmark to run
num_rollouts - number of rollouts to train across
num_cpus - number of cpus to use for training
"""

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="[Flow] Evaluates a Flow Garden solution on a benchmark.",
    epilog=EXAMPLE_USAGE)

# required input parameters
parser.add_argument(
    "--upload_dir", type=str, help="S3 Bucket to upload to.")

# required input parameters
parser.add_argument(
    "--benchmark_name", type=str, help="File path to solution environment.")

# optional input parameters
parser.add_argument(
    '--num_rollouts',
    type=int,
    default=50,
    help="The number of rollouts to train over.")
parser.add_argument(
    '--training_iterations',
    type=int,
    default=500,
    help="The number of iterations to train over.")

# optional input parameters
parser.add_argument(
    '--memory',
    type=int,
    default=4,
    help="memory in GiB")


# optional input parameters
parser.add_argument(
    '--num_cpus',
    type=int,
    default=2,
    help="The number of cpus to use.")

parser.add_argument(
    '--num_gpus',
    type=int,
    default=0,
    help="The number of gpus to use.")

parser.add_argument("-s", "--seeds_file", dest="seeds_file",
                    help="pickle file containing seeds", default=None)

parser.add_argument(
        '--lr',
        type=float,
        default=5e-4,
        help="learning rate")


if __name__ == "__main__":
    benchmark_name = 'grid0'
    args = parser.parse_args()
    # benchmark name
    benchmark_name = args.benchmark_name
    # number of rollouts per training iteration
    num_rollouts = args.num_rollouts
    # number of parallel workers
    num_cpus = args.num_cpus

    upload_dir = args.upload_dir

    # Import the benchmark and fetch its flow_params
    benchmark = __import__(
        "flow.benchmarks.%s" % benchmark_name, fromlist=["flow_params"])
    flow_params = benchmark.flow_params

    # get the env name and a creator for the environment
    create_env, env_name = make_create_env(params=flow_params, version=0, seeds_file=args.seeds_file)

    # initialize a ray instance
    ray.init(object_store_memory=args.memory*1024*1024*1024)

    alg_run = "PPO"

    horizon = flow_params["env"].horizon
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = min(num_cpus, num_rollouts)
    config["num_gpus"] = args.num_gpus
    config["train_batch_size"] = int(horizon * num_rollouts)
    #config["sgd_minibatch_size"] = 1024
    config["use_gae"] = True
    config["horizon"] = horizon
    gae_lambda = 0.97
    step_size = 5e-4
    if benchmark_name == "grid0":
        gae_lambda = 0.5
        step_size = 5e-5
    elif benchmark_name == "grid1":
        gae_lambda = 0.3
    config["gamma"] = 0.999
    config["lambda"] = gae_lambda
    config["lr"] = args.lr
    
    config["lr_schedule"] = [
            [0, args.lr],
            [1000000,args.lr/1.5],
            [4000000,args.lr/5]]
    
    config["vf_clip_param"] = 1e6
    config["num_sgd_iter"] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["model"]["fcnet_hiddens"] = [100, 50, 25]
    config["observation_filter"] = "NoFilter"
    #config["entropy_coeff"] = 0.01
    #config["kl_coeff"] = 0.5
    #config["kl_target"] = 0.02
    #config["seed"] = 123 # seed for PPO?

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    # Register as rllib env
    register_env(env_name, create_env)

    exp_tag = {
        "run": alg_run,
        "env": env_name,
        "config": {
            **config
        },
        "checkpoint_freq": 25,
        "max_failures": 999,
        "stop": {
            "training_iteration": args.training_iterations,
        },
        "num_samples": 2,

    }

    if upload_dir:
        exp_tag["upload_dir"] = "s3://" + upload_dir

    trials = run_experiments({
        flow_params["exp_tag"]: exp_tag
    })
