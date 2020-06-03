from flow.utils.rllib import get_flow_params, get_rllib_config, get_rllib_pkl
import argparse

def create_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'result_dir',type=str)
	parser.add_argument(
		'checkpoint_num',type=str)
	return parser

def printConfig(args):
	result_dir = args.result_dir if args.result_dir[-1]!='/' else args.result_dir[:-1]
	config = get_rllib_config(result_dir)
	print(config)

if __name__ == '__main__':
	parser = create_parser()
	args = parser.parse_args()
	printConfig(args)
