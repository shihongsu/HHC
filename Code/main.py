import numpy as np
from ppo_agent import PPOAgent


# sort pat
def sort_map(job):
    # add indices
    idx = list(range(len(job)))
    # sort by lv -> twS -> twE
    sorted_lv_tw = sorted(idx, key = lambda i: (-job[i][2], job[i][0], job[i][1]))
    return sorted_lv_tw


if __name__ == '__main__':
	config = {
		"gpu": False, # True,
		"training_steps": 1e8, # 1e8,
		"update_sample_count": 128, # 10000,
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.2,
		"max_gradient_norm": 0.5,
		"batch_size": 128,
		"logdir": 'log/',
		"update_ppo_epoch": 3,
		"learning_rate": 2.5e-6,
		"value_coefficient": 0.5,
		"entropy_coefficient": 0.01,
		"horizon": 128, # 128,
		"eval_interval": 100, # 100,
		"eval_episode": 5,
	}

	dist = np.loadtxt("10-1-dist.txt")
	jobs = np.loadtxt("10-1-job.txt")
	agent = PPOAgent(config, jobs, dist)
	# agent.load("./log/")
	agent.train()
	agent.evaluate()
