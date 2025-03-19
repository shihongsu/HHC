import numpy as np
from ppo_agent import PPOAgent


if __name__ == '__main__':
	config = {
		"gpu": False,
		"training_steps": 100, # 1e8,
		"update_sample_count": 10, # 10000,
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
		"horizon": 16, # 128,
		"eval_interval": 10, # 100,
		"eval_episode": 5,
	}

	distance = np.loadtxt("1-1n10-walk.txt")
	patients = np.loadtxt("1-1n10-job.txt")
	agent = PPOAgent(config, patients, distance, caregivers = [])
	# agent.load("./log/")
	agent.train()
	agent.evaluate()


