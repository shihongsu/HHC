import numpy as np
from ppo_agent import PPOAgent


if __name__ == '__main__':

	config = {
		"gpu": False,
		"training_steps": 1e8,
		"update_sample_count": 10000,
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
		"horizon": 128,
		"eval_interval": 100,
		"eval_episode": 5,
	}

	dist = [[0, 3, 5], 
		 	[3, 0, 4], 
			[5, 4, 0]]
	job = [[20, 100, 3, 40], 
			[40, 80, 1, 20], 
			[0, 60, 2, 30]]

	agent = PPOAgent(config)
	# agent.load("./log/")
	agent.train()
	agent.evaluate()


