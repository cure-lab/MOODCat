def GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config):
	def train(x, y, epoch, batch_size, target_map = None, r_mixup=0.0):
		G.optim.zero_grad()#  altertive G.optim.zero_grad(set_to_none=True): can modestly improv performance.
		D.optim.zero_grad()
		

		if config["unet_mixup"]:
			real_target = torch.tensor([1.0]).cuda()
			fake_target = torch.tensor([0]).cuda()
	
		if config