import openke
from openke.config import Trainer, Tester
from openke.module.model import RotatE
from openke.module.loss import SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/LUMB/",
	batch_size = 20,
	threads = 8,
	sampling_mode = "cross", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 64,
	neg_rel = 0
)

# dataloader for test
#test_dataloader = TestDataLoader("./benchmarks/LUMB/", "link")

# define the model
rotate = RotatE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 1024,
	margin = 6.0,
	epsilon = 2.0,
)

# define the loss function
model = NegativeSampling(
	model = rotate, 
	loss = SigmoidLoss(adv_temperature = 2),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.0
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 60, alpha = 0.5, use_gpu = False, opt_method = "adam")
trainer.run()
rotate.save_checkpoint('./checkpoint/rotate_lumb.ckpt')

# test the model
# rotate.load_checkpoint('./checkpoint/rotate.ckpt')
# tester = Tester(model = rotate, data_loader = test_dataloader, use_gpu = False)
# tester.run_link_prediction(type_constrain = False)
