__author__ = 'Rio'
from yacs.config import CfgNode as CN
config = CN()
config.image_dir = '../data/images'
config.ann_path = '../data/my_annotation.json'
config.models = '../save_models/model_best.pth'

# Data loader settings
config.max_seq_length = 60
config.threshold = 3
config.num_workers = 0  # the number of workers for dataloader
config.batch_size = 16  # the number of samples for a batch
config.evaluate_batch = 1 # the number of samples in evaluate part

# Model settings (for visual extractor)
config.visual_extractor = 'resnet101'
config.visual_extractor_pretrained = True  # 'whether to load the pretrained visual

# Model settings (for Transformer)
config.d_model = 512  # the dimension of Transformer.
config.d_ff = 512  # the dimension of FFN.
config.d_vf = 2048  # the dimension of the patch features.
config.num_heads = 8  # the number of heads in Transformer.
config.num_layers = 3  # the number of layers of Transformer.
config.dropout = 0.1  # the dropout rate of Transformer.
config.logit_layers = 1  # the number of the logit layer.
config.bos_idx = 0  # the index of <bos>.
config.eos_idx = 0  # the index of <eos>.
config.pad_idx = 0  # the index of <pad>.
config.use_bn = 0  # whether to use batch normalization.
config.drop_prob_lm = 0.5  # the dropout rate of the output layer.

# Sample related
config.sample_n = 1  # the sample number per image.
config.output_logsoftmax = 1  # whether to output the probabilities.
config.decoding_constraintt = 0  # whether decoding constraint.


# Trainer settings
config.n_gpu = 1  # the number of gpus to be used.
config.epochs = 50  # the number of training epochs.
config.save_dir = '../result/models/'  # the patch to save the save_models.
config.record_dir = '../result/records/'  # the patch to save the results of experiments
config.save_period = 1  # the saving period.
config.monitor_mode = 'max'  # choices=['min', 'max'],help='whether to max or min the metric.
config.monitor_metric = 'BLEU_4'  # the metric to be monitored.
config.early_stop = 100  # the patience of training.

# Optimization
config.optim = 'Adam'  # the type of the optimizer.
config.lr_ve = 5e-5  # the learning rate for the visual extractor.
config.lr_ed = 1e-3  # the learning rate for the remaining parameters.
config.weight_decay = 5e-5  # 'the weight decay.
config.amsgrad = True

# Learning Rate Scheduler
config.lr_scheduler = 'StepLR'  # the type of the learning rate scheduler.
config.step_size = 28  # he step size of the learning rate scheduler.
config.gamma = 0.1  # the gamma of the learning rate scheduler.

# Others
config.seed = 9233
config.resume = None  # whether to resume the training from existing checkpoints.

# RNN_Encoder
config.embedding_vector = 300
config.nhidden = 512
config.nlayers = 1
config.bidirectional = True
config.rnn_type = 'LSTM'

# text_image_losses.py
config.cuda = True
config.train_smooth_gamma3 = 10.0
config.train_smooth_gamma2 = 5.0
config.train_smooth_gamma1 = 4.0
config.attn_pth = '../attn_pth/'
