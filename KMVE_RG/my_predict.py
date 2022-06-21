import torch
import numpy as np
from modules.tokenizers import  Tokenizer
from modules.dataloaders import MyDataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.predicter import Predictor
from modules.loss import compute_loss
from models.SGF_model import SGF
from config import config as args

def main():
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    test_dataloader = MyDataLoader(args, tokenizer, split='test', shuffle=False, evaluate=True)

    # build model architecture
    model = SGF(args, tokenizer)
    best_pth = args.models
    pretrained_dict = torch.load(best_pth)['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('init from:', best_pth)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Predictor(model, criterion, metrics, optimizer, args, lr_scheduler, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()
