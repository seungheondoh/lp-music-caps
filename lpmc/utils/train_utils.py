import os
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf

def save_hparams(args, save_path):
    save_config = OmegaConf.create(vars(args))
    os.makedirs(save_path, exist_ok=True)
    OmegaConf.save(config=save_config, f= os.path.join(save_path, "hparams.yaml"))

class EarlyStopping():
    def __init__(self, min_max="min", tolerance=20, min_delta=1e-9):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.min_max = min_max
        self.counter = 0
        self.early_stop = False

    def min_stopping(self, valid_loss, best_valid_loss):
        if (valid_loss - best_valid_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
        else:
            self.counter = 0

    def max_stopping(self, valid_acc, best_valid_acc):
        if (best_valid_acc - valid_acc) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
        else:
            self.counter = 0

    def __call__(self, valid_metric, best_metic):
        if self.min_max == "min":
            self.min_stopping(valid_metric, best_metic)
        elif self.min_max == "max":
            self.max_stopping(valid_metric, best_metic)
        else:
            raise ValueError(f"Unexpected split name: {self.min_max}")

class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)

    def log_train_loss(self, loss, steps):
        self.add_scalar('train_loss', loss.item(), steps)

    def log_val_loss(self, loss, epochs):
        self.add_scalar('valid_loss', loss.item(), epochs)
    
    def log_caption_matric(self, metric, epochs, name="acc"):
        self.add_scalar(f'{name}', metric, epochs)

    def log_logitscale(self, logitscale, epochs):
        self.add_scalar('logit_scale', logitscale.item(), epochs)

    def log_learning_rate(self, lr, epochs):
        self.add_scalar('lr', lr, epochs)

    def log_learning_rate(self, lr, epochs):
        self.add_scalar('lr', lr, epochs)

    def log_roc(self, roc, epochs):
        self.add_scalar('roc', roc, epochs)

    def log_pr(self, pr, epochs):
        self.add_scalar('pr', pr, epochs)

class AverageMeter(object):
    def __init__(self,name, fmt, init_steps=0):
        self.name = name
        self.fmt = fmt
        self.steps = init_steps
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.num = 0
        self.avg = 0.0

    def step(self, val, num=1):
        self.val = val
        self.sum += num*val
        self.num += num
        self.steps += 1
        self.avg = self.sum/self.num

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def load_pretrained(pretrain_path, model):
    checkpoint= torch.load(pretrain_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.1.mlp'):
            state_dict[k[len("module.encoder_q.0."):]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict)
    return model