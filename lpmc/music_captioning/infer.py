import argparse
import os
import json
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from lpmc.music_captioning.datasets.mc import MC_Dataset
from lpmc.music_captioning.model.bart import BartCaptionModel
from lpmc.utils.eval_utils import load_pretrained
from tqdm import tqdm
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--data_dir', type=str, default="../../dataset")
parser.add_argument('--framework', type=str, default="supervised")
parser.add_argument("--caption_type", default="gt", type=str)
parser.add_argument('--arch', default='transformer')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=1e-9, type=float)
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--print_freq', default=50, type=int)
parser.add_argument("--cos", default=True, type=bool)
parser.add_argument("--label_smoothing", default=0.1, type=float)
parser.add_argument("--max_length", default=128, type=int)
parser.add_argument("--num_beams", default=5, type=int)
parser.add_argument("--model_type", default="last", type=str)


def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    test_dataset = MC_Dataset(
        data_path = args.data_dir,
        split="test",
        caption_type = "gt"
    )
    print(len(test_dataset))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    model = BartCaptionModel(
        max_length = args.max_length,
        label_smoothing = args.label_smoothing,
    )
    eval(args, model, test_dataset, test_loader, args.num_beams)
 
def eval(args, model, test_dataset, test_loader, num_beams=5):
    save_dir = f"exp/{args.framework}/{args.caption_type}/"
    config = OmegaConf.load(os.path.join(save_dir, "hparams.yaml"))
    model, save_epoch = load_pretrained(args, save_dir, model, mdp=config.multiprocessing_distributed)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    model.eval()
    
    inference_results = {}
    idx = 0
    for batch in tqdm(test_loader):
        fname, text,audio_tensor = batch
        if args.gpu is not None:
            audio_tensor = audio_tensor.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            output = model.generate(
                samples=audio_tensor,
                num_beams=num_beams,
            )
        for audio_id, gt, pred in zip(fname, text, output):
            inference_results[idx] = {
                "predictions": pred,
                "true_captions": gt,
                "audio_id": audio_id
            }
            idx += 1
    
    with open(os.path.join(save_dir, f"inference.json"), mode="w") as io:
        json.dump(inference_results, io, indent=4)

if __name__ == '__main__':
    main()

    