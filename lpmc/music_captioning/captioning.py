import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from lpmc.music_captioning.model.bart import BartCaptionModel
from lpmc.utils.eval_utils import load_pretrained
from lpmc.utils.audio_utils import load_audio, STR_CH_FIRST
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument("--framework", default="transfer", type=str)
parser.add_argument("--caption_type", default="lp_music_caps", type=str)
parser.add_argument("--max_length", default=128, type=int)
parser.add_argument("--num_beams", default=5, type=int)
parser.add_argument("--model_type", default="last", type=str)
parser.add_argument("--audio_path", default="../../dataset/samples/orchestra.wav", type=str)

def get_audio(audio_path, duration=10, target_sr=16000):
    n_samples = int(duration * target_sr)
    audio, sr = load_audio(
        path= audio_path,
        ch_format= STR_CH_FIRST,
        sample_rate= target_sr,
        downmix_to_mono= True,
    )
    if len(audio.shape) == 2:
        audio = audio.mean(0, False)  # to mono
    input_size = int(n_samples)
    if audio.shape[-1] < input_size:  # pad sequence
        pad = np.zeros(input_size)
        pad[: audio.shape[-1]] = audio
        audio = pad
    ceil = int(audio.shape[-1] // n_samples)
    audio = torch.from_numpy(np.stack(np.split(audio[:ceil * n_samples], ceil)).astype('float32'))
    return audio
    
def main():
    args = parser.parse_args()
    captioning(args)
 
def captioning(args):
    save_dir = f"exp/{args.framework}/{args.caption_type}/"
    config = OmegaConf.load(os.path.join(save_dir, "hparams.yaml"))
    model = BartCaptionModel(max_length = config.max_length)
    model, save_epoch = load_pretrained(args, save_dir, model, mdp=config.multiprocessing_distributed)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    model.eval()
    
    audio_tensor = get_audio(audio_path = args.audio_path)
    if args.gpu is not None:
        audio_tensor = audio_tensor.cuda(args.gpu, non_blocking=True)

    with torch.no_grad():
        output = model.generate(
            samples=audio_tensor,
            num_beams=args.num_beams,
        )
    inference = {}
    number_of_chunks = range(audio_tensor.shape[0])
    for chunk, text in zip(number_of_chunks, output):
        time = f"{chunk * 10}:00-{(chunk + 1) * 10}:00"
        item = {"text":text,"time":time}
        inference[chunk] = item
        print(item)

if __name__ == '__main__':
    main()

    