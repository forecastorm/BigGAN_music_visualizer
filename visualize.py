import librosa
import argparse
import numpy as np
import moviepy.editor as mpy
import random
import torch
from scipy.misc import toimage
from tqdm import tqdm
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)

if __name__ == '__main__':
    # get input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--song",default='coffin_dance.mp3')
    parser.add_argument("--resolution", default='512')

    parser.add_argument("--pitch_sensitivity", type=int, default=220)
    parser.add_argument("--tempo_sensitivity", type=float, default=0.25)
    parser.add_argument("--depth", type=float, default=1)
    parser.add_argument("--frame_length", type=int, default=512)
    args = parser.parse_args()

    # read song
    if args.song:
        song = args.song
    else:
        raise ValueError("name of the song must be specified in --song argument")

    # set model name based on resolution
    # model_name = 'big_gan_deep_512'
    model_name = 'big_gan_deep_'+args.resolution
    # frame length = 512
    frame_length = args.frame_length

    # set pitch sensitivity
    # pitch sensitivity = 80.0
    pitch_sensitivity = (300-args.pitch_sensitivity) * 512 / frame_length

    # set tempo sensitivity
    # tempo_sensitivity = 0.25
    tempo_sensitivity = args.tempo_sensitivity * frame_length / 512

    # set depth
    # depth = 1
    depth = args.depth

    # set number of classes

