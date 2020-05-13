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

