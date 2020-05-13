import librosa
import argparse
import numpy as np
import moviepy.editor as mpy
import random
import torch
from scipy.misc import toimage
from tqdm import tqdm
from keras.utils.vis_utils import plot_model
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)

if __name__ == '__main__':
    # get input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--song", default='coffin_dance.mp3')
    # scale resolution up when using GPU
    parser.add_argument("--resolution", default='128')
    parser.add_argument("--duration", type=int)
    parser.add_argument("--pitch_sensitivity", type=int, default=220)
    parser.add_argument("--tempo_sensitivity", type=float, default=0.25)
    parser.add_argument("--depth", type=float, default=1)
    parser.add_argument("--num_classes", type=int, default=12)
    parser.add_argument("--sort_classes_by_power", type=int, default=0)
    parser.add_argument("--jitter", type=float, default=0.5)
    parser.add_argument("--frame_length", type=int, default=512)
    parser.add_argument("--truncation", type=float, default=1)
    parser.add_argument("--smooth_factor", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--use_previous_classes", type=int, default=0)
    parser.add_argument("--use_previous_vectors", type=int, default=0)
    parser.add_argument("--output_file", default="coffin_dance.mp4")

    args = parser.parse_args()

    # read song
    if args.song:
        song = args.song
        print('\nReading audio \n')
        y, sr = librosa.load(song)
    else:
        raise ValueError("name of the song must be specified in --song argument")

    # set model name based on resolution
    # model_name = 'biggan-deep-512'
    model_name = 'biggan-deep-' + args.resolution
    # frame length = 512
    frame_length = args.frame_length

    # set pitch sensitivity
    # pitch sensitivity = 80.0
    pitch_sensitivity = (300 - args.pitch_sensitivity) * 512 / frame_length

    # set tempo sensitivity
    # tempo_sensitivity = 0.25
    tempo_sensitivity = args.tempo_sensitivity * frame_length / 512

    # set depth
    # depth = 1
    depth = args.depth

    # set number of classes
    # num_classes = 12
    num_classes = args.num_classes

    # set sort_classes_by_power
    # sort_classes_by_power = 0
    sort_classes_by_power = args.sort_classes_by_power

    # set jitter
    # jitter = 0.5
    jitter = args.jitter

    # set truncation
    # truncation = 1
    truncation = args.truncation

    # set batch size (number of samples for each batch)
    # batch_size = 30
    batch_size = args.batch_size

    # set use_previous_classes
    # use_previous_classes = 0
    use_previous_classes = args.use_previous_classes

    # set use_previous_vectors
    # use_previous_vectors = 0
    use_previous_vectors = args.use_previous_vectors

    # set output name
    # 'coffin_dance.mp4'
    output_file_name = args.output_file

    # set smooth factor
    # smooth_factor = 20
    if args.smooth_factor > 1:
        smooth_factor = int(args.smooth_factor * 512 / frame_length)
    else:
        smooth_factor = args.smooth_factor

    # set duration
    # frame_lim = 222
    if args.duration:
        seconds = args.duration
        frame_lim = int(np.floor(seconds * 22050 / frame_length / batch_size))
    else:
        frame_lim = int(np.floor(len(y) / sr * 22050 / frame_length / batch_size))

    # load pre-trained model
    model = BigGAN.from_pretrained(model_name)

    # set device
    # no cuda on mac
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create spectrogram
    # spec shape by (128, 6688)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, hop_length=frame_length)

    # get mean power at each time point
    # specm shape by (6688)
    specm = np.mean(spec,axis=0)
    print(specm.shape)