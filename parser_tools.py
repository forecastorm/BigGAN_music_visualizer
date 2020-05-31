import argparse


def get_arguments():
    # get input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--song", default='coffin_dance.mp3')
    # scale resolution up when using GPU
    parser.add_argument("--resolution", default='128')
    parser.add_argument("--duration", type=int)
    parser.add_argument("--pitch_sensitivity", type=int, default=220)
    parser.add_argument("--tempo_sensitivity", type=float, default=0.25)
    parser.add_argument("--depth", type=float, default=1)
    parser.add_argument("--classes", nargs='+', type=int)
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
    return args