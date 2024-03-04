
import os
import argparse 
import configparser
import subprocess
import pandas as pd
from pathlib import Path
from transformers.trainer_utils import get_last_checkpoint
from src.utils.prepare_dataset import DataConfig
import torchaudio
import librosa
import soundfile as sf

def write_pred(model_id_or_path, results, wer, accent, cols=None, output_dir="./results"):
    """
    Write model predictions to file
    :param cols: List[str]
    :param output_dir: str
    :param model_id_or_path: str
    :param results: Dataset instance
    :param wer: float
    :return: DataFrame
    """
    model_id_or_path = model_id_or_path.replace("/", "-")
    output_path = f"{output_dir}/{accent}-{model_id_or_path}-wer-{round(wer, 4)}-{len(results)}.csv"
    results.to_csv(output_path, index=False)

def parse_argument():
    config = configparser.ConfigParser()
    parser = argparse.ArgumentParser(prog="Train")
    parser.add_argument("-c", "--config", dest="config_file",
                        help="Pass a training config file", metavar="FILE")
    parser.add_argument("--local_rank", type=int,
                        default=0)
    parser.add_argument(
        "--data_csv_path",
        type=str,
        default="../../data/intron-train-public-58000-clean.csv",
        help="path to data csv file",
    )
    args = parser.parse_args()
    config.read(args.config_file)
    return args, config

def train_setup(config, args):
    exp_dir = config['experiment']['dir']
    checkpoints_path = config['checkpoints']['checkpoints_path']
    figure_path = config['logs']['figure_path']
    predictions_path = config['logs']['predictions_path']

    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    subprocess.call(['cp', args.config_file, f"{exp_dir}/{args.config_file.split('/')[-1]}"])
    Path(checkpoints_path).mkdir(parents=True, exist_ok=True)
    Path(figure_path).mkdir(parents=True, exist_ok=True)
    Path(predictions_path).mkdir(parents=True, exist_ok=True)

    print(f"using exp_dir: {exp_dir}. Starting...")

    return checkpoints_path


def data_setup(config):
    data_config = DataConfig(
        train_path=config['data']['train'],
        val_path=config['data']['val'],
        aug_path=config['data']['aug'] if 'aug' in config['data'] else None,
        aug_percent=float(config['data']['aug_percent']) if 'aug_percent' in config['data'] else None,
        exp_dir=config['experiment']['dir'],
        ckpt_path=config['checkpoints']['checkpoints_path'],
        model_path=config['models']['model_path'],
        audio_path=config['audio']['audio_path'],
        max_audio_len_secs=int(config['hyperparameters']['max_audio_len_secs']),
        min_transcript_len=int(config['hyperparameters']['min_transcript_len']),
        domain=config['data']['domain'],
        seed=int(config['hyperparameters']['data_seed']),
    )
    return data_config

def get_checkpoint(checkpoint_path, model_path):
    last_checkpoint_ = None
    if os.path.isdir(checkpoint_path):
        last_checkpoint_ = get_last_checkpoint(checkpoint_path)
        if last_checkpoint_ is None and len(os.listdir(checkpoint_path)) > 0:
            print(
                f"Output directory ({checkpoint_path}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint_ is not None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint_}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    if last_checkpoint_:
        checkpoint = last_checkpoint_
    elif os.path.isdir(model_path):
        checkpoint = None
    else:
        checkpoint = None
    return last_checkpoint_, checkpoint