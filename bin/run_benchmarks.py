import os
import argparse
from src.inference.inference import run_benchmarks
from src.utils.prepare_dataset import load_afri_speech_data


def main():
    dataset_path = "./data/intron-dev-public-3232.csv"
    intron_dataset = load_afri_speech_data(dataset_path)
    models_list = [
        'jonatasgrosman/wav2vec2-large-xlsr-53-english',
        "facebook/wav2vec2-large-960h",
        "jonatasgrosman/wav2vec2-xls-r-1b-english",
        "facebook/wav2vec2-large-960h-lv60-self",
        "facebook/hubert-large-ls960-ft",
        "facebook/wav2vec2-large-robust-ft-swbd-300h",
    ]

    for model_id in models_list:
        print(f"starting: {model_id}")
        run_benchmarks(model_id, intron_dataset)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_csv_path",
        type=str,
        default="./data/intron-dev-public-3232.csv",
        help="path to data csv file",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="./data/",
        help="directory to locate the audio",
    )
    parser.add_argument(
        "--model_id_or_path",
        type=str,
        default="facebook/hubert-large-ls960-ft",
        help="id of the model or path to huggingface model",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", 
        help="directory to store results"
    )
    parser.add_argument(
        "--max_audio_len_secs",
        type=int,
        default=17,
        help="maximum audio length passed to the inference model should",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="set gpu to -1 to use cpu",
    )

    return parser.parse_args()


if __name__ == "__main__":
    """Run main script"""

    args = parse_argument()

    # Make output directory if does not already exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    if "whisper" in args.model_id_or_path:
        test_dataset = load_afri_speech_data(
            data_path=args.data_csv_path,
            max_audio_len_secs=args.max_audio_len_secs,
            audio_dir=args.audio_dir,
            return_dataset=False, gpu=args.gpu
        )
    else:
        test_dataset = load_afri_speech_data(
            data_path=args.data_csv_path,
            max_audio_len_secs=args.max_audio_len_secs,
            audio_dir=args.audio_dir, gpu=args.gpu
        )

    run_benchmarks(
        model_id_or_path=args.model_id_or_path,
        test_dataset=test_dataset,
        output_dir=args.output_dir,
        gpu=args.gpu
    )
