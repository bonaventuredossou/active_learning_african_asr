import os
import argparse
import configparser
import random
import subprocess
import time
import warnings
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm

os.environ['TRANSFORMERS_CACHE'] = '/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/results/'
os.environ['XDG_CACHE_HOME'] = '/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/results/'
os.environ["WANDB_DISABLED"] = "true"

import torch
from torch.utils.data import DataLoader
from datasets import load_metric
from transformers import (
    Wav2Vec2ForCTC,
    HubertForCTC,
    TrainingArguments)
from transformers.trainer_utils import get_last_checkpoint

from src.utils.text_processing import clean_text
from src.utils.prepare_dataset import DataConfig, data_prep, DataCollatorCTCWithPaddingGroupLen
from src.utils.sampler import IntronTrainer

num_gpus = [i for i in range(torch.cuda.device_count())]
if len(num_gpus) > 1:
    print("Let's use", num_gpus, "GPUs!")
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in num_gpus)

warnings.filterwarnings('ignore')
wer_metric = load_metric("wer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLING_RATE = 16000
PROCESSOR = None


def parse_argument():
    config = configparser.ConfigParser()
    parser = argparse.ArgumentParser(prog="Train")
    parser.add_argument("-c", "--config", dest="config_file",
                        help="Pass a training config file", metavar="FILE")
    parser.add_argument("--local_rank", type=int,
                        default=0)
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


def get_data_collator():
    return DataCollatorCTCWithPaddingGroupLen(processor=PROCESSOR, padding=True)


def compute_metric(pred):
    wer, _, _ = compute_wer(pred.predictions, pred.label_ids)
    return wer


def compute_wer(logits, label_ids):
    label_ids[label_ids == -100] = PROCESSOR.tokenizer.pad_token_id

    pred_ids = torch.argmax(torch.tensor(logits), axis=-1)
    predicted_transcription = PROCESSOR.batch_decode(pred_ids)[0]

    text = PROCESSOR.batch_decode(label_ids, group_tokens=False)[0]
    target_transcription = text.lower()

    wer = wer_metric.compute(predictions=[predicted_transcription],
                             references=[target_transcription])
    return {"wer": wer}, target_transcription, predicted_transcription


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

    # use last checkpoint if exist
    if last_checkpoint_:
        checkpoint = last_checkpoint_
    elif os.path.isdir(model_path):
        checkpoint = None
    else:
        checkpoint = None

    return last_checkpoint_, checkpoint


def set_dropout(trained_model):
    trained_model.eval()
    for name, module in trained_model.named_modules():
        if 'dropout' in name:
            module.train()


def get_pretrained_model(checkpoint_pretrained, config_):
    models_with_different_vocab = ['jonatasgrosman/wav2vec2-large-xlsr-53-english',
                                   'facebook/wav2vec2-large-960h-lv60-self',
                                   'Harveenchadha/vakyansh-wav2vec2-hindi-him-4200'
                                   ]
    CTC_model_class = Wav2Vec2ForCTC if 'hubert' not in config_['models']['model_path'] else HubertForCTC
    if config_['models']['model_path'] in models_with_different_vocab:
        from transformers.file_utils import hf_bucket_url, cached_path

        archive_file = hf_bucket_url(
            config_['models']['model_path'],
            filename='pytorch_model.bin'
        )
        resolved_archive_file = cached_path(archive_file)

        state_dict = torch.load(resolved_archive_file, map_location='cpu')
        state_dict.pop('lm_head.weight')
        state_dict.pop('lm_head.bias')

        model_ = CTC_model_class.from_pretrained(
            config_['models']['model_path'],
            state_dict=state_dict,
            attention_dropout=float(config_['hyperparameters']['attention_dropout']),
            hidden_dropout=float(config_['hyperparameters']['hidden_dropout']),
            feat_proj_dropout=float(config_['hyperparameters']['feat_proj_dropout']),
            mask_time_prob=float(config_['hyperparameters']['mask_time_prob']),
            layerdrop=float(config_['hyperparameters']['layerdrop']),
            ctc_loss_reduction=config_['hyperparameters']['ctc_loss_reduction'],
            ctc_zero_infinity=True,
            pad_token_id=PROCESSOR.tokenizer.pad_token_id,
            vocab_size=len(PROCESSOR.tokenizer)
        )

    else:
        model_ = CTC_model_class.from_pretrained(
            checkpoint_pretrained if checkpoint_pretrained else config_['models']['model_path'],
            attention_dropout=float(config_['hyperparameters']['attention_dropout']),
            hidden_dropout=float(config_['hyperparameters']['hidden_dropout']),
            feat_proj_dropout=float(config_['hyperparameters']['feat_proj_dropout']),
            mask_time_prob=float(config_['hyperparameters']['mask_time_prob']),
            layerdrop=float(config_['hyperparameters']['layerdrop']),
            ctc_loss_reduction=config_['hyperparameters']['ctc_loss_reduction'],
            ctc_zero_infinity=True,
            pad_token_id=PROCESSOR.tokenizer.pad_token_id,
            vocab_size=len(PROCESSOR.tokenizer)
        )
    if config_['hyperparameters']['gradient_checkpointing'] == "True":
        model_.gradient_checkpointing_enable()
    if config_['hyperparameters']['ctc_zero_infinity'] == "True":
        model_.config.ctc_zero_infinity = True

    print(f"\n...Model loaded in {time.time() - start:.4f}.\n")

    if config['hyperparameters']['freeze_feature_encoder'] == "True":
        model_.freeze_feature_encoder()

    if len(num_gpus) > 1:
        model_ = torch.nn.DataParallel(model_, device_ids=num_gpus)
    model_.to(device)
    return model_

def select_best_al_generated_transcript(generated_transcripts):
    predicted_transcript_wer_dict = {}
    for index_transcript in range(len(generated_transcripts)):
        current_transcript_list = list()
        current_target = generated_transcripts[index_transcript]
        for other_index_transcript in range(len(generated_transcripts)):
            if other_index_transcript != index_transcript:
                wer = wer_metric.compute(predictions=[generated_transcripts[other_index_transcript]],
                                         references=[current_target])

                current_transcript_list.append(wer)
        predicted_transcript_wer_dict[index_transcript] = np.array(current_transcript_list).mean()
    
    ordered_dict = dict(sorted(predicted_transcript_wer_dict.items(), key=lambda item: item[1], reverse=False))
    best_transcript = generated_transcripts[list(ordered_dict.keys())[0]]
    return best_transcript

def run_inference(trained_model, dataloader, mode='most', mc_dropout_rounds=10):
    if 'random' in mode.lower():
        audios_ids = [batch['audio_idx'] for batch in dataloader]
        random.shuffle(audios_ids)
        return {key[0]: 1.0 for key in
                audios_ids}  # these values are just dummy ones, to have a format similar to the two other cases
    else:
        audio_wers = {}
        for batch in tqdm(dataloader, desc="Uncertainty Inference"):
            input_val = batch['input_values'].to(device)
            wer_list = []
            for mc_dropout_round in range(mc_dropout_rounds):
                with torch.no_grad():
                    logits = trained_model(input_val).logits
                    batch["logits"] = logits

                pred_ids = torch.argmax(torch.tensor(batch["logits"]), dim=-1)
                pred = PROCESSOR.batch_decode(pred_ids)[0]
                batch["predictions"] = clean_text(pred)
                try:
                    batch["wer"] = wer_metric.compute(
                        predictions=[batch["predictions"]], references=[batch["transcript"]]
                    )
                    wer_list.append(batch['wer'])
                except:
                    pass

            if len(wer_list) > 0:
                uncertainty_score = np.array(wer_list).std()
                audio_wers[batch['audio_idx'][0]] = uncertainty_score

        return dict(sorted(audio_wers.items(), key=lambda item: item[1], reverse=True))

if __name__ == "__main__":

    args, config = parse_argument()
    checkpoints_path = train_setup(config, args)
    data_config = data_setup(config)
    train_dataset, val_dataset, aug_dataset, PROCESSOR = data_prep(data_config)
    data_collator = get_data_collator()

    start = time.time()
    # Detecting last checkpoint.
    last_checkpoint, checkpoint_ = get_checkpoint(checkpoints_path, config['models']['model_path'])

    CTC_model_class = Wav2Vec2ForCTC if 'hubert' not in config['models']['model_path'] else HubertForCTC

    models_with_different_vocab = ['jonatasgrosman/wav2vec2-large-xlsr-53-english',
                                   'facebook/wav2vec2-large-960h-lv60-self',
                                   'Harveenchadha/vakyansh-wav2vec2-hindi-him-4200'
                                   ]

    print(f"model starting...from last checkpoint:{last_checkpoint}")
    model = get_pretrained_model(last_checkpoint, config)
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        overwrite_output_dir=True if config['hyperparameters']['overwrite_output_dir'] == "True" else False,
        group_by_length=True if config['hyperparameters']['group_by_length'] == "True" else False,
        length_column_name=config['hyperparameters']['length_column_name'],
        data_seed=int(config['hyperparameters']['data_seed']),
        per_device_train_batch_size=int(config['hyperparameters']['train_batch_size']),
        per_device_eval_batch_size=int(config['hyperparameters']['val_batch_size']),
        gradient_accumulation_steps=int(config['hyperparameters']['gradient_accumulation_steps']),
        gradient_checkpointing=True if config['hyperparameters']['gradient_checkpointing'] == "True" else False,
        ddp_find_unused_parameters=True if config['hyperparameters']['ddp_find_unused_parameters'] == "True" else False,
        evaluation_strategy="steps",
        num_train_epochs=int(config['hyperparameters']['num_epochs']),
        fp16=torch.cuda.is_available(),
        save_steps=int(config['hyperparameters']['save_steps']),
        eval_steps=int(config['hyperparameters']['eval_steps']),
        logging_steps=int(config['hyperparameters']['logging_steps']),
        learning_rate=float(config['hyperparameters']['learning_rate']),
        warmup_steps=int(config['hyperparameters']['warmup_steps']),
        save_total_limit=int(config['hyperparameters']['save_total_limit']),
        dataloader_num_workers=int(config['hyperparameters']['dataloader_num_workers']),
        logging_first_step=True,
        load_best_model_at_end=True if config['hyperparameters']['load_best_model_at_end'] == 'True' else False,
        metric_for_best_model='eval_wer',
        greater_is_better=False,
        ignore_data_skip=True if config['hyperparameters']['ignore_data_skip'] == 'True' else False,
        report_to=None
    )

    aug_batch_size = int(config['hyperparameters']['aug_batch_size'])
    sampling_mode = str(config['hyperparameters']['sampling_mode']).strip()
    mc_dropout_round = int(config['hyperparameters']['mc_dropout_round'])
    augmentation_dataloader = DataLoader(aug_dataset, batch_size=aug_batch_size)
    k = int(config['hyperparameters']['top_k'])
    results = pd.DataFrame()    
    mc_dropout_rounds_list = [1, mc_dropout_round]

    for round in range(3):
        filename = 'Top-{}_AL_Round_{}_Mode_{}_WERS.csv'.format(k, round, 'most')
        filepath = os.path.join('../../data/', filename)
        for mc_round in mc_dropout_rounds_list:
            model.eval()
            if mc_round > 1: # meaning we are running uncertainty
                set_dropout(model)
            samples_uncertainty = run_inference(model, augmentation_dataloader,
                                                    mode='most', mc_dropout_rounds=mc_round)
            if mc_round > 1:
                results['uncertainty_wer'] = list(samples_uncertainty.values())
            else:
                results['audios_ids'] = list(samples_uncertainty.keys())
                results['inference_wer'] = list(samples_uncertainty.values())

        results.to_csv(filepath, index = False)