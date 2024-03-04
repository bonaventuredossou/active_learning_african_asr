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

import torch
from torch.utils.data import DataLoader
from datasets import load_metric
from transformers import (
    Wav2Vec2ForCTC,
    HubertForCTC,
    TrainingArguments)
from transformers.trainer_utils import get_last_checkpoint

from src.utils.text_processing import clean_text
from src.utils.prepare_dataset import *
from src.utils.utils import *

num_gpus = [i for i in range(torch.cuda.device_count())]
if len(num_gpus) > 1:
    print("Let's use", num_gpus, "GPUs!")
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in num_gpus)

warnings.filterwarnings('ignore')
wer_metric = load_metric("wer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLING_RATE = 16000
PROCESSOR = None


def get_data_collator():
    return DataCollatorCTCWithPaddingGroupLen(processor=PROCESSOR, padding=True)


def compute_metric(pred):
    wer, _, _ = compute_wer(pred.predictions, pred.label_ids)
    return wer

def isNaN(string):
    return string != string

def compute_wer(logits, reference):
    pred_ids = torch.argmax(torch.tensor(logits), axis=-1)
    predicted_transcription = clean_text(PROCESSOR.batch_decode(pred_ids)[0])
    target_transcription = reference.lower()
    predicted_transcription = clean_text(predicted_transcription.lower())
    target_transcription = clean_text(target_transcription.lower())
    
    if isNaN(predicted_transcription):
        predicted_transcription = ""

    if isNaN(target_transcription):
        target_transcription = ""

    wer = wer_metric.compute(predictions=[predicted_transcription],
                             references=[target_transcription])
    return wer, target_transcription, predicted_transcription


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


def evaluate(trained_model, dataloader):
    wers, references, predictions, audio_paths, accents = [], [], [], [], []

    for batch in tqdm(dataloader, desc="Inference"):
        input_val = batch['input_values'].to(device)
        batch["reference"] = clean_text(batch['transcript'][0])

        with torch.no_grad():
            logits = trained_model(input_val).logits
            batch["logits"] = logits

        pred_ids = torch.argmax(torch.tensor(batch["logits"]), dim=-1)
        pred = PROCESSOR.batch_decode(pred_ids)[0]
        batch["predictions"] = clean_text(pred)

        wer, reference, prediction = compute_wer(logits, batch['transcript'][0])
        
        if (len(str(reference).strip()) > 0) and (len(str(prediction).strip()) > 0):        
            wers.append(wer)
            references.append(reference)
            predictions.append(prediction)
            audio_paths.extend(batch['audio_paths'])
            accents.extend(batch['accent'])

    return np.array(wers).mean(), references, predictions, audio_paths, accents

if __name__ == "__main__":

    args, config = parse_argument()
    checkpoints_path = train_setup(config, args)
    data_config = data_setup(config)

    train_dataset, val_dataset, aug_dataset, PROCESSOR = data_prep(data_config)

    test_data_set = load_custom_dataset(data_config, args.data_csv_path, 'test', 
                        transform_audio, transform_labels)
    
    start = time.time()
    # last_checkpoint, checkpoint_ = get_checkpoint(checkpoints_path, config['models']['model_path'])

    CTC_model_class = Wav2Vec2ForCTC if 'hubert' not in config['models']['model_path'] else HubertForCTC

    models_with_different_vocab = ['jonatasgrosman/wav2vec2-large-xlsr-53-english',
                                   'facebook/wav2vec2-large-960h-lv60-self',
                                   'Harveenchadha/vakyansh-wav2vec2-hindi-him-4200'
                                   ]
    model = get_pretrained_model(config['models']['model_path'], config)
    
    training_args = TrainingArguments(
        output_dir=config['models']['model_path'],
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
    augmentation_dataloader = DataLoader(test_data_set, batch_size=1)
    results = pd.DataFrame()

    wer, transcripts, predictions, paths, accents = evaluate(model, augmentation_dataloader)

    results['references'] = transcripts
    results['predictions'] = predictions
    results['audio_paths'] = paths
    results['accents'] = accents

    write_pred(config['models']['model_path'], results, wer, accents[0])
