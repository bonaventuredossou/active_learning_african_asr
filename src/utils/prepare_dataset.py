import logging
import os
import time
import json
import sys
from datetime import datetime
import pandas as pd
import subprocess

os.environ['TRANSFORMERS_CACHE'] = '/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/results/'
os.environ['XDG_CACHE_HOME'] = '/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/results/'

from datasets import load_dataset, load_metric, Dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import librosa
import torch
from transformers import (
    Wav2Vec2Tokenizer,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from src.utils.audio_processing import AudioConfig, load_audio_file
from src.utils.text_processing import clean_text

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging_level = logging.DEBUG
logger.setLevel(logging_level)

PROCESSOR = None
CONFIG = None
MAX_MODEL_AUDIO_LEN_SECS = 87


class DataConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

            
def load_afri_speech_data(
    data_path, max_audio_len_secs=17, audio_dir="/network/scratch/b/bonaventure.dossou/AfriSpeech-100/", 
    return_dataset=True, split="dev", gpu=-1, domain='all',
    max_transcript_len=-1, min_transcript_len=-1
):
    """
    load train/dev/test data from csv path.
    :param max_transcript_len:
    :param min_transcript_len:
    :param domain:
    :param gpu:
    :param split:
    :param return_dataset:
    :param audio_dir:
    :param max_audio_len_secs: int
    :param data_path: str
    :return: Dataset instance
    """
    data = pd.read_csv(data_path)
    if split == 'aug':
        data["audio_paths"] = data["audio_paths"].apply(
            lambda x: x.replace(f"/AfriSpeech-100/", audio_dir)
        )
    else:
        data["audio_paths"] = data["audio_paths"].apply(
            lambda x: x.replace(f"/AfriSpeech-100/", audio_dir)
        )
    
    if max_audio_len_secs > -1 and gpu != -1:
        # when gpu is available, it cannot fit long samples
        data = data[data.duration < max_audio_len_secs]
    elif gpu == -1 and max_audio_len_secs > MAX_MODEL_AUDIO_LEN_SECS:
        # if cpu, infer all samples, no filtering
        pass
    elif gpu == -1 and max_audio_len_secs != -1:
        # if cpu, infer only long samples
        # assuming gpu has inferred on all short samples
        data = data[data.duration >= max_audio_len_secs]
    else:
        # Check if any of the sample is longer than
        # the GPU global MAX_MODEL_AUDIO_LEN_SECS
        if (gpu != -1) and (data.duration.to_numpy() > MAX_MODEL_AUDIO_LEN_SECS).any():
            raise ValueError(
                f"Detected speech longer than {MAX_MODEL_AUDIO_LEN_SECS} secs"
                "-- set `max_audio_len_secs` to filter longer speech!"
            )
    
    if domain != 'all':
        data = data[data.domain == domain]
    if min_transcript_len != -1:
        data = data[data.transcript.str.len() >= min_transcript_len]
    if max_transcript_len != -1:
        data = data[data.transcript.str.len() < max_transcript_len]
    
    data["text"] = data["transcript"]
    print("before dedup", data.shape)
    data.drop_duplicates(subset=["audio_paths"], inplace=True)
    print("after dedup", data.shape)
    if return_dataset:
        return Dataset.from_pandas(data)
    else:
        return data
            
        
def data_prep(config):
    # Prepare data for the model
    global CONFIG, PROCESSOR
    CONFIG = config
    start = time.time()
    aug_dataset = None

    raw_dataset = load_data(config.train_path, config.val_path, config.aug_path)
    logger.debug(f"...Data Read Complete in {time.time() - start:.4f}. Starting Tokenizer...")

    vocab_file_name = load_vocab(config.model_path, config.ckpt_path, config.exp_dir, raw_dataset)
    PROCESSOR = load_processor(vocab_file_name)
    logger.debug(f"...Load vocab and processor complete in {time.time() - start:.4f}.\n"
                 f"Loading dataset...")

    val_dataset = load_custom_dataset(config, config.val_path, 'dev', transform_audio, transform_labels)
    if config.aug_percent and config.aug_percent > 1:
        train_df = load_custom_dataset(config, config.train_path, 'train', 
                                       transform_audio, transform_labels, return_dataset=False)
        aug_df = train_df.sample(frac=config.aug_percent, random_state=config.seed)
        train_df = train_df[~train_df.audio_ids.isin(aug_df.audio_ids.to_list())]
        aug_dataset = Dataset.from_pandas(aug_df)
        train_dataset = Dataset.from_pandas(train_df)
    elif config.aug_path:
        train_dataset = load_custom_dataset(config, config.train_path, 'train', transform_audio, transform_labels)
        aug_dataset = load_custom_dataset(config, config.aug_path, 'aug', transform_audio, transform_labels)
    else:
        train_dataset = load_custom_dataset(config, config.train_path, 'train', transform_audio, transform_labels)

    logger.debug(f"Load train and val dataset done in {time.time() - start:.4f}.")
    return train_dataset, val_dataset, aug_dataset, PROCESSOR


def load_custom_dataset(config, data_path, split, 
                        transform_audio_, transform_labels_=None, 
                        prepare=None, return_dataset=True):
    return CustomASRDataset(data_path, transform_audio_, transform_labels_,
                            config.audio_path, split=split, domain=config.domain,
                            max_audio_len_secs=config.max_audio_len_secs,
                            min_transcript_len=config.min_transcript_len,
                            prepare=prepare, return_dataset=return_dataset)


def load_vocab(model_path, checkpoints_path, exp_dir, raw_datasets):
    create_new_vocab = False
    vocab_file_name = None

    if os.path.isdir(model_path) and 'vocab.json' in os.listdir(model_path):
        vocab_files = ['preprocessor_config.json', 'tokenizer_config.json', 'vocab.json', 'special_tokens_map.json']
        for v in vocab_files:
            subprocess.call(['cp', os.path.join(model_path, v), os.path.join(checkpoints_path, v)])
        vocab_file_name = os.path.join(checkpoints_path, 'vocab.json')
        if os.path.isfile(vocab_file_name):
            print(f"vocab detected at {vocab_file_name}")
        else:
            create_new_vocab = True

    elif os.path.isdir(checkpoints_path) and len(os.listdir(checkpoints_path)) > 0:
        vocab_file_name = [x for x in os.listdir(checkpoints_path) if 'vocab' in x]
        if len(vocab_file_name) > 0:
            vocab_file_name = os.path.join(checkpoints_path, vocab_file_name[0])
            print(f"vocab detected at {vocab_file_name}")
        else:
            create_new_vocab = True
    else:
        create_new_vocab = True

    if create_new_vocab:
        vocab_dict = create_vocab(raw_datasets)
        vocab_file_name = f'vocab-{datetime.now().strftime("%d-%m-%Y--%H:%M:%S")}.json'
        vocab_file_name = os.path.join(exp_dir, 'checkpoints', vocab_file_name)
        logger.debug(f"creating new vocab {vocab_file_name}")
        with open(vocab_file_name, 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)
    elif vocab_file_name:
        with open(vocab_file_name, 'r') as vocab_file:
            vocab_dict = json.load(vocab_file)
    else:
        vocab_dict = {}

    logger.info(f"---vocab dict: {len(vocab_dict)}\n{vocab_dict}")
    return vocab_file_name


def load_data(train_path, val_path, aug_path=None):
    if aug_path:
        return load_dataset('csv', data_files={'train': train_path, 'val': val_path, 'aug': aug_path})
    else:
        return load_dataset('csv', data_files={'train': train_path, 'val': val_path})


def remove_special_characters(batch):
    batch['transcript'] = clean_text(batch['transcript']) + " "
    return batch


def extract_chars_vocab(batch):
    all_text = " ".join(batch['transcript'])
    vocab = list(set(all_text))
    return {'vocab': [vocab], 'all_text': [all_text]}


def special_tokens(vocab_dict):
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    return vocab_dict


def create_vocab(raw_datasets):
    raw_datasets = raw_datasets.map(remove_special_characters, num_proc=6)
    vocabs = raw_datasets.map(extract_chars_vocab,
                              batched=True, batch_size=-1, keep_in_memory=True,
                              remove_columns=raw_datasets.column_names["train"])

    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["val"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict = special_tokens(vocab_dict)
    return vocab_dict


def get_feature_extractor():
    return Wav2Vec2FeatureExtractor(feature_size=1,
                                    sampling_rate=AudioConfig.sr,
                                    padding_value=0.0,
                                    do_normalize=True,
                                    return_attention_mask=True)


def load_processor(vocab_file_name):
    tokenizer = Wav2Vec2CTCTokenizer(vocab_file_name, unk_token="[UNK]",
                                     pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = get_feature_extractor()
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor


def transform_audio(audio_path):
    try:
        speech = load_audio_file(audio_path)
    except Exception as e:
        print(f"{audio_path} not found {str(e)}")
        speech, fs = librosa.load(
            '/data/data/intron/e809b58c-4f05-4754-b98c-fbf236a88fbc/544bbfe5e1c6f8afb80c4840b681908d.wav',
            sr=AudioConfig.sr)

    return PROCESSOR(speech, sampling_rate=AudioConfig.sr).input_values


def transform_labels(text):
    text = clean_text(text)
    with PROCESSOR.as_target_processor():
        labels = PROCESSOR(text.lower()).input_ids
    return labels


class CustomASRDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, transform=None, transform_target=None, audio_dir=None,
                 split=None, domain="all", max_audio_len_secs=-1, min_transcript_len=10,
                 prepare=False, max_transcript_len=-1, gpu=1, 
                 length_column_name='duration', return_dataset=True):
        
        self.prepare = prepare
        self.split = split
        self.asr_data = load_afri_speech_data(data_file, min_transcript_len=min_transcript_len,
                                              max_audio_len_secs=max_audio_len_secs, 
                                              split=split, gpu=gpu, 
                                              audio_dir=audio_dir,
                                              max_transcript_len=max_transcript_len,
                                              domain=domain, return_dataset=return_dataset)
        self.transform = transform
        self.target_transform = transform_target

    def set_dataset(self, new_data):
        self.asr_data = Dataset.from_pandas(new_data, preserve_index=False)

    def get_dataset(self):
        return self.asr_data.to_pandas()

    def __len__(self):
        return len(self.asr_data)

    def __getitem__(self, idx):
        audio_path = self.asr_data[idx]['audio_paths']
        text = self.asr_data[idx]['transcript']
        accent = self.asr_data[idx]['accent']
        audio_idx = self.asr_data[idx]['audio_ids']
        
        if self.prepare:
            input_audio, label = self.transform(audio_path, text)
            result = {'input_values': input_audio, 'input_lengths': len(input_audio)}
        else:
            input_audio = self.transform(audio_path)
            label = self.target_transform(text)
            result = {'input_values': input_audio[0], 'input_lengths': len(input_audio[0])}

        result.update({'labels': label, 'accent': accent, 'audio_idx': audio_idx, 'transcript': clean_text(str(text).lower()), 'audio_paths': audio_path})
        return result


@dataclass
class DataCollatorCTCWithPaddingGroupLen:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods

        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch
