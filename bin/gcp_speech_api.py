import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from google.cloud import speech
from datasets import load_metric


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ""  # credentials file
wer_metric = load_metric("wer")

google_client = speech.SpeechClient()


def get_config(gcp_service):
    g_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
        sample_rate_hertz=44100,
        audio_channel_count=2,
        model='medical_dictation' if 'medical' in gcp_service else 'default'
    )
    return g_config


def gcp_transcribe_file(speech_file, recognition_config):
    """
    Transcribe the given audio file asynchronously.
    Note that transcription is limited to a 60 seconds audio file.
    Use a GCS file for audio longer than 1 minute.
    """

    with open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    response = google_client.recognize(config=recognition_config, audio=audio)
    if len(response.results) > 0:
        return response.results[0].alternatives[0].transcript
    return ""


def cleanup(text):
    """
    post processing to normalized reference and predicted transcripts
    :param text: str
    :return: str
    """
    text = text.lower()
    text = text.replace(" [ comma ] ", ", ") \
        .replace(" [ hyphen ] ", "-") \
        .replace(" [ full stop ] ", ".") \
        .replace(" [ full stop", ".") \
        .replace(" [ full", ".") \
        .replace(" [ question mark ]", "?") \
        .replace(" [ question mark", "?") \
        .replace(" [ question", "?") \
        .replace("[ next line ]", "next line") \
        .strip()
    text = " ".join(text.split())
    return text


def main_transcribe_medical(data, gcp_service):
    g_config = get_config(gcp_service)
    preds_raw = []
    preds_clean = []
    wers = []

    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        pred = gcp_transcribe_file(row['audio_paths'], g_config)
        preds_raw.append(pred)
        pred_clean = cleanup(pred)
        preds_clean.append(pred_clean)
        wers.append(wer_metric.compute(predictions=[pred_clean], references=[cleanup(row['transcript'])]))

    assert len(data) == len(preds_raw) == len(wers) == len(preds_clean)
    data['predictions_raw'] = preds_raw
    data['predictions'] = preds_clean
    data['wer'] = wers

    return data


def write_gcp_results(data, model_id_or_path='gcp-transcribe',
                      output_dir="./results", split='dev'):
    """
    Write predictions and wer to disk
    :param split: str, train/test/dev
    :param output_dir: str
    :param data: DataFrame
    :param model_id_or_path: str
    :return:
    """
    all_wer = np.mean(data['wer'])
    out_path = f'{output_dir}/intron-open-{split}-{model_id_or_path}-wer-{round(all_wer, 4)}-{len(data)}.csv'
    data_df.to_csv(out_path, index=False)
    print(out_path)


if __name__ == '__main__':
    dataset_path = "./data/intron-dev-public-3232.csv"
    data_df = pd.read_csv(dataset_path)
    data_df = data_df[data_df.duration < 17]

    service = 'gcp-transcribe-medical'

    data_df = main_transcribe_medical(data_df, service)

    write_gcp_results(data_df, service)
