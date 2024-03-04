import boto3
import pandas as pd
from tqdm import tqdm
import time
from datasets import load_metric
import numpy as np

from src.utils.utils import get_s3_file, get_json_result, cleanup

wer_metric = load_metric("wer")

transcribe = boto3.client(
    'transcribe',
    region_name='eu-west-2'
)

bucket_name = "intron-open-source"

s3 = boto3.resource(
    service_name='s3',
    region_name='eu-west-2',
)


def get_aws_transcript_medical(job_name, job_uri, service):
    """
    Make calls to transcribe service
    :param service: str [transcribe-medical or transcribe]
    :param job_name: str, unique job name
    :param job_uri: str, s3 path
    :return:
    """
    transcribe.start_medical_transcription_job(
        MedicalTranscriptionJobName=job_name,
        Media={
            'MediaFileUri': job_uri
        },
        OutputBucketName=bucket_name,
        OutputKey=f'aws-{service}-output-files/',
        LanguageCode='en-US',
        Specialty='PRIMARYCARE',
        Type='DICTATION'
    )
    status = transcribe.get_medical_transcription_job(MedicalTranscriptionJobName=job_name)


def get_aws_transcript(job_name, job_uri, service):
    """
    Make calls to transcribe medical service
    :param service: str [transcribe-medical or transcribe]
    :param job_name: str, unique job name
    :param job_uri: str, s3 path
    :return:
    """
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={
            'MediaFileUri': job_uri
        },
        MediaFormat='wav',
        LanguageCode='en-US',
        OutputBucketName=bucket_name,
        OutputKey=f'aws-{service}-output-files/',
    )
    status = transcribe.get_transcription_job(TranscriptionJobName=job_name)


def main_transcribe_medical(data_df, service):
    """
    Send requests to AWS transcribe medical service
    :param service: str [transcribe-medical or transcribe]
    :param data_df: DataFrame
    :return:
    """
    for idx, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        audio_path = row['audio_paths']
        s3_uri = f"s3://intron-open-source{audio_path}"
        s3_job_name = f"dev-transcription-job-{service}-{idx}"
        get_aws_transcript_medical(s3_job_name, s3_uri, service)
        if idx % 400 == 0:
            # avoid RateLimitExceeded error
            time.sleep(60)


def main_transcribe(data_df, service):
    """
    Send requests to AWS transcribe service
    :param service: str [transcribe-medical or transcribe]
    :param data_df: DataFrame
    :return:
    """
    for idx, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        audio_path = row['audio_paths']
        s3_uri = f"s3://intron-open-source{audio_path}"
        s3_job_name = f"dev-transcription-job-{service}-{idx}"
        get_aws_transcript(s3_job_name, s3_uri, service)
        if idx % 400 == 0:
            # avoid RateLimitExceeded error
            time.sleep(60)


def get_aws_results_from_s3(data_df, service):
    """
    Get transcription results written to s3 directory
    :param data_df: DataFrame
    :param service: [transcribe-medical or transcribe]
    :return: List[str], List[float]
    """
    preds = []
    wers = []
    for idx, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        s3_job_name = f"dev-transcription-job-{service}-{idx}"
        predicted_transcript_file = f'https://s3.eu-west-2.amazonaws.com/' \
                                    f'intron-open-source/aws-{service}-output-files/{s3_job_name}.json'
        s3_prefix = f"https://s3.eu-west-2.amazonaws.com/intron-open-source/aws-{service}-output-files"
        local_dev_fname = get_s3_file(predicted_transcript_file,
                                      s3_prefix=s3_prefix,
                                      local_prefix="/data/saved_models/predictions/aws/dev",
                                      bucket_name=bucket_name,
                                      s3=s3)
        pred = cleanup(get_json_result(local_dev_fname))
        preds.append(pred)
        wers.append(wer_metric.compute(predictions=[pred], references=[cleanup(row['transcript'])]))

    return preds, wers


def write_aws_results(data_df, predictions, wer_list,
                      model_id_or_path='aws-transcribe', output_dir="./results"):
    """
    Write predictions and wer to disk
    :param output_dir: str
    :param data_df: DataFrame
    :param predictions: List[str]
    :param wer_list: List[float]
    :param model_id_or_path: str
    :return:
    """
    data_df['predictions'] = predictions
    data_df['wer'] = wer_list
    all_wer = np.mean(data['wer'])

    out_path = f'{output_dir}/intron-open-test-{model_id_or_path}-wer-{round(all_wer, 4)}-{len(data)}.csv'
    data_df.to_csv(out_path, index=False)
    print(out_path)


if __name__ == '__main__':
    dataset_path = "./data/intron-dev-public-3232.csv"
    data = pd.read_csv(dataset_path)

    aws_service = 'transcribe-medical'

    if "medical" in aws_service:
        main_transcribe_medical(data, aws_service)
    else:
        main_transcribe(data, aws_service)

    prediction_list, all_wer_list = get_aws_results_from_s3(data, aws_service)

    write_aws_results(data, prediction_list, all_wer_list, aws_service)
