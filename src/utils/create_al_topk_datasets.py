import pandas as pd
import numpy as np

data = pd.read_csv('/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/data/intron-train-public-58000-clean.csv')
top_k_format = "/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/src/experiments/wav2vec2-large-xlsr-53-general_most/checkpoints/Top-{}_AL_Round_{}_Mode_'{}'.npy"
top_k_dataset_format = "/home/mila/b/bonaventure.dossou/AfriSpeech-Dataset-Paper/data/Top-{}_AL_Round_{}_Mode_'{}'_dataset.csv"
mode = 'most'
al_rounds = 3
k = 2000

def get_al_subset(dataframe, ids_list):
    return dataframe[dataframe['audio_ids'].isin(ids_list)]

for al_round in range(al_rounds):
    filename = top_k_format.format(k, al_round, mode)
    al_round_stats = np.load(filename, allow_pickle=True)
    audio_ids, uncertainty_stats = al_round_stats[:k], al_round_stats[k:]  # should be k+3 elements in this list
    accents_dataframe = get_al_subset(data, audio_ids)
    accents_dataframe.to_csv(top_k_dataset_format.format(k, al_round, mode), index=False)