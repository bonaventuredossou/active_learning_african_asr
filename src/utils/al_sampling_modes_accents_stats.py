import random

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import math
from wordcloud import WordCloud
from scipy import stats

import matplotlib.colors as colors

data = pd.read_csv('../../data/intron-train-public-58000-clean.csv')
data.drop_duplicates(subset=["audio_paths"], inplace=True)
train_data = pd.read_csv('../../data/intron-al-train-public-17400.csv')
aug_data = pd.read_csv('../../data/intron-al-aug-public-40600.csv')

top_k_format = "../src/experiments/wav2vec2-large-xlsr-53-general_most/checkpoints/Top-{}_AL_Round_{}_Mode_'{}'.npy"
wers_file = '../../Top-{}_AL_Round_{}_Mode_{}_WERS.csv'
mode = 'most'
al_rounds = 3
k = 2000
top_k_accents = 15


def get_accents_stats(round_al):
    wer_file = wers_file.format(k, round_al, mode)
    data_wer = pd.read_csv(wer_file)
    data_wer['accents'] = data_wer['audios_ids'].apply(
        lambda x: data[data['audio_ids'] == x]['accent'].values[0])
    accents = data_wer['accents'].tolist()
    accent_frequencies_dict = {accent: accents.count(accent) for accent in list(set(accents))}
    accents_frequencies_dict = dict(sorted(accent_frequencies_dict.items(), key=lambda item: item[1], reverse=True))
    return accents_frequencies_dict


def plot_stats(list_of_language, fig, axes, description):
    for round_al in range(al_rounds):
        wer_file = wers_file.format(k, round_al, mode)
        data_wer = pd.read_csv(wer_file)
        data_wer['accents'] = data_wer['audios_ids'].apply(
            lambda x: data[data['audio_ids'] == x]['accent'].values[0])
        data_wer = data_wer[data_wer['accents'].isin(list_of_language)]
        sns_ax_top = sns.boxplot(y="uncertainty_wer", x='accents', data=data_wer,
                                 ax=axes[round_al])
        sns_ax_top.set(xlabel=None, ylabel=None)
        sns_ax_top.tick_params(axis='x')
    fig.suptitle('{}'.format(description))
    fig.savefig('../src/experiments/wav2vec2-large-xlsr-53-general_most/figures/{}.png'.format(description), bbox_inches='tight', pad_inches=.3)


rounds_difference_stats = []
for al_round in range(al_rounds):
    filename = top_k_format.format(k, al_round, mode)
    al_round_stats = np.load(filename, allow_pickle=True)
    audio_ids, uncertainty_stats = al_round_stats[:k], al_round_stats[k:]  # should be k+3 elements in this list
    rounds_difference_stats.append(get_accents_stats(al_round))

# get the intersection of the top-15 for the three runs
al_0, al_1, al_2 = rounds_difference_stats
common_accents = set(list(al_0.keys())[:top_k_accents]).intersection(
    set(list(al_1.keys())[:top_k_accents])).intersection(set(list(al_2.keys())[:top_k_accents]))
common_accents_list = list(common_accents)

fig_common, axs_common = plt.subplots(al_rounds, 1, figsize=(20, 20))
plot_stats(common_accents_list, fig_common, axs_common, 'common_accents_wer_analysis')
plt.show()

frequencies_round_1 = [al_0[accent] for accent in common_accents_list]
frequencies_round_2 = [al_1[accent] for accent in common_accents_list]
frequencies_round_3 = [al_2[accent] for accent in common_accents_list]

df = pd.DataFrame(
    {'Round 1': frequencies_round_1, 'Round 2': frequencies_round_2, 'Round 3': frequencies_round_3},
    index=common_accents_list)

fig_bars, axs_bars = plt.subplots(1, 1, figsize=(15, 15))
df.plot.bar(rot=0, ax=axs_bars)
axs_bars.set_ylabel('Frequency')
fig_bars.suptitle(
    'Accents Appearing across AL rounds (from the top-{} uncertain samples)'.format(k))
fig_bars.savefig(
    '../src/experiments/wav2vec2-large-xlsr-53-general_most/figures/most_common_accents_distributions.png',
    bbox_inches='tight', pad_inches=.3)
plt.show()

all_common_accents = set(list(al_0.keys())).intersection(
    set(list(al_1.keys()))).intersection(set(list(al_2.keys())))

all_common_accents_list = list(all_common_accents)
frequencies_difference_round_0_round_1 = [al_0[accent] - al_1[accent] for accent in all_common_accents_list]
frequencies_difference_round_1_round_2 = [al_1[accent] - al_2[accent] for accent in all_common_accents_list]
difference_one = {accent: difference for accent, difference in
                  zip(all_common_accents_list, frequencies_difference_round_0_round_1)}

difference_two = {accent: difference for accent, difference in
                  zip(all_common_accents_list, frequencies_difference_round_1_round_2)}

most_variance_accents_one = dict(sorted(difference_one.items(), key=lambda item: item[1], reverse=True))
most_variance_accents_two = dict(sorted(difference_two.items(), key=lambda item: item[1], reverse=True))
all_most_variant_accents = set(list(most_variance_accents_one.keys())).intersection(
    set(list(most_variance_accents_two.keys())))
all_most_variant_accents_list = list(all_most_variant_accents)

most_variance_one = [most_variance_accents_one[accent] for accent in all_most_variant_accents_list]
most_variance_two = [most_variance_accents_two[accent] for accent in all_most_variant_accents_list]

df_variance = pd.DataFrame(
    {'Round 1 $\\rightarrow$ Round 2': most_variance_one, 'Round 2 $\\rightarrow$ Round 3': most_variance_two},
    index=all_most_variant_accents_list)

fig_bars_variance, axs_bars_variance = plt.subplots(1, 1, figsize=(20, 20))
df_variance.plot.bar(rot=90, ax=axs_bars_variance)
axs_bars_variance.set_ylabel('Frequency')
fig_bars_variance.suptitle(
    'Variational Distribution of Accents Frequencies (from the top-{} samples) Across AL rounds'.format(k))
fig_bars_variance.savefig('../src/experiments/wav2vec2-large-xlsr-53-general_most/figures/most_varying_accents.png',
                          bbox_inches='tight', pad_inches=.3)
plt.show()

# descending/ascending for the first AL rounds transition
descending_one = {accent: variation for accent, variation in most_variance_accents_one.items() if variation < 0}
ascending_one = {accent: variation for accent, variation in most_variance_accents_one.items() if variation > 0}

# descending/ascending for the second AL round
descending_two = {accent: variation for accent, variation in most_variance_accents_two.items() if variation < 0}
ascending_two = {accent: variation for accent, variation in most_variance_accents_two.items() if variation > 0}

descending_accents = set(list(descending_one.keys())).intersection(
    set(list(descending_two.keys())))
descending_accents = list(descending_accents)

fig_descending, axs_descending = plt.subplots(al_rounds, 1, figsize=(20, 20))
plot_stats(descending_accents, fig_descending, axs_descending, 'decreasing_accents_wer_analysis')
plt.show()

descending_variance_one = [descending_one[accent] for accent in descending_accents]
descending_variance_two = [descending_two[accent] for accent in descending_accents]

df_descending_variance = pd.DataFrame(
    {'Round 1 $\\rightarrow$ Round 2': descending_variance_one,
     'Round 2 $\\rightarrow$ Round 3': descending_variance_two},
    index=descending_accents)

ascending_accents = set(list(ascending_one.keys())).intersection(
    set(list(ascending_two.keys())))
ascending_accents = list(ascending_accents)

fig_ascending, axs_ascending = plt.subplots(al_rounds, 1, figsize=(20, 20))
plot_stats(ascending_accents, fig_ascending, axs_ascending, 'increasing_accents_wer_analysis')
plt.show()

ascending_variance_one = [ascending_one[accent] for accent in ascending_accents]
ascending_variance_two = [ascending_two[accent] for accent in ascending_accents]

df_ascending_variance = pd.DataFrame(
    {'Round 1 $\\rightarrow$ Round 2': ascending_variance_one,
     'Round 2 $\\rightarrow$ Round 3': ascending_variance_two},
    index=ascending_accents)

fig_bars_variances, axs_bars_variances = plt.subplots(1, 2, figsize=(20, 20))
df_descending_variance.plot.bar(rot=90, ax=axs_bars_variances[0])
df_ascending_variance.plot.bar(rot=90, ax=axs_bars_variances[1])
axs_bars_variances[0].set_ylabel('Frequency')
axs_bars_variances[1].set_ylabel('Frequency')

axs_bars_variances[0].set_title("Descending Accents Across Top-{} '{}' Uncertain Samples".format(k, mode))
axs_bars_variances[1].set_title("Ascending Accents Across Top-{} '{}' Uncertain Samples".format(k, mode))

fig_bars_variances.savefig('../data/descending_ascending_accents.png',
                           bbox_inches='tight', pad_inches=.3)
plt.show()