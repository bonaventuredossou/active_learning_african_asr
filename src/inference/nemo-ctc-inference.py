import nemo.collections.asr as nemo_asr
import glob
import os

ABS = "/scratch/pbsjobs/axy327/dev/" # this is the path where the audio files exist
wavfiles =  glob.glob(ABS+"*/"+"*.wav")
uttids = []
predictions = []
cmd = "ffmpeg -i "
cmd1 = " -ac 1 -ar 16000 "
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_conformer_ctc_large")
for wavfile in wavfiles:
    os.system(cmd + wavfile + cmd1 + wavfile.split("/")[-1])
    transcription = asr_model.transcribe([wavfile.split("/")[-1]])
    predictions.append(transcription)
    os.system("rm *.wav")

with open("../../results/african-nlp-nemo-ctc-predictons", "w") as f:
    for i in range(len(wavfiles)):
        f.write(f"{wavfiles[i]}\t{predictions[i]}\n")