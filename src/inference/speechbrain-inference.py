from speechbrain.pretrained import EncoderDecoderASR
import glob

ABS = "/scratch/pbsjobs/axy327/dev/" # this is the path where the audio files exist
wavfiles =  glob.glob(ABS+"*/"+"*.wav")
uttids = []
predictions = []
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-transformerlm-librispeech", savedir="pretrained_models/asr-crdnn-transformerlm-librispeech", run_opts={"device":"cuda"})
for wavfile in wavfiles:
    predictions.append(asr_model.transcribe_file(wavfile))

with open("../../results/african-nlp-speechbrain-predictons", "w") as f:
    for i in range(len(wavfiles)):
        f.write(f"{wavfiles[i]}\t{predictions[i]}\n")