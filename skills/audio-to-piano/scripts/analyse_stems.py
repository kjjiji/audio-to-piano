"""Independent full-song pitch and amplitude evidence for a sung piece."""
from pathlib import Path
import json
import time
import librosa
import numpy as np
import soundfile as sf
import essentia.standard as es
import onnxruntime as ort
from scipy.ndimage import gaussian_filter1d
from context import WORK, STEMS, CONFIG
SR=22050
HOP=256
FIRST_PITCH=24
reports=[]

def load(path):
    y,sr=sf.read(path,dtype='float32',always_2d=True)
    return y,sr,librosa.resample(y.mean(axis=1),orig_sr=sr,target_sr=SR)

parts={}
for name in ['piano','guitar','other']:
    audio,sr=sf.read(STEMS/(name+'.wav'),dtype='float32',always_2d=True)
    parts[name]=audio
accompaniment=sum(parts.values())
sf.write(WORK/'accompaniment.wav',accompaniment,sr,subtype='FLOAT')
print('ACCOMPANIMENT_READY',flush=True)
del parts,accompaniment,audio

original_session=ort.InferenceSession
def cpu_session(path,*args,**kwargs):
    options=ort.SessionOptions()
    options.intra_op_num_threads=1
    options.inter_op_num_threads=1
    kwargs['sess_options']=options
    return original_session(path,*args,**kwargs)
ort.InferenceSession=cpu_session
from basic_pitch.inference import predict
from basic_pitch.note_creation import model_frames_to_time
import basic_pitch
MODEL=str(Path(basic_pitch.__file__).parent/'saved_models/icassp_2022/nmp.onnx')

for name in ['vocals','piano','guitar','other','bass','accompaniment','source']:
    started=time.monotonic()
    path=STEMS/(name+'.wav') if name in ['vocals','piano','guitar','other','bass'] else WORK/(name+'.wav')
    audio,original_sr,mono=load(path)
    cq=abs(librosa.cqt(mono,sr=SR,hop_length=HOP,fmin=librosa.midi_to_hz(FIRST_PITCH),n_bins=84))
    times=librosa.frames_to_time(np.arange(cq.shape[1]),sr=SR,hop_length=HOP)
    stft=abs(librosa.stft(mono,n_fft=1024,hop_length=HOP))
    log=np.log1p(stft)
    flux=gaussian_filter1d(np.maximum(np.diff(log,axis=1,prepend=log[:,:1]),0).sum(axis=0),.6)
    rms=librosa.feature.rms(y=mono,frame_length=1024,hop_length=HOP)[0]
    evidence={'cqt':cq,'cqt_times':times,'audio_times':times,'flux':flux,'rms':rms,
              'first_pitch':np.array(FIRST_PITCH)}
    report={'stem':name,'rms':float(np.sqrt(np.mean(audio**2)))}
    if name in ['vocals','bass','accompaniment']:
        settings=(95,1300) if name=='vocals' else (32,400) if name=='bass' else (130,2200)
        f0,conf=es.PredominantPitchMelodia(sampleRate=SR,frameSize=2048,hopSize=128,
            minFrequency=settings[0],maxFrequency=settings[1],minDuration=70,
            guessUnvoiced=False)(mono)
        valid=f0>0
        evidence.update({'f0_times':np.arange(len(f0))*128/SR,
            'pitch':librosa.hz_to_midi(np.maximum(f0,1e-5)),'voiced':valid,'confidence':conf})
        report.update({'voiced_seconds':float(valid.sum()*128/SR),
            'pitch_quantiles':np.percentile(evidence['pitch'][valid],[5,25,50,75,95]).tolist() if valid.any() else []})
    if name in ['vocals','piano','guitar','other','bass'] or (name=='accompaniment' and CONFIG.get('backend')=='basic-pitch'):
        output,midi,events=predict(str(path),model_or_model_path=MODEL,
            onset_threshold=.40,frame_threshold=.32,minimum_note_length=90,
            minimum_frequency=librosa.midi_to_hz(28 if name=='bass' else 36),
            maximum_frequency=librosa.midi_to_hz(95),multiple_pitch_bends=False)
        midi.write(str(WORK/(name+'-basicpitch.mid')))
        evidence.update(output)
        evidence['times']=model_frames_to_time(output['note'].shape[0])
        report['basic_pitch_notes']=sum(len(i.notes) for i in midi.instruments)
    np.savez_compressed(WORK/(name+'-evidence.npz'),**evidence)
    report['analysis_seconds']=time.monotonic()-started
    reports.append(report)
    print('EVIDENCE',json.dumps(report),flush=True)

(WORK/'stem_analysis.json').write_text(json.dumps(reports,indent=2))
