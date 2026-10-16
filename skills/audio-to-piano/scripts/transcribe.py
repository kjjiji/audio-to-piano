"""MuScriptor worker. Public API, same backend used by FuFumidi."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--role',choices=['vocals','bass','accompaniment'],required=True)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    threads=max(1,args.threads)
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
        os.environ[name]=str(threads)
    import torch
    from muscriptor import TranscriptionModel
    from muscriptor.events import ProgressEvent
    from context import WORK,STEMS,CONFIG
    torch.set_num_threads(threads);torch.set_num_interop_threads(1);torch.manual_seed(0)
    source=WORK/'accompaniment.wav' if args.role=='accompaniment' else STEMS/(args.role+'.wav')
    decoded=WORK/(args.role+'-16000.wav')
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(source),'-ar','16000','-ac','1','-c:a','pcm_s16le',str(decoded)],check=True)
    device=CONFIG.get('device','cpu')
    dtype='bfloat16' if device=='cpu' or (device.startswith('cuda') and torch.cuda.is_bf16_supported()) else 'float16'
    weights=CONFIG.get('model_path') or CONFIG.get('model_size','medium')
    groups={'vocals':['voice'],'bass':['electric_bass','acoustic_bass'],
        'accompaniment':CONFIG.get('accompaniment_groups',[
            'acoustic_piano','acoustic_guitar','clean_electric_guitar',
            'string_ensemble','synth_strings','synth_pad'])}
    output=sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        model=TranscriptionModel.load_model(weights,device=device,dtype=dtype)
        def progress(events):
            for event in events:
                if isinstance(event,ProgressEvent):
                    print(json.dumps({'stage':'transcribe','role':args.role,
                        'completed':event.completed,'total':event.total}),file=output,flush=True)
                yield event
        events=model.transcribe(decoded,instruments=groups[args.role],prelude_forcing=True)
        data=model.events_to_midi_bytes(progress(events))
    target=WORK/(args.role+'-transcribed.mid')
    temporary=target.with_suffix('.mid.tmp');temporary.write_bytes(data);temporary.replace(target)
    print(json.dumps({'stage':'transcribe','role':args.role,'status':'complete','bytes':len(data)}),flush=True)

if __name__=='__main__':main()
