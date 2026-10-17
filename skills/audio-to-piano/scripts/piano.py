"""Portable CLI for the skill; no server and no song-specific paths."""
import argparse
import concurrent.futures
import ctypes.util
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import threading

SCRIPTS=Path(__file__).parent
VERSION='0.1.0'
EMIT_LOCK=threading.Lock()

def emit(stage,**fields):
    with EMIT_LOCK:sys.stdout.write(json.dumps({'stage':stage,**fields},ensure_ascii=False)+'\n');sys.stdout.flush()
def save(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def cpu_budget():
    count=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    try:
        quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota!='max':count=min(count,max(1,int(quota)//int(period)))
    except (OSError,ValueError):pass
    return max(1,min(8,count))

def cache_candidates():
    roots=[Path.cwd(),Path.cwd()/'scratch',Path.home()/'.cache',Path.home()/'.local/share']
    if os.environ.get('FUFUMIDI_MODELS_DIR'):roots.insert(0,Path(os.environ['FUFUMIDI_MODELS_DIR']))
    return roots

def resolve_model(explicit,size):
    if explicit:
        # Keep the HF snapshot symlink path: config.json is beside it, not in blobs/.
        path=Path(os.path.abspath(Path(explicit).expanduser()))
        if path.is_dir():path=path/'model.safetensors'
        if not path.is_file():raise ValueError('Model not found: '+str(path))
        return str(path)
    for root in cache_candidates():
        for candidate in [root/'muscriptor'/size/'model.safetensors',root/'FuFumidi-models/muscriptor'/size/'model.safetensors']:
            if candidate.is_file():return str(candidate.absolute())
    try:
        from huggingface_hub import hf_hub_download
        path=hf_hub_download(repo_id='MuScriptor/muscriptor-'+size,
            filename='model.safetensors',local_files_only=True)
        if Path(path).is_file():return str(Path(path).absolute())
    except Exception:pass  # Local-cache lookup is optional and never downloads.
    return None

def resolve_font(explicit):
    if explicit:
        path=Path(explicit).expanduser().resolve()
        if not path.is_file():raise ValueError('SoundFont not found: '+str(path))
        return str(path)
    if os.environ.get('PIANO_SOUNDFONT'):return resolve_font(os.environ['PIANO_SOUNDFONT'])
    for root in cache_candidates():
        candidates=[root/'piano-soundfonts/Salamander_Grand_Piano_6L_continuous.sf2',
            root/'FuFumidi/renderer/vendor/soundfonts/GeneralUser.sf2',root/'soundfonts/piano.sf2']
        for path in candidates:
            if path.is_file():return str(path.resolve())
    for name in ['/usr/share/sounds/sf2/FluidR3_GM.sf2','/usr/share/soundfonts/default.sf2','/usr/share/sounds/sf2/default-GM.sf2']:
        if Path(name).is_file():return str(Path(name).resolve())
    return None

def resolve_beats(explicit):
    if explicit:
        path=Path(explicit).expanduser().resolve()
        if not path.is_file():raise ValueError('Beat checkpoint not found')
        return str(path)
    for root in cache_candidates():
        for path in [root/'hub/checkpoints/beat_this-final0.ckpt',root/'FuFumidi-models/hub/checkpoints/beat_this-final0.ckpt',root/'torch/hub/checkpoints/beat_this-final0.ckpt']:
            if path.is_file():return str(path.resolve())
    return None

def font_credit(path,explicit=None):
    if explicit:return explicit
    if 'salamander' in Path(path).name.lower():
        return ('Alexander Holm / Salamander Grand Piano (Yamaha C5), CC BY 3.0 '
            'https://creativecommons.org/licenses/by/3.0/ ; '
            'https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html#SalamanderGrandPiano'+
            (' ; velocity-zone mapping adapted' if 'continuous' in Path(path).name.lower() else ''))
    return 'Configured SoundFont: '+Path(path).name+'; consult its supplied attribution and license.'

def dependencies():
    modules=['numpy','scipy','librosa','soundfile','pretty_midi','mido','fluidsynth',
             'torch','torchaudio','demucs','muscriptor','beat_this','onnxruntime','basic_pitch','essentia']
    found={name:importlib.util.find_spec(name) is not None for name in modules}
    commands={name:shutil.which(name) for name in ['ffmpeg','ffprobe']}
    fluid=ctypes.util.find_library('fluidsynth')
    return {'python':sys.executable,'python_version':sys.version.split()[0],'modules':found,
        'commands':commands,'fluidsynth_library':fluid,'cpu_budget':cpu_budget(),
        'ready':all(found.values()) and all(commands.values()) and bool(fluid)}

def inspect_source(path):
    path=Path(path).expanduser().resolve()
    if not path.is_file():raise ValueError('Input file not found: '+str(path))
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(path)]))
    audio=[stream for stream in probe.get('streams',[]) if stream.get('codec_type')=='audio']
    if not audio:raise ValueError('Input contains no audio stream')
    return {'path':str(path),'file_name':path.name,'bytes':path.stat().st_size,'sha256':digest(path),'probe':probe}

def device_choice(value):
    from runtime_support import select_device
    return select_device(value)

def doctor(args):
    from runtime_support import import_diagnostics,model_diagnostics,model_smoke_test,torch_diagnostics,onnx_diagnostics,package_conflicts
    info=dependencies();info['dependencies_ready']=info['ready']
    imports,versions=import_diagnostics()
    info.update(imports=imports,package_versions=versions)
    info['dependencies_ready']=info['dependencies_ready'] and all(item['ready'] for item in imports.values())
    info['package_compatibility']=package_conflicts()
    info['onnx']=onnx_diagnostics()
    info['dependencies_ready']=info['dependencies_ready'] and info['package_compatibility']['ready'] and info['onnx']['ready']
    info['runtime']=torch_diagnostics(args.device,args.dtype)
    errors=[]
    try:model=resolve_model(args.model_path,args.model_size or 'medium')
    except (ValueError,OSError) as error:model=None;errors.append(str(error))
    info['model_path']=model;info['model']=model_diagnostics(model)
    font=None;font_info={'ready':False,'synthesis_test':{'status':'not_run'}}
    try:
        font=resolve_font(args.soundfont)
        if not font:raise ValueError('No piano SoundFont found; pass --soundfont or set PIANO_SOUNDFONT')
        from soundfont_checks import piano_coverage
        mask=piano_coverage(font)
        coverage=bool(mask[21:109,1:].all())
        font_info.update(path=font,bank=0,program=0,covers_88_keys_all_velocities=coverage,
            uncovered_key_velocity_pairs=int((~mask[21:109,1:]).sum()))
        if not coverage:raise ValueError('SoundFont has silent key/velocity gaps; select continuous piano coverage')
        import fluidsynth
        import numpy as np
        synth=fluidsynth.Synth(samplerate=22050)
        try:
            font_id=synth.sfload(font)
            if font_id<0 or synth.program_select(0,font_id,0,0)!=0:raise ValueError('Cannot load bank 0/program 0 piano')
            synth.noteon(0,60,80);samples=synth.get_samples(11025)
            if not np.isfinite(samples).all() or not np.any(samples):raise ValueError('SoundFont synthesis is silent or non-finite')
            font_info['synthesis_test']={'status':'passed','pitch':60,'velocity':80}
        finally:synth.delete()
        font_info['ready']=True
    except Exception as error:font_info['error']=f'{type(error).__name__}: {error}'
    info['soundfont']=font;info['soundfont_checks']=font_info
    info['soundfont_covers_88_keys_all_velocities']=font_info.get('covers_88_keys_all_velocities',False)
    try:info['beat_checkpoint']=resolve_beats(args.beat_checkpoint)
    except (ValueError,OSError) as error:info['beat_checkpoint']=None;errors.append(str(error))
    info['beat_checkpoint_status']='local' if info['beat_checkpoint'] else 'not_cached; pipeline may download final0'
    backend=args.backend or 'auto'
    model_required=backend=='muscriptor' or (backend=='auto' and bool(model))
    info['selected_backend']='muscriptor' if model_required else 'basic-pitch'
    info['model_inference']={'status':'not_run','reason':'Use --smoke-test to load local weights and run multi-instrument inference'}
    if args.smoke_test:
        if not info['model']['ready'] or not info['runtime']['ready']:
            info['model_inference']={'status':'failed','error':'Local model metadata and device checks must pass first'}
        else:
            try:
                import torch
                torch.set_num_threads(min(max(1,args.threads or cpu_budget()),cpu_budget()))
                info['model_inference']=model_smoke_test(model,info['runtime']['selected_device'],
                    info['runtime']['selected_dtype'],args.smoke_audio)
                info['model']['weights_loaded']=True
            except Exception as error:info['model_inference']={'status':'failed','error':f'{type(error).__name__}: {error}'}
    info['gpu_inference_verified']=(info['model_inference']['status']=='passed'
        and info['runtime'].get('selected_device','').startswith('cuda'))
    info['errors']=errors
    info['ready']=(info['dependencies_ready'] and info['runtime']['ready'] and font_info['ready']
        and (not model_required or info['model']['ready']) and not errors
        and (not args.smoke_test or info['model_inference']['status']=='passed'))
    return info

def configure(args,source=None):
    work=Path(args.work_dir).expanduser().resolve();work.mkdir(parents=True,exist_ok=True)
    existing=json.loads((work/'config.json').read_text()) if (work/'config.json').exists() else {}
    if source and existing.get('source_sha256') and existing['source_sha256']!=source['sha256']:
        raise ValueError('Work directory belongs to another input; choose a new --work-dir')
    font=resolve_font(args.soundfont or existing.get('soundfont'))
    if not font:raise ValueError('No piano SoundFont found. Supply --soundfont /absolute/path/piano.sf2')
    model_size=args.model_size or existing.get('model_size') or 'medium'
    prior_model=existing.get('model_path') if existing.get('model_size')==model_size else None
    model=resolve_model(args.model_path or prior_model,model_size)
    backend=args.backend or existing.get('backend') or 'auto'
    if backend=='auto':backend='muscriptor' if model or args.download_model else 'basic-pitch'
    if backend=='muscriptor' and not model and not args.download_model:
        raise ValueError('MuScriptor weights missing. Supply --model-path, enable --download-model with valid model access, or choose --backend basic-pitch.')
    if backend=='muscriptor' and model and args.command!='render-midi':
        from runtime_support import model_diagnostics
        model_info=model_diagnostics(model)
        if not model_info['ready']:raise ValueError(model_info['error'])
    device=device_choice(args.device)
    from runtime_support import select_dtype
    dtype=select_dtype(args.dtype,device)
    out=Path(args.output_dir or existing.get('output_dir') or work/'output').expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    original=source['file_name'] if source else existing.get('source_name','Piano-Arrangement')
    stem=re.sub(r'[^\w.\-]+','-',Path(original).stem).strip('.-')[:70] or 'Piano-Arrangement'
    basename=args.basename or existing.get('basename') or stem+'-Piano'
    if basename in ['.','..'] or '/' in basename or '\\' in basename:raise ValueError('Basename must be a filename without directories')
    if source and any((out/(basename+suffix)).resolve()==Path(source['path']).resolve() for suffix in ['.mid','.mp3','.zip']):
        raise ValueError('Output would overwrite the supplied source; choose a new basename or output directory')
    config={**existing,'pipeline_version':VERSION,'output_dir':str(out),'basename':basename,
        'title':args.title or existing.get('title') or Path(original).stem+' — Piano',
        'soundfont':font,'soundfont_credit':font_credit(font,args.soundfont_credit or existing.get('soundfont_credit')),
        'model_path':model,'model_size':model_size,'backend':backend,
        'transcription_label':('MuScriptor local weights' if model else 'MuScriptor '+model_size) if backend=='muscriptor' else 'Basic Pitch ONNX (fallback)',
        'beat_checkpoint':resolve_beats(args.beat_checkpoint or existing.get('beat_checkpoint')),
        'vocal_mode':args.vocal_mode,'device':device,'dtype':dtype,
        'threads':min(max(1,args.threads or cpu_budget()),cpu_budget()),
        'tempo_bpm':args.tempo or existing.get('tempo_bpm'),'meter':args.meter or existing.get('meter'),
        'include_primary_melody':not args.accompaniment_only,
        'velocity_baselines':[88+args.melody_offset,63+args.harmony_offset,61+args.rhythm_offset,65+args.bass_offset]}
    if source:config.update(source_sha256=source['sha256'],source_name=source['file_name'])
    if args.bitrate:config['bitrate']=args.bitrate
    else:
        duration=float(source['probe']['format']['duration']) if source else float(existing.get('duration',300))
        target_rate=(9*1024*1024-400000)*8/max(duration+75,1)/1000
        config['bitrate']=next((b for b in [192,160,128,96,64,48,32] if b<=target_rate),32)
    save(work/'config.json',config);os.environ['PIANO_WORK_DIR']=str(work)
    return work,config

def execute(work,name,args=(),threads=1):
    env=dict(os.environ,PIANO_WORK_DIR=str(work),OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS='1')
    role=str(args[list(args).index('--role')+1]) if '--role' in args else None
    log=work/(name+('-'+role if role else '')+'.log')
    with log.open('w') as output:
        process=subprocess.Popen([sys.executable,str(SCRIPTS/(name+'.py')),*map(str,args)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            output.write(line);output.flush()
            try:
                message=json.loads(line)
                if isinstance(message,dict) and 'stage' in message:emit('worker',worker=name,progress=message)
            except json.JSONDecodeError:pass
        code=process.wait()
    if code:raise RuntimeError(name+' failed; inspect '+str(log))

def state_file(work):return work/'stage_state.json'
def cached(work,stage,fingerprint,paths):
    state=json.loads(state_file(work).read_text()) if state_file(work).exists() else {}
    item=state.get(stage,{})
    return item.get('fingerprint')==fingerprint and all(Path(path).is_file() and Path(path).stat().st_size>0 for path in paths)
def completed(work,stage,fingerprint):
    state=json.loads(state_file(work).read_text()) if state_file(work).exists() else {}
    state[stage]={'fingerprint':fingerprint,'completed_at':time.time()};save(state_file(work),state)

def transcription_fingerprint(base,role,config):
    model_identity={'path':config['model_path'],'size':config['model_size']} if config['backend']=='muscriptor' else None
    if model_identity and config['model_path']:
        stat=Path(config['model_path']).stat();model_identity.update(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
        model_identity['config_sha256']=digest(Path(config['model_path']).with_name('config.json'))
    return signature({**base,'role':role,'backend':config['backend'],'model':model_identity,
        'inference_runtime':2,'device':config['device'],'dtype':config['dtype'],
        'muscriptor_version':importlib.metadata.version('muscriptor') if config['backend']=='muscriptor' else None})

def run_audio(args):
    ready=dependencies()
    if not ready['ready']:raise ValueError('Runtime dependencies missing; run doctor and bootstrap first: '+json.dumps(ready))
    source=inspect_source(args.input);work,config=configure(args,source)
    if config['device'].startswith('cuda'):
        import torch
        from runtime_support import kernel_check
        kernel_check(torch,config['device'],config['dtype'])
    if args.force_from:
        stages=['decode','separate','analyse','transcribe','export']
        state=json.loads(state_file(work).read_text()) if state_file(work).exists() else {}
        start=stages.index(args.force_from)
        state={key:value for key,value in state.items() if stages.index(key.split('-')[0])<start}
        save(state_file(work),state)
    save(work/'source_info.json',source)
    base={'source':source['sha256'],'version':VERSION}
    fingerprint=signature(base)
    if not cached(work,'decode',fingerprint,[work/'source.wav']):
        emit('decode',input_name=source['file_name'])
        subprocess.run(['ffmpeg','-v','error','-y','-i',source['path'],'-map','0:a:0','-vn','-ar','44100','-ac','2','-c:a','pcm_f32le',str(work/'source.wav')],check=True)
        import soundfile as sf
        if sf.info(work/'source.wav').duration<1:raise ValueError('Audio is shorter than one second')
        completed(work,'decode',fingerprint)
    stems=work/'separated/htdemucs_6s/source'
    if not cached(work,'separate',fingerprint,[stems/(name+'.wav') for name in ['vocals','bass','drums','piano','guitar','other']]):
        emit('separate',model='htdemucs_6s')
        log=work/'separate.log'
        with log.open('w') as output:
            env=dict(os.environ,OMP_NUM_THREADS=str(min(4,config['threads'])),MKL_NUM_THREADS=str(min(4,config['threads'])),OPENBLAS_NUM_THREADS='1')
            subprocess.run([sys.executable,'-m','demucs.separate','--name','htdemucs_6s','--shifts','0','--overlap','.25',
                '-d',config['device'],'--float32','-o',str(work/'separated'),str(work/'source.wav')],env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
        completed(work,'separate',fingerprint)
    analysis_fingerprint=signature({**base,'backend':config['backend']})
    evidence=[work/(name+'-evidence.npz') for name in ['vocals','bass','piano','guitar','other','accompaniment','source']]
    if not cached(work,'analyse',analysis_fingerprint,[*evidence,work/'stem_analysis.json',work/'tempo_analysis.json',work/'accompaniment.wav']):
        emit('analyse')
        execute(work,'analyse_stems')
        execute(work,'analyse_tempo')
        completed(work,'analyse',analysis_fingerprint)
    import numpy as np
    import soundfile as sf
    analysis=json.loads((work/'stem_analysis.json').read_text())
    lookup={item['stem']:item for item in analysis}
    ratio=lookup['vocals']['rms']/max(lookup['source']['rms'],1e-6)
    with np.load(work/'vocals-evidence.npz') as e:
        stable_fraction=float(np.mean(e['voiced']&(e['confidence']>.08)))
    if args.vocal_mode=='auto':resolved='vocal' if ratio>=.10 and stable_fraction>=.10 else 'instrumental'
    else:resolved=args.vocal_mode
    config['resolved_vocal_mode']=resolved
    config['vocal_detection']={'rms_ratio':ratio,'stable_voicing_fraction':stable_fraction,
        'method':'Heuristic only; stem labels and this classification do not establish vocal presence.'}
    config['duration']=sf.info(work/'source.wav').duration
    save(work/'config.json',config)
    emit('vocal_mode',mode=resolved,evidence=config['vocal_detection'],backend=config['transcription_label'])
    roles=['bass','accompaniment']
    if resolved=='vocal' and config['include_primary_melody']:roles.insert(0,'vocals')
    pending=[]
    for role in roles:
        key=transcription_fingerprint(base,role,config)
        target=work/(role+'-transcribed.mid')
        if not cached(work,'transcribe-'+role,key,[target]):pending.append((role,key))
    if config['backend']=='basic-pitch':
        for role,key in pending:
            shutil.copy2(work/(role+'-basicpitch.mid'),work/(role+'-transcribed.mid'))
            completed(work,'transcribe-'+role,key)
    else:
        total=config['threads'];workers=min(3,total,len(pending)) if config['device']=='cpu' else min(1,len(pending))
        allocation={'vocals':1,'bass':1,'accompaniment':max(1,total-2)}
        if len(pending)==1:allocation[pending[0][0]]=total
        if workers:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                jobs={pool.submit(execute,work,'transcribe',['--role',role,'--threads',allocation[role]],allocation[role]):(role,key) for role,key in pending}
                failures=[]
                for future in concurrent.futures.as_completed(jobs):
                    role,key=jobs[future]
                    try:future.result();completed(work,'transcribe-'+role,key)
                    except Exception as error:failures.append(str(error))
                if failures:raise RuntimeError('; '.join(failures))
    export_key=signature({**base,'config':config})
    manifest_path=Path(config['output_dir'])/'file_manifest.json'
    if cached(work,'export',export_key,[manifest_path]):
        manifest=json.loads(manifest_path.read_text())
        if all(Path(value['path']).is_file() and digest(value['path'])==value['sha256'] for value in manifest.values()):
            emit('complete',cached=True,files=manifest);return
    emit('arrange_and_render')
    execute(work,'export_piano')
    completed(work,'export',export_key)

def import_work(args):
    source=inspect_source(args.input);work,config=configure(args,source)
    old=Path(args.from_work).expanduser().resolve()
    if old==work:raise ValueError('Import into a new work directory')
    info=json.loads((old/'source_info.json').read_text())
    if info.get('sha256')!=source['sha256']:raise ValueError('Cached work belongs to a different audio input')
    save(work/'source_info.json',source)
    files=['source.wav','accompaniment.wav','tempo_analysis.json','stem_analysis.json']
    files += [name+'-evidence.npz' for name in ['vocals','bass','piano','guitar','other','accompaniment','source']]
    files += [name+'-basicpitch.mid' for name in ['vocals','bass','piano','guitar','other']]
    if config['backend']=='basic-pitch':files.append('accompaniment-basicpitch.mid')
    for name in files:
        if not (old/name).is_file():raise ValueError('Missing cached file: '+name)
        shutil.copy2(old/name,work/name)
    shutil.copytree(old/'separated',work/'separated',dirs_exist_ok=True)
    for role in ['vocals','bass','accompaniment']:
        candidates=[old/(role+'-transcribed.mid'),old/(role+'-fufumidi.mid')]
        original=next((p for p in candidates if p.is_file()),None)
        if original:shutil.copy2(original,work/(role+'-transcribed.mid'))
    # Import is explicit reuse. Save config, then validate/render from these
    # complete references; it does not certify their musical correctness.
    config['resolved_vocal_mode']='vocal' if args.vocal_mode=='auto' else args.vocal_mode
    save(work/'config.json',config)
    emit('imported_cache',input_sha256=source['sha256'])
    execute(work,'export_piano')

def render_midi(args):
    import copy
    import pretty_midi
    source=Path(args.input).expanduser().resolve()
    try:original=pretty_midi.PrettyMIDI(str(source),charset='utf-8')
    except UnicodeDecodeError:original=pretty_midi.PrettyMIDI(str(source))
    record={'path':str(source),'file_name':source.name,'sha256':digest(source),'bytes':source.stat().st_size,
        'probe':{'format':{'duration':original.get_end_time()}}}
    work,config=configure(args,record)
    from midi_checks import write_midi,validate,track_stats
    from soundfont_checks import validate_notes
    from render_audio import render
    from export_piano import package
    midi=copy.deepcopy(original);removed=sum(len(i.notes) for i in midi.instruments if i.is_drum)
    midi.instruments=[i for i in midi.instruments if not i.is_drum and i.notes]
    for track in midi.instruments:
        track.program=0
        for n in track.notes:
            if not 21<=n.pitch<=108:raise ValueError('Input note outside 88-key piano range; choose explicit octave transposition instead of silently dropping it')
    if not midi.instruments:raise ValueError('MIDI contains no pitched notes')
    target=Path(config['output_dir'])/(config['basename']+'.mid');write_midi(midi,target,resolution=max(1920,midi.resolution))
    parsed,error=validate(midi,target,midi.get_end_time()+.002,one_keyboard=False)
    coverage=validate_notes(config['soundfont'],parsed)
    wav,mp3,details=render(target,midi.get_end_time()+2)
    report={'title':config['title'],'pipeline_version':VERSION,'note_count':sum(len(i.notes) for i in parsed.instruments),
        'tracks':[track_stats(i) for i in parsed.instruments],'removed_drum_notes':removed,
        'maximum_roundtrip_timing_error_seconds':error,'soundfont_coverage':coverage,'render':details,
        'comparison_cues':[],'limitations':'Pitched notes and tempo map from supplied MIDI preserved; original audio fidelity and two-hand playability were not assessed.'}
    report_file=Path(config['output_dir'])/'analysis_report.json';save(report_file,report)
    package([target,mp3,report_file],report)

def parser():
    root=argparse.ArgumentParser(description='Audio analysis and piano arrangement workflow')
    commands=root.add_subparsers(dest='command',required=True)
    for name in ['doctor','inspect','run','import-work','render-midi','compare']:
        p=commands.add_parser(name)
        if name in ['inspect','run','import-work','render-midi']:p.add_argument('input')
        p.add_argument('--work-dir',default='piano-work')
        p.add_argument('--output-dir');p.add_argument('--title');p.add_argument('--basename')
        p.add_argument('--soundfont');p.add_argument('--soundfont-credit')
        p.add_argument('--model-path');p.add_argument('--model-size',choices=['small','medium','large'])
        p.add_argument('--backend',choices=['auto','muscriptor','basic-pitch'])
        p.add_argument('--download-model',action='store_true')
        p.add_argument('--beat-checkpoint');p.add_argument('--device',default='auto')
        p.add_argument('--dtype',choices=['auto','float32','float16','bfloat16'],default='auto',
            help='MuScriptor precision; auto preserves CPU bfloat16 and uses native CUDA bfloat16 when supported')
        p.add_argument('--threads',type=int);p.add_argument('--tempo',type=float);p.add_argument('--meter')
        p.add_argument('--vocal-mode',choices=['auto','vocal','instrumental'],default='auto')
        p.add_argument('--accompaniment-only',action='store_true')
        p.add_argument('--melody-offset',type=int,default=0);p.add_argument('--harmony-offset',type=int,default=0)
        p.add_argument('--rhythm-offset',type=int,default=0);p.add_argument('--bass-offset',type=int,default=0)
        p.add_argument('--bitrate',type=int,choices=[32,48,64,96,128,160,192])
        p.add_argument('--force-from',choices=['decode','separate','analyse','transcribe','export'])
        if name=='import-work':p.add_argument('--from-work',required=True)
        if name=='compare':p.add_argument('--start',type=float,required=True);p.add_argument('--length',type=float,default=8)
        if name=='doctor':
            p.add_argument('--strict',action='store_true',help='Exit 1 if the selected runtime/backend/font is not ready')
            p.add_argument('--smoke-test',action='store_true',help='Load local MuScriptor weights and transcribe five seconds; no model downloads')
            p.add_argument('--smoke-audio',help='Local diagnostic audio (first five seconds); implies --smoke-test')
    return root

def main():
    args=parser().parse_args()
    if args.tempo is not None and not 20<=args.tempo<=300:raise ValueError('Tempo must be between 20 and 300 BPM')
    if args.meter:
        if not re.fullmatch(r'(?:[1-9]|1[0-2])/(?:1|2|4|8|16)',args.meter):raise ValueError('Unsupported meter; use N/D, e.g. 4/4 or 6/8')
    if args.command=='doctor':
        if args.smoke_audio:args.smoke_test=True
        info=doctor(args)
        emit('doctor',**info)
        if args.strict and not info['ready']:raise SystemExit(1)
    elif args.command=='inspect':emit('inspect',**inspect_source(args.input))
    elif args.command=='run':run_audio(args)
    elif args.command=='import-work':import_work(args)
    elif args.command=='render-midi':render_midi(args)
    elif args.command=='compare':
        work=Path(args.work_dir).expanduser().resolve()
        if not (work/'config.json').is_file():raise ValueError('Existing work directory with config.json required')
        os.environ['PIANO_WORK_DIR']=str(work)
        from context import STEM
        from render_audio import comparison
        target,cues=comparison(work/(STEM+'.wav'),work/'source.wav',[args.start],args.length,output_name=f'{STEM}-Compare-{args.start:g}s')
        emit('comparison',path=str(target),cues=cues)

if __name__=='__main__':
    try:main()
    except Exception as error:
        emit('error',error_type=type(error).__name__,message=str(error));raise SystemExit(1)
