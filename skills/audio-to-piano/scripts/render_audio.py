"""Render every active MIDI channel and generate synchronized comparisons."""
from pathlib import Path
import subprocess
import mido
import numpy as np
import soundfile as sf
from context import WORK,DEST,STEM,TITLE,CONFIG

def encode(wav,target,bitrate,title=None):
    args=['ffmpeg','-v','error','-y','-i',str(wav),'-af','loudnorm=I=-18:TP=-1.5:LRA=11',
        '-ar','44100','-c:a','libmp3lame','-b:a',str(bitrate)+'k','-metadata','title='+str(title or TITLE)]
    if CONFIG.get('soundfont_credit'):args+=['-metadata','comment='+CONFIG['soundfont_credit']]
    subprocess.run(args+[str(target)],check=True)
    subprocess.run(['ffmpeg','-v','error','-i',str(target),'-f','null','-'],check=True)

def render(path,duration,stem=None):
    import fluidsynth
    sr=44100;total=round(duration*sr)
    if total<1:raise ValueError('Empty render')
    font_path=Path(CONFIG['soundfont']).expanduser().resolve()
    if not font_path.is_file():raise ValueError('SoundFont not found: '+str(font_path))
    raw=mido.MidiFile(path,charset='utf-8')
    channels=sorted({e.channel for t in raw.tracks for e in t if hasattr(e,'channel')})
    def synthesize(gain):
        synth=fluidsynth.Synth(samplerate=sr,gain=gain)
        try:
            synth.setting('synth.chorus.active',0);synth.setting('synth.reverb.active',1)
            synth.setting('synth.reverb.level',.14);synth.setting('synth.reverb.room-size',.55)
            synth.setting('synth.reverb.damp',.7)
            font=synth.sfload(str(font_path))
            if font<0:raise ValueError('Cannot load SoundFont')
            for channel in channels:
                if synth.program_select(channel,font,0,0)!=0:raise ValueError('SoundFont has no bank 0 piano preset')
            audio=np.zeros((total,2),np.float32);cursor=0;elapsed=0.
            def fill(position):
                nonlocal cursor
                position=min(total,position)
                while cursor<position:
                    count=min(65536,position-cursor)
                    audio[cursor:cursor+count]=synth.get_samples(count).reshape(-1,2)/32768.
                    cursor+=count
            for event in raw:
                elapsed+=event.time;fill(round(elapsed*sr))
                if event.type=='note_on':
                    if event.velocity:synth.noteon(event.channel,event.note,event.velocity)
                    else:synth.noteoff(event.channel,event.note)
                elif event.type=='note_off':synth.noteoff(event.channel,event.note)
                elif event.type=='control_change':synth.cc(event.channel,event.control,event.value)
                elif event.type=='pitchwheel':synth.pitch_bend(event.channel,event.pitch)
                elif event.type=='program_change' and event.program!=0:raise ValueError('Non-piano program')
            fill(total)
            return audio
        finally:synth.delete()
    gain=.35
    for attempt in range(4):
        audio=synthesize(gain);peak=float(abs(audio).max())
        if not np.isfinite(audio).all():raise ValueError('Non-finite audio samples')
        if peak<.00001:raise ValueError('Silent render: check SoundFont key/velocity coverage')
        if peak<.95:break
        gain*=.40
    else:raise ValueError('Render clips even at reduced gain')
    fade=min(sr,len(audio));audio[-fade:]*=np.linspace(1,0,fade)[:,None];audio*=.88/peak
    basename=stem or STEM
    wav=WORK/(basename+'.wav');mp3=DEST/(basename+'.mp3')
    sf.write(wav,audio,sr,subtype='PCM_24')
    encode(wav,mp3,int(CONFIG.get('bitrate',192)))
    return wav,mp3,{'raw_peak':peak,'synth_gain':gain,'channels_initialized':channels,'sample_rate':sr}

def comparison(wav,source,starts=None,length=None,output_name=None):
    duration=sf.info(source).duration
    length=float(length or min(8,duration))
    if length<=0:raise ValueError('Empty comparison')
    available=max(0,duration-length)
    if starts is None:
        fractions=[.02,.25,.60,.88] if duration>=24 else [0,1] if duration>length+1 else [0]
        starts=[round(float(np.clip(f*duration,0,available)),3) for f in fractions]
        starts=list(dict.fromkeys(starts))
    parts=[];cues=[];position=0.
    for section,start in enumerate(starts):
        if not 0<=start<duration:raise ValueError('Comparison start outside source')
        for index,(label,path) in enumerate([('piano',wav),('source',source)]):
            segment=WORK/f'compare-{section}-{index}.wav'
            subprocess.run(['ffmpeg','-v','error','-y','-ss',str(start),'-t',str(length),'-i',str(path),
                '-af',f'loudnorm=I=-18:TP=-1.5:LRA=11,aresample=44100,asetpts=PTS-STARTPTS,apad=whole_dur={length},atrim=duration={length}',
                '-ar','44100','-ac','2','-c:a','pcm_f32le',str(segment)],check=True)
            audio,sr=sf.read(segment,dtype='float32',always_2d=True)
            if sr!=44100 or len(audio)!=round(length*sr) or not np.isfinite(audio).all() or abs(audio).max()>=1:
                raise ValueError('Invalid synchronized comparison segment')
            cues.append({'source_start_seconds':start,'version':label,'start_seconds':position,'length_seconds':length})
            parts.extend([audio,np.zeros((sr,2),np.float32)]);position+=length+1
    combined=WORK/'comparison.wav';sf.write(combined,np.concatenate(parts[:-1]),44100,subtype='PCM_24')
    target=DEST/((output_name or STEM+'-Source-Comparison')+'.mp3')
    encode(combined,target,min(192,int(CONFIG.get('bitrate',192))),TITLE+' / source comparison')
    return target,cues
