"""Piano arrangement of independent sung melody, accompaniment, and bass."""
from pathlib import Path
from collections import defaultdict
import copy
import json
import warnings
import numpy as np
import pretty_midi
import soundfile as sf
from context import WORK, CONFIG, TITLE

DURATION=sf.info(WORK/'source.wav').duration
ROLE_NAMES=['Piano Sung Melody','Piano Harmony and Countermelody','Piano Rhythmic Accompaniment','Piano Bass']
AUDIT={}

def arrays(path):
    with np.load(path) as f:return {k:f[k] for k in f.files}

EVIDENCE={name:arrays(WORK/(name+'-evidence.npz')) for name in
    ['vocals','bass','piano','guitar','other','accompaniment','source']}

def window(times,a,b):
    start=int(np.searchsorted(times,a));end=int(np.searchsorted(times,b))
    start=min(start,len(times)-1)
    return slice(start,min(len(times),max(start+1,end)))

def parse(path,origin):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        m=pretty_midi.PrettyMIDI(str(path))
    return [{'pitch':int(n.pitch),'start':float(n.start),'end':min(float(n.end),DURATION),
        'program':int(i.program),'origin':origin} for i in m.instruments if not i.is_drum
        for n in i.notes if 21<=n.pitch<=103 and n.end-n.start>=.065 and n.start<DURATION]

def spectral(name,pitch,a,b):
    e=EVIDENCE[name];p=pitch-int(e['first_pitch'])
    if not 0<=p<len(e['cqt']):return 0.
    s=window(e['cqt_times'],a,b)
    return float(np.median(e['cqt'][p,s]))

def relative_spectral(name,pitch,a,b):
    e=EVIDENCE[name];s=window(e['cqt_times'],a,b)
    low=26 if name=='bass' else 48 if name=='vocals' else 36
    high=62 if name=='bass' else 86 if name=='vocals' else 96
    strongest=float(np.median(e['cqt'][low-24:high-24,s].max(axis=0)))
    return spectral(name,pitch,a,b)/max(strongest,.025)

def probability(name,pitch,a,b,onset=False):
    e=EVIDENCE[name]
    if 'note' not in e or not 21<=pitch<109:return 0.
    s=window(e['times'],a,b)
    return float(np.max(e['onset'][s,pitch-21]) if onset else np.mean(e['note'][s,pitch-21]))

def rms(name,a,b):
    e=EVIDENCE[name];s=window(e['audio_times'],a,b)
    return float(np.percentile(e['rms'][s],75))

def pitch_support(name,pitch,a,b):
    e=EVIDENCE[name];s=window(e['f0_times'],a,b)
    v=e['voiced'][s]&(e['confidence'][s]>.055)
    amplitude=np.interp(e['f0_times'][s],e['audio_times'],e['rms'])
    v &= amplitude>(.007 if name=='vocals' else .010)
    weights=(e['confidence'][s]+.025)*v
    if weights.sum()<1e-6:return 0.,0.
    match=abs(e['pitch'][s]-pitch)<.65
    return float(np.sum(weights*match)/weights.sum()),float(v.mean())

def coverage(n,notes):
    matching=[x for x in notes if x['pitch']==n['pitch'] and x['end']>n['start'] and x['start']<n['end']]
    return max((max(0,min(n['end'],x['end'])-max(n['start'],x['start'])) for x in matching),default=0)/max(.01,n['end']-n['start'])

def sung_melody():
    raw=parse(WORK/'vocals-transcribed.mid',CONFIG.get('transcription_label','MuScriptor')+' isolated voice')
    candidates=[];corrected=[];rejected=[];recovered=[];trimmed=[]
    for n in raw:
        if not 47<=n['pitch']<=86:continue
        a=n['start']+.020;b=max(a+.020,n['end']-.020)
        match,voiced=pitch_support('vocals',n['pitch'],a,b)
        ratio=relative_spectral('vocals',n['pitch'],a,b)
        prob=probability('vocals',n['pitch'],a,b)
        if rms('vocals',a,b)<.006:
            rejected.append({**n,'reason':'very weak voice amplitude'});continue
        if voiced<.18 and prob<.34:
            rejected.append({**n,'reason':'neither stable vocal fundamental nor independent note activation'});continue
        if match<.20 and voiced>.65:
            e=EVIDENCE['vocals'];s=window(e['f0_times'],a,b)
            valid=e['voiced'][s]&(e['confidence'][s]>.07)
            q=np.rint(e['pitch'][s]).astype(int)
            weights=(e['confidence'][s]+.025)*valid
            counts=np.bincount(np.clip(q,0,127),weights=weights,minlength=128)
            best=int(counts.argmax())
            best_match,best_voiced=pitch_support('vocals',best,a,b)
            fundamental=spectral('vocals',best,a,b)
            proposed=spectral('vocals',n['pitch'],a,b)
            if best_match>.72 and 47<=best<=86 and fundamental>1.35*max(proposed,.025) and probability('vocals',best,a,b)>.36:
                corrected.append({'start':n['start'],'end':n['end'],'old_pitch':n['pitch'],'pitch':best})
                n={**n,'pitch':best};match=best_match
                ratio=relative_spectral('vocals',best,a,b);prob=probability('vocals',best,a,b)
        if match<.12 and voiced>.60 and (prob<.45 or ratio<.25):
            rejected.append({**n,'reason':'weak support from independent vocal pitch'});continue
        if match>.50 and voiced>.35 and n['end']-n['start']>.30:
            e=EVIDENCE['vocals'];s=window(e['f0_times'],n['start'],n['end'])
            times=e['f0_times'][s]
            tonal=e['voiced'][s]&(e['confidence'][s]>.055)&(abs(e['pitch'][s]-n['pitch'])<.85)
            if tonal.any():
                first=float(times[tonal][0]);last=float(times[tonal][-1])
                original_start=n['start'];original_end=n['end']
                if first-n['start']>.22 and probability('vocals',n['pitch'],n['start'],first)<.28:
                    n={**n,'start':max(n['start'],first-.035)}
                if n['end']-last>.22 and probability('vocals',n['pitch'],last,n['end'])<.28:
                    n={**n,'end':min(n['end'],last+.060)}
                if n['start']!=original_start or n['end']!=original_end:
                    trimmed.append({'pitch':n['pitch'],'old_start':original_start,'old_end':original_end,
                        'start':n['start'],'end':n['end']})
        candidates.append({**n,'role':0,'match':match,'ratio':ratio,'prob':prob,'priority':.18})
    # Restore independently strong sung notes missed by the universal model.
    for n in parse(WORK/'vocals-basicpitch.mid','independent vocal note'):
        if not 47<=n['pitch']<=86 or n['end']-n['start']<.13 or coverage(n,candidates)>.50:continue
        a=n['start']+.030;b=max(a+.020,n['end']-.020)
        match,voiced=pitch_support('vocals',n['pitch'],a,b)
        ratio=relative_spectral('vocals',n['pitch'],a,b)
        prob=probability('vocals',n['pitch'],a,b)
        if match<.58 or voiced<.45 or prob<.45 or ratio<.18 or rms('vocals',a,b)<.008:continue
        proposal={**n,'role':0,'match':match,'ratio':ratio,'prob':prob,'priority':0.}
        candidates.append(proposal);recovered.append(n)
    # The fundamental trajectory selects the sung line, rather than the
    # highest simultaneous estimate. Preserve the identity of a held note.
    edges=sorted(set(n[k] for n in candidates for k in ['start','end']))
    line=[]
    for a,b in zip(edges,edges[1:]):
        if b-a<.001:continue
        active=[n for n in candidates if n['start']<=(a+b)/2<n['end']]
        if not active:continue
        def score(n):
            match,voiced=pitch_support('vocals',n['pitch'],a,b)
            return 2.8*match+.40*n['prob']+.16*min(n['ratio'],1)+n['priority']
        chosen=max(active,key=score)
        if line and line[-1]['pitch']==chosen['pitch'] and line[-1]['source_start']==chosen['start'] and abs(line[-1]['end']-a)<.002:
            line[-1]['end']=b
        else:line.append({**chosen,'start':a,'end':b,'source_start':chosen['start']})
    line=[n for n in line if n['end']-n['start']>=.070]
    # Tiny gaps within a sustained syllable are not a new piano attack.
    joined=[]
    for n in line:
        if joined and n['pitch']==joined[-1]['pitch'] and 0<=n['start']-joined[-1]['end']<.055:
            onset=probability('vocals',n['pitch'],n['start']-.035,n['start']+.055,onset=True)
            if onset<.55:
                joined[-1]['end']=n['end'];continue
        joined.append(n)
    AUDIT['vocal']={'raw_notes':len(raw),'primary_notes':len(joined),
        'pitch_corrections':corrected,'rejected_notes':rejected,'recovered_notes':recovered,
        'weak_tonal_leads_and_tails_trimmed':trimmed}
    return joined

def instrumental_melody():
    raw=parse(WORK/'accompaniment-transcribed.mid',CONFIG.get('transcription_label','MuScriptor')+' instrumental reference')
    candidates=[]
    def support(n,a,b):
        match,voiced=pitch_support('accompaniment',n['pitch'],a,b)
        upper,_=pitch_support('accompaniment',n['pitch']+12,a,b)
        low=spectral('accompaniment',n['pitch'],a,b)
        high=spectral('accompaniment',n['pitch']+12,a,b)
        if low>2.4*max(high,.025):match=max(match,.75*upper)
        return match,voiced
    for n in raw:
        if not 48<=n['pitch']<=97:continue
        a=n['start']+.025;b=max(a+.020,n['end']-.015)
        match,voiced=support(n,a,b)
        ratio=relative_spectral('accompaniment',n['pitch'],a,b)
        if match<.18 or voiced<.25 or ratio<.12:continue
        candidates.append({**n,'role':0,'match':match,'ratio':ratio,'source_start':n['start']})
    for stem in ['piano','guitar','other']:
        for n in parse(WORK/(stem+'-basicpitch.mid'),'independent instrumental melody'):
            if not 48<=n['pitch']<=97 or n['end']-n['start']<.15 or coverage(n,candidates)>.5:continue
            a=n['start']+.025;b=max(a+.020,n['end']-.015)
            match,voiced=support(n,a,b)
            ratio=relative_spectral('accompaniment',n['pitch'],a,b)
            if match>.60 and voiced>.50 and ratio>.20 and probability(stem,n['pitch'],a,b)>.60:
                candidates.append({**n,'role':0,'match':match,'ratio':ratio,'source_start':n['start']})
    edges=sorted(set(n[k] for n in candidates for k in ['start','end']))
    result=[]
    for a,b in zip(edges,edges[1:]):
        active=[n for n in candidates if n['start']<=(a+b)/2<n['end']]
        if not active:continue
        def score(n):
            match,voiced=support(n,a,b)
            continuation=.10 if result and result[-1]['pitch']==n['pitch'] else 0
            return 2.2*match+.40*min(n['ratio'],1)+continuation
        chosen=max(active,key=score)
        if result and result[-1]['pitch']==chosen['pitch'] and result[-1]['source_start']==chosen['source_start'] and abs(result[-1]['end']-a)<.002:
            result[-1]['end']=b
        else:result.append({**chosen,'start':a,'end':b})
    result=[n for n in result if n['end']-n['start']>=.070]
    AUDIT['instrumental_melody']={'raw_candidates':len(raw),'primary_notes':len(result),
        'method':'Independent predominant pitch and original spectrum; never highest-active-note selection.',
        'uncertain':not bool(result)}
    return result

def accompaniment():
    raw=parse(WORK/'accompaniment-transcribed.mid',CONFIG.get('transcription_label','MuScriptor')+' vocal-free accompaniment')
    notes=[];rejected=[];recovered=[]
    def describe(n):
        a=n['start']+.018;b=max(a+.020,min(n['end']-.010,n['start']+.30))
        relative=relative_spectral('accompaniment',n['pitch'],a,b)
        original=spectral('accompaniment',n['pitch'],a,b)
        values={name:spectral(name,n['pitch'],a,b)*(.45+probability(name,n['pitch'],a,b))
                for name in ['piano','guitar','other']}
        role=2 if values['guitar']>1.15*max(values['piano'],values['other']) else 1
        if n['program'] in [48,49,50,51,88,89,90,91,92,93,94,95]:role=1
        return relative,original,role
    for n in raw:
        if not 28<=n['pitch']<=97:continue
        relative,original,role=describe(n)
        if original<.022 or relative<.055:
            rejected.append({**n,'reason':'weak accompaniment fundamental'});continue
        if n['pitch']<36 and probability('piano',n['pitch'],n['start']+.02,n['end']-.01)<.55:continue
        notes.append({**n,'role':role,'spectral_ratio':relative})
    for name in ['piano','guitar','other']:
        for n in parse(WORK/(name+'-basicpitch.mid'),'independent '+name):
            if not 36<=n['pitch']<=97 or n['end']-n['start']<.14 or coverage(n,notes)>.50:continue
            a=n['start']+.035;b=max(a+.02,n['end']-.015)
            prob=probability(name,n['pitch'],a,b)
            relative,original,role=describe(n)
            if prob<.67 or relative<.16 or relative_spectral(name,n['pitch'],a,b)<.21 or original<.045:continue
            lower=[x for x in notes if x['pitch']==n['pitch']-12 and
                max(0,min(n['end'],x['end'])-max(n['start'],x['start']))>.7*(n['end']-n['start'])]
            if lower and n['pitch']>=72 and original<.45*spectral('accompaniment',n['pitch']-12,a,b):continue
            notes.append({**n,'role':role,'spectral_ratio':relative});recovered.append(n)
    # Retain real octave voicings; discard only weak upper estimates with a
    # strong, time-coincident lower fundamental in the vocal-free mixture.
    filtered=[];octaves=[]
    by_pitch=defaultdict(list)
    for n in notes:by_pitch[n['pitch']].append(n)
    for n in notes:
        if n['pitch']>=76:
            lower=[x for x in by_pitch[n['pitch']-12] if
                abs(x['start']-n['start'])<.045 and
                max(0,min(n['end'],x['end'])-max(n['start'],x['start']))>.75*(n['end']-n['start'])]
            if lower:
                a=n['start']+.035;b=max(a+.03,min(n['end'],n['start']+.40))
                high=spectral('accompaniment',n['pitch'],a,b)
                low=spectral('accompaniment',n['pitch']-12,a,b)
                if high<.30*max(low,.025):octaves.append(n);continue
        filtered.append(n)
    AUDIT['accompaniment']={'raw_notes':len(raw),'candidate_notes':len(filtered),
        'spectral_rejections':rejected,'independent_recovered_notes':recovered,'weak_upper_octaves_removed':octaves}
    return filtered

def bass_line():
    raw=parse(WORK/'bass-transcribed.mid',CONFIG.get('transcription_label','MuScriptor')+' isolated bass')
    candidates=[];recovered=[]
    for n in raw:
        if not 24<=n['pitch']<=60:continue
        a=n['start']+.025;b=max(a+.020,n['end']-.015)
        if rms('bass',a,b)<.008 or relative_spectral('bass',n['pitch'],a,b)<.08:continue
        candidates.append({**n,'role':3,'priority':.28})
    for n in parse(WORK/'bass-basicpitch.mid','independent bass note'):
        if not 28<=n['pitch']<=57 or n['end']-n['start']<.13 or coverage(n,candidates)>.50:continue
        a=n['start']+.025;b=max(a+.020,n['end']-.015)
        match,voiced=pitch_support('bass',n['pitch'],a,b)
        if match>.60 and voiced>.45 and probability('bass',n['pitch'],a,b)>.57 and relative_spectral('bass',n['pitch'],a,b)>.16:
            candidates.append({**n,'role':3,'priority':0.});recovered.append(n)
    edges=sorted(set(n[k] for n in candidates for k in ['start','end']))
    result=[]
    for a,b in zip(edges,edges[1:]):
        active=[n for n in candidates if n['start']<=(a+b)/2<n['end']]
        if not active:continue
        model_active=[n for n in active if n['priority']>.1]
        latest=max(n['start'] for n in model_active) if model_active else max(n['start'] for n in active)
        active=[n for n in active if n['priority']<=.1 or n['start']>=latest-.024]
        def score(n):
            match,voiced=pitch_support('bass',n['pitch'],a,b)
            # Melodia frequently follows the bass's second harmonic. The
            # actual low fundamental and independent note activation take
            # precedence over that monophonic pitch estimate.
            fundamental=relative_spectral('bass',n['pitch'],a,b)
            activation=probability('bass',n['pitch'],a,b)
            return 1.30*fundamental+.45*activation+.45*match+n['priority']+(.20 if n['start']>=latest-.024 else 0)
        chosen=max(active,key=lambda n:(score(n),-n['pitch']))
        if result and result[-1]['pitch']==chosen['pitch'] and result[-1]['source_start']==chosen['start'] and abs(result[-1]['end']-a)<.002:
            result[-1]['end']=b
        else:result.append({**chosen,'start':a,'end':b,'source_start':chosen['start']})
    result=[n for n in result if n['end']-n['start']>=.065]
    AUDIT['bass']={'raw_notes':len(raw),'monophonic_notes':len(result),'independent_recovered_notes':recovered}
    return result

def dynamics(notes):
    values=[]
    for n in notes:
        a=n['start']+.018;b=max(a+.020,min(n['end']-.010,n['start']+.16))
        if n['role']==0:value=rms('vocals' if CONFIG.get('resolved_vocal_mode','vocal')=='vocal' else 'accompaniment',a,b)
        elif n['role']==3:value=rms('bass',a,b)
        else:
            stem='guitar' if n['role']==2 else 'piano'
            direct=spectral(stem,n['pitch'],a,b)
            other=spectral('other',n['pitch'],a,b)
            value=max(direct,other*.70,.001)
        values.append(max(value,.001))
    by_role=defaultdict(list);by_pitch=defaultdict(list);groups=defaultdict(list)
    for n,v in zip(notes,values):
        by_role[n['role']].append(v)
        by_pitch[(n['role'],n['pitch'])].append(v)
        groups[(n['role'],round(n['start']/.025))].append(n)
    bases=CONFIG.get('velocity_baselines',[88,63,61,65])
    for n,v in zip(notes,values):
        role=n['role'];same_pitch=by_pitch[(role,n['pitch'])]
        # Vocal and bass RMS should retain loud and quiet phrases; spectral
        # instrument levels also need per-pitch compensation for timbre.
        reference=float(np.median(by_role[role]))
        if role in [1,2] and len(same_pitch)>=5:
            reference=float(np.sqrt(reference*np.median(same_pitch)))
        change=float(np.clip(12*np.log2(v/max(reference,.001)),-20,14))
        voices=len(groups[(role,round(n['start']/.025))])
        budget=2.0*np.log2(max(1,voices)) if role in [1,2] else 0
        n['velocity']=int(np.clip(round(bases[role]+change-budget),30,104 if role==0 else 84))
    return notes

def single_keyboard(notes):
    priorities={0:4,3:3,1:2,2:1}
    result=[];merged=0;suppressed=0;truncated=0
    for pitch in sorted(set(n['pitch'] for n in notes)):
        seq=sorted([copy.deepcopy(n) for n in notes if n['pitch']==pitch],key=lambda n:n['start'])
        kept=[]
        for n in seq:
            if kept and n['start']-kept[-1]['start']<.035:
                prev=kept[-1]
                if priorities[n['role']]>priorities[prev['role']]:
                    end=n['end'] if n['role']==0 else max(n['end'],prev['end'])
                    start=n['start'] if n['role']==0 else min(n['start'],prev['start'])
                    kept[-1]={**n,'start':start,'end':end}
                else:
                    if prev['role']!=0:prev['end']=max(prev['end'],n['end'])
                    if prev['role']!=0 or n['role']==0:prev['velocity']=max(prev['velocity'],n['velocity'])
                merged+=1;continue
            if kept and n['start']<kept[-1]['end']:
                prev=kept[-1]
                if priorities[prev['role']]>priorities[n['role']]:
                    # The key already sounds in the primary voice. Do not
                    # invent a delayed attack when the sung note releases.
                    suppressed+=1;continue
                prev['end']=n['start']-.008;truncated+=1
                if prev['end']-prev['start']<.055:kept.pop()
            if n['end']-n['start']>=.065:kept.append(n)
        result.extend(kept)
    bass=sorted([n for n in result if n['role']==3],key=lambda n:n['start'])
    moved=0
    for a,b in zip(bass,bass[1:]):
        if a['end']>b['start']+.001:a['role']=1;moved+=1
    AUDIT['keyboard']={'duplicate_notes_merged':merged,'held_primary_keys_protected':suppressed,
        'repeated_key_durations_shortened':truncated,'shared_bass_keys_assigned_to_harmony':moved,
        'onsets_shifted_to_beat_grid':0}
    return sorted(result,key=lambda n:(n['start'],n['pitch']))

def build():
    vocal=[] if not CONFIG.get('include_primary_melody',True) else sung_melody() if CONFIG.get('resolved_vocal_mode','vocal')=='vocal' else instrumental_melody()
    print('PRIMARY_MELODY',len(vocal),flush=True)
    backing=accompaniment();print('ACCOMPANIMENT',len(backing),flush=True)
    bass=bass_line();print('BASS',len(bass),flush=True)
    notes=single_keyboard(dynamics(vocal+backing+bass))
    tempo=json.loads((WORK/'tempo_analysis.json').read_text())
    bpm=float(CONFIG.get('tempo_bpm') or tempo['fitted_bpm_with_skipped_beats'])
    if not np.isfinite(bpm) or not 20<=bpm<=300:raise ValueError('Unreliable tempo: supply --tempo explicitly')
    midi=pretty_midi.PrettyMIDI(initial_tempo=bpm,resolution=1920)
    meter=CONFIG.get('meter')
    if meter:numerator,denominator=map(int,meter.split('/'))
    else:numerator,denominator=int(tempo.get('conservative_beats_per_bar') or 4),4
    midi.time_signature_changes=[pretty_midi.TimeSignature(numerator,denominator,0)]
    midi.text_events=[pretty_midi.Text(TITLE+' - independently transcribed melody, accompaniment and bass; all piano',0),
        pretty_midi.Text('Original audio timing retained; first detected downbeat '+str(tempo['first_downbeat_seconds'])+' seconds',0)]
    for role,name in enumerate(ROLE_NAMES):
        if role==0 and CONFIG.get('resolved_vocal_mode')=='instrumental':name='Piano Primary Melody'
        track=pretty_midi.Instrument(program=0,name=name)
        for n in notes:
            if n['role']==role:
                # A small release gap prevents identical-key overlaps after
                # MIDI tick rounding, without moving the original attack.
                end=max(n['start']+.030,n['end']-.003)
                track.notes.append(pretty_midi.Note(n['velocity'],n['pitch'],n['start'],end))
        track.control_changes=[pretty_midi.ControlChange(7,100,0),pretty_midi.ControlChange(11,127,0),
            pretty_midi.ControlChange(10,[66,64,61,63][role],0),pretty_midi.ControlChange(64,0,0)]
        if track.notes:midi.instruments.append(track)
    if not midi.instruments:raise ValueError('No supported musical notes; cannot produce a piano arrangement')
    AUDIT['method']='Isolated sung melody selected by independent fundamental pitch; actual source accompaniment pitches and bass attacks retained. No fixed accompaniment template or added metrical accent.'
    return midi,notes
