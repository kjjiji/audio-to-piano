"""Meaningful serialization checks; these do not assess musical accuracy."""
from collections import defaultdict
import math
import mido
import pretty_midi

def write_midi(midi,path,resolution=None):
    """Serialize seconds against exact integer tempo values, including maps.

    Explicit UTF-8 text and note-off ordering avoid filename/title failures,
    long-song tempo drift, and repeated-key note truncation.
    """
    resolution=int(resolution or midi.resolution)
    tempo_times,bpms=midi.get_tempo_changes()
    tempos=[round(60_000_000/float(bpm)) for bpm in bpms]
    tick_origins=[0.]
    for k in range(1,len(tempos)):
        tick_origins.append(tick_origins[-1]+(float(tempo_times[k])-float(tempo_times[k-1]))*resolution*1e6/tempos[k-1])
    def ticks(seconds):
        k=0
        while k+1<len(tempo_times) and seconds>=tempo_times[k+1]:k+=1
        return round(tick_origins[k]+(seconds-float(tempo_times[k]))*resolution*1e6/tempos[k])
    def append_events(track,events):
        previous=0
        for tick,priority,event in sorted(events,key=lambda item:(item[0],item[1])):
            track.append(event.copy(time=tick-previous));previous=tick
        track.append(mido.MetaMessage('end_of_track',time=0))
    raw=mido.MidiFile(type=1,ticks_per_beat=resolution,charset='utf-8')
    conductor=mido.MidiTrack();raw.tracks.append(conductor)
    events=[(ticks(float(t)),0,mido.MetaMessage('set_tempo',tempo=us)) for t,us in zip(tempo_times,tempos)]
    for ts in midi.time_signature_changes:
        events.append((ticks(ts.time),1,mido.MetaMessage('time_signature',numerator=ts.numerator,denominator=ts.denominator)))
    majors=['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
    minors=['Cm','C#m','Dm','Ebm','Em','Fm','F#m','Gm','G#m','Am','Bbm','Bm']
    for key in midi.key_signature_changes:
        name=(majors if key.key_number<12 else minors)[key.key_number%12]
        events.append((ticks(key.time),1,mido.MetaMessage('key_signature',key=name)))
    for text in midi.text_events:events.append((ticks(text.time),2,mido.MetaMessage('text',text=text.text)))
    for lyric in midi.lyrics:events.append((ticks(lyric.time),2,mido.MetaMessage('lyrics',text=lyric.text)))
    append_events(conductor,events)
    channels=[k for k in range(16) if k!=9]
    if len(midi.instruments)>len(channels):raise ValueError('More than 15 piano tracks: merge MIDI ports/tracks explicitly before rendering')
    for channel,instrument in zip(channels,midi.instruments):
        track=mido.MidiTrack();raw.tracks.append(track)
        events=[(0,-2,mido.MetaMessage('track_name',name=instrument.name)),
                (0,-1,mido.Message('program_change',channel=channel,program=0))]
        for control in instrument.control_changes:
            events.append((ticks(control.time),1,mido.Message('control_change',channel=channel,control=control.number,value=control.value)))
        for bend in instrument.pitch_bends:
            events.append((ticks(bend.time),1,mido.Message('pitchwheel',channel=channel,pitch=bend.pitch)))
        for note in instrument.notes:
            start=ticks(note.start);end=ticks(note.end)
            if end<=start:raise ValueError('Note shorter than MIDI tick resolution')
            events.extend([(start,2,mido.Message('note_on',channel=channel,note=note.pitch,velocity=note.velocity)),
                           (end,0,mido.Message('note_off',channel=channel,note=note.pitch,velocity=0))])
        append_events(track,events)
    raw.save(path)

def track_stats(track):
    notes=track.notes
    held=set();polyphony=span=0
    edges=sorted([(n.start,1,k) for k,n in enumerate(notes)]+[(n.end,-1,k) for k,n in enumerate(notes)])
    for _,change,k in edges:
        if change>0:held.add(k)
        else:held.discard(k)
        pitches=[notes[j].pitch for j in held]
        if len(pitches)!=len(set(pitches)):
            raise ValueError('Overlapping identical keys in '+track.name)
        polyphony=max(polyphony,len(pitches))
        if pitches:span=max(span,max(pitches)-min(pitches))
    return {'name':track.name,'notes':len(notes),
        'pitch_range':[min(n.pitch for n in notes),max(n.pitch for n in notes)] if notes else [],
        'maximum_keydown_polyphony':polyphony,'maximum_keydown_span_semitones':span}

def validate(expected,path,duration,one_keyboard=True):
    actual=pretty_midi.PrettyMIDI(str(path),charset='utf-8');error=0.
    if len(actual.instruments)!=len(expected.instruments):raise ValueError('Track count changed after MIDI serialization')
    for before,after in zip(expected.instruments,actual.instruments):
        if before.program!=after.program or before.is_drum!=after.is_drum:raise ValueError('Instrument changed')
        first=sorted(before.notes,key=lambda n:(n.pitch,n.start,n.end))
        second=sorted(after.notes,key=lambda n:(n.pitch,n.start,n.end))
        if len(first)!=len(second):raise ValueError('Note count changed after MIDI serialization')
        for a,b in zip(first,second):
            if a.pitch!=b.pitch or a.velocity!=b.velocity:raise ValueError('Pitch/velocity changed after MIDI serialization')
            if not (math.isfinite(b.start) and math.isfinite(b.end) and 0<=b.start<b.end<=duration+.002):
                raise ValueError('Invalid note time')
            if not (21<=b.pitch<=108 and 1<=b.velocity<=127):raise ValueError('Note outside acoustic piano range')
            error=max(error,abs(a.start-b.start),abs(a.end-b.end))
    if error>=.001:raise ValueError('MIDI roundtrip changed timing by at least 1 ms')
    raw=mido.MidiFile(path,charset='utf-8')
    if raw.type!=1:raise ValueError('Expected Type 1 MIDI')
    for track in raw.tracks[1:]:
        if any(e.type in ['set_tempo','time_signature','key_signature'] for e in track):
            raise ValueError('Tempo/meter/key metadata must be in conductor track')
    if any(track.program!=0 or track.is_drum for track in actual.instruments):raise ValueError('Non-piano track in piano output')
    if any(e.type=='note_on' and e.velocity and e.channel==9 for track in raw.tracks for e in track):
        raise ValueError('Drum channel in piano output')
    if one_keyboard:
        by_pitch=defaultdict(list)
        for track in actual.instruments:
            for n in track.notes:by_pitch[n.pitch].append(n)
        for pitch,notes in by_pitch.items():
            notes.sort(key=lambda n:n.start)
            if any(a.end>b.start+.001 for a,b in zip(notes,notes[1:])):
                raise ValueError('Overlapping identical key across tracks: '+str(pitch))
    return actual,error

def maximum_polyphony(midi):
    edges=sorted([(n.start,1) for i in midi.instruments for n in i.notes]+
                 [(n.end,-1) for i in midi.instruments for n in i.notes])
    count=maximum=0
    for _,change in edges:count+=change;maximum=max(maximum,count)
    return maximum
