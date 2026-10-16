"""Regression cases for failures that can change the audible result."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'skills/audio-to-piano/scripts'))
import mido
import numpy as np
import pretty_midi
from midi_checks import validate,write_midi
from soundfont_checks import piano_coverage,validate_notes


class PianoRegressions(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def piece(self,bpm=137.01):
        midi=pretty_midi.PrettyMIDI(initial_tempo=bpm,resolution=1920)
        midi.time_signature_changes.append(pretty_midi.TimeSignature(4,4,0))
        track=pretty_midi.Instrument(0,name='钢琴／あなたがいない')
        midi.instruments.append(track)
        return midi,track

    def test_touching_same_key_keeps_second_note_duration(self):
        midi,track=self.piece(120)
        track.notes=[pretty_midi.Note(72,60,0,1),pretty_midi.Note(88,60,1,2)]
        path=self.root/'repeat.mid';write_midi(midi,path)
        parsed,error=validate(midi,path,2)
        self.assertEqual([(n.start,n.end) for n in parsed.instruments[0].notes],[(0.,1.),(1.,2.)])
        raw=mido.MidiFile(path,charset='utf-8');events=[];position=0
        for event in raw.tracks[1]:
            position+=event.time
            if position==3840 and event.type in ['note_on','note_off']:events.append(event.type)
        self.assertEqual(events,['note_off','note_on'])

    def test_unicode_titles_and_long_song_do_not_drift(self):
        midi,track=self.piece()
        midi.text_events.append(pretty_midi.Text('纯钢琴・全曲',0))
        track.notes=[pretty_midi.Note(80,60,286.321,287.190)]
        path=self.root/'中文と日本語.mid';write_midi(midi,path)
        parsed,error=validate(midi,path,287.2)
        self.assertEqual(parsed.instruments[0].name,track.name)
        self.assertEqual(parsed.text_events[0].text,'纯钢琴・全曲')
        self.assertLess(error,.0003)

    def test_variable_tempo_survives_increased_resolution(self):
        raw=mido.MidiFile(type=1,ticks_per_beat=480,charset='utf-8')
        conductor=mido.MidiTrack();raw.tracks.append(conductor)
        conductor.extend([mido.MetaMessage('set_tempo',tempo=333333,time=0),
            mido.MetaMessage('set_tempo',tempo=750000,time=1440)])
        track=mido.MidiTrack();raw.tracks.append(track)
        track.extend([mido.Message('program_change',program=0),
            mido.Message('note_on',note=60,velocity=90,time=480),
            mido.Message('note_off',note=60,time=1440),
            mido.Message('note_on',note=64,velocity=70,time=480),
            mido.Message('note_off',note=64,time=480)])
        source=self.root/'low-resolution.mid';raw.save(source)
        expected=pretty_midi.PrettyMIDI(str(source),charset='utf-8')
        before=expected.get_tempo_changes()
        target=self.root/'piano.mid';write_midi(expected,target,resolution=1920)
        parsed,error=validate(expected,target,expected.get_end_time()+.001)
        self.assertEqual(expected.resolution,480)
        self.assertEqual(mido.MidiFile(target).ticks_per_beat,1920)
        np.testing.assert_allclose(parsed.get_tempo_changes()[0],before[0],atol=.0001)
        np.testing.assert_allclose(parsed.get_tempo_changes()[1],before[1],atol=.00001)
        self.assertLess(error,.0001)

    def test_cross_track_same_key_collision_is_rejected(self):
        midi,first=self.piece(120)
        first.notes=[pretty_midi.Note(80,48,0,2)]
        second=pretty_midi.Instrument(0,name='bass')
        second.notes=[pretty_midi.Note(70,48,1,2.5)];midi.instruments.append(second)
        target=self.root/'collision.mid';write_midi(midi,target)
        with self.assertRaisesRegex(ValueError,'Overlapping identical key across tracks'):
            validate(midi,target,2.5)

    def test_soundfont_velocity_hole_cannot_silently_drop_notes(self):
        # Minimal SF2 metadata fixture with valid key zones but missing velocities 51..79.
        def chunk(name,data):return name+struct.pack('<I',len(data))+data+(b'\0' if len(data)%2 else b'')
        phdr=struct.pack('<20sHHHIII',b'Piano',0,0,0,0,0,0)+struct.pack('<20sHHHIII',b'EOP',0,0,1,0,0,0)
        pbag=struct.pack('<HHHH',0,0,1,0);pgen=struct.pack('<HH',41,0)
        inst=struct.pack('<20sH',b'Piano',0)+struct.pack('<20sH',b'EOI',2)
        ibag=b''.join(struct.pack('<HH',k,0) for k in [0,3,6])
        igen=b''.join(struct.pack('<HH',op,value) for op,value in [
            (43,21|(108<<8)),(44,1|(50<<8)),(53,0),
            (43,21|(108<<8)),(44,80|(127<<8)),(53,1)])
        pdta=b'pdta'+b''.join(chunk(tag,data) for tag,data in [
            (b'phdr',phdr),(b'pbag',pbag),(b'pgen',pgen),(b'inst',inst),(b'ibag',ibag),(b'igen',igen)])
        body=b'sfbk'+chunk(b'LIST',pdta)
        path=self.root/'gapped.sf2';path.write_bytes(b'RIFF'+struct.pack('<I',len(body))+body)
        coverage=piano_coverage(path)
        self.assertTrue(coverage[60,50]);self.assertTrue(coverage[60,80])
        self.assertFalse(coverage[60,64])
        midi,track=self.piece();track.notes=[pretty_midi.Note(64,60,0,1)]
        with self.assertRaisesRegex(ValueError,'silent key/velocity gaps'):validate_notes(path,midi)
        track.notes[0].velocity=80
        self.assertEqual(validate_notes(path,midi)['uncovered_pairs'],0)


if __name__=='__main__':unittest.main(verbosity=2)
