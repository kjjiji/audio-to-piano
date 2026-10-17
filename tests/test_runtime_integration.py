"""Offline CPU integration with real dependencies; no published weights or user audio.

Run after bootstrap. The tiny random MuScriptor fixture tests execution only,
never transcription quality. Set TEST_PIANO_SOUNDFONT for the render check.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]/'skills/audio-to-piano/scripts'
sys.path.insert(0,str(SCRIPTS))


class RuntimeIntegration(unittest.TestCase):
    def test_real_cpu_kernels_and_local_muscriptor_inference(self):
        import torch
        from muscriptor.transcription_model import _build_model,_ModelConfig
        from safetensors.torch import save_file
        import mido
        import numpy as np
        import soundfile as sf
        from piano import resolve_font
        from runtime_support import kernel_check
        torch.set_num_threads(2);torch.manual_seed(0)
        kernel_check(torch,'cpu','bfloat16')
        config={'dim':32,'num_heads':2,'num_layers':1,'card':1395}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.json').write_text(json.dumps(config))
            model=_build_model(torch.device('cpu'),_ModelConfig(**config))
            save_file(model.state_dict(),str(root/'model.safetensors'))
            result=subprocess.run([sys.executable,str(SCRIPTS/'piano.py'),'doctor','--device','cpu',
                '--backend','muscriptor','--model-path',str(root),'--smoke-test','--strict','--threads','2',
                '--soundfont',resolve_font(os.environ.get('TEST_PIANO_SOUNDFONT'))],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            report=json.loads(result.stdout)
            self.assertEqual(report['model_inference']['status'],'passed')
            self.assertEqual(report['model_inference']['completed_chunks'],1)
            self.assertFalse(report['gpu_inference_verified'])
            self.assertFalse(report['model_inference']['musical_accuracy_evaluated'])
            (root/'config.json').replace(root/'model-config.json')
            model_dir=root/'weights';model_dir.mkdir()
            (root/'model.safetensors').replace(model_dir/'model.safetensors')
            (root/'model-config.json').replace(model_dir/'config.json')
            (root/'config.json').write_text(json.dumps({'model_path':str(model_dir/'model.safetensors'),
                'device':'cpu','dtype':'bfloat16'}))
            t=np.arange(5*16000)/16000;sf.write(root/'accompaniment.wav',.1*np.sin(2*np.pi*261.63*t),16000)
            result=subprocess.run([sys.executable,str(SCRIPTS/'transcribe.py'),'--role','accompaniment','--threads','2'],
                env=dict(os.environ,PIANO_WORK_DIR=str(root)),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            self.assertEqual(mido.MidiFile(root/'accompaniment-transcribed.mid').type,1)

    def test_basic_pitch_onnx_and_essentia_execute(self):
        import numpy as np
        import essentia.standard as es
        import basic_pitch
        from basic_pitch.inference import predict
        import soundfile as sf
        from runtime_support import onnx_diagnostics
        self.assertTrue(onnx_diagnostics()['ready'])
        signal=(.1*np.sin(2*np.pi*261.63*np.arange(3*22050)/22050)).astype('float32')
        pitch,confidence=es.PredominantPitchMelodia(sampleRate=22050)(signal)
        self.assertTrue(np.isfinite(pitch).all());self.assertTrue(np.isfinite(confidence).all())
        with tempfile.TemporaryDirectory() as directory:
            audio=Path(directory)/'tone.wav';sf.write(audio,signal,22050)
            path=Path(basic_pitch.__file__).parent/'saved_models/icassp_2022/nmp.onnx'
            output,midi,events=predict(str(audio),model_or_model_path=str(path))
            self.assertTrue(output['note'].size>0)
            self.assertTrue(np.isfinite(output['note']).all())

    def test_real_midi_render_and_manifest(self):
        import pretty_midi
        from piano import digest,resolve_font
        font=resolve_font(os.environ.get('TEST_PIANO_SOUNDFONT'))
        self.assertIsNotNone(font,'Provide TEST_PIANO_SOUNDFONT or install a system piano SF2')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.mid';work=root/'job'
            midi=pretty_midi.PrettyMIDI()
            track=pretty_midi.Instrument(0);midi.instruments.append(track)
            track.notes=[pretty_midi.Note(80,pitch,k*.5,k*.5+.4) for k,pitch in enumerate((48,60,64,67))]
            midi.write(str(source))
            result=subprocess.run([sys.executable,str(SCRIPTS/'piano.py'),'render-midi',str(source),
                '--work-dir',str(work),'--soundfont',font,'--device','cpu'],
                capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            report=json.loads((work/'output/analysis_report.json').read_text())
            self.assertEqual(report['note_count'],4)
            manifest=json.loads((work/'output/file_manifest.json').read_text())
            for item in manifest.values():self.assertEqual(digest(item['path']),item['sha256'])


if __name__=='__main__':unittest.main(verbosity=2)
