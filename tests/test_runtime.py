"""GPU failures, environment preservation and diagnostics without requiring a GPU."""
import contextlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

SCRIPTS=Path(__file__).resolve().parents[1]/'skills/audio-to-piano/scripts'
sys.path.insert(0,str(SCRIPTS))
import bootstrap
import piano
import runtime_support as runtime


def fake_torch(available=True):
    cuda=MagicMock()
    cuda.is_available.return_value=available
    cuda.device_count.return_value=2 if available else 0
    cuda.current_device.return_value=0
    cuda.get_device_capability.return_value=(8,6)
    cuda.is_bf16_supported.return_value=True
    cuda.get_arch_list.return_value=['sm_86']
    cuda.mem_get_info.return_value=(20*1024**3,24*1024**3)
    return SimpleNamespace(cuda=cuda,version=SimpleNamespace(cuda='12.4'),__version__='2.5.1+cu124',
        bfloat16='torch.bfloat16',float16='torch.float16',float32='torch.float32',
        autocast=MagicMock(return_value=contextlib.nullcontext()),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                 cudnn=SimpleNamespace(allow_tf32=True)))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_explicit_cuda_never_silently_falls_back(self):
        torch=fake_torch(False)
        self.assertEqual(runtime.select_device('auto',torch),'cpu')
        with self.assertRaisesRegex(ValueError,'unavailable'):runtime.select_device('cuda',torch)
        torch.version.cuda=None
        with self.assertRaisesRegex(ValueError,'no CUDA runtime'):runtime.select_device('cuda:0',torch)

    def test_device_index_and_precision_use_selected_gpu(self):
        torch=fake_torch()
        self.assertEqual(runtime.select_device('cuda:1',torch),'cuda:1')
        self.assertEqual(runtime.select_dtype('auto','cuda:1',torch),'bfloat16')
        torch.cuda.device.assert_called_once_with('cuda:1')
        torch.cuda.get_device_capability.assert_called_once_with('cuda:1')
        with self.assertRaisesRegex(ValueError,'index 2'):runtime.select_device('cuda:2',torch)
        for value in ('cuda:-1','mps','gpu'):
            with self.assertRaises(ValueError):runtime.select_device(value,torch)

    def test_older_gpu_uses_half_and_rejects_native_bf16(self):
        torch=fake_torch();torch.cuda.get_device_capability.return_value=(7,5)
        self.assertEqual(runtime.select_dtype('auto','cuda:1',torch),'float16')
        with self.assertRaisesRegex(ValueError,'native bfloat16'):runtime.select_dtype('bfloat16','cuda:1',torch)
        self.assertEqual(runtime.select_dtype('auto','cpu'),'bfloat16')
        with self.assertRaises(ValueError):runtime.select_dtype('float16','cpu')

    def test_cuda_kernel_failure_is_not_reported_ready(self):
        torch=fake_torch();audio=SimpleNamespace(__version__='2.5.1+cu124')
        with patch.dict(sys.modules,{'torch':torch,'torchaudio':audio}), \
             patch.object(runtime,'kernel_check',side_effect=RuntimeError('no kernel image available')), \
             patch.object(runtime.subprocess,'run',side_effect=FileNotFoundError()):
            result=runtime.torch_diagnostics('cuda:1')
        self.assertTrue(result['cuda_available'])
        self.assertFalse(result['ready'])
        self.assertIn('no kernel image',result['error'])

    def test_torchaudio_abi_mismatch_is_not_reported_ready(self):
        with patch.dict(sys.modules,{'torch':fake_torch(),'torchaudio':SimpleNamespace(__version__='2.6.0')}), \
             patch.object(runtime.subprocess,'run',side_effect=FileNotFoundError()):
            result=runtime.torch_diagnostics('cuda')
        self.assertFalse(result['ready']);self.assertIn('do not match',result['error'])

    def test_cuda_autocast_matches_requested_weight_precision(self):
        torch=fake_torch();core=SimpleNamespace(autocast='original fp16')
        model=SimpleNamespace(_model=core)
        loader=MagicMock(return_value=model)
        with patch.dict(sys.modules,{'torch':torch,'muscriptor':SimpleNamespace(
                TranscriptionModel=SimpleNamespace(load_model=loader))}):
            runtime.load_muscriptor('weights','cuda:1','bfloat16')
            torch.autocast.assert_called_once_with('cuda',dtype='torch.bfloat16')
            runtime.load_muscriptor('weights','cuda:1','float32')
            self.assertIsInstance(core.autocast,contextlib.nullcontext)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            core.autocast='cpu unchanged'
            runtime.load_muscriptor('weights','cpu','bfloat16')
            self.assertEqual(core.autocast,'cpu unchanged')

    def test_model_config_is_required_and_validated(self):
        weights=self.root/'model.safetensors';weights.write_bytes(b'fixture')
        self.assertFalse(runtime.model_diagnostics(weights)['ready'])
        config={'dim':32,'num_heads':2,'num_layers':1,'card':1395}
        (self.root/'config.json').write_text(json.dumps(config))
        result=runtime.model_diagnostics(weights)
        self.assertTrue(result['ready']);self.assertFalse(result['weights_loaded'])
        config['num_heads']=3;(self.root/'config.json').write_text(json.dumps(config))
        self.assertFalse(runtime.model_diagnostics(weights)['ready'])

    def test_hf_symlink_keeps_sibling_config(self):
        blob=self.root/'blob';blob.write_bytes(b'fixture')
        snapshot=self.root/'snapshot';snapshot.mkdir()
        weights=snapshot/'model.safetensors';weights.symlink_to(blob)
        (snapshot/'config.json').write_text(json.dumps({'dim':32,'num_heads':2,'num_layers':1,'card':1395}))
        path=piano.resolve_model(str(weights),'large')
        self.assertEqual(Path(path).parent,snapshot)
        self.assertTrue(runtime.model_diagnostics(path)['ready'])

    def test_transcription_cache_changes_with_precision_device_and_config(self):
        weights=self.root/'model.safetensors';weights.write_bytes(b'fixture')
        config_file=self.root/'config.json';config_file.write_text('{}')
        config={'model_path':str(weights),'model_size':'large','backend':'muscriptor','device':'cpu','dtype':'bfloat16'}
        with patch.object(piano.importlib.metadata,'version',return_value='0.3.0'):
            old=piano.transcription_fingerprint({},'bass',config)
            self.assertNotEqual(old,piano.transcription_fingerprint({},'bass',{**config,'dtype':'float32'}))
            self.assertNotEqual(old,piano.transcription_fingerprint({},'bass',{**config,'device':'cuda:0'}))
            config_file.write_text('{"changed":true}')
            self.assertNotEqual(old,piano.transcription_fingerprint({},'bass',config))

    def test_reuse_preserves_installed_cuda_muscriptor_and_numpy(self):
        target=self.root/'existing';(target/'bin').mkdir(parents=True);(target/'bin/python').touch()
        packages={'torch':'2.10.0+cu130','torchaudio':'2.10.0+cu130','muscriptor':'0.3.0',
                  'numpy':'2.3.5','onnxruntime-gpu':'1.31.0'}
        snapshot={'platform':'linux','version':[3,12],'prefix':str(target),
                  'base_prefix':'/usr','packages':packages}
        with patch.object(bootstrap,'snapshot',return_value=snapshot):
            details=bootstrap.plan(target,'ignored','reuse')
        self.assertEqual(details['preserved_packages'],packages)
        constraints=details['files_to_write'][str(target/'.audio-to-piano-preserved.txt')]
        self.assertIn('torch==2.10.0+cu130',constraints)
        missing=details['files_to_write'][str(target/'.audio-to-piano-missing.txt')]
        self.assertNotIn('numpy==',missing);self.assertNotIn('onnxruntime==',missing)
        install_commands=[c for c in details['commands'] if 'install' in c]
        self.assertTrue(any('--dry-run' in c for c in install_commands))
        self.assertFalse(any('--upgrade' in c for c in install_commands))
        self.assertFalse((target/'.audio-to-piano-preserved.txt').exists())

    def test_install_profiles_use_target_interpreter_and_keep_cpu_default(self):
        snapshot={'platform':'linux','version':[3,12],'prefix':'/usr','base_prefix':'/usr','packages':{}}
        with patch.object(bootstrap,'snapshot',return_value=snapshot) as probe:
            cpu=bootstrap.plan(self.root/'cpu','/custom/python3.12')
            cuda=bootstrap.plan(self.root/'cuda','/custom/python3.12','cuda')
        probe.assert_called_with('/custom/python3.12')
        self.assertIn('torch==2.5.1+cpu',cpu['commands'][2])
        self.assertIn('torch==2.5.1+cu124',cuda['commands'][2])
        self.assertEqual(cpu['commands'][0][0],'/custom/python3.12')

    def test_existing_or_system_environments_cannot_be_replaced(self):
        target=self.root/'existing';(target/'bin').mkdir(parents=True);(target/'bin/python').touch()
        snapshot={'platform':'linux','version':[3,11],'prefix':str(target),'base_prefix':'/usr','packages':{}}
        with patch.object(bootstrap,'snapshot',return_value=snapshot):
            with self.assertRaisesRegex(ValueError,'already exists'):bootstrap.plan(target,'python','cuda')
        snapshot['base_prefix']=snapshot['prefix']
        with patch.object(bootstrap,'snapshot',return_value=snapshot):
            with self.assertRaisesRegex(ValueError,'system Python'):bootstrap.plan(target,'python','reuse')

    def test_only_expected_basic_pitch_tensorflow_exclusion_is_ignored(self):
        result=SimpleNamespace(returncode=1,stdout='basic-pitch 0.4.0 requires tensorflow, which is not installed.\n',stderr='')
        with patch.object(runtime.subprocess,'run',return_value=result):
            self.assertTrue(runtime.package_conflicts()['ready'])
            result.stdout+='torchaudio has requirement torch==2.6.0, but you have torch 2.5.1.\n'
            self.assertFalse(runtime.package_conflicts()['ready'])

    def test_doctor_bad_assets_still_reports_all_diagnostics(self):
        args=piano.parser().parse_args(['doctor','--model-path',str(self.root/'absent'),
                                      '--soundfont',str(self.root/'absent.sf2'),'--device','cuda'])
        with patch.object(piano,'dependencies',return_value={'ready':True}), \
             patch.object(runtime,'import_diagnostics',return_value=({'torch':{'ready':True}},{})), \
             patch.object(runtime,'torch_diagnostics',return_value={'ready':False}), \
             patch.object(runtime,'onnx_diagnostics',return_value={'ready':True}), \
             patch.object(runtime,'package_conflicts',return_value={'ready':True}):
            result=piano.doctor(args)
        self.assertFalse(result['ready']);self.assertFalse(result['model']['ready'])
        self.assertFalse(result['soundfont_checks']['ready']);self.assertIn('runtime',result)

    def test_strict_doctor_exit_status(self):
        with patch.object(sys,'argv',['piano.py','doctor','--strict']), \
             patch.object(piano,'doctor',return_value={'ready':False}),patch.object(piano,'emit'):
            with self.assertRaises(SystemExit) as error:piano.main()
        self.assertEqual(error.exception.code,1)


if __name__=='__main__':unittest.main(verbosity=2)
