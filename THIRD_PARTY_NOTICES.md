# External components

No dependency source distributions, model weights, soundfont samples or user music are included in this plugin. The MIT license covers this plugin's original scripts and documentation only. Installation leaves upstream packages and their license/notice files in the separate runtime environment.

Relevant upstream projects:

- Demucs: https://github.com/adefossez/demucs
- MuScriptor: https://github.com/muscriptor/muscriptor
- Beat This!: https://github.com/CPJKU/beat_this
- Basic Pitch: https://github.com/spotify/basic-pitch
- Essentia: https://essentia.upf.edu/licensing_information.html — its AGPL/commercial terms must be considered for any redistribution or service deployment involving Essentia; this archive does not bundle it.
- PyTorch / Torchaudio: https://pytorch.org/
- librosa: https://librosa.org/
- Mido: https://mido.readthedocs.io/
- Pretty MIDI: https://github.com/craffel/pretty-midi
- FluidSynth and pyFluidSynth: https://www.fluidsynth.org/ and https://github.com/nwhitehead/pyfluidsynth
- FFmpeg: https://ffmpeg.org/legal.html — terms depend on the selected build.

MuScriptor model weights have their own access conditions and license. Runtime installation or use does not transfer permission to redistribute weights.

When using Salamander Grand Piano, credit Alexander Holm and its Yamaha C5 samples, CC BY 3.0: https://creativecommons.org/licenses/by/3.0/ and https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html#SalamanderGrandPiano . If velocity-zone mappings have been adapted, disclose that change. Do not apply this attribution to unrelated user-configured soundfonts; retain their actual attribution/license instead.

FuFumidi is referenced to identify the previously used transcription engine. This plugin calls MuScriptor directly; FuFumidi application code, branding and installer binaries are not included.
