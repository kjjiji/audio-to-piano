"""Export the independent voices; validate, render, compare, package and hash."""
from pathlib import Path
import hashlib
import json
import zipfile
import soundfile as sf
from context import WORK,DEST,STEM,TITLE,CONFIG,PIPELINE_VERSION,emit,save_json
from midi_checks import write_midi,validate,track_stats,maximum_polyphony
from soundfont_checks import validate_notes
from render_audio import render,comparison

def package(files,report):
    notice=('钢琴改编助手 '+PIPELINE_VERSION+'\n\n'
        '整首音频自动分析与钢琴改编；最多四个可编辑声部：旋律、和声、伴奏节奏、低音。\n'
        '保留估计的源音频时间，不添加固定低音或统一重拍。\n'
        '模型与频谱相符、文件校验通过，均不代表逐音正确或双手可弹。请用试听和同段对比判断听感。\n\n'
        '对比文件：每组先钢琴、再原曲；具体起点见 analysis_report.json 中的 comparison_cues。\n\n'
        '钢琴音色：'+str(CONFIG.get('soundfont_credit') or Path(CONFIG['soundfont']).name)+'\n')
    archive=DEST/(STEM+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in files:z.write(file,file.name)
        z.writestr('README.txt',notice)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Corrupted ZIP')
        for file in files:
            if z.read(file.name)!=file.read_bytes():raise ValueError('Packaged bytes differ: '+file.name)
    manifest={file.name:{'path':str(file.resolve()),'bytes':file.stat().st_size,
        'sha256':hashlib.sha256(file.read_bytes()).hexdigest()} for file in files+[archive]}
    save_json(DEST/'file_manifest.json',manifest)
    emit('complete',files=manifest,note_count=report['note_count'],delivery_oversize=[name for name,v in manifest.items() if v['bytes']>10*1024*1024])
    return manifest

def main():
    import build_piano
    DEST.mkdir(parents=True,exist_ok=True)
    midi,notes=build_piano.build();duration=sf.info(WORK/'source.wav').duration
    target=DEST/(STEM+'.mid');write_midi(midi,target)
    parsed,error=validate(midi,target,duration)
    stats=[track_stats(track) for track in parsed.instruments]
    for item in stats:
        if item['name'] in ['Piano Sung Melody','Piano Primary Melody','Piano Bass'] and item['maximum_keydown_polyphony']>1:
            raise ValueError('Primary or bass line ceased to be monophonic')
    coverage=validate_notes(CONFIG['soundfont'],parsed)
    emit('midi_verified',tracks=stats,roundtrip_error_seconds=error)
    wav,mp3,render_report=render(target,duration+1)
    compare,cues=comparison(wav,WORK/'source.wav')
    source=json.loads((WORK/'source_info.json').read_text())
    source.pop('path',None)
    source.get('probe',{}).get('format',{}).pop('filename',None)
    report={'title':TITLE,'pipeline_version':PIPELINE_VERSION,'source':source,
        'source_duration_seconds':duration,'tempo_bpm':float(parsed.get_tempo_changes()[1][0]),
        'tracks':stats,'note_count':sum(len(track.notes) for track in parsed.instruments),
        'maximum_total_polyphony':maximum_polyphony(parsed),'maximum_roundtrip_timing_error_seconds':error,
        'soundfont_coverage':coverage,'render':render_report,
        'tempo_analysis':json.loads((WORK/'tempo_analysis.json').read_text()),
        'stem_analysis':json.loads((WORK/'stem_analysis.json').read_text()),
        'transcription_backend':CONFIG.get('transcription_label','MuScriptor'),
        'vocal_mode':CONFIG.get('resolved_vocal_mode'),'integration_audit':build_piano.AUDIT,
        'comparison_cues':cues,'soundfont_credit':CONFIG.get('soundfont_credit'),
        'limitations':'Automatic transcription and piano arrangement. No note-by-note listening validation or two-hand performance test. '
            'Model agreement and file checks do not establish musical accuracy or expressive feel.'}
    report_file=DEST/'analysis_report.json';save_json(report_file,report)
    package([target,mp3,compare,report_file],report)

if __name__=='__main__':main()
