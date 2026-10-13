"""Run identify_players --split in a per-variant copy of a window's output and score it."""
import sys, json, shutil, subprocess
from pathlib import Path
sys.path.insert(0, '/home/user/Veo')
from eval_soccertrack_v2 import truth, score_dir

w, tag, weights, cut, join = sys.argv[1:6]
src = Path(f'/home/user/Veo/output_{w}')
d = src / f'reid_{tag}'
if d.exists():
    shutil.rmtree(d)
d.mkdir()
for f in ('clip.json', 'pitch.json', 'profile.json', 'tracks.parquet', 'tracks_teams.parquet',
          'tracks_grass.parquet', 'tracks_merged.parquet'):
    shutil.copy(src / f, d / f)
cmd = [sys.executable, 'identify_players.py', str(d), '--split', '--reid', weights,
       '--cut-threshold', cut, '--look-threshold', join]
subprocess.run(cmd, check=True, cwd='/home/user/Veo')
gt = truth(Path(f'/home/user/Veo/data/soccertrack_v2/{w}_gt.parquet'))
r = score_dir(d, gt, 2)['feet']
(d / 'soccertrack_v2_scores.json').write_text(json.dumps(r, indent=1))
keep = ('identities', 'split', 'purity', 'IDF1', 'recall', 'precision')
print(json.dumps({'window': w, 'variant': tag, 'cut': cut, 'join': join,
                  're-joined': {k: r['re-joined'][k] for k in keep},
                  'identities': {k: r['identities'][k] for k in keep}}))
