"""Run the pipeline's later stages with a given team per track and score them."""
import sys, json, shutil, io, contextlib
from pathlib import Path
import pandas as pd
sys.path.insert(0, '/home/user/Veo')
from score_soccernet import run_pipeline
from eval_soccertrack_v2 import truth, score_dir

S = Path('/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad')
COPY = ('clip.json', 'profile.json', 'tracks.parquet', 'pitch.json',
        'tracks_teams.parquet', 'tracks_grass.parquet', 'ground_plane.json')


def run_variant(w: str, team_map: dict, tag: str, quiet: bool = True) -> dict:
    src = Path(f'/home/user/Veo/output_{w}')
    d = S / 'var' / w / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in COPY:
        if (src / f).exists():
            shutil.copy(src / f, d / f)
    cut = json.loads((src / 'pitch.json').read_text())
    from src.team_assignment_v2 import FIXED_VIEW_KIT_FEATURE, FIXED_VIEW_CORE_DEPTH
    # the override replaces the cached teams; the setting only has to match
    (d / 'pitch.json').write_text(json.dumps({
        'pitch': cut['pitch'], 'margin_px': cut['margin_px'],
        'kit_feature': FIXED_VIEW_KIT_FEATURE, 'core_depth': FIXED_VIEW_CORE_DEPTH}))
    clip = f'/home/user/Veo/data/soccertrack_v2/{w}.mp4'
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
        run_pipeline(clip, d, fixed_camera=True, model_name='yolov8s.pt', stride=2,
                     pitch=cut['pitch'], pitch_margin_px=cut['margin_px'],
                     team_override=team_map)
    gt = truth(Path(f'/home/user/Veo/data/soccertrack_v2/{w}_gt.parquet'))
    r = score_dir(d, gt, 2)['feet']['re-joined']
    out = {k: v for k, v in r.items() if k.startswith(('team', 'recall', 'precision', 'IDF1', 'split', 'purity'))}
    # detections kept in a team that are not an outfield player, per frame
    from eval_soccertrack_v2 import boxes, match_feet
    g = gt[gt.frame % 2 == 0]
    m = boxes(pd.read_parquet(d / 'tracks_merged.parquet'))
    m = m[m.frame % 2 == 0]
    person = match_feet(m, g)
    role = g.groupby('person').role.agg(lambda s: s.mode().iloc[0])
    outfield = person.map(role) == 'player'
    in_team = m.team.isin(('team_A', 'team_B'))
    out['non-outfield in a team /frame'] = round(float((in_team & ~outfield).sum()) / g.frame.nunique(), 2)
    out['n_outfield'] = int((g.role == 'player').sum())
    return out


if __name__ == '__main__':
    w = sys.argv[1]
    t = pd.read_parquet(f'/home/user/Veo/output_{w}/tracks_teams.parquet')
    p = t[t.cls == 'player']
    cur = p.groupby('track_id').team.first().dropna().to_dict()
    print('current assignment via override:', run_variant(w, cur, 'current'))
