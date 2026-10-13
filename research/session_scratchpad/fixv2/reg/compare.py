"""HEAD vs working tree reuse: legacy keys identical; v3 keys identical where measured."""
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
LEG = ("free-flight hit", "per pass", "feet share", "action hit", "candidates per frame",
       "candidates")
bad = []
for ra, rb in zip(a["results"], b["results"]):
    for k, v in ra.items():
        if isinstance(v, dict) and "candidates" in v:
            for key in LEG:
                if v[key] != rb[k][key]:
                    bad.append((ra["window"], k, key))
            for key in ("placebo pre", "placebo post", "placebo pre hits", "placebo post hits"):
                nb = {kk: vv for kk, vv in rb[k][key].items() if kk != "not measured"}                     if isinstance(rb[k][key], dict) else rb[k][key]
                if v[key] != nb:
                    bad.append((ra["window"], k, key))
            fa, fb = v["ff frames"], rb[k]["ff frames"]
            for ha, hb in zip(fa, fb):
                if None in hb:
                    if any(x != 0 for x, y in zip(ha, hb) if y is None):
                        bad.append((ra["window"], k, "ff unmeasured not 0 in HEAD"))
                    if [x for x, y in zip(ha, hb) if y is not None] != [y for y in hb if y is not None]:
                        bad.append((ra["window"], k, "ff measured frames differ"))
                elif ha != hb:
                    bad.append((ra["window"], k, "ff frames"))
            inc = rb[k]["ff incomplete passes"]
            print(f"  {ra['window']} {k}: incomplete passes {inc} of {len(fb)}; "
                  f"ff-auto same as HEAD: {v['ff-auto'] == rb[k]['ff-auto']}")
        elif k not in ("window",) and k in rb and v != rb[k]:
            bad.append((ra["window"], k))
sa, sb = a["summary"], b["summary"]
for key in ("windows", "paired"):
    if sa.get(key) != sb.get(key):
        bad.append(("summary", key))
print("legacy + placebo identical, ff identical where measured:", not bad, bad[:10])
