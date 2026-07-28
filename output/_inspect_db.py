import sqlite3
import pathlib
import json

db = pathlib.Path('output/match.db')
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
indexes = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
schema = {
    t: [dict(zip(['cid', 'name', 'type', 'notnull', 'dflt_value', 'pk'], row))
        for row in cur.execute(f"PRAGMA table_info({t})").fetchall()]
    for t in tables
}
counts = {t: cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
samples = {t: [dict(r) for r in cur.execute(f"SELECT * FROM {t} LIMIT 3").fetchall()] for t in tables}

payload = {
    'db': str(db.resolve()),
    'table_count': len(tables),
    'tables': tables,
    'index_count': len(indexes),
    'indexes': indexes,
    'counts': counts,
    'schema': schema,
    'samples': samples,
}

pathlib.Path('output/db_inspect.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
print(pathlib.Path('output/db_inspect.json').resolve())
