import sys, ast, pandas as pd, json
src = open('/home/user/Veo/tune_st2_split.py').read()
tree = ast.parse(src)
fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'choose'][0]
ns = {'pd': pd}
exec(compile(ast.Module([fn], []), 'choose', 'exec'), ns)
t = pd.DataFrame([
 {"cut": None, "join": None, "IDF1": 0.520},
 {"cut": None, "join": 0.97, "IDF1": 0.521},
 {"cut": 0.7, "join": 0.85, "IDF1": 0.523},
])
r = ns['choose'](t); print(json.dumps(r))
# identify()'s mapping
THRESHOLD, JOIN = 0.4, 0.7
cut = THRESHOLD if r['cut'] is None else r['cut']
look = JOIN if r['join'] is None else r['join']
print("identify() applies cut", -2.0 if cut < 0 else cut, "join", look)
t2 = pd.DataFrame([{"cut": None, "join": None, "IDF1": 0.52},{"cut": 0.5, "join": 0.9, "IDF1": 0.51}])
r2 = ns['choose'](t2); print(json.dumps(r2))
cut = THRESHOLD if r2['cut'] is None else r2['cut']; look = JOIN if r2['join'] is None else r2['join']
print("identify() applies cut", cut, "join", look)
