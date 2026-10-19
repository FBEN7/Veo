import ast, sys, types, numpy as np
src = open('/home/user/Veo/train_player_reid.py').read()
tree = ast.parse(src)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
ns = {'np': np}
# stub torch
torch = types.ModuleType('torch')
class NG:
    def __enter__(self): pass
    def __exit__(self,*a): pass
torch.no_grad = lambda: NG()
sys.modules['torch'] = torch
ns['to_tensor'] = lambda crops: np.array(crops, dtype=float)
exec(compile(ast.Module([fn], []), 'evaluate', 'exec'), ns)
class M:
    def eval(self): pass
    def __call__(self, x):
        class A:
            def __init__(s, a): s.a = a
            def numpy(s): return s.a
        v = x / np.linalg.norm(x, axis=1, keepdims=True)
        return A(v)
rng = np.random.default_rng(0)
def clip(base):
    # 4 people, 2 per team, 3 crops each; embedding near a base direction
    return {f"p{i}": ("L" if i < 2 else "R", [base + 0.05*rng.standard_normal(4) for _ in range(3)]) for i in range(4)}
# clip A: very spread (different people dissimilar), clip B: collapsed
A = {f"a{i}": ("L" if i<2 else "R", [np.eye(4)[i] + 0.01*rng.standard_normal(4) for _ in range(3)]) for i in range(4)}
B = clip(np.ones(4))
print("A then B:"); ns['evaluate'](M(), [A, B], "x")
print("B then A:"); ns['evaluate'](M(), [B, A], "x")
print("empty:")
try:
    ns['evaluate'](M(), [], "x")
except Exception as e:
    print(type(e).__name__, e)
