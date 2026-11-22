import sys, numpy as np
from fast_sa import Context, load_moves
from fastcubes import FastCubes, peek_move, stays_connected, DX, DY, DZ
problem, path = sys.argv[1], sys.argv[2]
ctx = Context(problem)
sim = FastCubes(ctx.init, extra_positions=ctx.target_positions)
sim.apply_moves(load_moves(path))
pos = sim.pos
flags = ctx.tgt_type_grid[pos[:,0],pos[:,1],pos[:,2]] == ctx.types
tgt = ctx.rematch(pos, flags)
dist = np.abs(pos - tgt).sum(axis=1)
occ_mask, emp_mask, opts, offs, disp = sim.tables
def mobility(c):
    legal = 0
    for cmd in range(6):
        ok, nx, ny, nz = peek_move(sim.grid, sim.pos, c, cmd, offs, occ_mask, emp_mask, opts, disp)
        if ok and stays_connected(sim.grid, sim.pos, c, sim.queue, sim.visit, sim.stampbox, sim.nbbuf, DX, DY, DZ):
            legal += 1
    return legal
mis = np.flatnonzero(~flags)
print("correct", int(flags.sum()), "of", ctx.n, "| misplaced", len(mis))
free_targets = {}
for t in range(ctx.num_types):
    cells = ctx.target_g[ctx.target_types == t]
    free = [tuple(cell) for cell in cells if sim.grid[cell[0],cell[1],cell[2]] < 0]
    free_targets[t] = free
print("free target cells per type:", {t: len(v) for t, v in free_targets.items() if v})
print("wrong cubes per type     :", {t: int(((~flags) & (ctx.types == t)).sum()) for t in range(ctx.num_types) if ((~flags) & (ctx.types == t)).any()})
print(" cube type dist mob  matched-target-status")
for c in mis[np.argsort(-dist[mis])]:
    t = tgt[c]; occ = sim.grid[t[0], t[1], t[2]]
    st = "free" if occ < 0 else f"occupied by cube {occ} (type {ctx.types[occ]}, mob {mobility(occ)}, {'correct' if flags[occ] else 'wrong'})"
    print(f" {c:4d} {ctx.types[c]:3d} {dist[c]:4d} {mobility(c):3d}  {st}")
