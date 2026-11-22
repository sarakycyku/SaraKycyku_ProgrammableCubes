"""fastcubes.py: a fast, exact re-implementation of the Programmable Cubes simulator.

It follows the official rules of programmable_cubes_UDP.ProgrammableCubes + CubeMoveset.MoveSetRev
exactly (a move is legal if the ensemble stays connected when the cube is removed AND one of the
8 pivot patterns of the command fits in the 5x5 plane around the cube), but it keeps the cubes in
a 3D occupancy grid, so one move costs a few dozen array reads instead of scanning all cubes.
Besides apply, it can PEEK: where a command would move a cube (pattern check only, no
connectivity check, no change of state). That is what makes accept-before-apply possible.

Coordinates inside the grid are shifted by `origin`; FastCubes.positions() returns real ones.
A cube closer than 4 cells to the grid border is not moved (margin is 12, never reached in practice).

Self-test against the official simulator:  python fastcubes.py ISS 20000
"""
import sys

import numpy as np
from numba import njit

import CubeMoveset as cm

DX = np.array([1, -1, 0, 0, 0, 0], dtype=np.int64)
DY = np.array([0, 0, 1, -1, 0, 0], dtype=np.int64)
DZ = np.array([0, 0, 0, 0, 1, -1], dtype=np.int64)


def build_tables():
    occ_mask = np.zeros(16, dtype=np.int64)
    emp_mask = np.zeros(16, dtype=np.int64)
    for i, (occupied, empty) in enumerate(zip(cm.move_patterns_occupied, cm.move_patterns_empty)):
        for k in occupied:
            occ_mask[i] |= 1 << k
        for k in empty:
            emp_mask[i] |= 1 << k
    opts = np.array([np.array(a) for a in cm.move_pattern_mapping], dtype=np.int64)  # (2, 8)
    offs = np.array(cm.diffs_to_surrounding, dtype=np.int64)  # (3, 24, 3)
    disp = np.array(cm.displacements, dtype=np.int64).reshape(3, 16, 3)
    return occ_mask, emp_mask, opts, offs, disp


TABLES = build_tables()


@njit(cache=True)
def which_move(grid, x, y, z, axis, rdir, offs, occ_mask, emp_mask, opts):
    """Index (0..15) of the first pivot pattern of the command that fits, or -1."""
    occ = 0
    for k in range(24):
        if grid[x + offs[axis, k, 0], y + offs[axis, k, 1], z + offs[axis, k, 2]] >= 0:
            occ |= (1 << k)
    for t in range(8):
        i = opts[rdir, t]
        if (occ & occ_mask[i]) == occ_mask[i] and (occ & emp_mask[i]) == 0:
            return i
    return -1


@njit(cache=True)
def stays_connected(grid, pos, cube, queue, visit, stampbox, nbbuf, dx, dy, dz):
    """True if the ensemble stays connected (through faces) when `cube` is removed."""
    x = pos[cube, 0]
    y = pos[cube, 1]
    z = pos[cube, 2]
    cnt = 0
    for d in range(6):
        v = grid[x + dx[d], y + dy[d], z + dz[d]]
        if v >= 0:
            nbbuf[cnt] = v
            cnt += 1
    if cnt <= 1:
        return True
    stampbox[0] += 1
    st = stampbox[0]
    qh = 0
    qt = 1
    queue[0] = nbbuf[0]
    visit[nbbuf[0]] = st
    found = 1
    while qh < qt:
        u = queue[qh]
        qh += 1
        ux = pos[u, 0]
        uy = pos[u, 1]
        uz = pos[u, 2]
        for d in range(6):
            v = grid[ux + dx[d], uy + dy[d], uz + dz[d]]
            if v < 0 or v == cube or visit[v] == st:
                continue
            visit[v] = st
            queue[qt] = v
            qt += 1
            if abs(pos[v, 0] - x) + abs(pos[v, 1] - y) + abs(pos[v, 2] - z) == 1:
                found += 1
                if found == cnt:
                    return True
    return False


@njit(cache=True)
def peek_move(grid, pos, cube, cmd, offs, occ_mask, emp_mask, opts, disp):
    """Pattern check only. Returns (ok, new_x, new_y, new_z); state is not changed."""
    x = pos[cube, 0]
    y = pos[cube, 1]
    z = pos[cube, 2]
    if (x < 4 or y < 4 or z < 4 or x >= grid.shape[0] - 4 or y >= grid.shape[1] - 4
            or z >= grid.shape[2] - 4):
        return False, 0, 0, 0
    axis = cmd >> 1
    which = which_move(grid, x, y, z, axis, cmd & 1, offs, occ_mask, emp_mask, opts)
    if which < 0:
        return False, 0, 0, 0
    return True, x + disp[axis, which, 0], y + disp[axis, which, 1], z + disp[axis, which, 2]


@njit(cache=True)
def commit_move(grid, pos, cube, nx, ny, nz):
    grid[pos[cube, 0], pos[cube, 1], pos[cube, 2]] = -1
    pos[cube, 0] = nx
    pos[cube, 1] = ny
    pos[cube, 2] = nz
    grid[nx, ny, nz] = cube


@njit(cache=True)
def apply_move(grid, pos, cube, cmd, offs, occ_mask, emp_mask, opts, disp,
               queue, visit, stampbox, nbbuf, dx, dy, dz):
    """Same contract as ProgrammableCubes.apply_single_update_step: 1 if applied, 0 if illegal."""
    ok, nx, ny, nz = peek_move(grid, pos, cube, cmd, offs, occ_mask, emp_mask, opts, disp)
    if not ok:
        return 0
    if not stays_connected(grid, pos, cube, queue, visit, stampbox, nbbuf, dx, dy, dz):
        return 0
    commit_move(grid, pos, cube, nx, ny, nz)
    return 1


@njit(cache=True)
def apply_list(grid, pos, cubes_arr, cmds_arr, count, offs, occ_mask, emp_mask, opts, disp,
               queue, visit, stampbox, nbbuf, dx, dy, dz):
    """Apply a command list; returns the number of illegal commands (they are skipped)."""
    illegal = 0
    for i in range(count):
        if apply_move(grid, pos, cubes_arr[i], cmds_arr[i], offs, occ_mask, emp_mask, opts, disp,
                      queue, visit, stampbox, nbbuf, dx, dy, dz) == 0:
            illegal += 1
    return illegal


class FastCubes:
    """Drop-in for the parts of ProgrammableCubes that the search needs."""

    def __init__(self, positions, extra_positions=None, margin=12):
        positions = np.asarray(positions, dtype=np.int64)
        allp = positions if extra_positions is None else np.vstack([positions, extra_positions])
        self.origin = allp.min(axis=0) - margin
        shape = tuple((allp.max(axis=0) + margin - self.origin + 1).tolist())
        self.grid = np.full(shape, -1, dtype=np.int32)
        self.pos = (positions - self.origin).astype(np.int64)
        self.grid[self.pos[:, 0], self.pos[:, 1], self.pos[:, 2]] = np.arange(len(positions), dtype=np.int32)
        n = len(positions)
        self.n = n
        self.queue = np.empty(n + 6, dtype=np.int64)
        self.visit = np.zeros(n, dtype=np.int64)
        self.stampbox = np.zeros(1, dtype=np.int64)
        self.nbbuf = np.empty(6, dtype=np.int64)
        self.tables = TABLES

    def apply(self, cube, cmd):
        occ_mask, emp_mask, opts, offs, disp = self.tables
        return apply_move(self.grid, self.pos, cube, cmd, offs, occ_mask, emp_mask, opts, disp,
                          self.queue, self.visit, self.stampbox, self.nbbuf, DX, DY, DZ)

    def apply_moves(self, moves):
        """Apply a list of (cube, command); returns the number of illegal commands."""
        occ_mask, emp_mask, opts, offs, disp = self.tables
        arr = np.asarray(moves, dtype=np.int64).reshape(-1, 2)
        return apply_list(self.grid, self.pos, np.ascontiguousarray(arr[:, 0]),
                          np.ascontiguousarray(arr[:, 1]), len(arr), offs, occ_mask, emp_mask, opts,
                          disp, self.queue, self.visit, self.stampbox, self.nbbuf, DX, DY, DZ)

    def positions(self):
        return self.pos + self.origin


def self_test(problem, n_moves, seed=0, from_solution=None):
    """Compare with the official simulator on random (or recorded) commands."""
    import time
    from programmable_cubes_UDP import programmable_cubes_UDP, ProgrammableCubes
    udp = programmable_cubes_UDP(problem)
    init = np.load("{}/Initial_Config.npy".format(udp.setup["path"]))
    official = ProgrammableCubes(init)
    fast = FastCubes(init, extra_positions=udp.target_cube_positions)
    rng = np.random.default_rng(seed)
    n = len(init)
    if from_solution:
        import json
        ch = json.load(open(from_solution))["chromosome"]
        seq = [(int(ch[i]), int(ch[i + 1])) for i in range(0, len(ch) - 1, 2) if ch[i] != -1]
    else:
        seq = [(int(rng.integers(n)), int(rng.integers(6))) for _ in range(n_moves)]
    mismatches, legal_count = 0, 0
    t_off = t_fast = 0.0
    for step, (c, m) in enumerate(seq):
        t0 = time.perf_counter()
        a = official.apply_single_update_step(c, m)
        t1 = time.perf_counter()
        b = fast.apply(c, m)
        t2 = time.perf_counter()
        t_off += t1 - t0
        t_fast += t2 - t1
        legal_count += a
        if a != b or not np.array_equal(official.cube_position[c], fast.positions()[c]):
            mismatches += 1
            if mismatches <= 5:
                print("MISMATCH at step", step, "cube", c, "cmd", m, "official", a, "fast", b,
                      official.cube_position[c], fast.positions()[c])
    final_equal = np.array_equal(official.cube_position, fast.positions())
    print("{}: {} commands, {} legal | mismatches: {} | final positions equal: {} | "
          "official {:.1f} us/cmd, fast {:.2f} us/cmd".format(
              problem, len(seq), legal_count, mismatches, final_equal,
              1e6 * t_off / max(1, len(seq)), 1e6 * t_fast / max(1, len(seq))))
    return mismatches == 0 and final_equal


if __name__ == "__main__":
    problem = sys.argv[1] if len(sys.argv) > 1 else "ISS"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    solution = sys.argv[3] if len(sys.argv) > 3 else None
    sys.exit(0 if self_test(problem, count, from_solution=solution) else 1)
