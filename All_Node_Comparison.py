"""Disk-backed, sequential all-node recording for the Radau MALD replay.

No biology or solver tolerances are defined here. Run one writer per directory.
Only a prefix of fully written nodes is committed; unfinished nodes are replayed.
"""
from pathlib import Path
import json
import hashlib
import os
import shutil
import time
import numpy as np

STATE_NAMES = ["APAP", "NAPQI", "GSH", "H", "Z", "Lys", "Regen", "AST", "ALT", "Clot"]
CHEMICAL_NAMES = STATE_NAMES[:3]
_S = np.array([0., 1./3., 2./3., 1.])
_INV = np.linalg.inv(np.vander(_S, 4, increasing=True))


class RunLock:
    """OS lock released even when the process crashes; lock file may remain."""
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0)
        if self.handle.read(1) == b"":
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise RuntimeError("Another MALD replay is using this project; wait for it to finish") from exc
        return self

    def __exit__(self, *args):
        if os.name == "nt":
            import msvcrt
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        self.handle.close()


def array_digest(value):
    """Hash in bounded blocks, including array layout-independent shape/dtype."""
    a = np.asarray(value)
    h = hashlib.sha256(str((a.shape, str(a.dtype))).encode())
    for first in range(0, len(a), 256):
        h.update(np.ascontiguousarray(a[first:first + 256]).tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    # Persist the rename on systems that support directory fsync.
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def interval_diagnostics(solution, reference_hzl=None):
    """Integrate and find extrema of each accepted Radau cubic interpolant.

    Four evaluations reconstruct each cubic using the public sol(t) interface.
    Simpson 3/8 integrates a cubic exactly (up to roundoff). Stationary points
    and both endpoints supply extrema. These remain numerical approximations
    of the true ODE solution, and must be included in tolerance comparisons.
    Returns AUC, minima, maxima, min/max local times, sampled minima of all
    states, and sampled H+Z+Lys drift. Never integrates across an APAP jump.
    """
    if not solution.success or solution.sol is None:
        raise RuntimeError("Successful Radau dense output is required")
    left = solution.t[:-1]
    width = np.diff(solution.t)
    if not len(width) or np.any(width <= 0):
        raise RuntimeError("Invalid accepted-step times")
    times = left[:, None] + width[:, None] * _S
    values = solution.sol(times.ravel()).reshape(10, len(width), 4)
    if not np.isfinite(values).all() or not np.isfinite(solution.y).all():
        raise RuntimeError("Nonfinite MALD state within an interval")
    chem = values[:3]
    auc = np.sum((chem @ np.array([1., 3., 3., 1.]) / 8.) * width, axis=1)
    coeff = chem @ _INV.T
    a, b, c = 3 * coeff[:, :, 3], 2 * coeff[:, :, 2], coeff[:, :, 1]
    scale = np.max(np.abs(coeff), axis=2)
    tiny = 128 * np.finfo(float).eps * np.maximum(scale, np.finfo(float).tiny)
    quadratic = np.abs(a) > tiny
    linear = (~quadratic) & (np.abs(b) > tiny)
    disc = b*b - 4*a*c
    real = quadratic & (disc >= 0)
    root_disc = np.sqrt(np.maximum(disc, 0))
    q = -.5 * (b + np.copysign(root_disc, b))
    r1 = np.full_like(a, np.nan)
    r2 = np.full_like(a, np.nan)
    np.divide(q, a, out=r1, where=real)
    np.divide(c, q, out=r2, where=real & (q != 0))
    np.divide(-c, b, out=r1, where=linear)
    roots = np.stack((np.zeros_like(a), np.ones_like(a), r1, r2), axis=2)
    valid = np.isfinite(roots) & (roots >= 0) & (roots <= 1)
    roots = np.where(valid, roots, 0.)
    candidate = ((coeff[:, :, 3, None]*roots + coeff[:, :, 2, None])*roots
                 + coeff[:, :, 1, None])*roots + coeff[:, :, 0, None]
    # Use original endpoint evaluations rather than reconstructed endpoints.
    candidate[:, :, 0] = chem[:, :, 0]
    candidate[:, :, 1] = chem[:, :, 3]
    flat = candidate.reshape(3, -1)
    candidate_time = (left[None, :, None] + width[None, :, None]*roots).reshape(3, -1)
    lo = np.argmin(flat, axis=1)
    hi = np.argmax(flat, axis=1)
    row = np.arange(3)
    hzl = values[3:6].sum(axis=0)
    hzl_start = solution.y[3:6, 0].sum() if reference_hzl is None else reference_hzl
    return (auc, flat[row, lo], flat[row, hi], candidate_time[row, lo],
            candidate_time[row, hi], values.min(axis=(1, 2)),
            float(np.max(np.abs(hzl - hzl_start))))


class AllNodeStore:
    def __init__(self, directory, identity, snapshot, resume=False,
                 checkpoint_nodes=100, checkpoint_seconds=300.):
        self.directory = Path(directory)
        self.identity = identity
        self.n = int(identity["node_count"])
        self.t = int(identity["total_iterations"] - identity["first_iteration"])
        self.checkpoint_nodes = checkpoint_nodes
        self.checkpoint_seconds = checkpoint_seconds
        self.last_checkpoint_time = time.monotonic()
        self.spec = {
            "states": ((self.n, self.t + 1, 10), "float64"),
            "added_apap": ((self.n, self.t), "float64"),
            "status": ((self.n, self.t), "uint8"),
            "auc": ((self.n, 3), "float64"),
            "minima": ((self.n, 3), "float64"),
            "maxima": ((self.n, 3), "float64"),
            "min_times": ((self.n, 3), "float64"),
            "max_times": ((self.n, 3), "float64"),
            # nfev,njev,nlu, sampled HZL drift, completed intervals, min10 states
            "diagnostics": ((self.n, 15), "float64"),
        }
        if resume:
            self.manifest = json.loads((self.directory / "checkpoint.json").read_text())
            if self.manifest["identity"] != identity:
                raise ValueError("Resume refused: settings, input, initialization, code or package versions changed")
            self.next_node = int(self.manifest["completed_nodes"])
            if not 0 <= self.next_node <= self.n:
                raise ValueError("Invalid checkpoint completed-node count")
        else:
            if self.directory.exists():
                raise FileExistsError(f"{self.directory} exists; use resume or a different run name")
            self.directory.parent.mkdir(parents=True, exist_ok=True)
            required = sum(int(np.prod(shape))*np.dtype(dtype).itemsize for shape, dtype in self.spec.values())
            if shutil.disk_usage(self.directory.parent).free < required + 256*1024**2:
                raise OSError(f"Need at least {required / 1024**3:.2f} GiB plus headroom for binary output")
            self.directory.mkdir()
            with (self.directory / "initial_snapshot.npz").open("wb") as f:
                np.savez(f, **snapshot)
                f.flush()
                os.fsync(f.fileno())
            self.next_node = 0
            self.manifest = {
                "schema_version": 1, "identity": identity, "completed_nodes": 0,
                "status": "running", "state_names": STATE_NAMES,
                "chemical_names": CHEMICAL_NAMES,
                "states_layout": "node, time, state; initial then every interval end",
                "APAP_jumps": "states[k,j,0] + added_apap[k,j] is the post-addition APAP",
                "auc_units": "mol*s per cell", "chemical_units": "mol per cell",
                "time_units": "s; absolute simulation time",
                "diagnostics_columns": ["nfev", "njev", "nlu", "sampled_HZL_drift",
                                         "completed_intervals"] + ["sampled_min_" + s for s in STATE_NAMES],
                "extrema_method": "endpoints and stationary points of each accepted Radau cubic; includes initial state and APAP jumps",
                "auc_method": "Simpson 3/8 on each accepted Radau cubic, exact for the interpolant up to roundoff",
                "scope": "this invocation only; first_iteration > 0 excludes earlier AUC",
            }
            print(f"All-node binary output: approximately {required / 1024**3:.2f} GiB")
        self.arrays = {}
        for name, (shape, dtype) in self.spec.items():
            path = self.directory / (name + ".npy")
            if resume:
                a = np.load(path, mmap_mode="r+", allow_pickle=False)
                if a.shape != shape or a.dtype != np.dtype(dtype):
                    raise ValueError(f"Invalid checkpoint array: {path}")
            else:
                a = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            self.arrays[name] = a
        self.committed = self.next_node
        if not resume:
            self.checkpoint()

    def write_node(self, k, states, added, status, auc, minima, maxima,
                   min_times, max_times, diagnostics):
        if k != self.next_node:
            raise ValueError("Nodes must be committed sequentially, exactly once")
        data = dict(states=states, added_apap=added, status=status, auc=auc,
                    minima=minima, maxima=maxima, min_times=min_times,
                    max_times=max_times, diagnostics=diagnostics)
        for name, value in data.items():
            if np.shape(value) != self.arrays[name].shape[1:] or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name} for node {k + 1}")
        if diagnostics[4] != self.t:
            raise ValueError("Incomplete node history")
        for name, value in data.items():
            self.arrays[name][k] = value
        self.next_node = k + 1
        if (self.next_node - self.committed >= self.checkpoint_nodes
                or time.monotonic() - self.last_checkpoint_time >= self.checkpoint_seconds):
            self.checkpoint()

    def checkpoint(self):
        # Persist arrays BEFORE atomically increasing the completed prefix.
        for name, array in self.arrays.items():
            array.flush()
            with (self.directory / (name + ".npy")).open("r+b") as f:
                os.fsync(f.fileno())
        self.manifest.update(completed_nodes=self.next_node, status="running")
        atomic_json(self.directory / "checkpoint.json", self.manifest)
        self.committed = self.next_node
        self.last_checkpoint_time = time.monotonic()
        print(f"Checkpoint: {self.committed}/{self.n} complete nodes", flush=True)

    def verify(self):
        if self.next_node != self.n:
            raise RuntimeError(f"Incomplete run: {self.next_node}/{self.n} nodes")
        minima = np.full(10, np.inf)
        negative_counts = np.zeros(10, dtype=np.int64)
        drift = 0.
        for first in range(0, self.n, 256):
            last = min(first + 256, self.n)
            for name, a in self.arrays.items():
                if not np.isfinite(a[first:last]).all():
                    raise RuntimeError(f"Nonfinite {name} in nodes {first + 1}..{last}")
            d = self.arrays["diagnostics"][first:last]
            if not np.all(d[:, 4] == self.t):
                raise RuntimeError("Missing intervals in saved node records")
            if np.any(self.arrays["status"][first:last] > 3):
                raise RuntimeError("Invalid categorical status")
            minima = np.minimum(minima, d[:, 5:].min(axis=0))
            negative_counts += (d[:, 5:] < 0).sum(axis=0)
            drift = max(drift, float(d[:, 3].max()))
        report = {
            "coverage_pass": True, "expected_nodes": self.n, "completed_nodes": self.n,
            "intervals_per_node": self.t, "successful_intervals": self.n*self.t,
            "final_time_s": self.identity["total_iterations"]*self.identity["interval_s"],
            "all_records_finite": True, "max_sampled_H_Z_Lys_drift": drift,
            "sampled_state_minima": dict(zip(STATE_NAMES, minima.tolist())),
            "nodes_with_sampled_negative_state": dict(zip(STATE_NAMES, negative_counts.tolist())),
            "note": "Coverage is not convergence or a biological validity pass. Negatives and conservation drift require interpretation against units and tolerances.",
        }
        atomic_json(self.directory / "coverage_report.json", report)
        return report

    def finish(self):
        self.checkpoint()
        report = self.verify()
        self.manifest["status"] = "completed"
        atomic_json(self.directory / "checkpoint.json", self.manifest)
        return report

    def populate_comparison(self, recorder):
        """Rebuild global/probe outputs from disk; no restart double counting."""
        states = self.arrays["states"]
        added = self.arrays["added_apap"]
        uptake_factor = (self.identity["apap_uptake_rate_s_inv"]
                         * self.identity["reference_hepatocyte_volume_m3"]
                         * self.identity["interval_s"])
        for first in range(0, self.n, 256):
            last = min(first + 256, self.n)
            block = states[first:last]
            w = recorder.weights[first:last]
            recorder.end_mean += np.einsum("k,ktv->tv", w, block[:, 1:])
            recorder.post_mean += np.einsum("k,ktv->tv", w, block[:, :-1])
            recorder.added_mean += w @ added[first:last]
            # The current run requires positive uptake, so input is reconstructible.
            recorder.external_mean += recorder.volumes[first:last] @ added[first:last] / uptake_factor
        recorder.post_mean[:, 0] += recorder.added_mean
        recorder.max_hzl_drift = float(self.arrays["diagnostics"][:, 3].max())
        for k in recorder.probes:
            for j in range(self.t):
                n = recorder.first + j
                post = states[k, j].copy()
                post[0] += added[k, j]
                recorder._probe(k, n, n*recorder.dt, "post_addition", post)
                recorder._probe(k, n, (n + 1)*recorder.dt, "interval_end", states[k, j + 1])

    def close(self):
        # Release mappings explicitly (important before Windows directory moves).
        for a in self.arrays.values():
            a._mmap.close()

