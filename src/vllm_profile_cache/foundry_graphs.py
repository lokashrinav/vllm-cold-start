"""Foundry integration for vLLM CUDA graph persistence.

Eliminates the CUDA graph capture phase (~3-60s depending on model size)
on subsequent cold starts by serializing captured graphs to disk using
Foundry's deterministic VMM-based CUDA graph serialization.

Save (first cold start):
  1. LD_PRELOAD=foundry hook intercepts all CUDA driver calls
  2. set_allocation_region() forces deterministic GPU memory addresses
  3. vLLM loads model, allocates buffers (all at fixed addresses)
  4. Graph capture uses Foundry CUDAGraph (drop-in for torch.cuda.CUDAGraph)
  5. After capture, graphs saved to disk with kernel binaries

Load (subsequent cold starts):
  1. Same LD_PRELOAD and allocation region setup
  2. vLLM loads model, allocates buffers (same addresses due to VMM)
  3. Graphs loaded from disk instead of captured
  4. Graph replay works because all memory addresses match

Requirements:
  - Foundry installed (pip install -e foundry/)
  - LD_PRELOAD=libcuda_hook.so set before Python starts
  - Linux + NVIDIA GPU (CUDA 12+)
  - tp_size=1 (single GPU — in-process engine core)

Usage:
    from vllm_profile_cache.foundry_graphs import cached_vllm_init_with_foundry

    llm = cached_vllm_init_with_foundry(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        graph_cache_dir="/root/.cache/foundry-graphs",
    )
"""

from __future__ import annotations

import atexit
import json
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

MODEL_REGION_BASE = 0x10000000000  # 64GB — safe for A100 GPU VA space
DEFAULT_REGION_SIZE = "36GB"
GRAPH_REGION_BASE = 0x30000000000  # 192GB — separate region for graph-private memory
GRAPH_REGION_SIZE = "4GB"
# Candidate base addresses for the Foundry allocation region.
# cuMemAddressReserve can fail if the requested VA range is already
# occupied (CUDA runtime, PyTorch caching allocator, etc). The code
# iterates until a test allocation lands in the region.
# The default Foundry region at 0x500000000000 is reserved by the
# cuCtxCreate hook; avoid overlap with that 2GB range.
CANDIDATE_REGION_BASES = [
    MODEL_REGION_BASE,         # 0x10000000000 (64GB)
    0x8000000000,              # 32GB
    0x18000000000,             # 96GB
    0x30000000000,             # 192GB
    0x4000000000,              # 16GB
    0x20000000000,             # 128GB
]
GRAPH_CACHE_KEY_KWARGS = (
    "dtype",
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "enforce_eager",
    "kv_cache_dtype",
    "quantization",
)


def get_foundry_hook_path() -> Optional[str]:
    try:
        import importlib.util
        spec = importlib.util.find_spec("foundry.ops")
        if spec and spec.origin:
            hook = Path(spec.origin).parent / "libcuda_hook.so"
            if hook.exists():
                return str(hook)
    except ImportError:
        pass
    return None


def is_foundry_available() -> bool:
    try:
        import foundry  # noqa: F401
        preload = os.environ.get("LD_PRELOAD", "")
        return "libcuda_hook.so" in preload
    except ImportError:
        return False


class FoundryGraphCache:
    """Manages serialized CUDA graphs on disk."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.graphs_dir = self.cache_dir / "graphs"
        self.wrapper_graphs_dir = self.cache_dir / "wrapper_graphs"
        self.hook_archive = self.cache_dir / "hook_archive"
        self.metadata_path = self.cache_dir / "metadata.json"

    def has_cached_graphs(self) -> bool:
        if not self.graphs_dir.exists():
            return False
        return len(list(self.graphs_dir.glob("graph_*.json"))) > 0

    def save_metadata(self, descs: list, model_id: str, **extra):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_id": model_id,
            "num_graphs": len(descs),
            "descs": [_describe_graph_key(desc) for desc in descs],
            "created_at": time.time(),
            **extra,
        }
        self.metadata_path.write_text(json.dumps(meta, indent=2))

    def load_metadata(self) -> Optional[dict]:
        if not self.metadata_path.exists():
            return None
        try:
            return json.loads(self.metadata_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def write_save_complete_marker(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / ".save_complete").write_text("")

    def is_save_complete(self) -> bool:
        return (self.cache_dir / ".save_complete").exists()

    def has_cached_wrapper_graphs(self) -> bool:
        return self.wrapper_graphs_dir.exists() and any(
            self.wrapper_graphs_dir.glob("graph_*.json")
        )

    def clear(self):
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)


def build_graph_cache_key(
    model: str,
    region_size: str = DEFAULT_REGION_SIZE,
    **vllm_kwargs,
) -> str:
    """Build a stable key for graph files that must match vLLM graph shapes."""
    try:
        import vllm
        vllm_version = vllm.__version__
    except Exception:
        vllm_version = "unknown"

    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda or "unknown"
    except Exception:
        torch_version = "unknown"
        cuda_version = "unknown"

    key_kwargs = {
        key: _json_safe(vllm_kwargs.get(key))
        for key in GRAPH_CACHE_KEY_KWARGS
        if key in vllm_kwargs
    }
    payload = {
        "model": model,
        "region_size": region_size,
        "vllm_version": vllm_version,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "kwargs": key_kwargs,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def resolve_graph_cache_dir(
    graph_cache_dir: str,
    model: str,
    region_size: str = DEFAULT_REGION_SIZE,
    **vllm_kwargs,
) -> tuple[str, str]:
    cache_key = build_graph_cache_key(model, region_size, **vllm_kwargs)
    return str(Path(graph_cache_dir) / cache_key), cache_key


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in sorted(value.items())}
    return str(value)


def _describe_graph_key(desc: Any) -> dict[str, Any]:
    return {
        "repr": repr(desc),
        "cg_mode": getattr(getattr(desc, "cg_mode", None), "name", None),
        "num_tokens": getattr(desc, "num_tokens", None),
        "num_reqs": getattr(desc, "num_reqs", None),
        "uniform_token_count": getattr(desc, "uniform_token_count", None),
        "uniform": getattr(desc, "uniform", None),
        "num_active_loras": getattr(desc, "num_active_loras", None),
    }


@dataclass
class _FoundryPatchState:
    cache: FoundryGraphCache
    model_id: str
    cache_key: str
    region_size: str
    region_base: Optional[int] = None
    manager_handled: bool = False
    saved_descs: list[Any] = field(default_factory=list)
    loaded_graphs: list[tuple[Any, Any]] = field(default_factory=list)
    _graph_files: Optional[list] = field(default=None, repr=False)
    load_index: int = 0
    save_seconds: float = 0.0
    load_seconds: float = 0.0
    finalized: bool = False
    pre_capture_offset: Optional[int] = None
    saved_offset_for_load: Optional[int] = None
    cursor_after_graph0: Optional[int] = None
    cursor_positions: list[int] = field(default_factory=list)
    _dummy_blocks: list = field(default_factory=list)
    profile_cudagraph_estimate: Optional[float] = None
    profile_cudagraph_cursor_delta: Optional[int] = None
    available_kv_cache_memory: Optional[int] = None
    determine_cursor_delta: Optional[int] = None
    _preallocated_early: bool = False
    _preparse_pending: Optional[Any] = field(default=None, repr=False)
    _preparse_iter: Optional[Any] = field(default=None, repr=False)
    _preloaded_graphs: Optional[list] = field(default=None, repr=False)
    passthrough_events: Optional[list] = field(default=None, repr=False)
    _extended_recording: bool = False

    @property
    def wrapper_graphs_dir(self) -> Path:
        return self.cache.wrapper_graphs_dir

    @property
    def has_cached_wrapper_graphs(self) -> bool:
        return self.cache.has_cached_wrapper_graphs()

    def load_all(self) -> None:
        """Prepare sorted graph file list for sequential loading."""
        if self._graph_files is not None:
            return
        if not self.has_cached_wrapper_graphs:
            self._graph_files = []
            return
        self._graph_files = sorted(
            self.wrapper_graphs_dir.glob("graph_*.json"),
            key=lambda path: int(path.stem.split("_")[1]),
        )
        if not self._graph_files:
            return
        logger.info("Found %d graph files for loading", len(self._graph_files))

    def next_loaded_graph(self) -> Optional[tuple[Any, Any]]:
        """Return next pre-loaded graph from cache."""
        if self._preloaded_graphs and self.load_index < len(self._preloaded_graphs):
            graph, output = self._preloaded_graphs[self.load_index]
            self.loaded_graphs.append((graph, output))
            self.load_index += 1
            return (graph, output)
        return None

    def save_graph(self, desc: Any, graph: Any, output: Any = None) -> None:
        self.wrapper_graphs_dir.mkdir(parents=True, exist_ok=True)
        index = len(self.saved_descs)
        path = self.wrapper_graphs_dir / f"graph_{index}.json"
        t0 = time.perf_counter()
        try:
            graph.save(str(path), output_tensors=output)
        except (TypeError, Exception):
            graph.save(str(path))
        self.save_seconds += time.perf_counter() - t0
        self.saved_descs.append(desc)

        # Pack fatbins and write metadata eagerly after each save.
        # The [CGE BUILD] background thread can SIGABRT at any moment;
        # even with pthread_exit the process may terminate via _exit(0).
        try:
            import foundry as fdry
            self.cache.hook_archive.mkdir(parents=True, exist_ok=True)
            fdry.pack_fatbins_to_folder(str(self.cache.hook_archive))
            n_files = len(list(self.cache.hook_archive.iterdir()))
            logger.info("Packed fatbins to %s (%d files)", self.cache.hook_archive, n_files)
        except Exception as e:
            logger.warning("Failed to pack fatbins: %s", e)
        self.cache.save_metadata(
            self.saved_descs,
            model_id=self.model_id,
            cache_key=self.cache_key,
            region_size=self.region_size,
            region_base=self.region_base,
            backend="vllm.compilation.cuda_graph.CUDAGraphWrapper",
            available_kv_cache_memory=self.available_kv_cache_memory,
            determine_cursor_delta=self.determine_cursor_delta,
            passthrough_events=self.passthrough_events,
        )
        self.cache.write_save_complete_marker()

    def finalize(self) -> None:
        if self.finalized:
            return
        self.finalized = True

        if not self.saved_descs:
            return

        self.cache.save_metadata(
            self.saved_descs,
            model_id=self.model_id,
            cache_key=self.cache_key,
            region_size=self.region_size,
            region_base=self.region_base,
            backend="vllm.compilation.cuda_graph.CUDAGraphWrapper",
            available_kv_cache_memory=self.available_kv_cache_memory,
            determine_cursor_delta=self.determine_cursor_delta,
            passthrough_events=self.passthrough_events,
        )
        logger.info(
            "Saved %d wrapper CUDA graphs in %.2fs",
            len(self.saved_descs),
            self.save_seconds,
        )


def _install_sigabrt_handler():
    """Install a C-level SIGABRT handler that calls _exit(0).

    The Foundry hook's [CGE BUILD] template builder can abort() when it
    encounters a JIT-compiled kernel binary hash it doesn't recognize.
    This happens in a background thread or C atexit handler shortly after
    graph capture. A C-level handler intercepts SIGABRT before abort()
    can re-raise it, allowing graph files already written to disk to
    survive.
    """
    try:
        import ctypes
        import ctypes.util

        libc_name = ctypes.util.find_library("c")
        if not libc_name:
            return
        libc = ctypes.CDLL(libc_name)

        SIGABRT = 6
        _HANDLER = ctypes.CFUNCTYPE(None, ctypes.c_int)

        @_HANDLER
        def _on_sigabrt(_signum):
            libc.fflush(None)
            libc._exit(0)

        libc.signal(SIGABRT, _on_sigabrt)
        _install_sigabrt_handler._prevent_gc = _on_sigabrt
    except Exception:
        pass


def setup_foundry_regions(
    region_size: str = DEFAULT_REGION_SIZE,
    force_base: Optional[int] = None,
) -> Optional[int]:
    """Initialize Foundry allocation region. Must be called BEFORE any GPU allocation.

    Returns the base address that was successfully reserved, or None if all
    candidates failed.  When *force_base* is given (load path), only that
    address is tried.  Otherwise every address in CANDIDATE_REGION_BASES is
    attempted in order.

    NOTE: set_allocation_region() can fail silently (cuMemAddressReserve
    returns a different address than requested). The C++ code disables the
    region but does NOT throw a Python exception. We verify with a test
    allocation after each attempt.
    """
    import torch
    import foundry as fdry

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")

    if not torch.cuda.is_initialized():
        torch.cuda.init()

    size_bytes = fdry.parse_size(region_size)

    bases_to_try = [force_base] if force_base is not None else CANDIDATE_REGION_BASES
    used_base = None
    for base in bases_to_try:
        try:
            fdry.set_allocation_region(base, size_bytes)
        except Exception as e:
            print(f"[FOUNDRY] set_allocation_region(0x{base:x}) exception: {e}", flush=True)
            continue

        # Flush cached blocks from the default region (0x500000000000)
        # so the test allocation goes through cuMemAlloc_v2.
        torch.cuda.empty_cache()

        # Verify: allocate and check address. Keep tensor alive —
        # cuMemFree_v2 calls cuMemAddressFree on sub-ranges of the
        # reservation which is undefined behavior per CUDA docs.
        test = torch.empty(1024, device="cuda")
        ptr = test.data_ptr()
        cursor = fdry.get_current_alloc_offset()
        if base <= ptr < (base + size_bytes) and cursor > 0:
            used_base = base
            print(
                f"[FOUNDRY] Region OK at 0x{base:x}: "
                f"test=0x{ptr:x}, cursor={cursor}",
                flush=True,
            )
            break

        # Region failed silently — try next candidate
        print(
            f"[FOUNDRY] Region 0x{base:x} failed: "
            f"test=0x{ptr:x}, cursor={cursor}. Trying next.",
            flush=True,
        )
        del test

    if used_base is None:
        logger.error(
            "All allocation region candidates failed. "
            "GPU addresses will be non-deterministic."
        )

    fdry.set_pack_fatbins_on_exit(False)
    _install_sigabrt_handler()
    return used_base


def _premap_non_foundry_addresses(
    graph_paths: list,
    device_index: int,
    region_base: Optional[int] = None,
    region_size: str = DEFAULT_REGION_SIZE,
):
    """Pre-map physical memory at non-Foundry addresses referenced by saved graphs.

    NCCL workspace and other non-Foundry allocations get baked into CUDA graph
    kernel arguments during capture. On reload, those addresses must have
    physical backing or the GPU faults (XID 31). This function scans all
    kernel parameter hex values for pointer-sized integers outside the Foundry
    region, then reserves VA + physical memory at those locations.
    """
    import json as _json
    import struct
    import ctypes

    base = region_base if region_base is not None else MODEL_REGION_BASE
    region_lo = base
    region_hi = base + _parse_size_bytes(region_size)

    GRANULARITY = 2 * 1024 * 1024  # 2MB
    MIN_GPU_ADDR = 0x1_0000_0000       # 4GB — below this is unlikely GPU memory
    MAX_GPU_ADDR = 0x800_0000_0000_0000  # way above any real GPU allocation

    non_foundry_addrs: set[int] = set()

    def _check_addr(addr: int):
        if addr == 0:
            return
        if addr < MIN_GPU_ADDR or addr >= MAX_GPU_ADDR:
            return
        if region_lo <= addr < region_hi:
            return
        if addr & 0xFFF:  # not page-aligned → probably not a pointer
            return
        non_foundry_addrs.add(addr)

    for path in graph_paths:
        try:
            with open(path, "r") as f:
                g = _json.load(f)
        except Exception:
            continue
        for node in g.get("nodes", []):
            ntype = node.get("type", "")
            p = node.get("params", {})

            if ntype == "MemsetNode":
                _check_addr(p.get("dst", 0))
            elif ntype == "MemcpyNode":
                _check_addr(p.get("srcDevice", 0))
                _check_addr(p.get("dstDevice", 0))

            if ntype == "KernelNode":
                for kp in p.get("kernelParams", []):
                    hex_val = kp.get("value_hex", "")
                    param_size = kp.get("size", 0)
                    if not hex_val or param_size < 8:
                        continue
                    try:
                        raw = bytes.fromhex(hex_val)
                    except ValueError:
                        continue
                    for off in range(0, len(raw) - 7, 8):
                        val = struct.unpack_from("<Q", raw, off)[0]
                        _check_addr(val)

                # Also check the arg_buffer (extra params)
                hex_val = p.get("arg_buffer_hex", "")
                if hex_val:
                    try:
                        raw = bytes.fromhex(hex_val)
                    except ValueError:
                        raw = b""
                    for off in range(0, len(raw) - 7, 8):
                        val = struct.unpack_from("<Q", raw, off)[0]
                        _check_addr(val)

    if not non_foundry_addrs:
        print(f"[FOUNDRY] No non-Foundry addresses in graphs for device {device_index}", flush=True)
        return

    sorted_addrs = sorted(non_foundry_addrs)
    sample = [f"0x{a:x}" for a in sorted_addrs[:5]]
    print(
        f"[FOUNDRY] Found {len(non_foundry_addrs)} unique non-Foundry pointer(s) "
        f"in graphs for device {device_index}, sample: {sample}",
        flush=True,
    )

    # Build allocation ranges: round each address down to 2MB, extend by 2MB
    range_set: dict[int, int] = {}
    for addr in non_foundry_addrs:
        a_lo = (addr // GRANULARITY) * GRANULARITY
        a_hi = a_lo + GRANULARITY
        prev = range_set.get(a_lo, a_lo)
        range_set[a_lo] = max(prev, a_hi)

    # Merge overlapping / adjacent ranges
    sorted_ranges = sorted(range_set.items())
    merged: list[tuple[int, int]] = []
    cur_lo, cur_hi = sorted_ranges[0]
    for lo, hi in sorted_ranges[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            merged.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi
    merged.append((cur_lo, cur_hi))

    # Pre-map each range using CUDA VMM APIs via ctypes.
    # Stop the Foundry region so cuMemAddressReserve passes through to
    # the real driver (hook checks tls_storage.enabled).
    import foundry as fdry
    fdry.stop_allocation_region()

    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        print("[FOUNDRY] WARNING: cannot load libcuda.so.1 for pre-mapping", flush=True)
        fdry.resume_allocation_region()
        return

    mapped_count = 0
    mapped_bytes = 0
    for lo, hi in merged:
        size = hi - lo
        reserved_ptr = ctypes.c_uint64(0)
        ret = libcuda.cuMemAddressReserve(
            ctypes.byref(reserved_ptr), ctypes.c_size_t(size),
            ctypes.c_size_t(GRANULARITY), ctypes.c_uint64(lo), ctypes.c_ulonglong(0),
        )
        if ret != 0 or reserved_ptr.value != lo:
            print(
                f"[FOUNDRY] cuMemAddressReserve failed: lo=0x{lo:x} size={size} "
                f"ret={ret} got=0x{reserved_ptr.value:x}",
                flush=True,
            )
            if ret == 0:
                libcuda.cuMemAddressFree(reserved_ptr, ctypes.c_size_t(size))
            continue

        # Create physical backing
        alloc_handle = ctypes.c_uint64(0)

        class CUmemAllocationProp(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_int),           # CU_MEM_ALLOCATION_TYPE_PINNED = 1
                ("requestedHandleTypes", ctypes.c_int),
                ("location_type", ctypes.c_int),   # CU_MEM_LOCATION_TYPE_DEVICE = 1
                ("location_id", ctypes.c_int),
                ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags_compressionType", ctypes.c_uint8),
                ("allocFlags_gpuDirectRDMACapable", ctypes.c_uint8),
                ("allocFlags_usage", ctypes.c_uint16),
                ("allocFlags_reserved", ctypes.c_uint8 * 4),
            ]

        prop = CUmemAllocationProp()
        prop.type = 1  # CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location_type = 1  # CU_MEM_LOCATION_TYPE_DEVICE
        prop.location_id = device_index

        ret = libcuda.cuMemCreate(
            ctypes.byref(alloc_handle), ctypes.c_size_t(size),
            ctypes.byref(prop), ctypes.c_ulonglong(0),
        )
        if ret != 0:
            libcuda.cuMemAddressFree(ctypes.c_uint64(lo), ctypes.c_size_t(size))
            continue

        ret = libcuda.cuMemMap(
            ctypes.c_uint64(lo), ctypes.c_size_t(size),
            ctypes.c_size_t(0), alloc_handle, ctypes.c_ulonglong(0),
        )
        if ret != 0:
            libcuda.cuMemRelease(alloc_handle)
            libcuda.cuMemAddressFree(ctypes.c_uint64(lo), ctypes.c_size_t(size))
            continue

        class CUmemAccessDesc(ctypes.Structure):
            _fields_ = [
                ("location_type", ctypes.c_int),
                ("location_id", ctypes.c_int),
                ("flags", ctypes.c_int),
            ]

        desc = CUmemAccessDesc()
        desc.location_type = 1  # CU_MEM_LOCATION_TYPE_DEVICE
        desc.location_id = device_index
        desc.flags = 1  # CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        libcuda.cuMemSetAccess(
            ctypes.c_uint64(lo), ctypes.c_size_t(size),
            ctypes.byref(desc), ctypes.c_size_t(1),
        )

        mapped_count += 1
        mapped_bytes += size

    fdry.resume_allocation_region()

    print(
        f"[FOUNDRY] Pre-mapped {mapped_count}/{len(merged)} non-Foundry ranges "
        f"({mapped_bytes / (1024**2):.1f} MB) for device {device_index}",
        flush=True,
    )


def _build_passthrough_addr_map(
    save_events: list[dict],
    load_events: list[dict],
) -> list[tuple[int, int, int, int]]:
    """Build old->new address range mapping from passthrough events.

    Uses size-based greedy matching: for each save event, find the first
    unmatched load event with the same allocation size.  This handles
    event count and ordering differences between save/load runs (e.g.
    different code paths producing different event sequences).

    Returns list of (old_base, old_end, new_base, delta) tuples sorted
    by old_base for binary-search patching.
    """
    from collections import defaultdict

    ranges: list[tuple[int, int, int, int]] = []

    if not save_events or not load_events:
        return ranges

    load_by_size: dict[int, list[dict]] = defaultdict(list)
    for ev in load_events:
        load_by_size[ev["size"]].append(ev)

    n_remapped = 0
    n_matched = 0
    n_unmatched = 0
    for s_ev in save_events:
        s_ptr, s_size = s_ev["ptr"], s_ev["size"]
        candidates = load_by_size.get(s_size)
        if not candidates:
            n_unmatched += 1
            continue
        l_ev = candidates.pop(0)
        n_matched += 1
        l_ptr = l_ev["ptr"]
        if s_ptr != l_ptr:
            delta = l_ptr - s_ptr
            ranges.append((s_ptr, s_ptr + s_size, l_ptr, delta))
            n_remapped += 1
            if n_remapped <= 10:
                print(
                    f"[FOUNDRY] addr remap: 0x{s_ptr:x} -> 0x{l_ptr:x} "
                    f"(size={s_size}, source={s_ev.get('source', '?')})",
                    flush=True,
                )

    if n_remapped > 10:
        print(f"[FOUNDRY] ... and {n_remapped - 10} more remappings", flush=True)

    print(
        f"[FOUNDRY] Passthrough matching: {n_matched} matched, "
        f"{n_remapped} remapped, {n_unmatched} save-only (no load match) "
        f"(save={len(save_events)} load={len(load_events)})",
        flush=True,
    )

    ranges.sort(key=lambda r: r[0])
    return ranges


def _remap_addr(addr: int, ranges: list[tuple[int, int, int, int]]) -> int:
    """Binary search for addr in sorted ranges, apply delta if found."""
    lo, hi = 0, len(ranges) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        r_lo, r_hi, _, delta = ranges[mid]
        if addr < r_lo:
            hi = mid - 1
        elif addr >= r_hi:
            lo = mid + 1
        else:
            return addr + delta
    return addr


def _patch_graph_json_addresses(
    graph_path: str,
    ranges: list[tuple[int, int, int, int]],
) -> str:
    """Patch kernel parameter addresses in a graph JSON file.

    For each 8-byte little-endian value in kernel params that falls inside
    a remapped range, applies the delta to produce the correct load-time
    address.  Handles interior pointers (base + offset within a buffer).

    Returns (patched_path, patch_count).
    """
    import struct

    with open(graph_path, "r") as f:
        graph = json.load(f)

    patch_count = 0

    def _patch_hex_field(hex_str: str) -> str:
        nonlocal patch_count
        if not hex_str or len(hex_str) < 16:
            return hex_str
        raw = bytes.fromhex(hex_str)
        result = bytearray(raw)
        changed = False
        for off in range(0, len(raw) - 7, 8):
            val = struct.unpack_from("<Q", raw, off)[0]
            new_val = _remap_addr(val, ranges)
            if new_val != val:
                struct.pack_into("<Q", result, off, new_val)
                changed = True
                patch_count += 1
        return result.hex() if changed else hex_str

    def _remap_int_field(val: int) -> int:
        nonlocal patch_count
        new_val = _remap_addr(val, ranges)
        if new_val != val:
            patch_count += 1
        return new_val

    for node in graph.get("nodes", []):
        params = node.get("params", {})
        if node.get("type") == "KernelNode":
            for kp in params.get("kernelParams", []):
                if "value_hex" in kp and kp.get("size", 0) >= 8:
                    kp["value_hex"] = _patch_hex_field(kp["value_hex"])
            if "extra_argBuffer_hex" in params:
                params["extra_argBuffer_hex"] = _patch_hex_field(
                    params["extra_argBuffer_hex"]
                )
        elif node.get("type") == "MemsetNode":
            if "dst" in params:
                params["dst"] = _remap_int_field(params["dst"])
        elif node.get("type") == "MemcpyNode":
            for key in ("srcDevice", "dstDevice"):
                if key in params:
                    params[key] = _remap_int_field(params[key])

    graph_dir = os.path.dirname(graph_path)
    base_name = os.path.basename(graph_path)
    patched_path = os.path.join(graph_dir, f".patched_{base_name}")
    with open(patched_path, "w") as f:
        json.dump(graph, f)

    return patched_path, patch_count


def _diagnose_unpatched_addresses(
    graph_paths: list[str],
    addr_ranges: list[tuple[int, int, int, int]],
    region_base: int,
    region_size_bytes: int,
) -> None:
    """Log non-Foundry addresses that are NOT covered by any passthrough range."""
    import struct as _struct

    MIN_GPU_ADDR = 0x1_0000_0000
    MAX_GPU_ADDR = 0x800_0000_0000_0000
    region_lo = region_base
    region_hi = region_base + region_size_bytes

    unpatched: dict[int, int] = {}

    for path in graph_paths:
        try:
            with open(path, "r") as f:
                graph = json.load(f)
        except Exception:
            continue
        for node in graph.get("nodes", []):
            params = node.get("params", {})
            ntype = node.get("type", "")

            def _check(val: int):
                if val == 0 or val < MIN_GPU_ADDR or val >= MAX_GPU_ADDR:
                    return
                if region_lo <= val < region_hi:
                    return
                if _remap_addr(val, addr_ranges) != val:
                    return
                unpatched[val] = unpatched.get(val, 0) + 1

            if ntype == "MemsetNode":
                _check(params.get("dst", 0))
            elif ntype == "MemcpyNode":
                _check(params.get("srcDevice", 0))
                _check(params.get("dstDevice", 0))

            if ntype == "KernelNode":
                for kp in params.get("kernelParams", []):
                    hex_val = kp.get("value_hex", "")
                    if not hex_val or kp.get("size", 0) < 8:
                        continue
                    try:
                        raw = bytes.fromhex(hex_val)
                    except ValueError:
                        continue
                    for off in range(0, len(raw) - 7, 8):
                        val = _struct.unpack_from("<Q", raw, off)[0]
                        _check(val)
                extra = params.get("extra_argBuffer_hex", "")
                if extra:
                    try:
                        raw = bytes.fromhex(extra)
                    except ValueError:
                        raw = b""
                    for off in range(0, len(raw) - 7, 8):
                        val = _struct.unpack_from("<Q", raw, off)[0]
                        _check(val)

    if not unpatched:
        print("[FOUNDRY] Diagnostic: all non-Foundry addresses are covered by passthrough ranges", flush=True)
        return

    sorted_addrs = sorted(unpatched.items(), key=lambda x: -x[1])
    print(
        f"[FOUNDRY] WARNING: {len(unpatched)} unique non-Foundry addresses "
        f"NOT covered by any passthrough range:",
        flush=True,
    )
    for addr, count in sorted_addrs[:20]:
        print(f"  0x{addr:x} (referenced {count} times)", flush=True)
    if len(sorted_addrs) > 20:
        print(f"  ... and {len(sorted_addrs) - 20} more", flush=True)


def _parse_size_bytes(s: str) -> int:
    """Parse a size string like '36GB' to bytes."""
    s = s.strip().upper()
    if s.endswith("GB"):
        return int(s[:-2]) * (1024 ** 3)
    if s.endswith("MB"):
        return int(s[:-2]) * (1024 ** 2)
    if s.endswith("KB"):
        return int(s[:-2]) * 1024
    return int(s)


def patch_vllm_for_foundry(
    graph_cache_dir: str,
    region_size: str = DEFAULT_REGION_SIZE,
    model_id: str = "unknown",
    cache_key: str = "",
    region_base: Optional[int] = None,
    load_mode: bool = False,
    profile_cudagraph_estimate: Optional[float] = None,
    profile_cudagraph_cursor_delta: Optional[int] = None,
    available_kv_cache_memory: Optional[int] = None,
    determine_cursor_delta: Optional[int] = None,
) -> _FoundryPatchState:
    """Monkey-patch vLLM's CUDA graph capture to use Foundry.

    Must be called BEFORE creating any vLLM LLM/Engine instances.
    Only works with tp_size=1 (in-process engine core, no subprocess).

    In vLLM V1 (0.20+), main model CUDA graphs are captured via
    CUDAGraphWrapper.__call__() triggered from capture_model(), NOT via
    CudaGraphManager.capture(). We patch CUDAGraphWrapper.__call__() to
    intercept FULL mode captures and use Foundry's CUDAGraph instead.

    A phase flag (set via patching GPUModelRunner.capture_model) ensures
    we only intercept during the real capture phase, not during the earlier
    profile_cudagraph_memory() phase.
    """
    import foundry as fdry
    import torch

    from vllm.compilation.counter import compilation_counter
    import vllm.compilation.cuda_graph as cuda_graph_mod
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    cache = FoundryGraphCache(graph_cache_dir)
    state = _FoundryPatchState(
        cache=cache,
        model_id=model_id,
        cache_key=cache_key,
        region_size=region_size,
        region_base=region_base,
    )
    state.profile_cudagraph_estimate = profile_cudagraph_estimate
    state.profile_cudagraph_cursor_delta = profile_cudagraph_cursor_delta

    # Phase flag: only True during capture_model(), not during profiling
    _capture_phase = [False]
    _pending_graphs = []  # (desc, graph, output) tuples deferred until finalize
    _is_load_mode = load_mode

    # Pre-copy graph files to local tmpfs in background so file I/O
    # overlaps with model loading instead of blocking capture_model.
    if _is_load_mode and cache.has_cached_wrapper_graphs():
        state.load_all()
        if state._graph_files:
            import shutil
            import threading
            _shm_dir = Path("/dev/shm/foundry_graphs")

            def _precopy():
                try:
                    _shm_dir.mkdir(parents=True, exist_ok=True)
                    src_dir = state._graph_files[0].parent
                    manifest = src_dir / "graph_manifest.json"
                    if manifest.exists():
                        shutil.copy2(str(manifest), str(_shm_dir / manifest.name))
                    for src in state._graph_files:
                        cg = src.with_suffix(".cugraph")
                        if cg.exists():
                            shutil.copy2(str(cg), str(_shm_dir / cg.name))
                        shutil.copy2(str(src), str(_shm_dir / src.name))
                    state._shm_graph_dir = _shm_dir
                    logger.info("Pre-copied %d graph files to /dev/shm", len(state._graph_files))
                except Exception as e:
                    logger.warning("Failed to pre-copy graph files: %s", e)
                    state._shm_graph_dir = None

            state._precopy_thread = threading.Thread(target=_precopy, daemon=True)
            state._precopy_thread.start()

    # --- Patch 0: GPUModelRunner.profile_cudagraph_memory ---
    _original_profile_cudagraph = GPUModelRunner.profile_cudagraph_memory

    if _is_load_mode and profile_cudagraph_estimate is not None:
        def _skip_profile_cudagraph(self):
            saved = state.profile_cudagraph_estimate
            delta = state.profile_cudagraph_cursor_delta
            pre = fdry.get_current_alloc_offset()
            if delta is not None and delta > 0:
                # Directly advance the bump allocator cursor. torch.empty
                # goes through PyTorch's caching allocator which may satisfy
                # the request from cached blocks without calling cuMemAlloc,
                # leaving the bump cursor unchanged.
                fdry.set_current_alloc_offset(pre + delta)
                post = fdry.get_current_alloc_offset()
                print(f"[FOUNDRY] Skipped profile_cudagraph_memory: "
                      f"cursor {pre} -> {post} (target_delta={delta})",
                      flush=True)
            print(f"[FOUNDRY] Returning saved cudagraph estimate: {saved}",
                  flush=True)
            return saved
        GPUModelRunner.profile_cudagraph_memory = _skip_profile_cudagraph
    else:
        def _instrumented_profile_cudagraph(self):
            if state._extended_recording:
                fdry.pause_passthrough_record()
            pre_cursor = fdry.get_current_alloc_offset()
            try:
                result = _original_profile_cudagraph(self)
            finally:
                if state._extended_recording:
                    fdry.resume_passthrough_record()
            post_cursor = fdry.get_current_alloc_offset()
            state.profile_cudagraph_estimate = result
            state.profile_cudagraph_cursor_delta = post_cursor - pre_cursor
            print(f"[FOUNDRY_DEBUG] profile_cudagraph_memory: result={result}, "
                  f"cursor_delta={post_cursor - pre_cursor}", flush=True)
            return result
        GPUModelRunner.profile_cudagraph_memory = _instrumented_profile_cudagraph

    # --- Patch 0b: GPUWorker.determine_available_memory ---
    # Skip the entire profiling phase (profile_run + profile_cudagraph_memory)
    # in load mode. profile_run runs a full forward pass (~16s) just to measure
    # peak memory. We saved the result and cursor delta from the save run.
    from vllm.v1.worker.gpu_worker import Worker as GPUWorker
    _original_determine = GPUWorker.determine_available_memory

    if _is_load_mode and available_kv_cache_memory is not None and determine_cursor_delta is not None:
        _saved_kv_mem = available_kv_cache_memory
        _saved_det_delta = determine_cursor_delta

        def _skip_determine(self):
            print("[FOUNDRY] Running original determine_available_memory "
                  "(for initialization side-effects only)",
                  flush=True)
            fdry.stop_allocation_region()
            try:
                result = _original_determine(self)
            finally:
                fdry.resume_allocation_region()
            print(f"[FOUNDRY] determine_available_memory returned {result} "
                  f"(using saved value {_saved_kv_mem} for cursor alignment)",
                  flush=True)
            return _saved_kv_mem
        GPUWorker.determine_available_memory = _skip_determine
    else:
        def _instrumented_determine(self):
            fdry.stop_allocation_region()
            pre_cursor = fdry.get_current_alloc_offset()
            try:
                result = _original_determine(self)
            finally:
                fdry.resume_allocation_region()
            post_cursor = fdry.get_current_alloc_offset()
            state.available_kv_cache_memory = result
            state.determine_cursor_delta = post_cursor - pre_cursor
            print(
                f"[FOUNDRY_DEBUG] determine_available_memory: "
                f"result={result}, cursor_delta={post_cursor - pre_cursor}",
                flush=True,
            )
            return result
        GPUWorker.determine_available_memory = _instrumented_determine

    # --- Debug: log create_kv_caches ---
    if hasattr(GPUWorker, 'create_kv_caches'):
        _original_create_kv = GPUWorker.create_kv_caches
        def _debug_create_kv(self, *args, **kwargs):
            print(f"[FOUNDRY] create_kv_caches ENTERED (args={args}, kwargs keys={list(kwargs.keys())})", flush=True)
            result = _original_create_kv(self, *args, **kwargs)
            print("[FOUNDRY] create_kv_caches DONE", flush=True)
            return result
        GPUWorker.create_kv_caches = _debug_create_kv

    # --- Patch 1: GPUModelRunner.capture_model to set phase flag ---
    _original_capture_model = GPUModelRunner.capture_model

    def _flagged_capture_model(self):
        _capture_phase[0] = True
        _pending_graphs.clear()

        # On LOAD: do NCCL warmup BEFORE ending passthrough recording
        # so NCCL's cudaMalloc allocations are captured as passthrough events.
        # On SAVE: recording continues through capture_model where NCCL ops
        # naturally trigger the same allocations during graph capture.
        if state._extended_recording and _is_load_mode:
            try:
                import torch.distributed as _dist
                if _dist.is_initialized():
                    import torch as _torch_warmup
                    _dev = _torch_warmup.cuda.current_device()
                    _warmup = _torch_warmup.zeros(1024, device=f"cuda:{_dev}")
                    _dist.all_reduce(_warmup)
                    _torch_warmup.cuda.synchronize()
                    del _warmup
                    print("[FOUNDRY] NCCL warmup allreduce completed (during recording)", flush=True)
            except Exception as e:
                print(f"[FOUNDRY] NCCL warmup failed: {e}", flush=True)

        # Stop extended passthrough recording and capture all events
        if state._extended_recording:
            fdry.end_passthrough_record()
            pt_events = fdry.get_passthrough_events()
            state.passthrough_events = pt_events
            state._extended_recording = False
            print(
                f"[FOUNDRY] Extended recording: captured {len(pt_events)} "
                f"total passthrough events (init + post-init)",
                flush=True,
            )
            for i, ev in enumerate(pt_events[:10]):
                print(
                    f"  [{i}] {ev['source']} ptr=0x{ev['ptr']:x} size={ev['size']}",
                    flush=True,
                )
            if len(pt_events) > 10:
                print(f"  ... and {len(pt_events) - 10} more", flush=True)

        if not _is_load_mode:
            state.pre_capture_offset = fdry.get_current_alloc_offset()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if _is_load_mode:
            print("[FOUNDRY] capture_model: calling state.load_all()", flush=True)
            state.load_all()
            print(f"[FOUNDRY] capture_model: load_all done, {len(state._graph_files)} graph files", flush=True)
            if state._graph_files:
                import torch as _torch_cap
                dev = _torch_cap.cuda.current_device()
                print(f"[FOUNDRY] capture_model: loading graphs on CUDA device={dev}", flush=True)
                _torch_cap.cuda.set_device(dev)

                all_paths = [str(p) for p in state._graph_files]

                # Build address range map from passthrough events and patch graph JSONs
                saved_meta = cache.load_metadata()
                save_pt = (saved_meta or {}).get("passthrough_events") or []
                load_pt = state.passthrough_events or []
                addr_ranges = _build_passthrough_addr_map(save_pt, load_pt)

                patched_paths = []
                if addr_ranges:
                    print(
                        f"[FOUNDRY] Patching {len(all_paths)} graph JSONs "
                        f"with {len(addr_ranges)} address range remappings",
                        flush=True,
                    )
                    total_patches = 0
                    for orig_path in all_paths:
                        patched_path, n_patches = _patch_graph_json_addresses(
                            orig_path, addr_ranges
                        )
                        patched_paths.append(patched_path)
                        total_patches += n_patches
                    print(
                        f"[FOUNDRY] Patched {total_patches} address references "
                        f"across {len(all_paths)} graph files",
                        flush=True,
                    )
                    load_paths = patched_paths
                    _diagnose_unpatched_addresses(
                        patched_paths,
                        addr_ranges,
                        region_base or MODEL_REGION_BASE,
                        _parse_size_bytes(region_size),
                    )
                else:
                    print("[FOUNDRY] No address remapping needed", flush=True)
                    load_paths = all_paths

                # Pre-map physical memory at non-Foundry addresses that
                # weren't covered by passthrough patching (e.g. cuBLAS
                # workspaces allocated during determine_available_memory).
                # These are scratch buffers — zero-init is fine.
                _premap_non_foundry_addresses(
                    load_paths,
                    dev,
                    region_base=region_base,
                    region_size=region_size,
                )

                t_load = time.perf_counter()
                all_loaded = []
                pool = _torch_cap.cuda.graph_pool_handle()
                for i, path in enumerate(load_paths):
                    try:
                        result = fdry.CUDAGraph.load(path, pool)
                        all_loaded.append(result)
                    except Exception as e:
                        print(f"[FOUNDRY] Failed to load graph {i} ({path}): {e}", flush=True)
                        break
                t_load = time.perf_counter() - t_load
                print(f"[FOUNDRY] Loaded {len(all_loaded)}/{len(all_paths)} graphs in {t_load:.3f}s", flush=True)

                # Clean up patched temp files
                for p in patched_paths:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

                state._preloaded_graphs = all_loaded
                state._preparse_pending = None
                state.load_seconds = t_load

        print(f"[FOUNDRY] capture_model started (load_mode={_is_load_mode})", flush=True)
        import sys
        sys.stdout.flush()
        sys.stderr.flush()

        if _is_load_mode and state._preloaded_graphs:
            # Direct-populate: set CUDAGraphWrapper entries without running
            # the capture loop. Each _dummy_run costs ~23ms (input prep +
            # model entry) × 35 batch sizes ≈ 0.8s — all wasted in load mode.
            _capture_phase[0] = False
            wrapper = (
                self.model
                if isinstance(self.model, CUDAGraphWrapper)
                else None
            )
            if wrapper is not None:
                graph_idx = 0
                for _, batch_descs in (
                    self.cudagraph_dispatcher.get_capture_descs()
                ):
                    for desc in batch_descs:
                        if graph_idx < len(state._preloaded_graphs):
                            graph, output = state._preloaded_graphs[graph_idx]
                            entry = cuda_graph_mod.CUDAGraphEntry(
                                batch_descriptor=desc,
                            )
                            entry.cudagraph = graph
                            entry.output = output
                            wrapper.concrete_cudagraph_entries[desc] = entry
                            state.loaded_graphs.append((graph, output))
                            graph_idx += 1
                state.load_index = graph_idx
            result = 0
        else:
            if _is_load_mode:
                n_expected = len(state._graph_files) if hasattr(state, '_graph_files') else 0
                n_loaded = len(state._preloaded_graphs) if state._preloaded_graphs else 0
                print(f"[FOUNDRY] Loaded {n_loaded}/{n_expected} graphs from cache, "
                      f"falling back to native capture for remaining", flush=True)
                if hasattr(GPUModelRunner, '_warmup_and_capture') and hasattr(state, '_orig_warmup_capture'):
                    GPUModelRunner._warmup_and_capture = state._orig_warmup_capture
                if hasattr(state, '_orig_dummy_sampler_run'):
                    GPUModelRunner._dummy_sampler_run = state._orig_dummy_sampler_run
            try:
                result = _original_capture_model(self)
            finally:
                _capture_phase[0] = False
        if not _is_load_mode:
            print(f"[FOUNDRY] capture_model finished. Pending graphs: {len(_pending_graphs)}", flush=True)
            _finalize_save()
        else:
            n_loaded = len(state.loaded_graphs)
            print(
                f"[FOUNDRY] Loaded {n_loaded} wrapper CUDA graphs "
                f"from cache in {getattr(state, 'load_seconds', 0):.2f}s",
                flush=True,
            )
            print(f"[FOUNDRY] capture_model finished (loaded graphs from cache)", flush=True)
        return result

    GPUModelRunner.capture_model = _flagged_capture_model

    # --- Patch 1b: GPUWorker.compile_or_warm_up_model ---
    if hasattr(GPUWorker, 'compile_or_warm_up_model'):
        _original_compile_warmup = GPUWorker.compile_or_warm_up_model

        if _is_load_mode:
            def _timed_compile_warmup(self):
                import time as _t
                print("[FOUNDRY] compile_or_warm_up_model ENTERED", flush=True)
                t0 = _t.perf_counter()
                result = _original_compile_warmup(self)
                elapsed = _t.perf_counter() - t0
                print(f"[FOUNDRY] compile_or_warm_up_model: {elapsed:.2f}s", flush=True)
                return result
        else:
            def _timed_compile_warmup(self):
                import time as _t
                t0 = _t.perf_counter()
                result = _original_compile_warmup(self)
                elapsed = _t.perf_counter() - t0
                print(f"[FOUNDRY] compile_or_warm_up_model: {elapsed:.2f}s", flush=True)
                return result

        GPUWorker.compile_or_warm_up_model = _timed_compile_warmup
    else:
        print("[FOUNDRY_DEBUG] GPUWorker has no compile_or_warm_up_model, skipping patch 1b", flush=True)

    # --- Patch 1b2: Skip warmup passes in capture loop (load mode) ---
    # _warmup_and_capture runs N warmup forward passes (CUDAGraphMode.NONE)
    # before each graph capture. In load mode these are wasted work (~30ms
    # each × 35 batch sizes ≈ 1s) since we return preloaded graphs.
    if _is_load_mode and hasattr(GPUModelRunner, '_warmup_and_capture'):
        _orig_warmup_capture = GPUModelRunner._warmup_and_capture
        state._orig_warmup_capture = _orig_warmup_capture

        def _skip_warmup_capture(self, desc, cudagraph_runtime_mode, **kwargs):
            self._dummy_run(
                desc.num_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                uniform_decode=desc.uniform,
                is_graph_capturing=True,
                skip_eplb=True,
                remove_lora=False,
                num_active_loras=getattr(desc, 'num_active_loras', 0),
            )

        GPUModelRunner._warmup_and_capture = _skip_warmup_capture

    # --- Patch 1b3: Skip _dummy_sampler_run in load mode ---
    # _dummy_sampler_run compiles sampling Triton kernels (top-k sort on
    # vocab_size * 256 logits). In load mode we skipped profile_run(),
    # so this triggers first-time JIT compilation (~15s). Skip it here;
    # kernels compile lazily on first real inference instead.
    if _is_load_mode:
        _original_dummy_sampler_run = GPUModelRunner._dummy_sampler_run
        state._orig_dummy_sampler_run = _original_dummy_sampler_run

        def _skip_dummy_sampler_run(self, *args, **kwargs):
            print("[FOUNDRY_DEBUG] Skipped _dummy_sampler_run (load mode)", flush=True)
            return None

        GPUModelRunner._dummy_sampler_run = _skip_dummy_sampler_run

    # --- Patch 2: CUDAGraphWrapper.__call__ to intercept FULL mode ---
    _original_wrapper_call = CUDAGraphWrapper.__call__

    def _foundry_wrapper_call(self, *args, **kwargs):
        if not _capture_phase[0]:
            return _original_wrapper_call(self, *args, **kwargs)

        # Skip NONE and PIECEWISE modes — only intercept FULL-style captures
        # (FULL, FULL_DECODE_ONLY, etc.)
        if self.runtime_mode == CUDAGraphMode.NONE:
            return _original_wrapper_call(self, *args, **kwargs)
        if hasattr(CUDAGraphMode, 'PIECEWISE') and self.runtime_mode == CUDAGraphMode.PIECEWISE:
            return _original_wrapper_call(self, *args, **kwargs)

        if not cuda_graph_mod.is_forward_context_available():
            return self.runnable(*args, **kwargs)

        forward_context = cuda_graph_mod.get_forward_context()
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
        batch_descriptor = forward_context.batch_descriptor

        if cudagraph_runtime_mode == CUDAGraphMode.NONE or cudagraph_runtime_mode != self.runtime_mode:
            return self.runnable(*args, **kwargs)

        assert batch_descriptor is not None

        if batch_descriptor not in self.concrete_cudagraph_entries:
            self.concrete_cudagraph_entries[batch_descriptor] = (
                cuda_graph_mod.CUDAGraphEntry(batch_descriptor=batch_descriptor)
            )

        entry = self.concrete_cudagraph_entries[batch_descriptor]
        if entry.cudagraph is not None:
            cuda_graph_mod.get_offloader().sync_prev_onload()
            entry.cudagraph.replay()
            return entry.output

        # --- Load path: load ALL graphs (including graph 0) from cache ---
        if _is_load_mode:
            if not getattr(state, '_load_abandoned', False):
                result = state.next_loaded_graph()
                if result is not None:
                    graph, output = result
                    entry.cudagraph = graph
                    entry.output = output
                    return entry.output
                state._load_abandoned = True
                loaded_count = len(state.loaded_graphs)
                n_total = len(state._graph_files or [])
                print(
                    f"[FOUNDRY] Loaded {loaded_count}/{n_total} graphs "
                    f"from cache, falling back to native capture for remaining",
                    flush=True,
                )
            return _original_wrapper_call(self, *args, **kwargs)

        # --- Save path: capture with Foundry ---

        _pre_cursor = fdry.get_current_alloc_offset()

        input_addresses = [
            x.data_ptr() for x in args if isinstance(x, torch.Tensor)
        ]
        entry.input_addresses = input_addresses

        cuda_graph_mod.validate_cudagraph_capturing_enabled()
        graph = fdry.CUDAGraph()

        with cuda_graph_mod.ExitStack() as stack:
            if self.cudagraph_options.gc_disable:
                stack.enter_context(cuda_graph_mod.patch("gc.collect", lambda: None))
                stack.enter_context(
                    cuda_graph_mod.patch(
                        "torch.accelerator.empty_cache", lambda: None
                    )
                )
            # Skip set_graph_pool_id — graph pool uses cudaMallocAsync
            # which bypasses Foundry's cuMemAlloc_v2 hook, producing
            # non-deterministic addresses that break graph loading.

            cuda_graph_mod.get_offloader().sync_prev_onload()
            with fdry.graph(graph):
                output = self.runnable(*args, **kwargs)
                cuda_graph_mod.get_offloader().join_after_forward()
                if self.cudagraph_options.weak_ref_output:
                    output = cuda_graph_mod.weak_ref_tensors(output)

        # capture_end() inside fdry.graph().__exit__ disables the allocation
        # region. Re-enable so allocations between captures stay deterministic.
        fdry.resume_allocation_region()

        entry.output = output
        entry.cudagraph = graph
        compilation_counter.num_cudagraph_captured += 1
        _pending_graphs.append((batch_descriptor, graph, output))
        state.cursor_positions.append(_pre_cursor)

        _post_save = fdry.get_current_alloc_offset()
        _graph_idx = len(_pending_graphs) - 1
        if state.cursor_after_graph0 is None:
            state.cursor_after_graph0 = _post_save
        return output

    CUDAGraphWrapper.__call__ = _foundry_wrapper_call

    def _finalize_save():
        if not _pending_graphs:
            print(f"[FOUNDRY] No graphs to save after capture_model", flush=True)
            return

        cache.wrapper_graphs_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()

        # Save all graphs rapidly before CGE BUILD can crash us
        for i, (desc, graph, output) in enumerate(_pending_graphs):
            path = str(cache.wrapper_graphs_dir / f"graph_{i}.json")
            try:
                graph.save(path, output_tensors=output)
            except (TypeError, Exception):
                graph.save(path)
            state.saved_descs.append(desc)

        state.save_seconds = time.perf_counter() - t0

        # Pack fatbins and write metadata immediately
        try:
            cache.hook_archive.mkdir(parents=True, exist_ok=True)
            fdry.pack_fatbins_to_folder(str(cache.hook_archive))
        except Exception as e:
            logger.warning("Failed to pack fatbins: %s", e)

        cache.save_metadata(
            state.saved_descs,
            model_id=model_id,
            cache_key=cache_key,
            region_size=region_size,
            region_base=state.region_base,
            pre_capture_offset=state.pre_capture_offset,
            cursor_after_graph0=state.cursor_after_graph0,
            cursor_positions=state.cursor_positions,
            profile_cudagraph_estimate=state.profile_cudagraph_estimate,
            profile_cudagraph_cursor_delta=state.profile_cudagraph_cursor_delta,
            available_kv_cache_memory=state.available_kv_cache_memory,
            determine_cursor_delta=state.determine_cursor_delta,
            passthrough_events=state.passthrough_events,
        )
        cache.write_save_complete_marker()
        state.finalized = True

        print(
            f"[FOUNDRY] Saved {len(state.saved_descs)} CUDA graphs "
            f"in {state.save_seconds:.2f}s (alloc_offset={state.pre_capture_offset})",
            flush=True,
        )

    # Fatbin loading is handled by cached_vllm_init_with_foundry before this call.
    # Graph loading happens lazily in state.load_all() during capture_model()
    # when the model and KV cache are already allocated at deterministic addresses.

    # --- Patch 3: Start graph builds before weight loading ---
    # Model architecture is created (tensors at deterministic addresses) before
    # load_weights is called. Weight loading takes ~25s (download + copy),
    # which is more than enough to overlap Phase 2a template building (~0.6-0.9s).
    if _is_load_mode:
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader
        _original_base_load_weights = BaseModelLoader.load_model

        def _load_model_with_early_builds(self_loader, **kwargs):
            import threading as _th

            _orig_load_weights = self_loader.load_weights

            def _hooked_load_weights(model, model_config):
                # DEBUG: skip preallocation + early graph builds to isolate crash
                print("[FOUNDRY] Calling _orig_load_weights (no prealloc/early builds)...", flush=True)
                result = _orig_load_weights(model, model_config)
                print("[FOUNDRY] _orig_load_weights completed", flush=True)
                return result

            self_loader.load_weights = _hooked_load_weights
            return _original_base_load_weights(self_loader, **kwargs)

        BaseModelLoader.load_model = _load_model_with_early_builds

    logger.info(
        "Patched vLLM for Foundry graph persistence "
        "(CUDAGraphWrapper.__call__ + GPUModelRunner.capture_model, cached=%s)",
        cache.has_cached_wrapper_graphs(),
    )
    return state




def cached_vllm_init_with_foundry(
    model: str,
    graph_cache_dir: str = "/root/.cache/foundry-graphs",
    region_size: str = DEFAULT_REGION_SIZE,
    force_save: bool = False,
    region_base_override: Optional[int] = None,
    **vllm_kwargs,
):
    """Drop-in replacement for vllm.LLM() with Foundry graph caching.

    On first cold start: captures graphs normally + saves to disk.
    On subsequent cold starts: loads graphs from disk, skipping FULL mode capture.

    Requires Foundry installed and LD_PRELOAD=libcuda_hook.so.
    Falls back to standard vLLM if Foundry is not available.

    For multi-GPU (tensor_parallel_size > 1), Foundry setup is deferred
    to each worker subprocess via a patched init_device. Requires the
    fork multiprocessing method (Linux default for vLLM).
    """
    tp_size = vllm_kwargs.get("tensor_parallel_size", 1)

    if tp_size > 1:
        # Multi-GPU: check LD_PRELOAD without importing foundry
        # (importing foundry in parent could init CUDA, forcing spawn)
        if "libcuda_hook.so" not in os.environ.get("LD_PRELOAD", ""):
            logger.warning(
                "LD_PRELOAD not set for multi-GPU Foundry. "
                "Falling back to standard vLLM."
            )
            from vllm import LLM
            return LLM(model=model, **vllm_kwargs)
        return _cached_vllm_init_multi_gpu(
            model=model,
            graph_cache_dir=graph_cache_dir,
            region_size=region_size,
            force_save=force_save,
            **vllm_kwargs,
        )

    if not is_foundry_available():
        logger.warning(
            "Foundry not available (missing install or LD_PRELOAD not set). "
            "Falling back to standard vLLM."
        )
        from vllm import LLM
        return LLM(model=model, **vllm_kwargs)

    effective_cache_dir, cache_key = resolve_graph_cache_dir(
        graph_cache_dir,
        model,
        region_size,
        **vllm_kwargs,
    )
    logger.info(
        "Foundry graph cache key=%s dir=%s",
        cache_key,
        effective_cache_dir,
    )

    pre_cache = FoundryGraphCache(effective_cache_dir)
    saved_meta = pre_cache.load_metadata()
    saved_base = saved_meta.get("region_base") if saved_meta else None

    has_any = pre_cache.has_cached_graphs() or pre_cache.has_cached_wrapper_graphs()
    if has_any and saved_base is None and pre_cache.is_save_complete():
        logger.warning(
            "Cached graphs have no region_base in metadata "
            "(saved with non-deterministic addresses). Clearing cache."
        )
        pre_cache.clear()
        has_any = False

    if region_base_override is not None:
        used_base = region_base_override
        logger.info("Using pre-configured allocation region base=0x%x", used_base)
    else:
        used_base = setup_foundry_regions(region_size, force_base=saved_base)
        if used_base is None and saved_base is not None:
            logger.warning(
                "Saved allocation base 0x%x unavailable. "
                "Clearing graph cache and trying alternative addresses.",
                saved_base,
            )
            pre_cache.clear()
            has_any = False
            used_base = setup_foundry_regions(region_size, force_base=None)
    if used_base is None:
        logger.error(
            "Foundry allocation region setup failed completely. "
            "Falling back to standard vLLM."
        )
        from vllm import LLM
        return LLM(model=model, **vllm_kwargs)

    # Determine if this is a load run (valid complete cache) or save run
    print(f"[FOUNDRY_DEBUG] force_save={force_save}, has_any={has_any}, is_save_complete={pre_cache.is_save_complete()}", flush=True)
    if force_save:
        is_load = False
        if has_any:
            print(f"[FOUNDRY_DEBUG] Clearing existing cache at {effective_cache_dir}", flush=True)
            pre_cache.clear()
    else:
        is_load = has_any and pre_cache.is_save_complete()
    print(f"[FOUNDRY_DEBUG] is_load={is_load}", flush=True)

    if is_load:
        import foundry as fdry
        hook_archive = str(Path(effective_cache_dir) / "hook_archive")
        if os.path.isdir(hook_archive):
            fdry.set_skip_fatbin_processing(True)
            fdry.load_cuda_modules_and_libraries(hook_archive)
            print(f"[FOUNDRY] Loaded CUDA modules from {hook_archive}", flush=True)

    # Extract profile data from metadata BEFORE patching so the patch
    # can decide whether to skip profile_cudagraph_memory at install time
    pcg_est = None
    pcg_delta = None
    kv_mem = None
    det_delta = None
    if is_load and saved_meta:
        pcg_est = saved_meta.get("profile_cudagraph_estimate")
        pcg_delta = saved_meta.get("profile_cudagraph_cursor_delta")
        kv_mem = saved_meta.get("available_kv_cache_memory")
        det_delta = saved_meta.get("determine_cursor_delta")

    state = patch_vllm_for_foundry(
        effective_cache_dir,
        region_size,
        model_id=model,
        cache_key=cache_key,
        region_base=used_base,
        load_mode=is_load,
        profile_cudagraph_estimate=pcg_est,
        profile_cudagraph_cursor_delta=pcg_delta,
        available_kv_cache_memory=kv_mem,
        determine_cursor_delta=det_delta,
    )

    if is_load and saved_meta:
        saved_offset = saved_meta.get("pre_capture_offset")
        if saved_offset is not None:
            state.saved_offset_for_load = saved_offset
            print(f"[FOUNDRY_DEBUG] Will restore alloc offset {saved_offset} on load", flush=True)
        cursor_g0 = saved_meta.get("cursor_after_graph0")
        if cursor_g0 is not None:
            state.cursor_after_graph0 = cursor_g0
            print(f"[FOUNDRY_DEBUG] cursor_after_graph0={cursor_g0} from metadata", flush=True)
        cursor_pos = saved_meta.get("cursor_positions")
        if cursor_pos is not None:
            state.cursor_positions = cursor_pos
            print(f"[FOUNDRY_DEBUG] Loaded {len(cursor_pos)} cursor positions from metadata", flush=True)

    atexit.register(state.finalize)

    t0 = time.perf_counter()
    from vllm import LLM
    llm = LLM(model=model, **vllm_kwargs)
    init_time = time.perf_counter() - t0

    logger.info("vLLM + Foundry init took %.2fs", init_time)
    return llm


def _cached_vllm_init_multi_gpu(
    model: str,
    graph_cache_dir: str,
    region_size: str,
    force_save: bool,
    **vllm_kwargs,
):
    """Multi-GPU Foundry init: defers setup to each worker subprocess.

    Patches Worker.init_device in the parent process. On Linux (fork),
    the patch propagates to each worker subprocess. Each worker sets up
    its own Foundry allocation region and applies rank-specific patches.

    Cache structure:
        {graph_cache_dir}/{cache_key}/rank_0/
        {graph_cache_dir}/{cache_key}/rank_1/
        ...
    """
    from vllm.v1.worker.gpu_worker import Worker as GPUWorker

    tp_size = vllm_kwargs.get("tensor_parallel_size", 1)

    # Build cache key without GPU-specific info (safe in parent — no CUDA init).
    # build_graph_cache_key only reads version strings from torch/vllm.
    _, cache_key = resolve_graph_cache_dir(
        graph_cache_dir, model, region_size, **vllm_kwargs
    )
    base_dir = Path(graph_cache_dir) / cache_key

    # All ranks must use the same mode to avoid NCCL deadlocks.
    # Check all rank caches in the parent (filesystem only, no CUDA).
    if force_save:
        global_load_mode = False
        for r in range(tp_size):
            FoundryGraphCache(str(base_dir / f"rank_{r}")).clear()
    else:
        def _rank_cache_ready(r: int) -> bool:
            rank_dir = base_dir / f"rank_{r}"
            if not (rank_dir / ".save_complete").exists():
                return False
            wg_dir = rank_dir / "wrapper_graphs"
            return wg_dir.exists() and any(wg_dir.glob("graph_*.json"))

        global_load_mode = all(_rank_cache_ready(r) for r in range(tp_size))

    print(
        f"[FOUNDRY] Multi-GPU: tp={tp_size}, key={cache_key}, "
        f"load_mode={global_load_mode}",
        flush=True,
    )

    _original_init_device = GPUWorker.init_device

    def _foundry_init_device(self):
        import torch
        import foundry as fdry

        rank = self.rank
        local_rank = self.local_rank

        torch.cuda.set_device(local_rank)

        # Per-rank cache
        rank_cache_dir = str(base_dir / f"rank_{rank}")
        rank_cache = FoundryGraphCache(rank_cache_dir)
        saved_meta = rank_cache.load_metadata() if global_load_mode else None
        saved_base = saved_meta.get("region_base") if saved_meta else None

        if global_load_mode:
            hook_archive = str(Path(rank_cache_dir) / "hook_archive")
            if os.path.isdir(hook_archive):
                torch.cuda.init()
                fdry.set_skip_fatbin_processing(True)
                fdry.load_cuda_modules_and_libraries(hook_archive)
                print(
                    f"[FOUNDRY] Rank {rank}: loaded CUDA modules from {hook_archive}",
                    flush=True,
                )

        region_base = setup_foundry_regions(
            region_size=region_size,
            force_base=saved_base,
        )
        if region_base is None and saved_base is not None:
            rank_cache.clear()
            region_base = setup_foundry_regions(region_size=region_size)

        if region_base is None:
            logger.error(
                "Foundry region setup failed for rank %d, "
                "falling back to standard init",
                rank,
            )
            _original_init_device(self)
            return

        pcg_est = saved_meta.get("profile_cudagraph_estimate") if saved_meta else None
        pcg_delta = saved_meta.get("profile_cudagraph_cursor_delta") if saved_meta else None
        kv_mem = saved_meta.get("available_kv_cache_memory") if saved_meta else None
        det_delta = saved_meta.get("determine_cursor_delta") if saved_meta else None

        state = patch_vllm_for_foundry(
            graph_cache_dir=rank_cache_dir,
            region_size=region_size,
            model_id=model,
            cache_key=cache_key,
            region_base=region_base,
            load_mode=global_load_mode,
            profile_cudagraph_estimate=pcg_est,
            profile_cudagraph_cursor_delta=pcg_delta,
            available_kv_cache_memory=kv_mem,
            determine_cursor_delta=det_delta,
        )

        if global_load_mode and saved_meta:
            saved_offset = saved_meta.get("pre_capture_offset")
            if saved_offset is not None:
                state.saved_offset_for_load = saved_offset
            cursor_g0 = saved_meta.get("cursor_after_graph0")
            if cursor_g0 is not None:
                state.cursor_after_graph0 = cursor_g0
            cursor_pos = saved_meta.get("cursor_positions")
            if cursor_pos is not None:
                state.cursor_positions = cursor_pos

        atexit.register(state.finalize)

        pre_init_cursor = fdry.get_current_alloc_offset()
        print(
            f"[FOUNDRY] Rank {rank} ready: base=0x{region_base:x}, "
            f"load={global_load_mode}, cursor_before_init={pre_init_cursor}, "
            f"dir={rank_cache_dir}",
            flush=True,
        )

        fdry.stop_allocation_region()
        fdry.start_passthrough_record()
        _original_init_device(self)
        # Keep recording running to capture post-init allocations
        # (cuBLAS workspaces, etc.) that get baked into CUDA graphs.
        # Recording will be stopped in _flagged_capture_model.
        fdry.resume_allocation_region()
        state._extended_recording = True
        init_count = len(fdry.get_passthrough_events())
        print(
            f"[FOUNDRY] Rank {rank}: {init_count} passthrough events so far "
            f"(extended recording active through capture_model)",
            flush=True,
        )

        post_init_cursor = fdry.get_current_alloc_offset()
        print(
            f"[FOUNDRY] Rank {rank} after init_device: "
            f"cursor={post_init_cursor} ({post_init_cursor/(1024**3):.2f}GB)",
            flush=True,
        )

    GPUWorker.init_device = _foundry_init_device

    vllm_kwargs["disable_custom_all_reduce"] = True

    t0 = time.perf_counter()
    from vllm import LLM
    llm = LLM(model=model, **vllm_kwargs)
    init_time = time.perf_counter() - t0

    logger.info("vLLM + Foundry (tp=%d) init took %.2fs", tp_size, init_time)
    return llm
