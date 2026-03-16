"""
Auto-detects NVIDIA GPU architecture and compiles/loads
the correct PTX kernel for the current device.
"""
import subprocess
import cupy as cp
from pathlib import Path

# Map compute capability to sm_ target
SM_MAP = {
    (7, 0): "sm_70",  # V100
    (7, 5): "sm_75",  # RTX 20xx, T4
    (8, 0): "sm_80",  # A100, RTX 30xx
    (8, 6): "sm_86",  # RTX 3060-3090
    (8, 9): "sm_89",  # RTX 40xx, L40
    (9, 0): "sm_90",  # H100
}

def get_sm_target() -> str:
    """Detect current GPU and return correct sm_ target."""
    device = cp.cuda.Device(0)
    major = device.compute_capability[0]
    minor = device.compute_capability[1]
    cc = (int(major), int(minor))

    # Exact match first
    if cc in SM_MAP:
        return SM_MAP[cc]

    # Fall back to closest lower architecture
    best = None
    for (ma, mi), sm in SM_MAP.items():
        if (ma, mi) <= cc:
            best = sm
    if best:
        return best

    # Default to sm_75 as safe baseline
    print(f"[GPU] Unknown compute capability {cc}, defaulting to sm_75")
    return "sm_75"

def get_device_info() -> dict:
    device = cp.cuda.Device(0)
    props  = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "name":    props["name"].decode(),
        "cc":      device.compute_capability,
        "sm":      get_sm_target(),
        "vram_gb": props["totalGlobalMem"] / 1024**3,
        "sm_count": props["multiProcessorCount"],
    }

def compile_ptx(cu_path: Path, ptx_path: Path, sm: str):
    """Compile CUDA kernel to PTX for target architecture."""
    print(f"[GPU] Compiling for {sm}...")
    result = subprocess.run(
        ["nvcc", "--ptx", f"-arch={sm}", "-O3",
         str(cu_path), "-o", str(ptx_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{result.stderr}")
    print(f"[GPU] Compiled {ptx_path.name} for {sm}")

def get_ptx(kernels_dir: Path, cu_name: str) -> Path:
    """
    Get the correct PTX for the current GPU.
    Compiles if needed, caches by architecture.
    """
    sm      = get_sm_target()
    cu_path = kernels_dir / f"{cu_name}.cu"
    ptx_path = kernels_dir / f"{cu_name}_{sm}.ptx"

    if not ptx_path.exists():
        compile_ptx(cu_path, ptx_path, sm)

    return ptx_path
