"""
Auto-detects NVIDIA GPU architecture and compiles/loads
the correct PTX kernel for the current device.
Supports any NVIDIA GPU with CUDA 11+ and nvcc installed.
"""
import subprocess
import cupy as cp
from pathlib import Path

# Map compute capability to sm_ target
SM_MAP = {
    (6, 0): "sm_60",  # P100
    (6, 1): "sm_61",  # GTX 10xx
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
        print(f"[GPU] No exact match for cc={cc}, using {best}")
        return best

    # Last resort
    print(f"[GPU] Unknown compute capability {cc}, defaulting to sm_75")
    return "sm_75"

def get_device_info() -> dict:
    device = cp.cuda.Device(0)
    props  = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "name":     props["name"].decode(),
        "cc":       device.compute_capability,
        "sm":       get_sm_target(),
        "vram_gb":  props["totalGlobalMem"] / 1024**3,
        "sm_count": props["multiProcessorCount"],
    }

def _nvcc_available() -> bool:
    """Check if nvcc compiler is installed."""
    result = subprocess.run(
        ["nvcc", "--version"],
        capture_output=True, text=True
    )
    return result.returncode == 0

def compile_ptx(cu_path: Path, ptx_path: Path, sm: str):
    """Compile CUDA kernel to PTX for target architecture."""
    print(f"[GPU] Compiling kernels for {sm} — first run only...")
    result = subprocess.run(
        ["nvcc", "--ptx", f"-arch={sm}", "-O3",
         str(cu_path), "-o", str(ptx_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvcc compilation failed:\n{result.stderr}")
    print(f"[GPU] Compiled {ptx_path.name} ✓")

def get_ptx(kernels_dir: Path, cu_name: str) -> Path:
    """
    Get the correct PTX for the current GPU.
    Compiles automatically if needed — works on any CUDA-capable GPU.
    """
    sm       = get_sm_target()
    cu_path  = kernels_dir / f"{cu_name}.cu"
    ptx_path = kernels_dir / f"{cu_name}_{sm}.ptx"

    if ptx_path.exists():
        return ptx_path

    # PTX not found — try to compile
    print(f"[GPU] No pre-compiled PTX found for {sm}")

    if not cu_path.exists():
        raise FileNotFoundError(
            f"Kernel source not found: {cu_path}\n"
            f"Clone the full repository to enable compilation."
        )

    if not _nvcc_available():
        raise RuntimeError(
            f"No PTX for {sm} and nvcc not found.\n"
            f"Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads\n"
            f"Or pre-compile manually:\n"
            f"  nvcc --ptx -arch={sm} -O3 {cu_path} -o {ptx_path}"
        )

    compile_ptx(cu_path, ptx_path, sm)
    return ptx_path

