import os
import math
import torch
import torch.fft as fft
import numpy as np
import xarray as xr
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from torch_cfd.grids import Grid
from torch_cfd.initial_conditions import filtered_vorticity_field
from torch_cfd.spectral import *
from fno.data_gen.trajectories import get_trajectory_imex
# --- (파일 상단 유틸 아래에 추가) ---
def choose_netcdf_engine_and_encoding():
    """
    netCDF 엔진 및 인코딩을 자동 선택:
    - netCDF4가 설치되어 있으면: engine='netcdf4' + zlib 압축 사용
    - 아니면: engine=None (SciPy backend) + 압축 인코딩 제거
    """
    try:
        import netCDF4  # noqa: F401
        engine = "netcdf4"
        encoding = {name: {"zlib": True, "complevel": 4}
                    for name in ["u", "v", "vorticity"]}
    except Exception:
        engine = None   # SciPy backend (압축 인코딩 미지원)
        encoding = None
    return engine, encoding
# -------------------------------------------------------
# 1️⃣ Utility Functions
# -------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def make_coords_full_from_linspace(x_vals, y_vals):
    X, Y = np.meshgrid(x_vals, y_vals, indexing="xy")
    coords_full = np.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)
    return coords_full

def to_torch_split_real_only(lst, device):
    yr = [torch.from_numpy(np.array(y, dtype=np.float32)).to(device) for y in lst]
    yi = [torch.zeros_like(r) for r in yr]
    return [torch.stack([r, i], dim=-1) for r, i in zip(yr, yi)]

# -------------------------------------------------------
# 2️⃣ Simulation + GIF + NetCDF
# -------------------------------------------------------
def run_torch_cfd_spectral_sim_and_save(
    n=256,
    T=20.0,
    dt=1e-3,
    viscosity=1e-3,
    max_velocity=2.0,
    batch_size=1, 
    num_snapshots=100,
    peak_wavenumber=2,
    random_state=0,
    out_dir="./torchcfd_spectral_out",
    gif_name="spectral_vorticity.gif",
    nc_name="spectral_vorticity.nc",
):
    ensure_dir(out_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    diam = 2 * torch.pi
    dx = 2 * torch.pi / n
    dt = stable_time_step(dx=dx, dt=dt, viscosity=viscosity, max_velocity=max_velocity)
    num_steps = int(T / dt)
    record_iters = int(num_steps / num_snapshots)

    # Grid and initial condition
    grid = Grid(shape=(n, n), domain=((0, diam), (0, diam)), device=device)
    vort_init = filtered_vorticity_field(
        grid, peak_wavenumber, random_state=random_state, batch_size=batch_size
    )
    vort_hat = fft.rfft2(vort_init).data.to(device)

    # Spectral Navier–Stokes
    ns2d = NavierStokes2DSpectral(
        viscosity=viscosity,
        grid=grid,
        smooth=True,
        step_fn=RK4CrankNicolsonStepper(),
    ).to(device)

    print(f"Simulating {num_steps} steps, saving every {record_iters}...")
    result = get_trajectory_imex(
        ns2d,
        vort_hat,
        dt,
        num_steps=num_steps,
        record_every_steps=record_iters,
        pbar=True,
    )

    vort = fft.irfft2(result["vorticity"]).cpu()  # (batch, snapshots, n, n)
    stream = result["stream"].to(device)
    velocity = spectral_rot_2d(stream, grid.rfft_mesh())  # tuple (u_hat, v_hat)

    # u_hat, v_hat each have shape (batch, num_snapshots, nx, ny)
    u_hat, v_hat = velocity

    u = fft.irfft2(u_hat).cpu()  # (batch, snapshots, nx, ny)
    v = fft.irfft2(v_hat).cpu()

    # Choose one batch
    u_b = u[0].numpy()  # (snapshots, nx, ny)
    v_b = v[0].numpy()
    vort_b = fft.irfft2(result["vorticity"])[0].cpu().numpy()  # (snapshots, nx, ny)
    # Build xarray dataset
    # (1) Grid normalization: [-1, 1]
    x_vals = np.linspace(-1.0, 1.0, n)
    y_vals = np.linspace(-1.0, 1.0, n)

    # (2) Flow normalization: [0, 1]
    def normalize_field(field):
        fmin, fmax = field.min(), field.max()
        if fmax == fmin:
            return np.zeros_like(field)
        return (field - fmin) / (fmax - fmin)

    u_b = normalize_field(u_b)
    v_b = normalize_field(v_b)
    vort_b = normalize_field(vort_b)

    # (3) Time
    t_vals = np.linspace(0, 1, u_b.shape[0])  # normalized 0~1 for time (optional)

    # (4) xarray dataset construction
    coords = {"time": t_vals, "x": x_vals, "y": y_vals}
    ds = xr.Dataset(
        data_vars=dict(
            u=(("time", "x", "y"), u_b),
            v=(("time", "x", "y"), v_b),
            vorticity=(("time", "x", "y"), vort_b),
        ),
        coords=coords,
    )
    nc_path = os.path.join(out_dir, nc_name)
    engine, encoding = choose_netcdf_engine_and_encoding()

    if encoding is None:
        ds.to_netcdf(nc_path, engine=engine)
    else:
        ds.to_netcdf(nc_path, engine=engine, encoding=encoding)

    print(f"💾 Saved NetCDF: {nc_path}")

    # Save GIF (vorticity evolution)
    frames_dir = os.path.join(out_dir, "frames_vorticity")
    ensure_dir(frames_dir)
    frames = []
    for i in tqdm(range(vort_b.shape[0]), desc="Saving frames"):
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(
            vort_b[i].T,
            origin="lower",
            extent=(-1, 1, -1, 1),
            cmap=sns.cm.icefire,
            vmin=0, vmax=1, 
        )
        plt.title(f"Vorticity t={t_vals[i]:.3f}")
        plt.axis("off")
        frame_path = os.path.join(frames_dir, f"vort_{i:04d}.png")
        plt.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frames.append(imageio.imread(frame_path))
    gif_path = os.path.join(out_dir, gif_name)
    imageio.mimsave(gif_path, frames, duration=0.08, loop=0)
    print(f"✅ Saved GIF: {gif_path}")

    return {
        "nc_path": nc_path,
        "gif_path": gif_path,
        "t_vals": t_vals,
        "x_vals": x_vals,
        "y_vals": y_vals,
    }

# -------------------------------------------------------
# 3️⃣ Gray–Scott Compatible Loader
# -------------------------------------------------------
def load_spectral_nc_as_grayscott_compatible(
    nc_path: str,
    sample_ratio: float = 0.2,
    normalize_t: bool = False,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
):
    np.random.seed(seed)
    ds = xr.open_dataset(nc_path)

    vort = ds["vorticity"].transpose("time", "x", "y").values  # (T, n, n)
    tvals = ds["time"].values
    xvals = ds["x"].values
    yvals = ds["y"].values

    coords_full = make_coords_full_from_linspace(xvals, yvals)
    n = coords_full.shape[0]
    m = int(n * sample_ratio)
    idx = np.random.choice(n, size=m, replace=False)

    y_list, y_list_full, coords_list = [], [], []
    for k in range(vort.shape[0]):
        flat = vort[k].reshape(-1)
        y_list_full.append(flat.copy())
        y_list.append(flat[idx])
        coords_list.append(coords_full[idx])

    # Normalize time
    if normalize_t:
        t_list = [np.float32(t / tvals[-1]) for t in tvals]
    else:
        t_list = [np.float32(t) for t in tvals]

    coords_torch = [torch.from_numpy(c).float().to(device) for c in coords_list]
    coords_full_torch = torch.from_numpy(coords_full).float().to(device)
    y_torch = to_torch_split_real_only(y_list, device)
    y_full_torch = to_torch_split_real_only(y_list_full, device)

    return t_list, coords_torch, y_torch, y_full_torch, coords_full_torch

# -------------------------------------------------------
# 4️⃣ Dataset Class
# -------------------------------------------------------
class VorticityDataset(torch.utils.data.Dataset):
    def __init__(self, t_list, coords_list, y_list):
        self.t_list = t_list
        self.coords_list = coords_list
        self.y_list = y_list
        self.length = len(t_list) - 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        t_prev = self.t_list[idx]
        t_next = self.t_list[idx + 1]
        coords = self.coords_list[idx + 1]
        y_prev = self.y_list[idx]
        y_next = self.y_list[idx + 1]
        return t_prev, t_next, coords, y_next, y_prev

# -------------------------------------------------------
# 5️⃣ Example Usage
# -------------------------------------------------------
if __name__ == "__main__":
    info = run_torch_cfd_spectral_sim_and_save(
        n=128,
        T=20.0,
        viscosity=1e-3,
        max_velocity=2,
        num_snapshots=100,
        batch_size=1,
        out_dir="./dataset/navier_stokes_flow",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_list, coords_list, y_list, y_full, coords_full = load_spectral_nc_as_grayscott_compatible(
        info["nc_path"], sample_ratio=0.1, normalize_t=True, device=device
    )
    dataset = VorticityDataset(t_list, coords_list, y_list)
    print(f"Dataset length: {len(dataset)}")
    tp, tn, c, yn, yp = dataset[0]
    print(f"Example shapes -> coords:{c.shape}, y_next:{yn.shape}, y_prev:{yp.shape}")