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
import scipy.interpolate as interp
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


def compute_particle_trajectory(u_b, v_b, x_vals, y_vals, t_vals, dt=0.01, start_pos=(-0.5, 0)):
    """
    u_b, v_b : (T, nx, ny)
    x_vals, y_vals : 1D arrays
    t_vals : time array
    dt : integration step
    start_pos : initial particle position
    """
    traj_segments = []
    segment = [start_pos]
    pos = np.array(start_pos, dtype=float)

    nx, ny = len(x_vals), len(y_vals)
    xmin, xmax = x_vals.min(), x_vals.max()
    ymin, ymax = y_vals.min(), y_vals.max()
    bound = (xmin, xmax, ymin, ymax)
    # 시간에 따라 velocity interpolation 함수 생성
    interp_u = [interp.RegularGridInterpolator((x_vals, y_vals), u_b[i], bounds_error=False, fill_value=None)
                for i in range(len(t_vals))]
    interp_v = [interp.RegularGridInterpolator((x_vals, y_vals), v_b[i], bounds_error=False, fill_value=None)
                for i in range(len(t_vals))]
    for ti in range(len(t_vals)-1):
        n_steps = int((t_vals[ti+1] - t_vals[ti]) / dt)
        for _ in range(n_steps):
            u = interp_u[ti](pos)
            v = interp_v[ti](pos)
            pos[0] += dt * u
            pos[1] += dt * v
            segment.append(pos.copy())

            # 범위 벗어나면 trajectory 저장 후 초기화
            if pos[0] < xmin or pos[0] > xmax or pos[1] < ymin or pos[1] > ymax:
                traj_segments.append(np.array(segment))
                pos = np.array(start_pos, dtype=float)
                segment = [pos.copy()]
                break

    if len(segment) > 1:
        traj_segments.append(np.array(segment))

    return traj_segments, bound

def save_trajectory_plot(traj_segments, out_dir="./dataset/navier_stokes_flow", fname="trajectory_plot.png", bound=None):
    plt.figure(figsize=(6,6))
    for seg in traj_segments:
        plt.plot(seg[:,0], seg[:,1], '-', lw=1)
        plt.plot(seg[0,0], seg[0,1], 'go', markersize=3)
        plt.plot(seg[-1,0], seg[-1,1], 'ro', markersize=3)
    if bound is not None:
        plt.xlim(bound[0], bound[1])
        plt.ylim(bound[2], bound[3])
    # plt.ylim(-1,1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Particle Trajectories')
    plt.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Trajectory plot saved: {os.path.join(out_dir, fname)}")
import matplotlib.cm as cm
def save_trajectories_plot(trajectories, out_dir="./dataset/navier_stokes_flow",
                         fname="trajectory_plot.png", bound=None):
    """
    trajectories: list of list
        예: [ [traj_0_seg_0, traj_0_seg_1], [traj_1_seg_0], [traj_2_seg_0, traj_2_seg_1, ...] ]
    bound: [xmin, xmax] 또는 [xmin, xmax, ymin, ymax]
    """
    plt.figure(figsize=(7, 7))
    colors = cm.get_cmap('tab10', len(trajectories))  # dataset별 고유 색상
    
    label_added = set()  # legend 중복 방지
    idx = 0  # trajectory index (전체)

    for exp_i, traj_list in enumerate(trajectories):
        color = colors(exp_i)
        label = f"Dataset {exp_i+1}"
        for seg in traj_list:
            x, y = seg[:, 0], seg[:, 1]
            if label not in label_added:
                plt.plot(x, y, '-', color=color, lw=1.5, label=label)
                label_added.add(label)
            else:
                plt.plot(x, y, '-', color=color, lw=1.0)
            
            # trajectory 번호 표시 (중앙 혹은 마지막 위치)
            # cx, cy = x[len(x)//2], y[len(y)//2]
            # plt.text(cx, cy, f"{idx}", fontsize=7, color=color, ha='center', va='center')
            idx += 1

    # 범위 설정
    # if len(bound) == 2:
    #     xmin, xmax = bound
    #     ymin, ymax = bound
    # elif len(bound) == 4:
    #     xmin, xmax, ymin, ymax = bound
    # else:
    #     xmin, xmax, ymin, ymax = -1, 1, -1, 1

    # plt.xlim(xmin, xmax)
    # plt.ylim(ymin, ymax)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Particle Trajectories from 3 Simulations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"✅ Combined trajectory plot saved: {path}")
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
    dataset_name="spectral_vorticity.nc",
    raw_dataset_name="spectral_vorticity.nc",
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
    u_b_raw = u[0].numpy()  # (snapshots, nx, ny)
    v_b_raw = v[0].numpy()
    vort_b_raw = fft.irfft2(result["vorticity"])[0].cpu().numpy()  # (snapshots, nx, ny)
    # Build xarray dataset
    # (1) Grid normalization: [-1, 1]
    x_vals = np.linspace(-1.0, 1.0, n)
    y_vals = np.linspace(-1.0, 1.0, n)
    x_raw_vals = np.linspace(0, diam, n, endpoint=False)
    y_raw_vals = np.linspace(0, diam, n, endpoint=False)

    # (2) Flow normalization: [-1, 1]
    def normalize_field(field):
        fmin, fmax = field.min(), field.max()
        if fmax == fmin:
            return np.zeros_like(field)
        return 2 * (field - fmin) / (fmax - fmin) - 1

    u_b = normalize_field(u_b_raw)
    v_b = normalize_field(v_b_raw)
    vort_b = normalize_field(vort_b_raw)

    # (3) Time
    t_vals = np.linspace(0, u_b.shape[0]*0.1, u_b.shape[0])  # normalized 0~1 for time (optional)

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
    coords_raw = {"time": t_vals, "x": x_raw_vals, "y": y_raw_vals}
    ds_raw = xr.Dataset(
        data_vars=dict(
            u=(("time", "x", "y"), u_b_raw),
            v=(("time", "x", "y"), v_b_raw),
            vorticity=(("time", "x", "y"), vort_b_raw),
        ),
        coords=coords_raw,
    )
    nc_path = os.path.join(out_dir, dataset_name)
    nc_path_raw = os.path.join(out_dir, raw_dataset_name)
    engine, encoding = choose_netcdf_engine_and_encoding()

    if encoding is None:
        ds.to_netcdf(nc_path, engine=engine)
        ds_raw.to_netcdf(nc_path_raw, engine=engine)
    else:
        ds.to_netcdf(nc_path, engine=engine, encoding=encoding)
        ds_raw.to_netcdf(nc_path_raw, engine=engine, encoding=encoding)

    print(f"💾 Saved NetCDF: {nc_path}")
    print(f"💾 Saved Raw NetCDF: {nc_path_raw}")
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
def merge_datasets(out_dir: str, num_traj: int, save_name: str = "dataset_merged.nc"):
    datasets = []

    for i in range(num_traj):
        dataset_name = f"dataset_{i}.nc"
        data_path = os.path.join(out_dir, dataset_name)

        if not os.path.exists(data_path):
            print(f"⚠️  Warning: {data_path} not found, skipping.")
            continue

        ds = xr.open_dataset(data_path)
        datasets.append(ds)

    if not datasets:
        raise ValueError("❌ No datasets were loaded. Check your paths or num_traj value.")

    # realization 차원으로 합치기
    ds_merged = xr.concat(datasets, dim="realization")

    # realization 좌표 추가 (0, 1, 2, ...)
    ds_merged = ds_merged.assign_coords(realization=("realization", list(range(len(datasets)))))

    # 저장
    merged_path = os.path.join(out_dir, save_name)
    ds_merged.to_netcdf(merged_path)

    print(f"✅ Merged {len(datasets)} datasets into {merged_path}")
    print(f"   → Dimensions: {ds_merged.dims}")

    return ds_merged
# -------------------------------------------------------
# 5️⃣ Example Usage
# -------------------------------------------------------
if __name__ == "__main__":
    out_dir = "./dataset/navier_stokes_flow/multiple_traj"
    trajectories = []
    nun_traj = 10
    for i in range(nun_traj):
        print(f"================ Experiment {i+1} ================")
        dataset_name = f"dataset_{i}.nc"
        raw_dataset_name = f"raw_dataset_{i}.nc"
        sim_info = run_torch_cfd_spectral_sim_and_save(
            n=128,
            T=10.0,
            viscosity=1e-3+np.random.rand()*5e-2,
            max_velocity=2+np.random.rand()*2,
            num_snapshots=30,
            batch_size=3,
            out_dir=out_dir,
            dataset_name=dataset_name,
            raw_dataset_name=raw_dataset_name,
        )
        data_path = os.path.join(out_dir, raw_dataset_name)
        ds = xr.open_dataset(data_path)
        u_b = ds["u"].values  # (T, nx, ny)
        v_b = ds["v"].values
        x_vals = ds["x"].values
        y_vals = ds["y"].values
        t_vals = ds["time"].values
        print(f"time: {t_vals.shape}, x: {x_vals.shape}, y: {y_vals.shape}, u/v: {u_b.shape}")
        trajectory, bound = compute_particle_trajectory(u_b, v_b, x_vals, y_vals, t_vals, dt=1e-3, start_pos=(3,3))
        trajectories.append(trajectory)
        np.save(os.path.join(out_dir, "particle_trajectory.npy"), trajectories)
        # print("💾 Saved trajectory data (.npy)")

    merge_datasets(out_dir, num_traj=nun_traj)

    # save_trajectories_plot(trajectories, out_dir=out_dir, fname="trajectory_plot.png", bound=bound)
    '''generate single dataset'''
    # info = run_torch_cfd_spectral_sim_and_save(
    #     n=128,
    #     T=20.0,
    #     viscosity=1e-3,
    #     max_velocity=2,
    #     num_snapshots=1000,
    #     batch_size=5,
    #     out_dir="./dataset/navier_stokes_flow",
    #     dataset_name="dataset_slow.nc",
    #     raw_dataset_name="raw_dataset_slow.nc",
    #     gif_name="spectral_vorticity_slow.gif",
    # )
    '''load dataset'''
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # out_dir = "./dataset/navier_stokes_flow"
    # data_path = "./dataset/navier_stokes_flow/raw_dataset_slow.nc"
    # ds = xr.open_dataset(data_path)
    # u_b = ds["u"].values  # (T, nx, ny)
    # v_b = ds["v"].values
    # x_vals = ds["x"].values
    # y_vals = ds["y"].values
    # t_vals = ds["time"].values
    # print(f"time: {t_vals.shape}, x: {x_vals.shape}, y: {y_vals.shape}, u/v: {u_b.shape}")

    