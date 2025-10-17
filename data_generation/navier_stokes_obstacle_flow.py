import os
import math
import numpy as np
import torch
import xarray as xr
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import seaborn as sns

import torch_cfd.finite_differences as fdm
from torch_cfd import advection, boundaries, grids
from torch_cfd.fvm import NavierStokes2DFVMProjection, PressureProjection, RKStepper
from torch_cfd.initial_conditions import velocity_field
from tqdm import tqdm

# ======================================================
# 0) 유틸
# ======================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def central_vorticity_np(u, v, dx, dy):
    # u, v: (nx, ny)
    # np.gradient은 (∂/∂x, ∂/∂y) 순으로 반환 (축 순서에 유의)
    du_dy = np.gradient(u, dy, axis=1, edge_order=2)
    dv_dx = np.gradient(v, dx, axis=0, edge_order=2)
    return dv_dx - du_dy  # (nx, ny)

def make_coords_full_from_linspace(x_vals, y_vals):
    X, Y = np.meshgrid(x_vals, y_vals, indexing="xy")
    coords_full = np.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)
    return coords_full

def to_torch_split_real_only(lst, device):
    # (real, imag) 2채널로 쪼개되 imag=0
    import torch
    yr = [torch.from_numpy(np.array(y, dtype=np.float32)).to(device) for y in lst]
    yi = [torch.zeros_like(r) for r in yr]
    return [torch.stack([r, i], dim=-1) for r, i in zip(yr, yi)]

# ======================================================
# 1) 시뮬레이션 + GIF + .nc 저장
# ======================================================
def run_torch_cfd_sim_and_save(
    nx=400, ny=200,
    density=1.0,
    dt=1e-3,           # 시뮬레이션 내부 dt
    T=2.0,            # 전체 물리시간
    viscosity=1/500,
    batch_size=8,
    domain=((0.0, 2.0), (0.0, 1.0)),
    cylinder_center=(0.45, 0.5),
    cylinder_radius=0.1,
    update_steps=100,  # 매 update_steps마다 스냅샷 저장
    batch_index=0,     # GIF/NC/로더에 사용할 배치 인덱스
    out_dir="./torchcfd_out",
    gif_name="obstacle_flow_vorticity.gif",
    nc_name="obstacle_flow.nc",
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    ensure_dir(out_dir)

    # ----- Grid & BC -----
    grid = grids.Grid((nx, ny), domain=domain, device=device)
    vel_bc, p_bc = boundaries.karman_vortex_boundary_conditions(
        grid, cylinder_center=cylinder_center, cylinder_radius=cylinder_radius
    )

    # ----- 초기 속도장 (유입: (1,0) + 노이즈) -----
    x_velocity_fn = lambda x, y: torch.ones_like(x)
    y_velocity_fn = lambda x, y: torch.zeros_like(x)
    v0 = velocity_field(
        (x_velocity_fn, y_velocity_fn),
        grid,
        velocity_bc=vel_bc,
        batch_size=batch_size,
        random_state=42,
        noise=0.1,
        device=device,
    )

    # ----- 압력 투영, 대류, 적분기 -----
    pressure_proj = PressureProjection(grid=grid, bc=p_bc, dtype=dtype, implementation="matmul")
    convection = advection.ConvectionVector(
        grid=grid,
        offsets=(v0[0].offset, v0[1].offset),
        bcs=vel_bc,
        advect=advection.AdvectionVanLeer,
    )
    step_fn = RKStepper.from_method(method="classic_rk4", requires_grad=False, dtype=dtype)

    ns2d = NavierStokes2DFVMProjection(
        viscosity=viscosity,
        grid=grid,
        bcs=vel_bc,
        density=density,
        step_fn=step_fn,
        pressure_proj=pressure_proj,
        convection=convection,
    ).to(v0.device)

    # ----- 시뮬레이션 루프 -----
    num_steps = int(T / dt)
    v = v0
    trajectory_u = []   # 각 저장시점: (batch, nx, ny)
    trajectory_v = []   # 각 저장시점: (batch, nx, ny)

    velocity_norm = torch.sqrt(v[0].L2norm ** 2 + v[1].L2norm ** 2)
    divergence = fdm.divergence(v)
    desc = f"u norm: {velocity_norm.mean().item():.3e} | div norm: {divergence.L2norm.mean().item():.3e}"
    nan_count = 0

    with tqdm(total=num_steps, desc=desc) as pbar:
        with torch.no_grad():
            for i in range(num_steps):
                v, p = ns2d(v, dt)

                if torch.isnan(v[0].data).any() or torch.isnan(v[1].data).any():
                    print(f"[Warn] NaN detected at step {i}")
                    nan_count += 1
                    break

                if i % update_steps == 0:
                    velocity_norm = torch.sqrt(v[0].L2norm ** 2 + v[1].L2norm ** 2)
                    divergence = fdm.divergence(v)
                    desc = f"u norm: {velocity_norm.mean().item():.3e} | div norm: {divergence.L2norm.mean().item():.3e}"
                    trajectory_u.append(v[0].data.detach().cpu().numpy())  # (batch, nx, ny)
                    trajectory_v.append(v[1].data.detach().cpu().numpy())
                    pbar.set_description(desc)
                pbar.update(1)

    if nan_count > 0 and len(trajectory_u) == 0:
        raise RuntimeError("Simulation aborted early due to NaNs before any snapshot was stored.")

    # ----- numpy 스택: (num_saved, batch, nx, ny)
    U = np.stack(trajectory_u, axis=0).astype(np.float64)
    V = np.stack(trajectory_v, axis=0).astype(np.float64)
    num_saved = U.shape[0]

    # ----- 좌표/시간 축 생성 -----
    # 저장 간격은 update_steps*dt
    t_saved = np.arange(num_saved, dtype=np.float64) * (update_steps * dt)  # shape: (num_saved,)
    x_vals = np.linspace(domain[0][0], domain[0][1], nx, dtype=np.float64)
    y_vals = np.linspace(domain[1][0], domain[1][1], ny, dtype=np.float64)

    # ----- 선택된 배치 인덱스의 (time, x, y) 배열 구성 -----
    u_b = U[:, batch_index, :, :]  # (num_saved, nx, ny)
    v_b = V[:, batch_index, :, :]

    # ----- 와도 계산 -----
    dx = (domain[0][1] - domain[0][0]) / (nx - 1)
    dy = (domain[1][1] - domain[1][0]) / (ny - 1)
    vort_b = np.empty_like(u_b)
    for k in range(num_saved):
        vort_b[k] = central_vorticity_np(u_b[k], v_b[k], dx, dy)

    # (1) Grid normalization: [-1, 1]
    x_vals = np.linspace(-1.0, 1.0, nx)
    y_vals = np.linspace(-1.0, 1.0, ny)

    def normalize_field(field):
        fmin, fmax = np.percentile(field, [2, 98])  # outlier 제거
        return np.clip(2 * (field - fmin) / (fmax - fmin) - 1, -1, 1)

    u_b = normalize_field(u_b)
    v_b = normalize_field(v_b)
    vort_b = normalize_field(vort_b)

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

    nc_path = os.path.join(out_dir, nc_name)
    encoding = {name: {"zlib": True, "complevel": 4} for name in ["u", "v", "vorticity"]}
    ds.to_netcdf(nc_path, encoding=encoding)

    # ----- GIF 저장 (vorticity) -----
    frames_dir = os.path.join(out_dir, "frames_vorticity")
    ensure_dir(frames_dir)
    frames = []
    for i in range(num_saved):
        fig = plt.figure(figsize=(6, 3))
        plt.imshow(
            vort_b[i].T,  # (x,y)->imshow는 y가 세로축이므로 보기 좋게 T
            origin="lower",
            extent=(-1, 1, -1, 1),
            vmin=-1, vmax=1,
            cmap=sns.cm.vlag,
            aspect="auto",
        )
        plt.title(f"Vorticity  t={t_saved[i]:.3f}")
        plt.axis("off")
        frame_path = os.path.join(frames_dir, f"vort_{i:04d}.png")
        plt.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frames.append(imageio.imread(frame_path))
    gif_path = os.path.join(out_dir, gif_name)
    imageio.mimsave(gif_path, frames, duration=0.08, loop=0)

    return {
        "nc_path": nc_path,
        "gif_path": gif_path,
        "num_saved": num_saved,
        "t_saved": t_saved,
        "x_vals": x_vals,
        "y_vals": y_vals,
    }

# ======================================================
# 2) Gray-Scott 호환 로더
#    (t_list, coords_list, y_list, y_list_full, coords_full)
#    여기서는 'vorticity'를 관측 y로 사용
# ======================================================
def load_torchcfd_nc_as_grayscott_compatible(
    nc_path: str,
    sample_ratio: float = 0.2,
    normalize_t: bool = False,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
):
    np.random.seed(seed)
    ds = xr.open_dataset(nc_path)

    # (time, x, y)
    vort = ds["vorticity"].transpose("time", "x", "y").values  # (T, nx, ny)
    tvals = ds["time"].values
    xvals = ds["x"].values
    yvals = ds["y"].values
    nx, ny = len(xvals), len(yvals)

    coords_full = make_coords_full_from_linspace(xvals, yvals)  # (nx*ny, 2)
    n = coords_full.shape[0]
    m = int(n * sample_ratio)
    idx = np.random.choice(n, size=m, replace=False)

    y_list, y_list_full, coords_list = [], [], []
    for k in range(vort.shape[0]):
        flat = vort[k].reshape(-1)  # (nx*ny,)
        y_list_full.append(flat.copy())
        y_list.append(flat[idx])
        coords_list.append(coords_full[idx])

    # 시간 정규화
    if normalize_t:
        t_list = [np.float32(t / tvals[-1]) for t in tvals]
    else:
        t_list = [np.float32(t) for t in tvals]

    # torch 변환 + (real, imag) 2채널
    import torch
    coords_torch = [torch.from_numpy(c).float().to(device) for c in coords_list]
    coords_full_torch = torch.from_numpy(coords_full).float().to(device)
    y_torch = to_torch_split_real_only(y_list, device)
    y_full_torch = to_torch_split_real_only(y_list_full, device)

    return t_list, coords_torch, y_torch, y_full_torch, coords_full_torch

# ======================================================
# 3) PyTorch Dataset (Gray-Scott과 동일)
# ======================================================
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

# ======================================================
# 4) 사용 예시
# ======================================================
if __name__ == "__main__":
    info = run_torch_cfd_sim_and_save(
        nx=32, ny=32,
        # nx=100, ny=100,
        dt=1e-3, T=20.0,
        update_steps=100,
        batch_index=0,
        out_dir="./dataset/navier_stokes_obstacle_flow",
        gif_name="obstacle_flow_vorticity_low_res.gif",
        nc_name="obstacle_flow_low_res.nc",
        batch_size=1,
    )
    print(f"[Saved] GIF: {info['gif_path']}")
    print(f"[Saved] NetCDF: {info['nc_path']}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_list, coords_list, y_list, y_list_full, coords_full = load_torchcfd_nc_as_grayscott_compatible(
        info["nc_path"], sample_ratio=0.1, normalize_t=True, device=device
    )
    dataset = VorticityDataset(t_list, coords_list, y_list)
    print(f"Dataset length: {len(dataset)}")
    tp, tn, c, yn, yp = dataset[0]
    print(f"Example shapes -> coords:{c.shape}, y_next:{yn.shape}, y_prev:{yp.shape}")