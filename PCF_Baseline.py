"""
============================================================
  Phase Field Crystal (PFC) Simulation
  Crystal Growth from a Hexagonal Seed

  Based on: elvissoares/PyPFC (github.com/elvissoares/PyPFC)
  Modified by: Ayush Shah & Tobias Li
  Georgia Institute of Technology
  NSF IRES Physical AI Design Program — Prof. Bo Zhu
============================================================

Run: Runtime > Run all  (~2-3 min on Colab CPU)
"""

# ── Cell 1: Imports ──────────────────────────────────────────
import numpy as np
from scipy.fft import fft2, ifft2
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "serif",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})
print("Libraries loaded.")

# ── Cell 2: PFC Parameters ───────────────────────────────────
# Physics
r    = -0.28    # temperature param: more negative = faster crystallization
M    =  1.0     # mobility
n0   = -0.285   # mean density (sits in hexagonal crystal phase)

# Grid
N    = 128              # grid points per side (128x128 — fast on CPU)
L    = 16 * np.pi       # domain size in PFC units
x    = np.linspace(0, L, N, endpoint=False)
dx   = x[1] - x[0]

# Time
dt        = 0.25         # timestep (semi-implicit allows large dt)
T         = 250.0      # total simulation time
Nsteps    = int(T / dt)
save_every = 25         # save a frame every this many steps
Nframes   = Nsteps // save_every

print(f"Grid: {N}x{N}  |  Steps: {Nsteps}  |  Frames: {Nframes}")
print(f"Domain size L = {L:.2f}  |  dx = {dx:.4f}")

# ── Cell 3: Fourier Setup ────────────────────────────────────
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
k  = np.array(np.meshgrid(kx, kx, indexing='ij'), dtype=np.float32)
k2 = np.sum(k * k, axis=0, dtype=np.float32)

# 2/3 dealiasing mask — zeros out high-freq modes that cause aliasing errors
kmax_dealias = kx.max() * 2.0 / 3.0
dealias = np.array(
    (np.abs(k[0]) < kmax_dealias) * (np.abs(k[1]) < kmax_dealias),
    dtype=bool
)

# Linear operator: eigenvalue of the linear PFC PDE in Fourier space
# dpsi/dt = M * nabla^2 * [ (r + (1+nabla^2)^2) * psi + psi^3 ]
# Linear part eigenvalue: -M * k^2 * (k^4 - 2k^2 + 1 + r) = -M*k^2*(k^2-1)^2 + M*k^2*r...
# Expanded: k^4 - 2k^2 + 1 = (k^2 - 1)^2, so:
L_operator = -M * k2 * (k2**2 - 2*k2 + 1 + r)

# Semi-implicit denominator (computed once, reused every step)
linear_denom = 1.0 / (1.0 - dt * L_operator)

print("Fourier operators ready.")
print(f"L_operator range: {L_operator.min():.2f} to {L_operator.max():.2f}")

# ── Cell 4: Nonlinear Operator ───────────────────────────────
def nonlinear_operator(n_field):
    """
    Computes the Fourier transform of the nonlinear term: -M * nabla^2(n^3)
    In Fourier space: -M * (-k^2) * fft(n^3) = M * k^2 * fft(n^3)
    Dealising zeros out high-frequency modes to prevent aliasing instability.
    """
    return -(k2 * M * fft2(n_field**3)) * dealias

# ── Cell 5: Initial Condition ────────────────────────────────
rng = np.random.default_rng(42)

# Storage
n_all  = np.zeros((Nframes, N, N), dtype=np.float32)
n_hat  = np.empty((N, N), dtype=np.complex64)

# Start: uniform liquid + small noise
n_init = n0 + 0.01 * rng.standard_normal((N, N))

# Plant a hexagonal crystal seed at the centre
# This means we don't have to wait for random nucleation
SX, SY   = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
cx, cy   = N // 2, N // 2
R2       = (SX - cx)**2 + (SY - cy)**2
sigma    = (N // 10)**2          # seed radius in grid units
envelope = np.exp(-R2 / (2 * sigma))

# Hexagonal pattern: three cosine waves at 0, 60, 120 degrees
# k0 = 1 in PFC units, but in physical space k0 = 2*pi / lattice_spacing
# lattice spacing = dx * (N / num_lattice_cells), here ~4*dx
k0_phys = 2 * np.pi / (4 * dx)  # adjust to get visible lattice
hex_pattern = (
    np.cos(k0_phys * x[SX]) +
    2 * np.cos(0.5 * k0_phys * x[SX]) * np.cos(np.sqrt(3)/2 * k0_phys * x[SY])
)

n_init += 0.20 * envelope * hex_pattern
n_all[0] = n_init.astype(np.float32)

plt.figure(figsize=(5, 4))
plt.imshow(n_all[0], cmap='viridis', origin='lower')
plt.colorbar(label=r'$n(x,y)$', shrink=0.8)
plt.title(f'Initial condition  (t = 0,  $n_0 = {n0}$)', fontweight='bold')
plt.tight_layout()
plt.savefig('pfc_initial.png', dpi=130, bbox_inches='tight')
plt.show()
print(f"Initial field: min={n_all[0].min():.3f}  max={n_all[0].max():.3f}")

# ── Cell 6: Run the PFC Solver ───────────────────────────────
import time

print(f"\nRunning PFC simulation ({Nsteps} steps)...")
print("This takes ~2-3 minutes on Colab CPU. Please wait.\n")

nn      = n_init.copy()
n_hat[:] = fft2(nn)
NL_hat  = n_hat.copy()

frame_idx = 1
t0 = time.time()

for i in range(1, Nsteps):
    # Compute nonlinear term in Fourier space
    NL_hat[:] = nonlinear_operator(nn)

    # Semi-implicit update: treat linear part implicitly, nonlinear explicitly
    # n_hat_new = (n_hat_old + dt * NL_hat) / (1 - dt * L_operator)
    n_hat[:] = (n_hat + dt * NL_hat) * linear_denom

    # Inverse FFT back to real space
    nn[:] = ifft2(n_hat).real

    # Save frame
    if i % save_every == 0 and frame_idx < Nframes:
        n_all[frame_idx] = nn.astype(np.float32)
        elapsed = time.time() - t0
        pct = 100 * i / Nsteps
        print(f"  {pct:5.1f}%  |  step {i:5d}/{Nsteps}  |  "
              f"field range [{nn.min():.3f}, {nn.max():.3f}]  |  "
              f"{elapsed:.0f}s elapsed", end='\r')
        frame_idx += 1

print(f"\n\nDone in {time.time()-t0:.1f}s.  Frames saved: {frame_idx}")

# Mass conservation check
N0_mean    = n_all[0].mean()
Nlast_mean = n_all[frame_idx-1].mean()
print(f"Mass conservation check: initial mean = {N0_mean:.4f}, "
      f"final mean = {Nlast_mean:.4f}, "
      f"relative drift = {abs(Nlast_mean/N0_mean - 1)*100:.4f}%")

# ── Cell 7: Plot Growth Sequence ─────────────────────────────

# Custom colormap: navy (liquid) -> gold (crystal)
cmap_pfc = LinearSegmentedColormap.from_list("pfc", [
    (0.00, "#0a1628"),
    (0.35, "#1a3a6b"),
    (0.58, "#c8a84b"),
    (0.78, "#e8d5a3"),
    (1.00, "#ffffff"),
])

n_show   = 6
indices  = np.linspace(0, frame_idx - 1, n_show, dtype=int)
times    = indices * save_every * dt

vmin = np.percentile(n_all[frame_idx-1], 2)
vmax = np.percentile(n_all[frame_idx-1], 98)

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes = axes.flatten()

for col, (idx, t_val) in enumerate(zip(indices, times)):
    ax = axes[col]
    im = ax.imshow(n_all[idx], cmap=cmap_pfc,
                   vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title(f"t = {t_val:.0f}", fontweight='bold')
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=r'$n(x,y)$')

fig.suptitle(
    f"PFC Crystal Growth from Hexagonal Seed\n"
    f"$r = {r}$,  $n_0 = {n0}$,  $N = {N}\\times{N}$",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('pfc_growth_sequence.png', dpi=150, bbox_inches='tight')
plt.show()
print("Growth sequence saved.")

# ── Cell 8: Free Energy Dissipation ──────────────────────────
print("Computing free energy over time...")

F      = np.zeros(frame_idx)
t_vals = np.arange(frame_idx) * save_every * dt
lapn   = np.empty((N, N), dtype=np.float32)
laplapn = np.empty((N, N), dtype=np.float32)

for i in range(frame_idx):
    nh      = fft2(n_all[i])
    lapn[:] = ifft2(-k2 * nh).real
    laplapn[:] = ifft2(k2**2 * nh).real
    F[i]    = np.sum(
        n_all[i] * (lapn + 0.5 * laplapn)
        + 0.5 * (1 + r) * n_all[i]**2
        + 0.25 * n_all[i]**4
    ) * dx**2

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(t_vals, F / L**2, color='#1a3a6b', linewidth=2)
ax.fill_between(t_vals, F / L**2, alpha=0.15, color='#1a3a6b')
ax.set_xlabel("Time $t$")
ax.set_ylabel(r"$\mathcal{F}[n] \;/\; L^2$")
ax.set_title("Free Energy Dissipates as Crystal Grows", fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pfc_free_energy.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Energy drop: {F[0]/L**2:.5f} -> {F[frame_idx-1]/L**2:.5f}  "
      f"({100*(F[0]-F[frame_idx-1])/abs(F[0]):.1f}% reduction)")

# ── Cell 9: Grain Boundary Extraction ────────────────────────
final = n_all[frame_idx - 1]

# Isolate crystal oscillations near k0 using a ring filter in Fourier space
k0_target = k0_phys
ring_mask  = (k2 > (0.5 * k0_target)**2) & (k2 < (1.5 * k0_target)**2)
nh_final   = fft2(final)
n_crystal  = np.real(ifft2(nh_final * ring_mask))

# Local amplitude envelope (proxy for solid fraction)
amplitude  = ndimage.gaussian_filter(np.abs(n_crystal), sigma=3.0)
amplitude /= amplitude.max()

# Label grains
solid_mask = amplitude > 0.3
labeled, n_grains = ndimage.label(solid_mask)
boundaries = solid_mask ^ ndimage.binary_erosion(solid_mask)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].imshow(final, cmap=cmap_pfc, origin='lower', vmin=vmin, vmax=vmax)
axes[0].set_title("Final Density Field $n(x,y)$", fontweight='bold')
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

axes[1].imshow(amplitude, cmap='magma', origin='lower')
axes[1].set_title("Crystal Amplitude Envelope", fontweight='bold')
axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

gv = labeled.astype(float); gv[gv == 0] = np.nan
axes[2].imshow(final, cmap='Greys_r', alpha=0.4, origin='lower')
axes[2].imshow(gv, cmap='tab20', alpha=0.75, origin='lower')
axes[2].set_title(f"Grain Network  ({n_grains} grains detected)",
                  fontweight='bold')
axes[2].set_xlabel("x"); axes[2].set_ylabel("y")

plt.suptitle("Grain Boundary Extraction", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('pfc_grains.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Grains detected: {n_grains}")

# ── Cell 10: Percolation Analysis ────────────────────────────
print("Running percolation analysis...")

thresholds = np.linspace(0.05, 0.95, 80)
perc_flags = []

for thresh in thresholds:
    mask          = amplitude > thresh
    lbl, _        = ndimage.label(mask)
    top_grains    = set(lbl[0,  :].ravel()) - {0}
    bottom_grains = set(lbl[-1, :].ravel()) - {0}
    left_grains   = set(lbl[:,  0].ravel()) - {0}
    right_grains  = set(lbl[:, -1].ravel()) - {0}
    percolates    = bool(top_grains & bottom_grains) or \
                    bool(left_grains & right_grains)
    perc_flags.append(int(percolates))

perc_flags = np.array(perc_flags)
crossing   = np.where(np.diff(perc_flags) != 0)[0]
phi_c      = thresholds[crossing[0]] if len(crossing) > 0 else None

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(thresholds, perc_flags,
             color='#1a3a6b', linewidth=2.5, drawstyle='steps-post')
axes[0].fill_between(thresholds, perc_flags,
                     alpha=0.12, color='#1a3a6b', step='post')
if phi_c is not None:
    axes[0].axvline(phi_c, color='#c8a84b', linewidth=2,
                    linestyle='--', label=f'$\\phi_c \\approx {phi_c:.2f}$')
    axes[0].legend(fontsize=11)
axes[0].set_xlabel("Solid fraction threshold $\\phi$")
axes[0].set_ylabel("Percolating  (1 = yes)")
axes[0].set_title("Percolation Transition", fontweight='bold')
axes[0].set_ylim(-0.05, 1.15)
axes[0].grid(True, alpha=0.3)

if phi_c is not None:
    mask_pc     = amplitude > phi_c
    lbl_pc, npc = ndimage.label(mask_pc)
    gv_pc       = lbl_pc.astype(float); gv_pc[gv_pc == 0] = np.nan
    axes[1].imshow(final, cmap='Greys_r', alpha=0.35, origin='lower')
    axes[1].imshow(gv_pc, cmap='tab20', alpha=0.75, origin='lower')
    axes[1].set_title(
        f"Crystal at $\\phi_c \\approx {phi_c:.2f}$\n"
        f"First fully connected solid  ({npc} grains)",
        fontweight='bold'
    )
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

plt.suptitle("Percolation Analysis: When Does the Crystal Become One Solid?",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('pfc_percolation.png', dpi=150, bbox_inches='tight')
plt.show()

if phi_c is not None:
    print(f"Percolation threshold phi_c = {phi_c:.3f}")

# ── Cell 11: Save Animation as GIF ───────────────────────────
print("Generating animation...")

fig_anim, ax_anim = plt.subplots(1, 1, figsize=(5, 4.5))
im_anim = ax_anim.imshow(n_all[0], cmap=cmap_pfc,
                          vmin=vmin, vmax=vmax, origin='lower')
fig_anim.colorbar(im_anim, ax=ax_anim,
                  label=r'$n(x,y)$', shrink=0.8)
time_text = ax_anim.text(
    0.03, 0.95, 't = 0', transform=ax_anim.transAxes,
    fontsize=10, color='white', fontweight='bold',
    verticalalignment='top',
    bbox=dict(boxstyle='round', fc='#1a3a6b', alpha=0.7)
)
ax_anim.set_title(
    f"PFC Crystal Growth  ($r={r}$, $n_0={n0}$)",
    fontweight='bold'
)
ax_anim.set_xlabel("x"); ax_anim.set_ylabel("y")

def animate(i):
    im_anim.set_data(n_all[i])
    time_text.set_text(f't = {i * save_every * dt:.0f}')
    return im_anim, time_text

ani = animation.FuncAnimation(
    fig_anim, animate,
    frames=frame_idx, interval=240, blit=True
)
ani.save('pfc_crystal_growth.gif',
         writer='pillow', fps=10, dpi=110)
plt.close()
print("Animation saved as pfc_crystal_growth.gif")

# ── Cell 12: Summary Figure ───────────────────────────────────
from matplotlib import gridspec

fig  = plt.figure(figsize=(14, 9))
gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

snap_idx   = [0, frame_idx//3, frame_idx-1]
snap_titles = ["Initial Seed", "Growing Crystal", "Final Microstructure"]

for col, (sidx, title) in enumerate(zip(snap_idx, snap_titles)):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(n_all[sidx], cmap=cmap_pfc,
              vmin=vmin, vmax=vmax, origin='lower')
    t_label = sidx * save_every * dt
    ax.set_title(f"{title}\n$t = {t_label:.0f}$",
                 fontweight='bold', fontsize=10)
    ax.set_xlabel("x", fontsize=8); ax.set_ylabel("y", fontsize=8)
    ax.tick_params(labelsize=7)

ax_fe = fig.add_subplot(gs[1, 0])
ax_fe.plot(t_vals, F / L**2, color='#1a3a6b', linewidth=2)
ax_fe.fill_between(t_vals, F / L**2, alpha=0.15, color='#1a3a6b')
ax_fe.set_xlabel("Time $t$", fontsize=8)
ax_fe.set_ylabel(r"$\mathcal{F}/L^2$", fontsize=9)
ax_fe.set_title("Free Energy Dissipation", fontweight='bold', fontsize=10)
ax_fe.grid(True, alpha=0.3); ax_fe.tick_params(labelsize=7)

ax_gr = fig.add_subplot(gs[1, 1])
ax_gr.imshow(final, cmap='Greys_r', alpha=0.35, origin='lower')
gv2 = labeled.astype(float); gv2[gv2 == 0] = np.nan
ax_gr.imshow(gv2, cmap='tab20', alpha=0.8, origin='lower')
ax_gr.set_title(f"Grain Network\n({n_grains} grains)",
                fontweight='bold', fontsize=10)
ax_gr.set_xlabel("x", fontsize=8); ax_gr.set_ylabel("y", fontsize=8)
ax_gr.tick_params(labelsize=7)

ax_pc = fig.add_subplot(gs[1, 2])
ax_pc.plot(thresholds, perc_flags,
           color='#1a3a6b', linewidth=2.5, drawstyle='steps-post')
ax_pc.fill_between(thresholds, perc_flags,
                   alpha=0.12, color='#1a3a6b', step='post')
if phi_c is not None:
    ax_pc.axvline(phi_c, color='#c8a84b', linewidth=2,
                  linestyle='--', label=f'$\\phi_c={phi_c:.2f}$')
    ax_pc.legend(fontsize=9)
ax_pc.set_xlabel("Solid fraction $\\phi$", fontsize=8)
ax_pc.set_ylabel("Percolating", fontsize=8)
ax_pc.set_title("Percolation Transition", fontweight='bold', fontsize=10)
ax_pc.set_ylim(-0.05, 1.15); ax_pc.grid(True, alpha=0.3)
ax_pc.tick_params(labelsize=7)

fig.suptitle(
    "PFC Crystal Growth Simulation  —  Proof of Concept Render\n"
    "Ayush Shah & Tobias Li  |  NSF IRES Physical AI Design Program",
    fontsize=12, fontweight='bold'
)
plt.savefig('pfc_summary.png', dpi=160, bbox_inches='tight')
plt.show()
print("\nAll figures and animation saved successfully.")
print("Files: pfc_initial.png, pfc_growth_sequence.png, pfc_free_energy.png,")
print("       pfc_grains.png, pfc_percolation.png, pfc_crystal_growth.gif, pfc_summary.png")
