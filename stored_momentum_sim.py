"""Interactive spacecraft stored-momentum explorer.

A rigid spacecraft carries reaction wheels with a fixed (uncontrolled) stored
angular momentum h_w, expressed in the body frame. You set the principal
inertias Ixx, Iyy, Izz and the wheel momentum, then apply external torque
(body frame) and watch the attitude dynamics.

Dynamics (body frame, diagonal inertia I, constant wheel momentum h_w):

    H_body_total = I*w + h_w
    I*w_dot = tau_ext + tau_diss - w x (I*w + h_w)
    q_dot   = 1/2 * q (x) [0, w]

The inertial total angular momentum R(q)·(I*w + h_w) is conserved whenever
the external torque is zero — watch the black arrow stay fixed while the
blue/green arrows (body momentum and wheel momentum) precess around it.

Energy dissipation uses the classic "energy sink" model of internal dampers
(fuel slosh, flexible booms, nutation dampers):

    tau_diss = -c * w_perp,   w_perp = w - (w . H_hat) H_hat

i.e. rate damping perpendicular to the total momentum H_b = I*w + h_w.
This removes kinetic energy at rate c*|w_perp|^2 while conserving |H|
exactly, so nutation damps until w is parallel to H. With no wheel momentum
this drives a tumbling body to spin about its MAJOR inertia axis (the
Explorer 1 flat-spin effect); with wheel momentum it damps nutation after a
torque pulse. Energy-sink caveat: the *direction* of H in inertial space
precesses slowly (rate ~ c|w_perp|/|H|) because the damper's own momentum
is not modeled; its magnitude is exact.

Controls
--------
Sliders : Ixx, Iyy, Izz     principal inertias [kg m^2]
          hx, hy, hz        stored wheel momentum, body frame [N m s]
          tx, ty, tz        external torque, body frame [N m], applied
                            continuously while nonzero
          c diss            energy dissipation coefficient [N m s/rad]
Buttons : Pause/Run, Reset (attitude + rates), Zero torque,
          Zero wheel momentum,
          Pulse: apply the current torque sliders for 0.5 s, then zero them
Mouse   : drag in the 3D view to rotate the camera

Run:  python stored_momentum_sim.py
"""

from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider

# ----------------------------------------------------------------------
# quaternion helpers  (scalar-first, q maps body -> inertial)
# ----------------------------------------------------------------------

def quat_mult(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# ----------------------------------------------------------------------
# rigid-body simulation
# ----------------------------------------------------------------------

class SpacecraftSim:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t = 0.0
        self.q = np.array([1.0, 0.0, 0.0, 0.0])   # body -> inertial
        self.w = np.zeros(3)                       # body rates [rad/s]

    @staticmethod
    def w_perp(w, I, h_w):
        """Body-rate component perpendicular to H_b = I*w + h_w."""
        H_b = I * w + h_w
        H2 = H_b @ H_b
        if H2 < 1e-18:
            return w
        return w - (w @ H_b) / H2 * H_b

    def _deriv(self, q, w, I, h_w, tau, c):
        H_b = I * w + h_w
        tau_d = -c * self.w_perp(w, I, h_w)
        w_dot = (tau + tau_d - np.cross(w, H_b)) / I
        q_dot = 0.5 * quat_mult(q, np.array([0.0, *w]))
        return q_dot, w_dot

    def step(self, dt, I, h_w, tau, c):
        q, w = self.q, self.w
        k1q, k1w = self._deriv(q, w, I, h_w, tau, c)
        k2q, k2w = self._deriv(q + 0.5 * dt * k1q, w + 0.5 * dt * k1w, I, h_w, tau, c)
        k3q, k3w = self._deriv(q + 0.5 * dt * k2q, w + 0.5 * dt * k2w, I, h_w, tau, c)
        k4q, k4w = self._deriv(q + dt * k3q, w + dt * k3w, I, h_w, tau, c)
        self.q = q + dt / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
        self.w = w + dt / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
        self.q /= np.linalg.norm(self.q)
        self.t += dt


# ----------------------------------------------------------------------
# interactive application
# ----------------------------------------------------------------------

DT = 0.02          # integrator step [s]
SUBSTEPS = 3       # integrator steps per animation frame
LIM = 2.0          # half-width of the 3D view
TRAIL_LEN = 700    # points kept in the arrow-tip trails

COL_HTOT = "black"
COL_HBODY = "tab:blue"
COL_HWHEEL = "tab:green"
COL_OMEGA = "tab:red"


class App:
    def __init__(self):
        self.sim = SpacecraftSim()
        self.paused = False
        self.pulse_tau = np.zeros(3)
        self.pulse_t_end = -1.0

        # slowly-decaying maxima used to auto-scale the arrows without jitter
        self.h_max = 1e-9
        self.w_max = 1e-9

        self.trail_htot = deque(maxlen=TRAIL_LEN)
        self.trail_omega = deque(maxlen=TRAIL_LEN)
        self.trail_bodyz = deque(maxlen=TRAIL_LEN)

        self._build_figure()
        self._build_widgets()
        self.quivers = []

        self.anim = FuncAnimation(self.fig, self._update, interval=33,
                                  cache_frame_data=False)

    # ------------------------------------------------------------ layout
    def _build_figure(self):
        self.fig = plt.figure("Spacecraft stored momentum", figsize=(13, 8))
        self.ax = self.fig.add_axes([0.0, 0.26, 0.62, 0.72], projection="3d")
        self.ax.set_xlim(-LIM, LIM)
        self.ax.set_ylim(-LIM, LIM)
        self.ax.set_zlim(-LIM, LIM)
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.set_xlabel("X (inertial)")
        self.ax.set_ylabel("Y (inertial)")
        self.ax.set_zlabel("Z (inertial)")

        # persistent artists: spacecraft box, body triad, trails
        self.box_lines = [self.ax.plot([], [], [], color="0.45", lw=1.0)[0]
                          for _ in range(12)]
        self.triad_lines = [
            self.ax.plot([], [], [], color=c, lw=1.4, ls="--", alpha=0.9)[0]
            for c in ("r", "g", "b")
        ]
        self.line_trail_htot = self.ax.plot([], [], [], color=COL_HTOT,
                                            lw=0.8, alpha=0.35)[0]
        self.line_trail_omega = self.ax.plot([], [], [], color=COL_OMEGA,
                                             lw=0.8, alpha=0.35)[0]
        self.line_trail_bodyz = self.ax.plot([], [], [], color="0.5",
                                             lw=0.8, alpha=0.35)[0]

        proxies = [
            Line2D([], [], color=COL_HTOT, lw=2.5,
                   label="H total (inertial, conserved when τ=0)"),
            Line2D([], [], color=COL_HBODY, lw=2.5, label="H body = I·ω"),
            Line2D([], [], color=COL_HWHEEL, lw=2.5, label="h wheels"),
            Line2D([], [], color=COL_OMEGA, lw=2.5, label="ω body rate"),
            Line2D([], [], color="0.5", lw=1.2, ls="--",
                   label="body axes x,y,z (r,g,b)"),
        ]
        self.ax.legend(handles=proxies, loc="upper left",
                       bbox_to_anchor=(0.0, 1.02), fontsize=8, framealpha=0.85)

        self.info = self.fig.text(0.645, 0.955, "", va="top", ha="left",
                                  family="monospace", fontsize=9)

    def _build_widgets(self):
        def slider(col, row, name, vmin, vmax, vinit):
            rect = [0.06 + 0.25 * col, 0.155 - 0.055 * row, 0.15, 0.028]
            return Slider(self.fig.add_axes(rect), name, vmin, vmax,
                          valinit=vinit)

        self.s_I = [slider(0, r, n, 0.1, 10.0, v) for r, (n, v) in
                    enumerate([("Ixx", 2.0), ("Iyy", 4.0), ("Izz", 6.0)])]
        self.s_h = [slider(1, r, n, -20.0, 20.0, v) for r, (n, v) in
                    enumerate([("hx", 0.0), ("hy", 0.0), ("hz", 5.0)])]
        self.s_tau = [slider(2, r, n, -2.0, 2.0, 0.0) for r, n in
                      enumerate([("τx", "τy", "τz")[i] for i in range(3)])]
        self.s_c = slider(3, 0, "c diss", 0.0, 1.0, 0.0)

        self.fig.text(0.79, 0.115,
                      "Pulse: applies slider τ for 0.5 s,\n"
                      "  then zeros it.\n"
                      "c diss: internal damping (energy\n"
                      "  sink) τ_d = -c·ω⊥ — removes\n"
                      "  kinetic energy, conserves |H|.",
                      va="top", fontsize=8, color="0.35")

        def button(i, label, cb):
            rect = [0.60 + 0.078 * i, 0.205, 0.072, 0.045]
            b = Button(self.fig.add_axes(rect), label)
            b.label.set_fontsize(8.5)
            b.on_clicked(cb)
            return b

        self.b_pause = button(0, "Pause", self._on_pause)
        self.b_reset = button(1, "Reset", self._on_reset)
        self.b_zero = button(2, "Zero τ", self._on_zero_tau)
        self.b_zeroh = button(3, "Zero h", self._on_zero_h)
        self.b_pulse = button(4, "Pulse τ 0.5s", self._on_pulse)

    # ------------------------------------------------------------ callbacks
    def _on_pause(self, _event):
        self.paused = not self.paused
        self.b_pause.label.set_text("Run" if self.paused else "Pause")

    def _on_reset(self, _event):
        self.sim.reset()
        self.pulse_t_end = -1.0
        self.h_max = self.w_max = 1e-9
        self.trail_htot.clear()
        self.trail_omega.clear()
        self.trail_bodyz.clear()

    def _on_zero_tau(self, _event):
        for s in self.s_tau:
            s.set_val(0.0)

    def _on_zero_h(self, _event):
        for s in self.s_h:
            s.set_val(0.0)

    def _on_pulse(self, _event):
        self.pulse_tau = np.array([s.val for s in self.s_tau])
        self.pulse_t_end = self.sim.t + 0.5
        self._on_zero_tau(None)

    # ------------------------------------------------------------ helpers
    def _params(self):
        I = np.array([s.val for s in self.s_I])
        h_w = np.array([s.val for s in self.s_h])
        tau = np.array([s.val for s in self.s_tau])
        if self.sim.t < self.pulse_t_end:
            tau = tau + self.pulse_tau
        return I, h_w, tau, self.s_c.val

    @staticmethod
    def _box_dims(I):
        """Side lengths of the unit-mass cuboid with these inertias."""
        Ix, Iy, Iz = I
        L = np.sqrt(np.maximum([6 * (Iy + Iz - Ix),
                                6 * (Ix + Iz - Iy),
                                6 * (Ix + Iy - Iz)], 0.05))
        return L / max(L.max(), 1e-9) * 1.5   # normalize: longest side 1.5

    _BOX_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                  (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]

    def _draw_box(self, R, I):
        L = self._box_dims(I) / 2.0
        corners = np.array([[sx * L[0], sy * L[1], sz * L[2]]
                            for sx in (-1, 1) for sy in (-1, 1)
                            for sz in (-1, 1)])          # index bits: x,y,z
        corners = corners @ R.T
        for line, (i, j) in zip(self.box_lines, self._BOX_EDGES):
            seg = np.array([corners[i], corners[j]])
            line.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
        for k, line in enumerate(self.triad_lines):
            tip = R[:, k] * 1.3
            line.set_data_3d([0, tip[0]], [0, tip[1]], [0, tip[2]])

    def _draw_trail(self, line, trail, scale):
        if len(trail) < 2:
            line.set_data_3d([], [], [])
            return
        a = np.asarray(trail) * scale
        line.set_data_3d(a[:, 0], a[:, 1], a[:, 2])

    # ------------------------------------------------------------ main loop
    def _update(self, _frame):
        I, h_w, tau, c = self._params()

        if not self.paused:
            for _ in range(SUBSTEPS):
                I, h_w, tau, c = self._params()
                self.sim.step(DT, I, h_w, tau, c)

        R = quat_to_rot(self.sim.q)
        w_b = self.sim.w
        Hb_b = I * w_b                    # body's own momentum, body frame
        Ht_b = Hb_b + h_w                 # total, body frame

        Ht_i = R @ Ht_b                   # inertial-frame vectors for display
        Hb_i = R @ Hb_b
        hw_i = R @ h_w
        w_i = R @ w_b

        if not self.paused:
            self.trail_htot.append(Ht_i)
            self.trail_omega.append(w_i)
            self.trail_bodyz.append(R[:, 2])

        # arrow auto-scaling with a slowly decaying peak-hold
        self.h_max = max(np.linalg.norm(Ht_i), np.linalg.norm(Hb_i),
                         np.linalg.norm(hw_i), self.h_max * 0.998, 1e-9)
        self.w_max = max(np.linalg.norm(w_i), self.w_max * 0.998, 1e-9)
        sH = 0.85 * LIM / self.h_max
        sW = 0.85 * LIM / self.w_max

        for artist in self.quivers:
            artist.remove()
        self.quivers = [
            self.ax.quiver(0, 0, 0, *(Ht_i * sH), color=COL_HTOT, lw=2.5,
                           arrow_length_ratio=0.07),
            self.ax.quiver(0, 0, 0, *(Hb_i * sH), color=COL_HBODY, lw=2.2,
                           arrow_length_ratio=0.07),
            self.ax.quiver(0, 0, 0, *(hw_i * sH), color=COL_HWHEEL, lw=2.2,
                           arrow_length_ratio=0.07),
            # wheel momentum drawn again from the body-momentum tip:
            # blue + green tip-to-tail = black (H total)
            self.ax.quiver(*(Hb_i * sH), *(hw_i * sH), color=COL_HWHEEL,
                           lw=1.0, alpha=0.45, arrow_length_ratio=0.07),
            self.ax.quiver(0, 0, 0, *(w_i * sW), color=COL_OMEGA, lw=2.2,
                           arrow_length_ratio=0.07),
        ]
        if self.sim.t < self.pulse_t_end:
            tau_i = R @ tau
            n = np.linalg.norm(tau_i)
            if n > 1e-9:
                self.quivers.append(
                    self.ax.quiver(0, 0, 0, *(tau_i / n * 0.9 * LIM),
                                   color="orange", lw=3.0,
                                   arrow_length_ratio=0.1))

        self._draw_box(R, I)
        self._draw_trail(self.line_trail_htot, self.trail_htot, sH)
        self._draw_trail(self.line_trail_omega, self.trail_omega, sW)
        self._draw_trail(self.line_trail_bodyz, self.trail_bodyz, 1.3)

        T_kin = 0.5 * float(w_b @ (I * w_b))
        P_diss = c * float(np.sum(self.sim.w_perp(w_b, I, h_w) ** 2))
        self.info.set_text(
            f"t = {self.sim.t:8.2f} s"
            f"{'   [PAUSED]' if self.paused else ''}\n\n"
            f"H total (inertial) [N·m·s]\n"
            f"  [{Ht_i[0]:8.3f} {Ht_i[1]:8.3f} {Ht_i[2]:8.3f}]"
            f"  |H| = {np.linalg.norm(Ht_i):7.3f}\n"
            f"H body = I·ω        |H_b| = {np.linalg.norm(Hb_b):7.3f}\n"
            f"h wheels            |h_w| = {np.linalg.norm(h_w):7.3f}\n\n"
            f"ω body [rad/s]\n"
            f"  [{w_b[0]:8.4f} {w_b[1]:8.4f} {w_b[2]:8.4f}]"
            f"  |ω| = {np.linalg.norm(w_b):7.4f}\n"
            f"T_rot = {T_kin:9.4f} J   "
            f"dT/dt = {-P_diss:9.5f} W\n"
            f"τ applied (body) = [{tau[0]:5.2f} {tau[1]:5.2f} {tau[2]:5.2f}]"
            f" N·m\n\n"
            f"Arrows auto-scale; read magnitudes here.\n"
            f"Blue + green (tip-to-tail) = black.\n\n"
            f"Try:  h=0, τx pulse  -> spin about x\n"
            f"      hz=5, τx pulse -> precession, not tumble\n"
            f"      hz=15          -> stiffer gyroscope\n"
            f"      h=0, spin about Iyy (intermediate\n"
            f"      axis) -> Dzhanibekov instability\n"
            f"      hz=5, τx pulse, c>0 -> nutation damps,\n"
            f"      ω realigns with H\n"
            f"      h=0, spin about Ixx (minor axis),\n"
            f"      c>0 -> flat spin about major axis"
        )
        return self.quivers


def main():
    app = App()
    plt.show()
    return app


if __name__ == "__main__":
    main()
