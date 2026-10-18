"""
Муха рисует в 3D: модель мухи летает в пространстве, а её телом
управляет нейросеть, собранная из таблицы связей коннектома FlyWire.
Камера медленно облетает вокруг, след светится.

Запуск (из папки с файлами):
  python fly_draw_3d.py 1 connections_princeton_csv.gz

Результат: fly3d_seedN.gif (анимация) и fly3d_seedN.png (кадр-рисунок).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
DATA = sys.argv[2] if len(sys.argv) > 2 else None
N = 1500
FRAMES = int(os.environ.get("FRAMES", 450))
STEPS_PER_FRAME = 8
rng = np.random.default_rng(SEED)


def load_real(path, n=N):
    """n самых связанных нейронов FlyWire; знак связи по нейромедиатору:
    GABA и GLUT тормозят, остальные возбуждают."""
    import pandas as pd
    df = pd.read_feather(path) if path.endswith(".feather") else pd.read_csv(path)
    pre = next(c for c in df.columns if "pre" in c.lower())
    post = next(c for c in df.columns if "post" in c.lower())
    wcol = next(c for c in df.columns if "syn" in c.lower())
    ids = df[pre].value_counts().index[:n]
    idx = {k: i for i, k in enumerate(ids)}
    df = df[df[pre].isin(idx) & df[post].isin(idx)]
    w = df[wcol].values.astype(float)
    if "nt_type" in df.columns:
        w = w * np.where(df["nt_type"].isin(["GABA", "GLUT"]), -1.0, 1.0)
    W = np.zeros((len(ids), len(ids)))
    np.add.at(W, (df[post].map(idx).values, df[pre].map(idx).values), w)
    return W


def make_demo(n=N):
    """Заглушка: случайная сеть. Честно называй её демо!"""
    W = (rng.random((n, n)) < 0.02) * rng.random((n, n))
    return W * np.where(rng.random((n, n)) < 0.25, -1.0, 1.0)


W = load_real(DATA) if DATA else make_demo()
n = W.shape[0]
print("Нейронов:", n, "| связей:", int((W != 0).sum()),
      "| режим:", "РЕАЛЬНЫЕ ДАННЫЕ" if DATA else "ДЕМО")
W = W * (1.6 / max(np.abs(np.linalg.eigvals(W)).max(), 1e-9))

grp = rng.choice(n, 4 * 30, replace=False).reshape(4, 30)
yaw_L, yaw_R, pit_U, pit_D = grp         # 4 группы "моторных" нейронов
sense = rng.choice(n, 60, replace=False)  # "глаза"

x = rng.normal(size=n) * 0.1
pos = np.zeros(3)
yaw = rng.uniform(0, 2 * np.pi)
pitch = 0.0
trail = [pos.copy()]
t = 0
R = 7.0


def step():
    global x, pos, yaw, pitch, t
    t += 1
    inp = np.zeros(n)
    inp[sense] = 0.6 * np.sin(0.05 * t + np.arange(len(sense)))
    x = x + 0.1 * (-x + W @ np.tanh(x) + inp) + 0.02 * rng.normal(size=n)
    yaw += 0.45 * np.tanh(2.0 * (x[yaw_L].mean() - x[yaw_R].mean()))
    pitch += 0.6 * np.tanh(2.0 * (x[pit_U].mean() - x[pit_D].mean()))
    pitch = float(np.clip(pitch, -1.1, 1.1))
    r = np.linalg.norm(pos)
    if r > R:                                # не вылетаем за границы
        yaw += 0.3 * np.sin(np.arctan2(-pos[1], -pos[0]) - yaw)
        pitch += 0.3 * (np.arcsin(np.clip(-pos[2] / r, -1, 1)) - pitch)
    speed = 0.10 + 0.06 * np.tanh(np.abs(x).mean())
    d = np.array([np.cos(pitch) * np.cos(yaw),
                  np.cos(pitch) * np.sin(yaw),
                  np.sin(pitch)])
    pos = pos + speed * d
    trail.append(pos.copy())


BG = "#0b0d17"
fig = plt.figure(figsize=(7, 7), facecolor=BG)
ax = fig.add_subplot(111, projection="3d", facecolor=BG)
ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
ax.set_box_aspect((1, 1, 1), zoom=1.55)
ax.set_axis_off()

lc = Line3DCollection([[(0, 0, 0), (0.01, 0, 0)]], cmap="plasma", linewidths=1.1)
ax.add_collection3d(lc)
(bodyL,) = ax.plot([], [], [], color="white", lw=4, solid_capstyle="round")
(headP,) = ax.plot([], [], [], "o", color="#ff4d4d", ms=6)
(wingA,) = ax.plot([], [], [], color="#8fd3ff", lw=2.5)
(wingB,) = ax.plot([], [], [], color="#8fd3ff", lw=2.5)


def draw_fly():
    d = np.array([np.cos(pitch) * np.cos(yaw),
                  np.cos(pitch) * np.sin(yaw),
                  np.sin(pitch)])
    s = np.cross(d, [0, 0, 1.0])
    s /= np.linalg.norm(s) + 1e-9
    up = np.cross(s, d)
    a, b = pos - 0.45 * d, pos + 0.45 * d
    bodyL.set_data_3d([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])
    h = pos + 0.55 * d
    headP.set_data_3d([h[0]], [h[1]], [h[2]])
    for w, sign in ((wingA, 1), (wingB, -1)):
        tip = pos - 0.2 * d + sign * 0.9 * s + 0.35 * up
        w.set_data_3d([pos[0], tip[0]], [pos[1], tip[1]], [pos[2], tip[2]])


def update(f):
    for _ in range(STEPS_PER_FRAME):
        step()
    a = np.array(trail)
    segs = np.stack([a[:-1], a[1:]], axis=1)
    lc.set_segments(segs)
    lc.set_array(np.linspace(0, 1, len(segs)))
    draw_fly()
    ax.view_init(elev=18 + 6 * np.sin(f / 60), azim=30 + 0.5 * f)
    return lc, bodyL, headP, wingA, wingB


ani = FuncAnimation(fig, update, frames=FRAMES, blit=False)
ani.save(f"fly3d_seed{SEED}.gif", writer=PillowWriter(fps=24))
fig.savefig(f"fly3d_seed{SEED}.png", dpi=220, facecolor=BG)
print("Готово:", f"fly3d_seed{SEED}.gif", f"fly3d_seed{SEED}.png")
