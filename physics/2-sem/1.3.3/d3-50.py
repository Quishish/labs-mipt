
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# НАСТРОЙКИ
# =========================
K = 0.2          # коэффициент наклона микроманометра
n_corr = 0.9953  # поправка n (по температуре)
DT_S = 30.0      # время измерения объёма (сек)  <-- ты сказал 30с
V_IS_LITERS = True

G = 9.8067       # ΔP[Па] = 9.8067 * N * K * n

# =========================
# ДАННЫЕ
# =========================
def make_tube_table(name, N, V, d_mm, l_cm):
    N = np.asarray(N, dtype=float)
    V = np.asarray(V, dtype=float)

    V_m3 = V * 1e-3 if V_IS_LITERS else V
    Q = V_m3 / DT_S
    dP = G * N * K * n_corr

    # ВНИМАНИЕ: d_mm трактуем как диаметр (как в методичке "d≈4 мм")
    R_m = (d_mm * 1e-3) / 2.0
    l_m = l_cm * 1e-2

    df = pd.DataFrame({
        "N_div": N,
        "dP_Pa": dP,
        "V_input": V,
        "Q_m3s": Q,
    })
    df.attrs["name"] = name
    df.attrs["d_mm"] = d_mm
    df.attrs["R_m"] = R_m
    df.attrs["l_m"] = l_m
    return df

data = {
    "d3(50)": make_tube_table(
        name="d3(50)",
        N=[10, 20, 30, 40, 50, 60, 69, 81, 90, 100, 110, 120, 127],
        V=[0.375, 0.777, 1.143, 1.549, 1.864, 2.305, 2.593, 2.864, 2.965, 3.019, 3.085, 3.202, 3.231],
        d_mm=3.95,
        l_cm=50
    ),
    "d2(30)": make_tube_table(
        name="d2(30)",
        N=[5,10,15,20,25,30,35],
        V=[0.254, 0.637, 0.986, 1.295, 1.700, 1.960, 2.222],
        d_mm=3.0,
        l_cm=30
    ),
    "d1(50)": make_tube_table(
        name="d1(50)",
        N=[5,10,15,20,25,30,40,50,60,70],
        V=[0.472, 1.104, 1.738, 2.358, 2.945, 3.577, 4.118, 4.370, 4.616, 4.797],
        d_mm=5.30,
        l_cm=50
    ),
}

# =========================
# МНК и выбор ламинарного участка
# =========================
def linear_fit_with_cov(x, y):
    (k, b), cov = np.polyfit(x, y, deg=1, cov=True)
    k_err = float(np.sqrt(cov[0, 0]))
    b_err = float(np.sqrt(cov[1, 1]))

    y_pred = k * x + b
    resid = y - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return float(k), float(b), k_err, b_err, float(R2)

def pick_laminar_prefix(dP, Q, min_points=4, r2_threshold=0.995):
    n = len(dP)
    best = None

    for k in range(min_points, n + 1):
        kk, bb, kk_err, bb_err, R2 = linear_fit_with_cov(dP[:k], Q[:k])
        item = (k, R2, kk, bb, kk_err, bb_err)
        if best is None:
            best = item
        else:
            if (item[1] >= r2_threshold) and (best[1] >= r2_threshold):
                if item[0] > best[0]:
                    best = item
            elif (item[1] >= r2_threshold) and (best[1] < r2_threshold):
                best = item
            elif (item[1] < r2_threshold) and (best[1] < r2_threshold):
                if item[1] > best[1]:
                    best = item

    k_pts, R2, k_slope, b_int, k_err, b_err = best
    mask = np.zeros(n, dtype=bool)
    mask[:k_pts] = True
    return mask, k_pts, R2, k_slope, b_int, k_err, b_err

def eta_from_slope(k_slope, k_err, R_m, l_m):
    # k = πR^4/(8ηl)  =>  η = πR^4/(8lk)
    eta = np.pi * (R_m**4) / (8.0 * l_m * k_slope)
    eta_err = abs(eta) * (k_err / abs(k_slope))
    return float(eta), float(eta_err)

# =========================
# АНАЛИЗ + ГРАФИКИ
# =========================
summary_rows = []

for key, df in data.items():
    name = df.attrs["name"]
    d_mm = df.attrs["d_mm"]
    R_m = df.attrs["R_m"]
    l_m = df.attrs["l_m"]

    dP = df["dP_Pa"].to_numpy()
    Q = df["Q_m3s"].to_numpy()

    lam_mask, lam_count, lam_R2, k_slope, b_int, k_err, b_err = pick_laminar_prefix(
        dP, Q,
        min_points=4 if len(dP) >= 6 else 3,
        r2_threshold=0.995
    )

    eta, eta_err = eta_from_slope(k_slope, k_err, R_m, l_m)

    dP_cr = dP[lam_count] if lam_count < len(dP) else np.nan

    summary_rows.append({
        "dataset": name,
        "d_mm": d_mm,
        "l_m": l_m,
        "lam_points": int(lam_count),
        "lam_R2": lam_R2,
        "k_slope_(m3/s)/Pa": k_slope,
        "k_err": k_err,
        "b_(m3/s)": b_int,
        "b_err": b_err,
        "eta_Pa_s": eta,
        "eta_err": eta_err,
        "dP_cr_est_Pa": dP_cr
    })

    # ---- График ----
    dP_line = np.linspace(dP.min(), dP.max(), 300)
    Q_line = k_slope * dP_line + b_int

    plt.figure()
    plt.plot(dP, Q, "o", label=f"{name}: эксперимент")
    plt.plot(dP[lam_mask], Q[lam_mask], "o", label=f"{name}: ламинарные точки (авто)")
    plt.plot(dP_line, Q_line, "-", label=f"МНК (ламинарн.): R^2={lam_R2:.3f}")
    if not np.isnan(dP_cr):
        plt.axvline(dP_cr, linestyle="--", label=f"граница (авто) ΔP≈{dP_cr:.1f} Па")
    plt.xlabel("ΔP, Па")
    plt.ylabel("Q, м³/с")
    plt.grid(True)
    plt.legend()
    plt.title(f"Q(ΔP) — {name}   (d={d_mm:.2f} мм, l={l_m:.2f} м, Δt={DT_S:g} c)")
    plt.show()

summary = pd.DataFrame(summary_rows)

print("\n=== ИТОГ ПО ВЯЗКОСТИ (по ламинарному МНК) ===")
print(summary[["dataset", "d_mm", "l_m", "lam_points", "lam_R2", "eta_Pa_s", "eta_err", "dP_cr_est_Pa"]])

print("\n=== Строки для отчёта (η ± δη) ===")
for _, r in summary.iterrows():
    print(
        f"{r['dataset']}: η = {r['eta_Pa_s']:.3e} ± {r['eta_err']:.1e} Па·с "
        f"(d={r['d_mm']:.2f} мм, l={r['l_m']:.2f} м, R²={r['lam_R2']:.3f}, N_lam={int(r['lam_points'])})"
    )
