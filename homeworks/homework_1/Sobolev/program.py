import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

BASE_DIR = Path(__file__).resolve().parent


def calc_ws(gamma_wat: float) -> float:
    ws = (
        1 / (gamma_wat * 1000)
        * (1.36545 * gamma_wat * 1000 - (3838.77 * gamma_wat * 1000 - 2.009 * (gamma_wat * 1000) ** 2) ** 0.5)
    )
    if ws > 0:
        return ws
    return 0


def calc_rho_w(ws: float, t: float) -> float:
    rho_w = 1000 * (1.0009 - 0.7114 * ws + 0.2605 * ws ** 2) ** (-1)
    return rho_w / (1 + (t - 273) * 1e-4 * (0.269 * (t - 273) ** 0.637 - 0.8))


def calc_mu_w(ws: float, t: float, p: float) -> float:
    a = (
        109.574
        - (0.840564 * 1000 * ws)
        + (3.13314 * 1000 * ws ** 2)
        + (8.72213 * 1000 * ws ** 3)
    )
    b = (
        1.12166
        - 2.63951 * ws
        + 6.79461 * ws ** 2
        + 54.7119 * ws ** 3
        - 155.586 * ws ** 4
    )
    mu_w = (
        a * (1.8 * t - 460) ** (-b)
        * (0.9994 + 0.0058 * (p * 1e-6) + 0.6534 * 1e-4 * (p * 1e-6) ** 2)
    )
    return mu_w


def calc_n_re(rho_w: float, q_ms: float, mu_w: float, d_tub: float) -> float:
    v = q_ms / (np.pi * d_tub ** 2 / 4)
    return rho_w * v * d_tub / mu_w * 1000


def calc_ff_churchill(n_re: float, roughness: float, d_tub: float) -> float:
    a = (-2.457 * np.log((7 / n_re) ** 0.9 + 0.27 * (roughness / d_tub))) ** 16
    b = (37530 / n_re) ** 16
    return 8 * ((8 / n_re) ** 12 + 1 / (a + b) ** 1.5) ** (1 / 12)


def calc_dp_dl_grav(rho_w: float, angle: float) -> float:
    return rho_w * 9.81 * np.sin(angle / 180 * np.pi)


def calc_dp_dl_fric(rho_w: float, mu_w: float, q_ms: float, d_tub: float, roughness: float) -> float:
    if q_ms != 0:
        n_re = calc_n_re(rho_w, q_ms, mu_w, d_tub)
        ff = calc_ff_churchill(n_re, roughness, d_tub)
        return ff * rho_w * q_ms ** 2 / d_tub ** 5
    return 0


def calc_dp_dl(rho_w: float, mu_w: float, angle: float, q_ms: float, d_tub: float, roughness: float) -> float:
    dp_dl_grav = calc_dp_dl_grav(rho_w, angle)
    dp_dl_fric = calc_dp_dl_fric(rho_w, mu_w, q_ms, d_tub, roughness)
    return dp_dl_grav - 0.815 * dp_dl_fric


def __integr_func(
    h: float,
    pt: tuple,
    temp_grad: float,
    gamma_wat: float,
    angle: float,
    q_ms: float,
    d_tub: float,
    roughness: float,
) -> tuple:
    p, t = pt
    ws = calc_ws(gamma_wat)
    rho_w = calc_rho_w(ws, t)
    mu_w = calc_mu_w(ws, t, p)

    dp_dl = calc_dp_dl(rho_w, mu_w, angle, q_ms, d_tub, roughness)
    dt_dl = temp_grad * 1e-2 * np.sin(angle / 180 * np.pi)

    return dp_dl, dt_dl


def calc_pipe(
    p_wh: float,
    t_wh: float,
    h0: float,
    md_vdp: float,
    temp_grad: float,
    gamma_wat: float,
    angle: float,
    q_ms: float,
    d_tub: float,
    roughness: float,
) -> tuple:
    sol = solve_ivp(
        __integr_func,
        t_span=(h0, md_vdp),
        y0=[p_wh, t_wh],
        args=(temp_grad, gamma_wat, angle, q_ms, d_tub, roughness),
    )
    return sol.y[0], sol.y[1], sol.t


def calc_p_wf(
    p_wh: float,
    t_wh: float,
    h0: float,
    md_vdp: float,
    temp_grad: float,
    gamma_wat: float,
    angle: float,
    q_ms: float,
    d_tub: float,
    roughness: float,
) -> float:
    p_res, _, _ = calc_pipe(
        p_wh, t_wh, h0, md_vdp, temp_grad, gamma_wat, angle, q_ms, d_tub, roughness
    )
    return p_res[-1]


if __name__ == "__main__":
    with open(BASE_DIR.parent / "input_data" / "12.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    q_vals = list(range(1, 401))
    p_wf_vals = []

    for q in q_vals:
        p_wf_pa = calc_p_wf(
            p_wh=data["p_wh"] * 101325,
            t_wh=data["t_wh"] + 273,
            h0=0,
            md_vdp=data["md_vdp"],
            temp_grad=data["temp_grad"],
            gamma_wat=data["gamma_water"],
            angle=data["angle"],
            q_ms=q / 86400,
            d_tub=data["d_tub"],
            roughness=data["roughness"],
        )
        p_wf_vals.append(float(p_wf_pa / 101325))

    result = {
        "q_liq": q_vals,
        "p_wf": p_wf_vals,
    }

    with open(BASE_DIR / "output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)