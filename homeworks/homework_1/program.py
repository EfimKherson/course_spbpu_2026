import json
import numpy as np
from scipy.integrate import solve_ivp


def calc_ws(gamma_wat: float) -> float:
    """Расчет солесодержания в воде, г/г."""
    ws = (
        1 / (gamma_wat * 1000)
        * (1.36545 * gamma_wat * 1000 - (3838.77 * gamma_wat * 1000 - 2.009 * (gamma_wat * 1000) ** 2) ** 0.5)
    )
    return ws if ws > 0 else 0.0


def calc_rho_w(ws: float, t: float) -> float:
    """Расчет плотности воды в зависимости от температуры (К) и солесодержания (г/г), кг/м3."""
    rho_w = 1000 * (1.0009 - 0.7114 * ws + 0.2605 * ws ** 2) ** (-1)
    return rho_w / (1 + (t - 273) * 1e-4 * (0.269 * (t - 273) ** 0.637 - 0.8))


def calc_mu_w(ws: float, t: float, p: float) -> float:
    """Расчет динамической вязкости воды по корреляции Matthews & Russel, сПз."""
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
    """Расчет числа Рейнольдса."""
    v = q_ms / (np.pi * d_tub ** 2 / 4)
    return rho_w * v * d_tub / mu_w * 1000


def calc_ff_churchill(n_re: float, roughness: float, d_tub: float) -> float:
    """Расчет коэффициента трения по корреляции Churchill."""
    a = (-2.457 * np.log((7 / n_re) ** 0.9 + 0.27 * (roughness / d_tub))) ** 16
    b = (37530 / n_re) ** 16
    return 8 * ((8 / n_re) ** 12 + 1 / (a + b) ** 1.5) ** (1 / 12)


def calc_dp_dl_grav(rho_w: float, angle: float) -> float:
    """Градиент давления на гравитацию (angle - угол к горизонтали в градусах), Па/м."""
    return rho_w * 9.81 * np.sin(angle / 180 * np.pi)


def calc_dp_dl_fric(rho_w: float, mu_w: float, q_ms: float, d_tub: float, roughness: float) -> float:
    """Градиент давления на трение, Па/м."""
    if q_ms != 0:
        n_re = calc_n_re(rho_w, q_ms, mu_w, d_tub)
        ff = calc_ff_churchill(n_re, roughness, d_tub)
        return ff * rho_w * q_ms ** 2 / d_tub ** 5
    return 0.0


def calc_dp_dl(rho_w: float, mu_w: float, angle: float, q_ms: float, d_tub: float, roughness: float) -> float:
    """Полный градиент давления в нагнетательной скважине, Па/м."""
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
) -> list:
    """Правая часть системы ОДУ для интегрирования по стволу скважины."""
    p, t = pt
    ws = calc_ws(gamma_wat)
    rho_w = calc_rho_w(ws, t)
    mu_w = calc_mu_w(ws, t, p)

    dp_dl = calc_dp_dl(rho_w, mu_w, angle, q_ms, d_tub, roughness)
    dt_dl = temp_grad * 1e-2
    return [dp_dl, dt_dl]


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
    """Интегрирование распределения давления и температуры от устья до ВДП."""
    sol = solve_ivp(
        fun=__integr_func,
        t_span=(h0, md_vdp),
        y0=[p_wh, t_wh],
        args=(temp_grad, gamma_wat, angle, q_ms, d_tub, roughness),
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
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
    """Расчет забойного давления (на глубине верхних дыр перфорации), Па."""
    p_arr, _, _ = calc_pipe(
        p_wh, t_wh, h0, md_vdp, temp_grad, gamma_wat, angle, q_ms, d_tub, roughness
    )
    return float(p_arr[-1])


def main():
    with open("3.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    p_wh_pa = data["p_wh"] * 101325.0
    t_wh_k = data["t_wh"] + 273.0
    md_vdp = data["md_vdp"]
    temp_grad = data["temp_grad"]
    gamma_wat = data["gamma_water"]
    angle = data["angle"]
    d_tub = data["d_tub"]
    roughness = data["roughness"]

    # Диапазон дебитов 1 - 400 м3/сут (шаг 1 м3/сут)
    q_liq_list = list(range(1, 401))
    p_wf_list = []

    for q_m3_day in q_liq_list:
        q_ms = q_m3_day / 86400.0
        p_wf_pa = calc_p_wf(
            p_wh=p_wh_pa,
            t_wh=t_wh_k,
            h0=0.0,
            md_vdp=md_vdp,
            temp_grad=temp_grad,
            gamma_wat=gamma_wat,
            angle=angle,
            q_ms=q_ms,
            d_tub=d_tub,
            roughness=roughness,
        )
        p_wf_list.append(p_wf_pa / 101325.0)

    output_data = {
        "q_liq": q_liq_list,
        "p_wf": p_wf_list,
    }

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    main()