import json
import numpy as np
import os
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def calc_ws(gamma_wat: float) -> float:
    """Расчет солесодержания в воде"""
    ws = (
            1 / (gamma_wat * 1000)
            * (1.36545 * gamma_wat * 1000 - (3838.77 * gamma_wat * 1000 - 2.009 * (gamma_wat * 1000) ** 2) ** 0.5)
    )
    if ws > 0:
        return ws
    else:
        return 0


def calc_rho_w(ws: float, t: float) -> float:
    """Расчет плотности воды в зависимости от температуры и солесодержания"""
    rho_w = 1000 * (1.0009 - 0.7114 * ws + 0.2605 * ws ** 2) ** (-1)
    return rho_w / (1 + (t - 273) * 1e-4 * (0.269 * (t - 273) ** 0.637 - 0.8))


def calc_mu_w(ws: float, t: float, p: float) -> float:
    """Расчет динамической вязкости воды по корреляции Matthews & Russel"""
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
    """Расчет числа Рейнольдса"""
    v = q_ms / (np.pi * d_tub ** 2 / 4)
    return rho_w * v * d_tub / mu_w * 1000


def calc_ff_churchill(n_re: float, roughness: float, d_tub: float) -> float:
    """Расчет коэффициента трения по корреляции Churchill"""
    a = (-2.457 * np.log((7 / n_re) ** 0.9 + 0.27 * (roughness / d_tub))) ** 16
    b = (37530 / n_re) ** 16
    ff = 8 * ((8 / n_re) ** 12 + 1 / (a + b) ** 1.5) ** (1 / 12)
    return ff


def calc_dp_dl_grav(rho_w: float, angle: float) -> float:
    """Расчет градиента на гравитацию"""
    dp_dl_grav = rho_w * 9.81 * np.sin(angle / 180 * np.pi)
    return dp_dl_grav


def calc_dp_dl_fric(rho_w: float, mu_w: float, q_ms: float, d_tub: float, roughness: float) -> float:
    """Расчет градиента давления на трение"""
    if q_ms != 0:
        n_re = calc_n_re(rho_w, q_ms, mu_w, d_tub)
        ff = calc_ff_churchill(n_re, roughness, d_tub)
        dp_dl_fric = ff * rho_w * q_ms ** 2 / d_tub ** 5
    else:
        dp_dl_fric = 0
    return dp_dl_fric


def calc_dp_dl(rho_w: float, mu_w: float, angle: float, q_ms: float, d_tub: float, roughness: float) -> float:
    """Расчет общего градиента давления в трубе"""
    dp_dl_grav = calc_dp_dl_grav(rho_w, angle)
    dp_dl_fric = calc_dp_dl_fric(rho_w, mu_w, q_ms, d_tub, roughness)
    dp_dl = dp_dl_grav - 8 / np.pi**2 * dp_dl_fric
    return dp_dl


def __integr_func(h: float, pt: tuple, temp_grad: float, gamma_wat: float, angle: float, 
                  q_ms: float, d_tub: float, roughness: float) -> tuple:
    """Функция для интегрирования трубы (система уравнений)"""
    p, t = pt
    ws = calc_ws(gamma_wat)
    rho_w = calc_rho_w(ws, t)
    mu_w = calc_mu_w(ws, t, p)
    
    # Градиент давления (Па/м)
    dp_dl = calc_dp_dl(rho_w, mu_w, angle, q_ms, d_tub, roughness)
    # Градиент температуры (К/м)
    dT_dl = temp_grad
    
    return [dp_dl, dT_dl]


def calc_pipe(p_wh: float, t_wh: float, h0: float, md_vdp: float, temp_grad: float, 
              gamma_wat: float, angle: float, q_ms: float, d_tub: float, roughness: float) -> tuple:
    """Функция для расчета распределения давления и температуры в трубе"""
    y0 = [p_wh, t_wh] # нач условия: давление, температура на устье
    t_span = [h0, md_vdp] # интервал интегрирования по глубине
    
    # ссиистема дифференциальных уравнений
    sol = solve_ivp(
        fun=__integr_func,
        t_span=t_span,
        y0=y0,
        args=(temp_grad, gamma_wat, angle, q_ms, d_tub, roughness),
        dense_output=True,
        max_step=50.0 # шагдля интегрирвоания
    )
    
    # sol.y[0] - массив давлений, sol.y[1] - массив температур, sol.t - массив глубин
    return sol.y[0], sol.y[1], sol.t


def calc_p_wf(p_wh: float, t_wh: float, h0: float, md_vdp: float, temp_grad: float, 
              gamma_wat: float, angle: float, q_ms: float, d_tub: float, roughness: float) -> float:
    """Функция для расчета забойного давления"""
    p_res, t_res, h_res = calc_pipe(p_wh, t_wh, h0, md_vdp, temp_grad, gamma_wat, angle, q_ms, d_tub, roughness)
    return p_res[-1] # Возвращаем последнее значение давления на забое в Паскалях


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '14.json')

    with open(file_path, 'r') as f:
        data = json.load(f)

    gamma_water = data["gamma_water"]
    md_vdp = data["md_vdp"]                 # Глубина, м
    d_tub = data["d_tub"]                   # Диаметр, м
    angle = data["angle"]                   # Угол, градусы
    roughness = data["roughness"]           # Шероховатость, м
    p_wh_atm = data["p_wh"]                 # Устьевое давление, атм
    t_wh_c = data["t_wh"]                   # Устьевая температура, С
    temp_grad_c_per_100m = data["temp_grad"] # Геотермический градиент, С/100м

    # Перевод в систему СИ
    p_wh = p_wh_atm * 101325                # Па
    t_wh = t_wh_c + 273.15                  # К
    temp_grad = temp_grad_c_per_100m * 1e-2 # К/м

    q_liq = np.arange(0, 400, 10) 

    # VLP зависимость забойного давления от дебита
    p_wf_list = []
    for q in q_liq:
        q_ms = q / 86400.0
        
        # Расчет давления в Па
        p_wf_pa = calc_p_wf(
            p_wh=p_wh, t_wh=t_wh, h0=0, md_vdp=md_vdp, 
            temp_grad=temp_grad, gamma_wat=gamma_water, angle=angle, 
            q_ms=q_ms, d_tub=d_tub, roughness=roughness
        )
        
        # Перевод обратно в атм для файла
        p_wf_atm = p_wf_pa / 101325
        p_wf_list.append(float(p_wf_atm))

    output_data = {
        "q_liq": q_liq.tolist(),
        "p_wf": p_wf_list
    }

    output_path = os.path.join(script_dir, 'output.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(output_path)

    # построение VLP
    plt.figure(figsize=(9, 6))

    plt.plot(q_liq, p_wf_list, marker='o', linewidth=2, markersize=4)

    plt.xlabel('Дебит закачиваемой жидкости, м³/сут')
    plt.ylabel('Забойное давление, атм')
    plt.title('VLP')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    graph_path = os.path.join(script_dir, 'vlp.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.show()