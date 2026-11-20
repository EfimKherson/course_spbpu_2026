import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import json


data = {
    "gamma_water": 1.0179115494421604,
    "md_vdp": 2560.7358194324115,
    "d_tub": 0.08581831634516876,
    "angle": 56.22825885617751,
    "roughness": 0.0003935839727709461,
    "p_wh": 151.32654595389886,
    "t_wh": 28.75610701819444,
    "temp_grad": 2.947060639486247
}

# Извлечение параметров
gamma_water = data['gamma_water']
md_vdp = data['md_vdp']
d_tub = data['d_tub']
angle_deg = data['angle']
roughness = data['roughness']
p_wh_atm = data['p_wh']
t_wh_c = data['t_wh']
temp_grad = data['temp_grad'] # К/100м

# Перевод в СИ
p_wh_pa = p_wh_atm * 101325
t_wh_k = t_wh_c + 273.15
angle_rad = math.radians(90 - angle_deg) # Угол к горизонтали
temp_grad_m = temp_grad / 100 # К/м

g = 9.81 # м/с^2

# --- 2. Функции для расчета свойств флюида ---

def calc_ws(gamma_wat: float) -> float:
    """Расчет солесодержания (массовой доли) по относительной плотности."""
    rho_w_sc = gamma_wat * 1000
    if rho_w_sc <= 992:
        return 0.0
    ws = (1 / rho_w_sc) * (1.36545 * rho_w_sc - (3838.77 * rho_w_sc - 2.009 * rho_w_sc**2)**0.5)
    return ws

def calc_rho_w(ws: float, t_k: float) -> float:
    """Расчет плотности воды в зависимости от температуры и солености."""
    rho_w_sc = 1000 * (1.0009 - 0.7114 * ws + 0.2605 * ws**2)**(-1)
    return rho_w_sc / (1 + (t_k - 273.15) * 1e-4 * (0.269 * (t_k - 273.15)**0.637 - 0.8))

def calc_mu_w(ws: float, t_k: float, p_pa: float) -> float:
    """Расчет динамической вязкости воды, сПз."""
    p_mpa = p_pa / 1e6
    a = 109.574 - 0.840564 * 1000 * ws + 3.13314 * 1000 * ws**2 + 8.72213 * 1000 * ws**3
    b = 1.12166 - 2.63951 * ws + 6.79461 * ws**2 + 54.7119 * ws**3 - 155.586 * ws**4
    mu_w = a * (1.8 * t_k - 460)**(-b) * (0.9994 + 0.0058 * p_mpa + 0.6534 * 1e-4 * p_mpa**2)
    return mu_w

def calc_n_re(rho: float, q_ms: float, mu_cp: float, d: float) -> float:
    """Расчет числа Рейнольдса."""
    v = q_ms / (np.pi * d**2 / 4)
    return rho * v * d / (mu_cp * 1e-3)

def calc_ff_churchill(n_re: float, roughness: float, d: float) -> float:
    """Расчет коэффициента трения по корреляции Черчилля."""
    if n_re < 1e-5: return 0
    a = (-2.457 * np.log((7 / n_re)**0.9 + 0.27 * (roughness / d)))**16
    b = (37530 / n_re)**16
    ff = 8 * ((8 / n_re)**12 + 1 / (a + b)**1.5)**(1/12)
    return ff

# --- 3. Основные функции для интегрирования ---

def integr_func(h, y, q_ms, ws, d_tub, angle_rad, roughness, temp_grad_m, t_wh_k):
    """Функция для solve_ivp. y = [p, t]"""
    p, t = y
    
    # Свойства при текущих p, t
    rho = calc_rho_w(ws, t)
    mu = calc_mu_w(ws, t, p)
    
    # Гидравлика
    n_re = calc_n_re(rho, q_ms, mu, d_tub)
    ff = calc_ff_churchill(n_re, roughness, d_tub)
    v = q_ms / (np.pi * d_tub**2 / 4)
    
    # Градиенты
    dp_dl_grav = rho * g * math.cos(angle_rad)
    dp_dl_fric = ff * rho * v**2 / (2 * d_tub)
    dp_dl = dp_dl_grav - dp_dl_fric
    
    # Температура меняется линейно
    dt_dl = temp_grad_m
    
    return [dp_dl, dt_dl]

def calc_p_wf(q_ms, data):
    """Расчет забойного давления для заданного дебита."""
    ws = calc_ws(data['gamma_water'])
    
    # Начальные условия
    y0 = [data['p_wh_pa'], data['t_wh_k']]
    
    # Интегрируем от устья (0) до глубины перфорации (md_vdp)
    sol = solve_ivp(
        fun=lambda h, y: integr_func(h, y, q_ms, ws, data['d_tub'], data['angle_rad'], 
                                     data['roughness'], data['temp_grad_m'], data['t_wh_k']),
        t_span=(0, data['md_vdp']),
        y0=y0,
        method='RK45',
        dense_output=True
    )
    
    # Возвращаем давление на забое (в Па)
    return sol.y[0][-1]

# --- 4. Генерация VLP и сохранение результатов ---

def main():
    # Подготовка данных для расчета
    calc_data = {
        'gamma_water': gamma_water,
        'md_vdp': md_vdp,
        'd_tub': d_tub,
        'angle_rad': angle_rad,
        'roughness': roughness,
        'p_wh_pa': p_wh_pa,
        't_wh_k': t_wh_k,
        'temp_grad_m': temp_grad_m
    }
    
    
    q_range_m3_day = np.linspace(1, 400, 50)
    q_range_m3_s = q_range_m3_day / 86400.0
    
    p_wf_list = []
    
    
    for q in q_range_m3_s:
        p_wf = calc_p_wf(q, calc_data)
        p_wf_list.append(p_wf)
    
    p_wf_atm = [p / 101325 for p in p_wf_list]
    
    # Сохранение в output.json
    output_data = {
        "q_m3_day": q_range_m3_day.tolist(),
        "p_wf_atm": p_wf_atm
    }
    with open('output.json', 'w') as f:
        json.dump(output_data, f, indent=4)
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(q_range_m3_day, p_wf_atm, 'b-', linewidth=2)
    plt.title('Кривая VLP для нагнетательной скважины')
    plt.xlabel('Дебит закачиваемой жидкости, м³/сут')
    plt.ylabel('Забойное давление, атм')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()