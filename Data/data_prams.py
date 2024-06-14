import json
import numpy as np
import torch

product_cost = np.array([16.19, 17.26, 16.60, 16.50, 19.70, 22.26, 27.74, 25.92])
start_up_cost = np.array([4500, 5000, 550, 560, 900, 170, 260, 30])
shutdown_cost = np.array([4500, 5000, 550, 560, 900, 170, 260, 30])
load_demand = np.array([1422*0.71, 1422*0.65, 1422*0.62, 1422*0.60, 1422*0.58, 1422*0.58, 1422*0.60, 1422*0.64, 1422*0.73, 1422*0.80, 1422*0.82, 1422*0.83,
                        1422*0.82, 1422*0.80, 1422*0.79,1422*0.79, 1422*0.83, 1422*0.91, 1422*0.90, 1422*0.88, 1422*0.85, 1422*0.84, 1422*0.79
                           , 1422*0.74])
min_up = [8, 8, 5, 5, 6, 3, 3, 1]
min_down = [8, 8, 5, 5, 6, 3, 3, 1]
power_l = [150, 150, 20, 20, 25, 20, 25, 10]
power_h = [455, 455, 130, 130, 162, 80, 85, 55]
data = {
    "product_cost": product_cost.tolist(),
    "start_up_cost": start_up_cost.tolist(),
    "shutdown_cost": shutdown_cost.tolist(),
    "load_demand": load_demand.tolist(),
    "min_up": min_up,
    "min_down": min_down,
    "power_l": power_l,
    "power_h": power_h
}
# save data as json file

with open('data.json', 'w') as f:
    json.dump(data, f)

def get_encoder_input(device):
    data = json.load(open("Data/data.json"))
    product_cost = np.array(data["product_cost"])
    start_up_cost = np.array(data["start_up_cost"])
    shutdown_cost = np.array(data["shutdown_cost"])
    min_up = np.array(data["min_up"])
    min_down = np.array(data["min_down"])
    power_l = np.array(data["power_l"])
    power_h = np.array(data["power_h"])
    e_in = np.hstack((product_cost, start_up_cost, shutdown_cost, min_up, min_down, power_l, power_h))
    e_in=torch.from_numpy(e_in).float().to(device)
    return e_in
