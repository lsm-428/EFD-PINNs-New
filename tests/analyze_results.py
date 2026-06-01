import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.pinn_two_//pinn_two_phase import TwoPhasePINN # 这里的路径需要正确
import json

# 这里的路径请根据实际情况调整，我先尝试一个通用路径
model_path = "/home/scnu/Gitee/EFD3D/outputs/train/pinn_20260529_180712/final_model.pth"
config_path = "/home/scnu/Gitee/EFD3D/outputs/train/pinn_20260529_180712/config.json"

try:
    with open(config_path) as f:
        config = json.load(f)
    
    # 模拟初始化模型（需要正确传递参数）
    # 注意：这里需要 TwoPhasePINN 的初始化参数，为了简单，我们尝试加载状态字典
    # 但 TwoPhasePINN 需要初始化后才能 load_state_dict
    print("Loading model...")
    model = torch.load(model_path)
    model.eval()

    # 采样底面 (z=0) 的 φ 场
    # 采样 100x100 个点
    Lx, Ly = 174e-6, 174e-6
    x = np.linspace(0, Lx, 100)
    y = np.linspace(0, Ly, 100)
    X, Y = np.meshgrid(x, y)
    
    pts = np.stack([X.flatten(), Y.flatten(), np.zeros(100*100), 
                    np.zeros(100*100), 30.0*np.ones(100*100), 0.020*np.ones(100*100)], axis=1)
    pts_t = torch.tensor(pts, dtype=torch.float32)
    
    with torch.no_grad():
        phi = model(pts_t)[:, 4].numpy()
    
    phi_grid = phi.reshape(100, 100)
    opening_rate = np.mean(phi_grid < 0.3) # 简单估计开口率
    print(f"Estimated Opening Rate (phi < 0.3): {opening_rate * 100:.2f}%")
    
    if opening_rate < 0.1:
        print("Result: Still Flat (平铺状态)")
    elif opening_rate > 0.5:
        print("Result: Opened (已开口)")
    else:
        print("Result: Partially Opened (部分开口)")

except Exception as e:
    print(f"Analysis failed: {e}")
