import torch
import numpy as np
import json
import sys
import os

# 路径定义
MODEL_PATH = "/home/scnu/Gitee/EFD3D/outputs/train/pinn_20260529_180712/final_model.pth"
CONFIG_PATH = "/home/scnu/Gitee/EFD3D/outputs/train/pinn_20260529_180712/config.json"

def analyze():
    print("=" * 50)
    print("Model Output Analysis")
    print("=" * 50)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    try:
        # 加载模型
        # 如果是 torch.save(model)，则直接 load
        model = torch.load(MODEL_PATH, map_location='cpu')
        model.eval()
        print("Model loaded successfully.")

        # 定义采样参数
        Lx, Ly = 174e-6, 174e-6
        n_grid = 100
        x = np.linspace(0, Lx, n_grid)
        y = np.linspace(0, Ly, n_grid)
        X, Y = np.meshgrid(x, y)
        
        # 采样点 (x, y, z=0, V_from=0, V_to=30, t_since=0.020)
        # 场景：升压后 20ms 的状态
        pts = np.stack([
            X.flatten(), 
            Y.flatten(), 
            np.zeros(n_grid*n_grid), 
            np.zeros(n_grid*n_grid), 
            30.0 * np.ones(n_grid*n_grid), 
            0.020 * np.ones(n_grid*n_//n_grid*n_grid) # 这里修正一下
        ], axis=1)
        # 修正 pts 构造
        pts = np.stack([
            X.flatten(), 
            Y.flatten(), 
            np.zeros(n_grid*n_grid), 
            np.zeros(n_grid*n_grid), 
            30.0 * np.ones(n_grid*n_//n_grid*n_grid) # 再次修正
        ], axis=1)
        
        # 重新构造点集
        pts = []
        for xi in x:
            for yi in y:
                pts.append([xi, yi, 0.0, 0.0, 30.0, 0.020])
        
        pts_t = torch.tensor(pts, dtype=torch.float32)
        
        with torch.no_grad():
            # model(pts_t) returns (u, v, w, p, phi)
            out = model(pts_t)
            phi = out[:, 4].numpy()
        
        phi_grid = phi.reshape(n_grid, n_grid)
        
        # 计算开口率 (phi < 0.3 为极性液体/开口区域)
        opening_mask = phi_grid < 0.3
        opening_rate = np.mean(opening_mask) * 100
        
        print(f"Analysis at z=0, V=30V, t=20ms:")
        print(f"  Average phi: {np.mean(phi_grid):.4f}")
        print(f"  Min phi: {np.min(phi_grid):.4f}")
        print(f"  Max phi: {np.max(phi_grid):.4f}")
        print(f"  Estimated Opening Rate: {opening_rate:.2f}%")
        
        if opening_rate < 1.0:
            print("\nConclusion: [STILL FLAT] - No significant opening detected.")
        elif opening_rate > 50.0:
            print("\nConclusion: [OPENED] - Significant opening detected.")
        else:
            print("\nConclusion: [PARTIAL] - Some opening detected.")

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze()
