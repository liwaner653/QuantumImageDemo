# Help by gemini

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile #, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit_aer import QasmSimulator


# 1. 定义 FRQI 制备函数 (基于前一环节逻辑)
def create_frqi_circuit(angles):
    qc = QuantumCircuit(n_pos + 1, n_pos + 1) # 增加经典比特用于存储测量结果
    
    # 步骤 1: Hadamard 变换生成位置叠加态 [cite: 49, 52]
    qc.h(range(1, n_pos + 1))
    qc.barrier()
    
    # 步骤 2: 受控旋转编码颜色 [cite: 49, 54]
    for i, theta in enumerate(angles):
        binary_idx = format(i, f'0{n_pos}b')
        for bit_idx, bit in enumerate(reversed(binary_idx)):
            if bit == '0': qc.x(bit_idx + 1)
        
        # 多受控 Ry 门：控制位是位置比特，目标位是颜色比特(0)
        qc.mcry(2 * theta, list(range(1, n_pos + 1)), 0)
        
        for bit_idx, bit in enumerate(reversed(binary_idx)):
            if bit == '0': qc.x(bit_idx + 1)
        qc.barrier()
        
    return qc

# 2. 准备实验数据 (4个像素的不同灰度)
# 角度 theta 范围 [0, pi/2]，0 为全黑，pi/2 为全白 [cite: 34]
pixel_angles = [0, np.pi/4, np.pi/3, np.pi/2] 
num_pixels = len(pixel_angles)
n_pos = int(np.log2(num_pixels))
qc = create_frqi_circuit(pixel_angles)

print(qc.draw('mpl', fold=-1))

# 3. 添加测量操作 [cite: 162]
# 将所有量子位测量到对应的经典比特上
qc.measure(range(n_pos + 1), range(n_pos + 1))

# 4. 执行仿真
# 使用 Aer 的 qasm_simulator 进行多次采样 (shots)
backend = QasmSimulator()
circ = transpile(qc, backend=backend)
shots=100000
job = backend.run(circ, shots=shots) # 采样次数越多，图像重建越精确 [cite: 172]
result = job.result()
counts = result.get_counts()

# 5. 可视化测量统计
print("测量统计结果 (bitstring: pos1 pos0 color):", counts)
plot_histogram(counts)
plt.show()

def reconstruct_image(counts, n_pos, shots):
    reconstructed_angles = np.zeros(2**n_pos)
    for bitstring, count in counts.items():
        # 分离颜色位和位置位
        color_bit = bitstring[-1]
        pos_bits = bitstring[:-1]
        pos_idx = int(pos_bits, 2)
        
        if color_bit == '1':
            # 概率 P(1, i) = count / shots
            prob = count / shots
            # 反推 theta = arcsin(sqrt(prob * total_pixels))
            val = np.sqrt(prob * (2**n_pos))
            # 裁剪范围防止浮点数误差
            reconstructed_angles[pos_idx] = np.arcsin(np.clip(val, 0, 1))
            
    return reconstructed_angles

angles = reconstruct_image(counts, 2, shots)
print("重建的角度序列:", angles)