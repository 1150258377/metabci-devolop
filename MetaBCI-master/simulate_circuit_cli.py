#!/usr/bin/env python3
"""
CLI 工具: 根据三输入 -> 两隐藏神经元 -> 1 输出的线性网络
计算节点电流和分类结果，方便硬件电路仿真验证。

使用方式：
    python simulate_circuit_cli.py
随后按提示输入权重矩阵、bias、输入电压。
如果你已在 Streamlit 页面看到权重表，可直接复制粘贴。

权重维度假设：
    W: (hidden, 3)
    b_hidden: (hidden,)
    v: (hidden,)  # 输出层权重
    b_out: float  # 输出层 bias

参考电阻 Rs 可自定，默认 10kΩ。
"""
import numpy as np
import re

def parse_list(prompt, expected_len):
    while True:
        raw = input(prompt)
        txt = raw.strip()
        # 使用正则按逗号、分号、空白(含制表)切分
        vals = [v for v in re.split(r'[\s,;]+', txt) if v]
        if len(vals) != expected_len:
            print(f"需要 {expected_len} 个值，实际 {len(vals)}，请重新输入")
            continue
        try:
            return np.array([float(x) for x in vals], dtype=float)
        except ValueError:
            print("输入包含非数字，请重新输入！")


def main():
    print("========== CLI 电路仿真 ==========")
    hid = int(input("隐藏层神经元数量 (默认2)：") or 2)
    print("请依次输入隐藏层权重矩阵 W ({} x 3) 行向量，每行回车一次：".format(hid))
    W = []
    for i in range(hid):
        row = parse_list(f"  W 行 {i} (3 值): ", 3)
        W.append(row)
    W = np.stack(W)  # (hid,3)
    b_hid = parse_list("隐藏层 bias b_hid ({} 值): ".format(hid), hid)
    v = parse_list("输出层权重 v ({} 值): ".format(hid), hid)
    b_out = float(input("输出层 bias b_out: "))

    Rs = float(input("参考电阻 Rs (Ω, 默认10000):") or 10000.0)

    Vin = parse_list("输入电压 V0 V1 V2 (μV): ", 3) * 1e-6  # 转 V

    # 线性层
    z = W @ Vin + b_hid           # (hid,)
    a = np.maximum(0, z)          # ReLU
    logit = v @ a + b_out
    prob = 1/(1+np.exp(-logit))

    # 电流
    I_pre = z / Rs
    I_post = a / Rs
    I_out = logit / Rs

    print("\n========= 结果 =========")
    print("隐藏层 z:", z)
    print("隐藏层电流 I_pre (A):", I_pre)
    print("ReLU 后 a:", a)
    print("ReLU 后电流 I_post (A):", I_post)
    print("输出 logit:", logit)
    print("输出 I_out (A):", I_out)
    print(f"Sigmoid 概率: {prob:.4f}")
    print("分类结果:", 1 if prob>0.5 else 0)

if __name__ == "__main__":
    main() 