import scipy.io
import numpy as np
import os
import sys
import pandas as pd

def read_mat_file(file_path):
    """
    读取.mat文件并展示其内容
    
    Args:
        file_path (str): .mat文件的路径
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误：文件 '{file_path}' 不存在")
            return
        
        # 检查文件扩展名
        if not file_path.lower().endswith('.mat'):
            print(f"错误：文件 '{file_path}' 不是.mat文件")
            return
        
        # 读取.mat文件
        print(f"正在读取文件: {file_path}")
        mat_data = scipy.io.loadmat(file_path)
        
        print("\n" + "="*50)
        print("MAT文件内容分析")
        print("="*50)
        
        # 显示所有变量名
        print(f"\n文件中的变量数量: {len(mat_data)}")
        print("\n变量列表:")
        for i, key in enumerate(mat_data.keys(), 1):
            # 跳过系统变量（以__开头的变量）
            if not key.startswith('__'):
                value = mat_data[key]
                print(f"{i}. {key}")
                print(f"   类型: {type(value).__name__}")
                
                if isinstance(value, np.ndarray):
                    print(f"   形状: {value.shape}")
                    print(f"   数据类型: {value.dtype}")
                    
                    # 如果是数值数组，显示一些统计信息
                    if np.issubdtype(value.dtype, np.number):
                        if value.size > 0:
                            print(f"   最小值: {np.min(value)}")
                            print(f"   最大值: {np.max(value)}")
                            print(f"   平均值: {np.mean(value):.4f}")
                            print(f"   标准差: {np.std(value):.4f}")
                    
                    # 显示数组的前几个元素
                    if value.size <= 10:
                        print(f"   内容: {value}")
                    else:
                        print(f"   前5个元素: {value.flatten()[:5]}")
                        print(f"   后5个元素: {value.flatten()[-5:]}")
                else:
                    print(f"   内容: {value}")
                print()
        
        # 显示系统变量信息
        system_vars = [key for key in mat_data.keys() if key.startswith('__')]
        if system_vars:
            print("系统变量:")
            for var in system_vars:
                print(f"  {var}: {mat_data[var]}")
        
        return mat_data
        
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return None

def mat_to_dataframe(mat_data, output_prefix="output"):
    """
    尝试将mat文件中的每个变量转为DataFrame并保存为csv
    """
    for key, value in mat_data.items():
        if key.startswith("__"):
            continue
        try:
            # 只处理二维数组或一维数组
            if isinstance(value, np.ndarray) and value.ndim in [1, 2]:
                df = pd.DataFrame(value)
                csv_path = f"{output_prefix}_{key}.csv"
                df.to_csv(csv_path, index=False)
                print(f"变量 {key} 已保存为 {csv_path}")
        except Exception as e:
            print(f"变量 {key} 转换失败: {e}")

def main():
    """主函数"""
    print("MAT文件读取器")
    print("="*30)
    
    # 用户指定的默认文件路径
    default_path = r"C:\Users\jing pengqiang\Downloads\DREAMER.mat"
    
    # 如果命令行提供了文件路径
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # 否则使用默认路径
        print(f"未检测到命令行参数，将使用默认路径: {default_path}")
        file_path = default_path
    
    # 读取并显示文件内容
    data = read_mat_file(file_path)
    
    if data:
        print("\n" + "="*50)
        print("文件读取完成！")
        print("="*50)
        
        # 新增：尝试转为DataFrame并保存
        print("\n正在尝试将可转换的变量保存为csv...")
        mat_to_dataframe(data, output_prefix="dreamer")
        
        # 提供交互式选项
        while True:
            print("\n选项:")
            print("1. 查看特定变量的详细信息")
            print("2. 保存变量到新的.mat文件")
            print("3. 退出")
            
            choice = input("\n请选择操作 (1-3): ").strip()
            
            if choice == '1':
                var_name = input("请输入变量名: ").strip()
                if var_name in data and not var_name.startswith('__'):
                    value = data[var_name]
                    print(f"\n变量 '{var_name}' 的详细信息:")
                    print(f"类型: {type(value).__name__}")
                    if isinstance(value, np.ndarray):
                        print(f"形状: {value.shape}")
                        print(f"数据类型: {value.dtype}")
                        print(f"完整内容:\n{value}")
                    else:
                        print(f"内容: {value}")
                else:
                    print(f"变量 '{var_name}' 不存在")
            
            elif choice == '2':
                output_file = input("请输入输出文件名 (例如: output.mat): ").strip()
                if output_file:
                    try:
                        scipy.io.savemat(output_file, data)
                        print(f"数据已保存到 {output_file}")
                    except Exception as e:
                        print(f"保存文件时发生错误: {str(e)}")
            
            elif choice == '3':
                print("再见！")
                break
            
            else:
                print("无效选择，请重试")

if __name__ == "__main__":
    main() 