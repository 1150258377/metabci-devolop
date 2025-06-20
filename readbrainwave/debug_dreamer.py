import scipy.io
import numpy as np

def inspect_dreamer_structure():
    file_path = r"C:\Users\jing pengqiang\Downloads\DREAMER.mat"
    mat_data = scipy.io.loadmat(file_path)
    
    print("MAT文件中的所有变量:")
    for key in mat_data.keys():
        if not key.startswith('__'):
            print(f"  {key}")
    
    print("\nDREAMER变量的结构:")
    dreamer_data = mat_data['DREAMER'][0, 0]
    print(f"类型: {type(dreamer_data)}")
    print(f"形状: {dreamer_data.shape}")
    print(f"数据类型: {dreamer_data.dtype}")
    
    if dreamer_data.dtype.names:
        print(f"字段名: {dreamer_data.dtype.names}")
        
        for name in dreamer_data.dtype.names:
            field = dreamer_data[name]
            print(f"\n字段 '{name}':")
            print(f"  类型: {type(field)}")
            print(f"  形状: {field.shape}")
            print(f"  数据类型: {field.dtype}")
            
            # 尝试访问数据
            try:
                if field.size > 0:
                    first_element = field[0, 0]
                    print(f"  第一个元素类型: {type(first_element)}")
                    print(f"  第一个元素形状: {first_element.shape if hasattr(first_element, 'shape') else 'N/A'}")
                    print(f"  第一个元素内容: {first_element}")
                    
                    if hasattr(first_element, 'dtype') and first_element.dtype.names:
                        print(f"  第一个元素字段: {first_element.dtype.names}")
            except Exception as e:
                print(f"  访问错误: {e}")

if __name__ == "__main__":
    inspect_dreamer_structure() 