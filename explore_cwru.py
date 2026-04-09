import numpy as np
import os

def explore_npz(file_path):
    print(f"正在加载文件: {file_path}")
    if not os.path.exists(file_path):
        print("错误：文件不存在！请检查路径。")
        return

    # 加载 npz 文件
    data = np.load(file_path, allow_pickle=True)
    
    print("\n" + "="*50)
    print("📦 NPZ 文件包含的数组列表 (Keys):")
    print(data.files)
    print("="*50)

    # 遍历打印每个数组的 shape 和数据类型
    for key in data.files:
        array = data[key]
        print(f"\n🔍 键名 (Key): '{key}'")
        print(f"   - 形状 (Shape): {array.shape}")
        print(f"   - 类型 (Dtype): {array.dtype}")
        
        # 如果是标签数组，我们来看看有哪几种标签
        if 'y' in key.lower() or 'label' in key.lower():
            unique_labels = np.unique(array)
            print(f"   - 唯一的标签值: {unique_labels}")
            print(f"   - 类别总数: {len(unique_labels)}")

if __name__ == "__main__":
    # 替换为你实际的 CWRU npz 文件路径
    cwru_npz_path = "data/CWRU/CWRU_48k_load_1_CNN_data.npz"
    explore_npz(cwru_npz_path)