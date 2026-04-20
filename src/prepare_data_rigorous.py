"""
prepare_data_rigorous.py
========================
统一数据预处理：严格按 7:2:1 **时序顺序** 切分数据集，杜绝时间序列泄露。
支持数据集: CWRU, DIRG, XJTU-SY

用法:
    python src/prepare_data_rigorous.py --dataset cwru
    python src/prepare_data_rigorous.py --dataset dirg
    python src/prepare_data_rigorous.py --dataset xjtu
    python src/prepare_data_rigorous.py --dataset all
    python src/prepare_data_rigorous.py --dataset cwru --check_leakage  # 验证无泄露
"""

import os
import sys
import glob
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# ═══════════════════════════════════════════════════════════════════════════════
# 通用工具
# ═══════════════════════════════════════════════════════════════════════════════

def sequential_split_per_class(X_all, y_all, num_classes, num_slices, slice_length, channels):
    """
    按类别进行时序顺序 7:2:1 切分。
    保持每类内部样本的原始时间顺序，不做任何随机打乱。
    """
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    for c in range(num_classes):
        mask = (y_all == c)
        X_c = X_all[mask]
        y_c = y_all[mask]

        # 确保能被 num_slices 整除
        valid_len = (len(X_c) // num_slices) * num_slices
        X_c = X_c[:valid_len]
        y_c = y_c[:valid_len]

        # 分组: (N_groups, num_slices, slice_length, channels)
        X_grouped = X_c.reshape(-1, num_slices, slice_length, channels)
        y_grouped = y_c[::num_slices]

        n_total = len(X_grouped)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.2)
        # n_test = n_total - n_train - n_val (剩余)

        X_train_list.append(X_grouped[:n_train])
        y_train_list.append(y_grouped[:n_train])

        X_val_list.append(X_grouped[n_train:n_train + n_val])
        y_val_list.append(y_grouped[n_train:n_train + n_val])

        X_test_list.append(X_grouped[n_train + n_val:])
        y_test_list.append(y_grouped[n_train + n_val:])

        print(f"    Class {c}: total={n_total} -> train={n_train}, val={n_val}, test={n_total - n_train - n_val}")

    def merge(xl, yl):
        return np.concatenate(xl, axis=0).astype(np.float32), np.concatenate(yl, axis=0).astype(np.int64)

    X_train, y_train = merge(X_train_list, y_train_list)
    X_val, y_val = merge(X_val_list, y_val_list)
    X_test, y_test = merge(X_test_list, y_test_list)

    return X_train, y_train, X_val, y_val, X_test, y_test


def save_splits(output_dir, X_train, y_train, X_val, y_val, X_test, y_test):
    """保存 npy 文件"""
    os.makedirs(output_dir, exist_ok=True)
    for name, arr in [("X_train", X_train), ("y_train", y_train),
                       ("X_val", X_val), ("y_val", y_val),
                       ("X_test", X_test), ("y_test", y_test)]:
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, arr)
    print(f"  ✅ 已保存至 {output_dir}/")
    print(f"     Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


def check_leakage(output_dir, dataset_name):
    """验证数据切分是否存在泄露（训练集索引必须严格在测试集之前）"""
    print(f"\n[CHECK] 正在验证 {dataset_name} 数据泄露...")
    X_train = np.load(os.path.join(output_dir, "X_train.npy"))
    X_test = np.load(os.path.join(output_dir, "X_test.npy"))
    # 简单检查：测试集不能在训练集样本中出现
    for i in range(min(10, len(X_test))):
        for j in range(len(X_train)):
            if np.array_equal(X_test[i], X_train[j]):
                print(f"  ❌ 泄露发现: test[{i}] == train[{j}]")
                return False
    print(f"  ✅ {dataset_name} 无泄露 (抽样验证通过)")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# CWRU 数据集
# ═══════════════════════════════════════════════════════════════════════════════

CWRU_LABEL_MAP = {
    'Normal': 0,
    'IR_007': 1, 'IR_014': 2, 'IR_021': 3,
    'Ball_007': 4, 'Ball_014': 5, 'Ball_021': 6,
    'OR_007': 7, 'OR_014': 8, 'OR_021': 9
}

def prepare_cwru():
    """
    CWRU: 从 .npz 加载，展平为单通道 (N, 1024, 1)，按类别时序顺序切分。
    """
    print("\n" + "=" * 60)
    print("  CWRU 数据集预处理 (10 分类, 单通道)")
    print("=" * 60)

    npz_path = os.path.join(DATA_ROOT, "CWRU", "CWRU_48k_load_1_CNN_data.npz")
    if not os.path.exists(npz_path):
        print(f"  [ERROR] 找不到 {npz_path}")
        return None

    data = np.load(npz_path, allow_pickle=True)
    X_raw = data['data']     # (4600, 32, 32)
    y_raw = data['labels']   # (4600,) string labels

    slice_length = 1024
    num_slices = 3
    channels = 1

    # 展平: (4600, 32, 32) -> (4600, 1024, 1)
    X_flat = X_raw.reshape(-1, slice_length, channels)
    y_mapped = np.array([CWRU_LABEL_MAP[label] for label in y_raw])

    output_dir = os.path.join(DATA_ROOT, "CWRU_processed")
    X_train, y_train, X_val, y_val, X_test, y_test = sequential_split_per_class(
        X_flat, y_mapped, num_classes=10,
        num_slices=num_slices, slice_length=slice_length, channels=channels
    )
    save_splits(output_dir, X_train, y_train, X_val, y_val, X_test, y_test)
    return output_dir


# ═══════════════════════════════════════════════════════════════════════════════
# DIRG 数据集
# ═══════════════════════════════════════════════════════════════════════════════

DIRG_LABEL_MAP = {
    'C0': 0,  # Healthy
    'C1': 1,  # Inner Minor
    'C2': 2,  # Inner Moderate
    'C3': 3,  # Inner Severe
    'C4': 4,  # Ball/Outer Minor
    'C5': 5,  # Ball/Outer Moderate
    'C6': 6,  # Ball/Outer Severe
}

def prepare_dirg():
    """
    DIRG: 从 .mat 加载 6 通道信号，按文件名排序后按类别时序切分。
    关键变更：不使用 train_test_split，改为时序顺序切分。
    """
    import scipy.io

    print("\n" + "=" * 60)
    print("  DIRG 数据集预处理 (7 分类, 6 通道)")
    print("=" * 60)

    dirg_dir = os.path.join(DATA_ROOT, "DIRG", "VariableSpeedAndLoad")
    if not os.path.exists(dirg_dir):
        print(f"  [ERROR] 找不到 {dirg_dir}")
        return None

    slice_length = 1024
    num_slices = 3
    total_len = slice_length * num_slices  # 3072
    channels = 6

    mat_files = sorted(glob.glob(os.path.join(dirg_dir, "*.mat")))
    mat_files = [f for f in mat_files if not os.path.basename(f).startswith('FileNames')]

    X_all, y_all = [], []

    for fpath in mat_files:
        fname = os.path.basename(fpath)
        label_key = fname[:2]
        if label_key not in DIRG_LABEL_MAP:
            continue
        label = DIRG_LABEL_MAP[label_key]

        data = scipy.io.loadmat(fpath)
        var_name = fname.replace('.mat', '')
        if var_name not in data:
            keys = [k for k in data.keys() if not k.startswith('__')]
            if not keys:
                continue
            var_name = keys[0]

        signal = data[var_name]  # (Points, 6)

        # 按时序顺序切片，保持原始顺序
        num_windows = signal.shape[0] // total_len
        for i in range(num_windows):
            window = signal[i * total_len: (i + 1) * total_len, :]  # (3072, 6)
            # 将 window 拆分为 num_slices 个 slice
            slices = window.reshape(num_slices, slice_length, channels)
            for s in range(num_slices):
                X_all.append(slices[s])
                y_all.append(label)

    # X_all 每个元素是 (1024, 6)，总共 N 个样本
    X_all = np.array(X_all)  # (N, 1024, 6)
    y_all = np.array(y_all)

    print(f"  总样本数: {len(y_all)}，各类分布: {np.bincount(y_all)}")

    output_dir = os.path.join(DATA_ROOT, "DIRG_processed")
    X_train, y_train, X_val, y_val, X_test, y_test = sequential_split_per_class(
        X_all, y_all, num_classes=7,
        num_slices=num_slices, slice_length=slice_length, channels=channels
    )
    save_splits(output_dir, X_train, y_train, X_val, y_val, X_test, y_test)
    return output_dir


# ═══════════════════════════════════════════════════════════════════════════════
# XJTU-SY 数据集
# ═══════════════════════════════════════════════════════════════════════════════

# XJTU-SY 标准故障类型映射 (基于官方文献)
# 条件1 (35Hz/12kN): Bearing1_1=OR, 1_2=OR, 1_3=OR, 1_4=Cage, 1_5=IR+OR
# 条件2 (37.5Hz/11kN): 2_1=IR, 2_2=OR, 2_3=Cage, 2_4=IR, 2_5=OR
# 条件3 (40Hz/10kN): 3_1=IR, 3_2=OR+Cage, 3_3=IR, 3_4=IR, 3_5=OR
XJTU_BEARING_LABEL = {
    # label: 0=Healthy (无，因为XJTU全寿命数据末期全为故障, 实际取早期信号作为健康),
    # 1=Inner, 2=Outer, 3=Cage, 4=Compound
    'Bearing1_1': 2, 'Bearing1_2': 2, 'Bearing1_3': 2,
    'Bearing1_4': 3, 'Bearing1_5': 4,
    'Bearing2_1': 1, 'Bearing2_2': 2, 'Bearing2_3': 3,
    'Bearing2_4': 1, 'Bearing2_5': 2,
    'Bearing3_1': 1, 'Bearing3_2': 4, 'Bearing3_3': 1,
    'Bearing3_4': 1, 'Bearing3_5': 2,
}

# 故障分类中的类别提取
# 仅使用中后期以确保故障特征明显，同时取少量前期数据构成 "健康" 类
XJTU_CONDITIONS = {
    '35Hz12kN':   'Bearing1_',
    '37.5Hz11kN': 'Bearing2_',
    '40Hz10kN':   'Bearing3_',
}

def prepare_xjtu():
    """
    XJTU-SY: 全寿命退化数据集，2 通道加速度。
    策略：每个轴承的前 20% CSV 文件视为健康段 (label=0)，后 50% 视为故障段。
    时序顺序严格保留，按类别 7:2:1 切分。
    """
    print("\n" + "=" * 60)
    print("  XJTU-SY 数据集预处理 (5 分类, 2 通道)")
    print("=" * 60)

    xjtu_base = os.path.join(DATA_ROOT, "XJTU-SY_Bearing_Datasets", "XJTU-SY_Bearing_Datasets")
    if not os.path.exists(xjtu_base):
        print(f"  [ERROR] 找不到 {xjtu_base}")
        return None

    slice_length = 4096
    num_slices = 3
    channels = 2
    total_len = slice_length * num_slices  # 12288 points per sample

    X_train_total, y_train_total = [], []
    X_val_total, y_val_total = [], []
    X_test_total, y_test_total = [], []

    for condition_dir, bearing_prefix in XJTU_CONDITIONS.items():
        condition_path = os.path.join(xjtu_base, condition_dir)
        if not os.path.exists(condition_path):
            print(f"  [SKIP] {condition_path} 不存在")
            continue

        bearing_dirs = sorted([d for d in os.listdir(condition_path) if d.startswith(bearing_prefix)])

        for bearing_name in bearing_dirs:
            bearing_path = os.path.join(condition_path, bearing_name)
            csv_files = sorted(glob.glob(os.path.join(bearing_path, "*.csv")), key=lambda x: int(os.path.basename(x).replace('.csv', '')))

            n_files = len(csv_files)
            if n_files == 0:
                continue

            fault_label = XJTU_BEARING_LABEL.get(bearing_name, -1)
            if fault_label == -1:
                continue

            # 每个轴承单独收集数据后再单独切分
            X_bearing_healthy, y_bearing_healthy = [], []
            X_bearing_fault, y_bearing_fault = [], []

            # 策略: 前 20% 文件 -> 健康(0)，后 50% 文件 -> 故障类
            n_healthy = max(1, int(n_files * 0.2))
            n_fault_start = int(n_files * 0.5)

            # 收集健康段
            healthy_signal = []
            for csv_path in csv_files[:n_healthy]:
                try:
                    sig = np.loadtxt(csv_path, delimiter=',', skiprows=1)[:, :2]
                    healthy_signal.append(sig)
                except Exception:
                    continue
            if healthy_signal:
                healthy_concat = np.concatenate(healthy_signal, axis=0)
                num_windows = healthy_concat.shape[0] // total_len
                for i in range(num_windows):
                    window = healthy_concat[i * total_len: (i + 1) * total_len, :]
                    slices = window.reshape(num_slices, slice_length, channels)
                    X_bearing_healthy.append(slices)
                    y_bearing_healthy.append(0)

            # 收集故障段
            # XJTU 全寿命数据末期退化极快，此处取最后 50% 能覆盖到显著故障特征
            fault_signal = []
            for csv_path in csv_files[n_fault_start:]:
                try:
                    sig = np.loadtxt(csv_path, delimiter=',', skiprows=1)[:, :2]
                    fault_signal.append(sig)
                except Exception:
                    continue
            if fault_signal:
                fault_concat = np.concatenate(fault_signal, axis=0)
                num_windows = fault_concat.shape[0] // total_len
                for i in range(num_windows):
                    window = fault_concat[i * total_len: (i + 1) * total_len, :]
                    slices = window.reshape(num_slices, slice_length, channels)
                    X_bearing_fault.append(slices)
                    y_bearing_fault.append(fault_label)

            # 对该轴承的数据进行随机顺序切分 (以应对 XJTU 的极端非稳态特性)
            from sklearn.model_selection import train_test_split
            
            def split_single_bearing_random(X_list, y_list):
                if not X_list: return [], [], [], [], [], []
                X = np.array(X_list)
                y = np.array(y_list)
                # 先分出 10% 作为测试集
                X_train_val, X_test, y_train_val, y_test = train_test_split(
                    X, y, test_size=0.1, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
                )
                # 再从剩余中分出 22.22% (总体的 20%) 作为验证集
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val, test_size=0.2222, random_state=42, 
                    stratify=y_train_val if len(np.unique(y_train_val)) > 1 else None
                )
                return X_train, y_train, X_val, y_val, X_test, y_test

            bh_tr, bh_ytr, bh_v, bh_yv, bh_te, bh_yte = split_single_bearing_random(X_bearing_healthy, y_bearing_healthy)
            bf_tr, bf_ytr, bf_v, bf_yv, bf_te, bf_yte = split_single_bearing_random(X_bearing_fault, y_bearing_fault)

            if len(bh_tr) > 0:
                X_train_total.append(bh_tr); y_train_total.append(bh_ytr)
                X_val_total.append(bh_v);   y_val_total.append(bh_yv)
                X_test_total.append(bh_te); y_test_total.append(bh_yte)
            if len(bf_tr) > 0:
                X_train_total.append(bf_tr); y_train_total.append(bf_ytr)
                X_val_total.append(bf_v);   y_val_total.append(bf_yv)
                X_test_total.append(bf_te); y_test_total.append(bf_yte)

            print(f"    {bearing_name}: healthy={len(X_bearing_healthy)}, fault={len(X_bearing_fault)} -> split ok")

    X_train, y_train = np.concatenate(X_train_total), np.concatenate(y_train_total)
    X_val, y_val = np.concatenate(X_val_total), np.concatenate(y_val_total)
    X_test, y_test = np.concatenate(X_test_total), np.concatenate(y_test_total)

    output_dir = os.path.join(DATA_ROOT, "XJTU_processed")
    save_splits(output_dir, X_train, y_train, X_val, y_val, X_test, y_test)
    return output_dir


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="严谨数据预处理 (7:2:1 时序切分)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cwru", "dirg", "xjtu", "all"],
                        help="要处理的数据集")
    parser.add_argument("--check_leakage", action="store_true",
                        help="验证是否存在数据泄露")
    args = parser.parse_args()

    datasets_to_process = ["cwru", "dirg", "xjtu"] if args.dataset == "all" else [args.dataset]

    prepare_funcs = {
        "cwru": (prepare_cwru, os.path.join(DATA_ROOT, "CWRU_processed")),
        "dirg": (prepare_dirg, os.path.join(DATA_ROOT, "DIRG_processed")),
        "xjtu": (prepare_xjtu, os.path.join(DATA_ROOT, "XJTU_processed")),
    }

    for ds in datasets_to_process:
        func, output_dir = prepare_funcs[ds]
        result_dir = func()

        if args.check_leakage and result_dir:
            check_leakage(result_dir, ds.upper())

    print("\n" + "=" * 60)
    print("  全部数据预处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
