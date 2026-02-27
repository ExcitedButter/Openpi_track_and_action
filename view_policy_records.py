#!/usr/bin/env python3
"""
查看和转换 policy_records 中的 .npy 文件

用法:
    python view_policy_records.py                    # 查看所有记录的基本信息
    python view_policy_records.py --step 6            # 查看 step_6.npy 的详细信息
    python view_policy_records.py --convert json      # 转换为 JSON 格式
    python view_policy_records.py --convert pickle    # 转换为 pickle 格式
"""

import argparse
import json
import pathlib
import pickle

import numpy as np


def load_record(file_path: pathlib.Path) -> dict:
    """加载单个记录文件"""
    data = np.load(file_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.size == 1:
        data = data.item()
    return data


def unflatten_dict(flat_dict: dict, sep: str = "/") -> dict:
    """将扁平化的字典恢复为嵌套字典"""
    result = {}
    for key, value in flat_dict.items():
        parts = key.split(sep)
        node = result
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return result


def print_record_info(record: dict, step: int = None):
    """打印记录的基本信息"""
    if step is not None:
        print(f"\n=== Step {step} ===")
    else:
        print("\n=== Record Info ===")
    
    print(f"Keys: {len(record)}")
    print("\nKey structure:")
    for key in sorted(record.keys()):
        value = record[key]
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value).__name__}")


def convert_to_json(record: dict, output_path: pathlib.Path):
    """转换为 JSON 格式（注意：numpy 数组会被转换为列表）"""
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    converted = convert_numpy(record)
    with open(output_path, 'w') as f:
        json.dump(converted, f, indent=2)
    print(f"Saved to {output_path}")


def convert_to_pickle(record: dict, output_path: pathlib.Path):
    """转换为 pickle 格式（保留 numpy 数组）"""
    with open(output_path, 'wb') as f:
        pickle.dump(record, f)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="查看和转换 policy_records")
    parser.add_argument(
        "--record-dir",
        type=str,
        default="policy_records",
        help="记录文件目录 (默认: policy_records)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="查看特定步骤的记录 (例如: --step 6)",
    )
    parser.add_argument(
        "--convert",
        type=str,
        choices=["json", "pickle"],
        default=None,
        help="转换格式: json 或 pickle",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (默认: 与输入目录相同)",
    )
    parser.add_argument(
        "--unflatten",
        action="store_true",
        help="将扁平化的字典恢复为嵌套结构",
    )
    
    args = parser.parse_args()
    
    record_dir = pathlib.Path(args.record_dir)
    if not record_dir.exists():
        print(f"错误: 目录 {record_dir} 不存在")
        return
    
    # 获取所有记录文件
    record_files = sorted(record_dir.glob("step_*.npy"))
    if not record_files:
        print(f"错误: 在 {record_dir} 中未找到 step_*.npy 文件")
        return
    
    print(f"找到 {len(record_files)} 个记录文件")
    
    if args.step is not None:
        # 查看特定步骤
        file_path = record_dir / f"step_{args.step}.npy"
        if not file_path.exists():
            print(f"错误: 文件 {file_path} 不存在")
            return
        
        record = load_record(file_path)
        if args.unflatten:
            record = unflatten_dict(record)
        
        print_record_info(record, args.step)
        
        # 如果需要转换
        if args.convert:
            output_dir = pathlib.Path(args.output_dir) if args.output_dir else record_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if args.convert == "json":
                output_path = output_dir / f"step_{args.step}.json"
                convert_to_json(record, output_path)
            elif args.convert == "pickle":
                output_path = output_dir / f"step_{args.step}.pkl"
                convert_to_pickle(record, output_path)
    else:
        # 查看所有记录的基本信息
        print("\n所有记录文件:")
        for file_path in record_files:
            step_num = file_path.stem.split("_")[1]
            record = load_record(file_path)
            print(f"\nStep {step_num}:")
            print(f"  文件: {file_path.name}")
            print(f"  键数量: {len(record)}")
            print(f"  主要键: {', '.join(sorted(record.keys())[:10])}")
            if len(record) > 10:
                print(f"  ... 还有 {len(record) - 10} 个键")
        
        # 如果需要转换所有文件
        if args.convert:
            output_dir = pathlib.Path(args.output_dir) if args.output_dir else record_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in record_files:
                step_num = file_path.stem.split("_")[1]
                record = load_record(file_path)
                if args.unflatten:
                    record = unflatten_dict(record)
                
                if args.convert == "json":
                    output_path = output_dir / f"step_{step_num}.json"
                    convert_to_json(record, output_path)
                elif args.convert == "pickle":
                    output_path = output_dir / f"step_{step_num}.pkl"
                    convert_to_pickle(record, output_path)


if __name__ == "__main__":
    main()
