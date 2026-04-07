import json
import os
import argparse
from itertools import combinations
from typing import List, Any, Tuple

from open_source_config import DEFAULT_RESULT_DIR

def calculate_strict_accuracy(gt: List[Any], llm: List[Any]) -> float:
    """
    方法1: 严格匹配 (Strict Match)
    - 检查两个列表是否100%完全相同。
    - 返回 1.0 (正确) 或 0.0 (错误)。
    """
    return 1.0 if gt == llm else 0.0

def calculate_precision_recall_f1(gt: List[Any], llm: List[Any]) -> Tuple[float, float, float]:
    """
    计算排序任务的 precision、recall、f1-score
    - 对于排序任务，顺序严格匹配才算对
    - 每个位置都是一个预测，如果预测的位置和标准答案的位置完全一致，则该位置正确
    - Precision = 正确预测的位置数 / 预测的总位置数
    - Recall = 正确预测的位置数 / 标准答案的总位置数
    - F1-score = 2 * (precision * recall) / (precision + recall)
    """
    if not gt and not llm:
        return 1.0, 1.0, 1.0
    
    if not gt or not llm:
        return 0.0, 0.0, 0.0
    
    # 计算正确预测的位置数（位置和值都完全匹配）
    correct_positions = 0
    min_len = min(len(gt), len(llm))
    
    for i in range(min_len):
        if gt[i] == llm[i]:
            correct_positions += 1
    
    # 计算 precision、recall、f1-score
    precision = correct_positions / len(llm) if len(llm) > 0 else 0.0
    recall = correct_positions / len(gt) if len(gt) > 0 else 0.0
    
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1_score

def calculate_pairwise_accuracy(gt: List[Any], llm: List[Any]) -> float:
    """
    方法2: 相对顺序匹配 (Pairwise Match) - (推荐)
    - 检查所有可能的“配对”的相对顺序是否正确。
    - 返回一个 0.0 到 1.0 之间的分数。
    """
    if not gt:
        return 1.0 if not llm else 0.0
    
    # 检查模型答案是否包含了所有标准答案的项，防止出错
    if set(gt) != set(llm):
        # 集合不同（例如模型漏答或错答了选项），直接判为 0 分
        return 0.0

    total_pairs = 0
    correct_pairs = 0
    
    # 为 llm 答案创建一个索引地图，便于快速查找
    llm_index_map = {item: i for i, item in enumerate(llm)}

    # 遍历标准答案 (gt) 中的所有唯一配对
    for item1, item2 in combinations(gt, 2):
        # 在标准答案中, item1 总是排在 item2 之前
        total_pairs += 1
        
        # 检查在模型答案 (llm) 中，item1 是否也排在 item2 之前
        if llm_index_map[item1] < llm_index_map[item2]:
            correct_pairs += 1
            
    if total_pairs == 0:
        return 1.0  # 如果列表只有1项, 没有配对, 算100%
        
    return correct_pairs / total_pairs

def process_jsonl_file(input_path: str, output_path: str, rank_ids: set):
    """
    读取 .jsonl 文件, 只处理 ranking 题目（id 在 rank_ids 范围内）, 计算准确率, 并写入新文件。
    """
    total_count = 0
    total_strict = 0
    total_pairwise = 0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    # [新增] 0分计数器
    strict_zero_count = 0
    pairwise_zero_count = 0

    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                if not line.strip():
                    continue # 跳过空行
                    
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"警告: 跳过格式错误的行: {line.strip()}")
                    continue

                # 只处理 ranking 题目（id 在 472-505 范围内）
                item_id = data.get("id")
                if item_id not in rank_ids:
                    continue

                # 1. 获取答案 (使用 .get() 保证安全, 如果键不存在则返回空列表)
                gt_answer = data.get("gt_answer", [])
                llm_answer = data.get("llm_answer", [])

                # 确保答案是列表格式
                if not isinstance(gt_answer, list):
                    gt_answer = []
                if not isinstance(llm_answer, list):
                    llm_answer = []

                # 2. 计算指标
                strict_acc = calculate_strict_accuracy(gt_answer, llm_answer)
                pairwise_acc = calculate_pairwise_accuracy(gt_answer, llm_answer)
                precision, recall, f1_score = calculate_precision_recall_f1(gt_answer, llm_answer)

                # 3. 将新指标添加回 data 字典
                data["strict_acc"] = strict_acc
                data["pairwise_acc"] = pairwise_acc
                data["precision"] = precision
                data["recall"] = recall
                data["f1_score"] = f1_score

                # 4. 将更新后的字典写回新文件
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write("\n") # 保持 .jsonl 格式

                # 5. 累加总分
                total_count += 1
                total_strict += strict_acc
                total_pairwise += pairwise_acc
                total_precision += precision
                total_recall += recall
                total_f1 += f1_score
                
                # [新增] 统计0分个数
                if strict_acc == 0.0:
                    strict_zero_count += 1
                if pairwise_acc == 0.0:
                    pairwise_zero_count += 1
        
        print(f"\n✅ 评测完成！已处理 {total_count} 条记录。")
        print(f"结果已保存至: {output_path}")
        
        if total_count > 0:
            print("\n--- 总体准确率 ---")
            print(f"平均 [严格匹配] 准确率:   { (total_strict / total_count) * 100 :.2f} %")
            print(f"平均 [相对顺序] 准确率: { (total_pairwise / total_count) * 100 :.2f} %")
            
            print("\n--- Precision/Recall/F1-Score ---")
            print(f"平均 Precision: { (total_precision / total_count) * 100 :.2f} %")
            print(f"平均 Recall:    { (total_recall / total_count) * 100 :.2f} %")
            print(f"平均 F1-Score:  { (total_f1 / total_count) * 100 :.2f} %")
            
            print("\n--- 0分统计 ---")
            print(f"[严格匹配]   0分题数: {strict_zero_count} / {total_count} 题")
            print(f"[相对顺序] 0分题数: {pairwise_zero_count} / {total_count} 题")
        else:
            print("⚠️  未找到任何 ranking 题目（id 在 472-505 范围内）")

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_path}")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ranking accuracy from all result files.")
    
    parser.add_argument(
        "--result_dir", 
        type=str, 
        default=str(DEFAULT_RESULT_DIR), 
        help="Directory containing model result folders."
    )
    
    args = parser.parse_args()
    
    # ranking 题目的 id 范围：472-505
    rank_ids = set(range(472, 506))
    
    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        print(f"错误: 找不到结果目录 {result_dir}")
        exit(1)
    
    # 遍历 result 目录下的所有子文件夹
    model_dirs = [d for d in os.listdir(result_dir) 
                  if os.path.isdir(os.path.join(result_dir, d))]
    
    if not model_dirs:
        print(f"警告: 在 {result_dir} 中未找到任何模型文件夹")
        exit(0)
    
    print(f"找到 {len(model_dirs)} 个模型文件夹，开始处理...")
    print("=" * 60)
    
    # 处理每个模型文件夹
    for model_dir in sorted(model_dirs):
        model_path = os.path.join(result_dir, model_dir)
        safe_model_name = model_dir.replace("/", "_")
        
        # 查找 results_{model_name}.jsonl 文件
        # 先尝试使用 safe_model_name，如果不存在则尝试使用原始 model_dir
        results_file = os.path.join(model_path, f"results_{safe_model_name}.jsonl")
        if not os.path.exists(results_file):
            results_file = os.path.join(model_path, f"results_{model_dir}.jsonl")
        
        if not os.path.exists(results_file):
            print(f"⚠️  跳过 {model_dir}: 未找到 results 文件")
            continue
        
        # 输出文件直接放在模型目录下
        output_file = os.path.join(model_path, f"ranking_{safe_model_name}.jsonl")
        
        print(f"\n处理模型: {model_dir}")
        print(f"输入文件: {results_file}")
        print(f"输出文件: {output_file}")
        
        # 处理该模型的结果文件
        process_jsonl_file(results_file, output_file, rank_ids)
        print("-" * 60)
    
    print("\n✅ 所有模型处理完成！")
