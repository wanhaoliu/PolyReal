      
import requests
import json
import time
import os
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import re
from tqdm import tqdm
from datetime import datetime

from open_source_config import DEFAULT_LOG_DIR, DEFAULT_RESULT_DIR, build_headers, get_api_config

url = None
headers = None

lock = Lock()
logger = logging.getLogger(__name__)  # 全局logger，将在main函数中配置


def setup_api_config():
    global url, headers

    base_url, api_key = get_api_config("POLYREAL_API_BASE_URL", "POLYREAL_API_KEY")
    url = base_url + "/v1/chat/completions"
    headers = build_headers(api_key)

def create_evaluation_prompt(item):
    """根据单个条目动态创建用于评分的Prompt"""
    gt_answer = item.get("gt_answer", "N/A")
    if isinstance(gt_answer, str):
        gt_answer = gt_answer.replace("\\", "") 

    llm_answer = item.get("llm_answer", "N/A")
    if isinstance(llm_answer, str):
        llm_answer = llm_answer.replace("\\", "")

    keywords = item.get("Keywords", [])

    # 将关键词列表格式化为带编号的字符串，方便模型理解
    formatted_keywords = "\n".join([f"{i+1}. {kw}" for i, kw in enumerate(keywords)])

    prompt = f"""
You are a rigorous, fair, and professional Benchmark Evaluator. Your task is to calculate the "Precision" of a "Model's Answer" and verify its coverage of the "Key Scoring Points."

Core Calculation Formula
Precision = (Total TP Count) / (Total TP Count + Total FP Count)

Evaluation Criteria
You must strictly adhere to the following definitions to count TP and FP:

TP (True Positive):

A specific unit of information (a phrase or sentence) within the "Model's Answer".

This unit clearly and directly corresponds to one of the points in the "Key Scoring Points (Keywords)" list.

FP (False Positive):

A specific unit of information within the "Model's Answer".

It falls into any of the following categories:

[FP-Irrelevant]: The information is correct but irrelevant to the current question.

[FP-Incorrect]: A factual error or hallucination.

[FP-Redundant]: A verbose, repetitive restatement of the same point already counted as a TP.

[FP-Filler]: "Empty" phrases with no informational value (e.g., "This is a good question," "In conclusion," "It is obvious that").

Note: Points from the "Key Scoring Points (Keywords)" list that are missed (FN - False Negatives) do not participate in the Precision calculation.

Input Data
---
### Ground Truth Answer:
{gt_answer}

---
### Key Scoring Points (Keywords):
{formatted_keywords}

---
### Model's Answer to Evaluate:
{llm_answer}

---
### Your Evaluation (in JSON format only):

Evaluation Task and Output Format
You must strictly follow the steps below and output only a single JSON object. Do not add any explanatory text or markdown tags outside the JSON.

Your ONLY task is to identify all TP and FP information units from the "Model's Answer".

Analyze TP (True Positives): Identify all information units in the [Model's Answer] that meet the TP criteria. Generate a TP list.

Analyze FP (False Positives): Identify all information units in the [Model's Answer] that meet the FP criteria (including their type). Generate an FP list.

Output in the following JSON format Please adhere strictly to this JSON format for your output:
{{
  "tp_string": "(Found first TP information unit)\\n(Found second TP information unit)\\n...",
  "fp_string": "(Found first FP information unit - [FP-Type])\\n(Found second FP information unit - [FP-Type])\\n..."
}}"""
    return prompt


def extract_json_from_llm_output(llm_output: str) -> dict:
    import json
    import re

    text = llm_output.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    def fix_multiline_json_string(s):
        result = []
        inside_string = False
        for line in s.splitlines():
            if not inside_string:
                if '"' in line:
                    inside_string = line.count('"') % 2 == 1
                result.append(line)
            else:
                result[-1] += '\\n' + line.strip()
                if '"' in line:
                    inside_string = line.count('"') % 2 == 0
        return '\n'.join(result)

    text = fix_multiline_json_string(text)

    try:
        parsed = json.loads(text)
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ 无法解析 JSON：{e}\n💡 内容预览：\n{text[:1000]}")


def evaluate_answer_threadsafe(item, fail_file,max_retries=5):
    """
    线程安全函数，用于获取单个条目的评分结果。
    """
    item_id = item.get("id", "N/A")
    prompt = create_evaluation_prompt(item)

    payload = {
        "model": eval_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, # 对于评分任务，使用低temperature以保证结果稳定性
    }
    error_message = "Unknown error" # 初始化错误消息
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            response.raise_for_status()
            
            response_text = response.json()['choices'][0]['message']['content']
            eval_data = extract_json_from_llm_output(response_text)

            # 验证返回的数据是否符合预期格式
            # if 'tp_list' in eval_data and 'fp_list' in eval_data:
            if 'tp_string' in eval_data and 'fp_string' in eval_data:
                
                final_record = item.copy()
                
                tp_string = eval_data.get('tp_string', "")
                fp_string = eval_data.get('fp_string', "")

                # 从字符串解析为列表：按换行符 \n 分割，并过滤掉空字符串
                tp_list = [item for item in tp_string.split('\n') if item.strip()]
                fp_list = [item for item in fp_string.split('\n') if item.strip()]
                
                # final_record = item.copy() # 复制原始条目
                
                # tp_list = eval_data.get('tp_list', [])
                # fp_list = eval_data.get('fp_list', [])

                # 在本地计算分数
                count_tp = len(tp_list)
                count_fp = len(fp_list)
                
                if (count_tp + count_fp) > 0:
                    precision_score_val = count_tp / (count_tp + count_fp)
                else:
                    # 如果没有 TP 和 FP (例如模型答案为空)，精确率计为 0
                    precision_score_val = 0.0
                
                # 按照参考代码的格式保存分数和详情
                final_record['score'] = round(precision_score_val, 3)
                final_record['precision_details'] = {
                    "tp_list": tp_list,
                    "fp_list": fp_list,
                    "counts": {
                        "count_tp": count_tp,
                        "count_fp": count_fp
                    }
                }
            
                
                with lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(final_record, ensure_ascii=False) + "\n")
                        f.flush()
                return item_id, "Success"
            else:
                error_message = f"Invalid JSON format from model: {response_text}"

        except Exception as e:
            error_message = f"API Error: {str(e)}"

        # 如果代码运行到这里，说明发生了错误
        if attempt < max_retries:
            time.sleep(5)
    logger.warning(f"ID {item_id} 在达到最大重试次数后仍然失败: {error_message}")
    
    # 将失败的条目和错误信息写入失败文件
    final_record_failed = item.copy()
    final_record_failed['status'] = 'FAILED'
    final_record_failed['error_message'] = error_message
        # else:
        #     print(f"ID {item_id} 在达到最大重试次数后仍然失败: {error_message}")
        #     return item_id, f"Failed: {error_message}"
    with lock:
        with open(fail_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(final_record_failed, ensure_ascii=False) + "\n")
            
    return item_id, f"Failed: {error_message}"

    # return item_id, "Failed after all retries"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM answers using another LLM with concurrency.")
    parser.add_argument("--eval_model", type=str, default="gemini-2.5-flash", help="The evaluator LLM model name.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="The LLM model name.")
    parser.add_argument("--workers", type=int, default=60, help="Number of concurrent threads.")
    parser.add_argument("--input_file", type=str, default=str(DEFAULT_RESULT_DIR), help="Directory containing model result folders")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_RESULT_DIR), help="Directory to save the output file")
    args = parser.parse_args()

    eval_model_name = args.eval_model
    model_name = args.model
    setup_api_config()

    num_workers = args.workers
    safe_model_name = model_name.replace("/", "_")

    input_file = args.input_file + "/" + safe_model_name + "/results_" + safe_model_name + ".jsonl"
    args.output_dir = args.output_dir + "/" + safe_model_name
    os.makedirs(args.output_dir, exist_ok=True) 

    output_file = os.path.join(args.output_dir, f"precision_{safe_model_name}.jsonl")
    fail_file = os.path.join(args.output_dir, f"errors_eval_precision_{safe_model_name}.jsonl")
    
    # 配置日志：同时输出到控制台和文件（追加模式）
    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
    log_file = os.path.join(DEFAULT_LOG_DIR, "precision.log")
    # 清除已有的handlers，避免重复添加
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    # 文件handler（追加模式）
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 记录开始时间
    logger.info("=" * 60)
    logger.info(f"开始执行评分任务 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info(f"评分模型: {eval_model_name}")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"失败记录将保存至: {fail_file}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            all_items = [json.loads(line) for line in f]
        logger.info(f"成功加载 {len(all_items)} 条记录。")
    except Exception as e:
        logger.error(f"加载输入文件时出错: {e}")
        exit(1)

    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'id' in data and 'score' in data:
                        processed_ids.add(data['id'])
                except json.JSONDecodeError:
                    continue # 跳过损坏的行
        logger.info(f"检测到已有 {len(processed_ids)} 条成功评分的记录，将跳过它们。")

    rank_ids = set(range(472, 506))  # 472-505的rank题目id范围
    tasks = [item for item in all_items 
             if item.get('id') not in processed_ids 
             and item.get('id') not in rank_ids]
    
    skipped_count = len([item for item in all_items if item.get('id') in rank_ids])
    if skipped_count > 0:
        logger.info(f"跳过 {skipped_count} 条rank题目（id范围: 472-505）")
    
    if not tasks:
        logger.info("所有记录都已评分，程序退出。")
        exit(0)
    logger.info(f"待处理任务数量: {len(tasks)}")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(evaluate_answer_threadsafe, item, fail_file) for item in tasks]
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc=f"并发评分 ({eval_model_name})"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"一个线程任务执行时发生严重错误: {e}")

    logger.info("=" * 60)
    logger.info(f"✅ 所有评分处理完成，结果保存在 {output_file}")
    logger.info(f"任务结束时间 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    
