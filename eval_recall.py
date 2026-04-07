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
        # 移除所有反斜杠
        gt_answer = gt_answer.replace("\\", "") 

    llm_answer = item.get("llm_answer", "N/A")
    if isinstance(llm_answer, str):
        # 移除所有反斜杠
        llm_answer = llm_answer.replace("\\", "")
    # llm_answer = item.get("llm_answer", "N/A")
    keywords = item.get("Keywords", [])

    # 将关键词列表格式化为带编号的字符串，方便模型理解
    formatted_keywords = "\n".join([f"{i+1}. {kw}" for i, kw in enumerate(keywords)])

    prompt = f"""
You are an exceptionally strict, meticulous, and critical grader specializing in polymer science. Your task is to evaluate a "Model's Answer" based on a list of "Key Scoring Points" and a "Ground Truth Answer".

### Evaluation Task ###
You must evaluate *each* "Key Scoring Point" *sequentially* based on the following two dimensions:

**1. Completeness (`met` - Binary Score):**
* Does the "Model's Answer" clearly and unambiguously cover the core concept of this "Key Scoring Point"?
* This is a strict binary (0 or 1) check.
* **1 (Met):** The point is clearly and directly addressed, and its explanation **has no critical information missing compared to the relevant explanation of this point in the "Ground Truth Answer".**
* **0 (Not Met):** The point is missing, glossed over, vaguely implied, or **has significant information omissions compared to the "Ground Truth Answer".**

**2. Professional Quality (`quality_score` - Float 0.0 to 1.0):**
* **If `met` is 0, this score MUST be 0.0.**
* If `met` is 1, you must then grade *how well* the point was covered, based on the following scale:
    * **1.0 (Perfect):** The explanation is impeccable in fact, depth, and accuracy. The terminology is professional, the logic is rigorous, and the clarity is at the level of a top-tier journal.
    * **0.5 (Average):** The point is covered, but the explanation is superficial, imprecise, logically flawed, uses casual language, **or is vague and unfocused, making the core concept difficult to grasp.**
    * **0.1 (Poor):** The point is mentioned, but its explanation contains severe factual errors or logical fallacies.
### Your Evaluation (in JSON format only):
You must output a SINGLE JSON object. Do not add any text before or after the JSON object.
Your JSON object must contain exactly three keys: "met", "quality_score", and "reasoning".
{{

  "met": [A list of integers, where each integer corresponds to a key point. Use 1 for met, and 0 for not met.],
  "quality_score": [A list of float number, A single float number for the actual score of this point ,A float number between 0.0 and 1.0],
  "reasoning": "A step-by-step, critical explanation. If a point is judged 'not met', clearly specify in which aspect  it failed to meet the expert standard. If judged 'met', provide a detailed justification for why the answer irrefutably meets all criteria.If 'Met', you *must* then provide a detailed justification for the specific `quality_score` given."
}}
---
### Key Scoring Points (Keywords):
(You must evaluate these sequentially)
{formatted_keywords}

---
### Ground Truth Answer:
(The benchmark for evaluating "Completeness" for any omissions)
{gt_answer}

---
### Model's Answer to Evaluate:
{llm_answer}

---
### Your Evaluation (in JSON format only):
"""
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


def evaluate_answer_threadsafe(item, fail_file, eval_model_name, output_file, max_retries=5):
    """
    线程安全函数，用于获取单个条目的评分结果。
    """
    item_id = item.get("id", "N/A")
    prompt = create_evaluation_prompt(item)
    final_record = item.copy() # 复制原始条目
    completeness_score_avg = 0.0
    completeness_call_succeeded = False # 用于标记成功
    eval_data = None

    payload = {
        "model": eval_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, # 对于评分任务，使用低temperature以保证结果稳定性
    }
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=3000)
            response.raise_for_status()
            
            response_text = response.json()['choices'][0]['message']['content']
            eval_data = extract_json_from_llm_output(response_text)
            
            # 验证返回的数据是否符合预期格式
            if 'met' in eval_data and 'quality_score' in eval_data:
                quality_scores = eval_data.get('quality_score', [])
                # 成功解析并验证格式
                # 计算平均 "质量" 分作为 "完整性" 的总分
                if quality_scores:
                    completeness_score_avg = sum(quality_scores) / len(quality_scores)
                final_record['completeness_score_avg'] = completeness_score_avg
                final_record['completeness_details'] = eval_data # 保存 met/quality 详情
                completeness_call_succeeded = True # <--- 标记成功
                break # <--- 成功后退出循环
                
            else:
                error_message = f"Invalid JSON format: {response_text}"    

        except Exception as e:
            error_message = f"API Error: {str(e)}"

        # 等待后重试
        if attempt < max_retries:
            time.sleep(5)
    if completeness_call_succeeded:
        try:
            with lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(final_record, ensure_ascii=False) + "\n")
            return item_id, "Success"
        except Exception as e:
             # 如果最终计分或写入成功文件时发生意外错误，也记录到失败文件
             final_record['final_save_error'] = str(e)
 
             # Fall through to the failure case below

    # 如果任何步骤失败，则执行以下操作
    final_record['status'] = 'FAILED'
    final_record['error_message'] = error_message
    final_record['eval_data']= eval_data
    with lock:
        with open(fail_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(final_record, ensure_ascii=False) + "\n")
    return item_id, "Failed and Logged"
    
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM answers using another LLM with concurrency.")
    parser.add_argument("--eval_model", type=str, default="gemini-2.5-flash", help="The evaluator LLM model name.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="The LLM model name.")
    parser.add_argument("--workers", type=int, default=80, help="Number of concurrent threads.")
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

    output_file = os.path.join(args.output_dir, f"recall_{safe_model_name}.jsonl")
    fail_file = os.path.join(args.output_dir, f"errors_eval_recall_{safe_model_name}.jsonl")
    
    # 配置日志：同时输出到控制台和文件（追加模式）
    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
    log_file = os.path.join(DEFAULT_LOG_DIR, "recall.log")
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

    # 2. 断点续传：检查哪些ID已经处理过
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 确保'score'和'reasoning'字段存在，才算处理过
                    if 'id' in data and 'completeness_score_avg' in data:
                        processed_ids.add(data['id'])
                except json.JSONDecodeError:
                    continue # 跳过损坏的行
        logger.info(f"检测到已有 {len(processed_ids)} 条成功评分的记录，将跳过它们。")

    # 3. 筛选出未处理的任务，并跳过id在472-505范围内的rank题目
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

    # 4. 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(evaluate_answer_threadsafe, item, fail_file, eval_model_name, output_file, max_retries=8) for item in tasks]
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc=f"并发评分 ({eval_model_name})"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"一个线程任务执行时发生严重错误: {e}")

    logger.info("=" * 60)
    logger.info(f"✅ 所有评分处理完成，结果保存在 {output_file}")
    logger.info(f"任务结束时间 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    
