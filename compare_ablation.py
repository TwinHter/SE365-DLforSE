import json
import os
import sys
import argparse
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Reconfigure stdout to use UTF-8 to avoid UnicodeEncodeError in Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def calculate_metrics_for_data(results):
    if not results:
        return {}
        
    y_true = [item.get("expected_answer", []) for item in results]
    y_pred = [item.get("predicted_answer", []) for item in results]
    
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    
    overall_accuracy = accuracy_score(y_true_bin, y_pred_bin)
    
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average='micro', zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average='macro', zero_division=0
    )
    
    # Split single vs multiple
    single_true = []
    single_pred = []
    multiple_true = []
    multiple_pred = []
    multiple_partial_count = 0
    
    for item in results:
        expected = item.get("expected_answer", [])
        predicted = item.get("predicted_answer", [])
        is_partial = item.get("is_partial_correct", False)
        
        if len(expected) == 1:
            single_true.append(expected)
            single_pred.append(predicted)
        else:
            multiple_true.append(expected)
            multiple_pred.append(predicted)
            if is_partial:
                multiple_partial_count += 1
                
    if single_true:
        single_true_bin = mlb.transform(single_true)
        single_pred_bin = mlb.transform(single_pred)
        single_accuracy = accuracy_score(single_true_bin, single_pred_bin)
        single_correct_count = int(np.sum(np.all(single_true_bin == single_pred_bin, axis=1)))
    else:
        single_accuracy = 0.0
        single_correct_count = 0
        
    if multiple_true:
        multiple_true_bin = mlb.transform(multiple_true)
        multiple_pred_bin = mlb.transform(multiple_pred)
        multiple_accuracy_exact = accuracy_score(multiple_true_bin, multiple_pred_bin)
        multiple_correct_exact_count = int(np.sum(np.all(multiple_true_bin == multiple_pred_bin, axis=1)))
        multiple_accuracy_partial = multiple_partial_count / len(multiple_true)
    else:
        multiple_accuracy_exact = 0.0
        multiple_correct_exact_count = 0
        multiple_accuracy_partial = 0.0
        
    return {
        "overall_accuracy": overall_accuracy,
        "precision_micro": p_micro,
        "recall_micro": r_micro,
        "f1_micro": f1_micro,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "single_total": len(single_true),
        "single_correct": single_correct_count,
        "single_accuracy": single_accuracy,
        "multiple_total": len(multiple_true),
        "multiple_correct_exact": multiple_correct_exact_count,
        "multiple_accuracy_exact": multiple_accuracy_exact,
        "multiple_partial_count": multiple_partial_count,
        "multiple_accuracy_partial": multiple_accuracy_partial
    }

def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs Ablation (No Hard Filter) RAG Results")
    parser.add_argument("baseline", help="Path to baseline evaluation JSON report")
    parser.add_argument("ablation", help="Path to ablation evaluation JSON report")
    parser.add_argument("--output", default="results/ablation_report.md", help="Path to save markdown report")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.baseline):
        print(f"Error: Không tìm thấy file baseline tại '{args.baseline}'")
        sys.exit(1)
    if not os.path.exists(args.ablation):
        print(f"Error: Không tìm thấy file ablation tại '{args.ablation}'")
        sys.exit(1)
        
    with open(args.baseline, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    with open(args.ablation, 'r', encoding='utf-8') as f:
        ablat_data = json.load(f)
        
    base_results = base_data.get("results", [])
    ablat_results = ablat_data.get("results", [])
    
    if len(base_results) != len(ablat_results):
        print(f"Warning: Số lượng câu hỏi không khớp! Baseline: {len(base_results)}, Ablation: {len(ablat_results)}")
        # Map them by id
        base_map = {item["question_id"]: item for item in base_results}
        ablat_map = {item["question_id"]: item for item in ablat_results}
        common_ids = set(base_map.keys()) & set(ablat_map.keys())
        print(f"Đang so sánh trên {len(common_ids)} câu hỏi chung...")
        base_results = [base_map[qid] for qid in sorted(common_ids)]
        ablat_results = [ablat_map[qid] for qid in sorted(common_ids)]
        
    base_metrics = calculate_metrics_for_data(base_results)
    ablat_metrics = calculate_metrics_for_data(ablat_results)
    
    # Calculate difference
    diff_accuracy = ablat_metrics["overall_accuracy"] - base_metrics["overall_accuracy"]
    diff_f1_micro = ablat_metrics["f1_micro"] - base_metrics["f1_micro"]
    diff_f1_macro = ablat_metrics["f1_macro"] - base_metrics["f1_macro"]
    
    # Performance differences
    base_avg_time = base_data.get("timing", {}).get("avg_duration_ms", 0) / 1000
    ablat_avg_time = ablat_data.get("timing", {}).get("avg_duration_ms", 0) / 1000
    diff_time = ablat_avg_time - base_avg_time
    
    # Candidate counts differences
    base_chunks = [item.get("retrieved_chunks", 0) for item in base_results]
    ablat_chunks = [item.get("retrieved_chunks", 0) for item in ablat_results]
    base_avg_chunks = sum(base_chunks) / len(base_chunks) if base_chunks else 0
    ablat_avg_chunks = sum(ablat_chunks) / len(ablat_chunks) if ablat_chunks else 0
    diff_chunks = ablat_avg_chunks - base_avg_chunks
    
    # Analyze changes per question
    improved = []
    degraded = []
    both_correct = []
    both_wrong = []
    
    for b_item, a_item in zip(base_results, ablat_results):
        qid = b_item["question_id"]
        q_text = b_item["question"]
        b_corr = b_item["is_correct"]
        a_corr = a_item["is_correct"]
        
        if b_corr and not a_corr:
            degraded.append({
                "id": qid,
                "question": q_text,
                "expected": b_item["expected_answer"],
                "base_pred": b_item["predicted_answer"],
                "ablat_pred": a_item["predicted_answer"]
            })
        elif not b_corr and a_corr:
            improved.append({
                "id": qid,
                "question": q_text,
                "expected": b_item["expected_answer"],
                "base_pred": b_item["predicted_answer"],
                "ablat_pred": a_item["predicted_answer"]
            })
        elif b_corr and a_corr:
            both_correct.append(qid)
        else:
            both_wrong.append({
                "id": qid,
                "question": q_text,
                "expected": b_item["expected_answer"],
                "base_pred": b_item["predicted_answer"],
                "ablat_pred": a_item["predicted_answer"]
            })

    # Prepare Markdown Report
    report = []
    report.append("# BÁO CÁO ABLATION STUDY: BỎ BƯỚC HARD FILTER")
    report.append(f"\n- **File Baseline:** `{os.path.basename(args.baseline)}`")
    report.append(f"- **File Ablation:** `{os.path.basename(args.ablation)}`")
    report.append(f"- **Tổng số câu so sánh:** {len(base_results)}")
    
    report.append("\n## 1. Kết quả so sánh chính")
    report.append("| Chỉ số | Baseline (Có Hard Filter) | Ablation (Không Hard Filter) | Thay đổi |")
    report.append("| :--- | :---: | :---: | :---: |")
    report.append(f"| **Accuracy (Exact Match)** | {base_metrics['overall_accuracy']*100:.2f}% | {ablat_metrics['overall_accuracy']*100:.2f}% | {diff_accuracy*100:+.2f}% |")
    report.append(f"| **F1-score (Micro)** | {base_metrics['f1_micro']*100:.2f}% | {ablat_metrics['f1_micro']*100:.2f}% | {diff_f1_micro*100:+.2f}% |")
    report.append(f"| **F1-score (Macro)** | {base_metrics['f1_macro']*100:.2f}% | {ablat_metrics['f1_macro']*100:.2f}% | {diff_f1_macro*100:+.2f}% |")
    report.append(f"| **Thời gian chạy TB / câu** | {base_avg_time:.2f}s | {ablat_avg_time:.2f}s | {diff_time:+.2f}s ({diff_time/base_avg_time*100:+.1f}%) |")
    report.append(f"| **Số lượng ứng viên lọc TB** | {base_avg_chunks:.1f} | {ablat_avg_chunks:.1f} | {diff_chunks:+.1f} |")
    
    report.append("\n## 2. Phân tích chi tiết theo loại câu hỏi")
    report.append("| Định dạng câu hỏi | Chỉ số | Baseline (Có Hard Filter) | Ablation (Không Hard Filter) | Thay đổi |")
    report.append("| :--- | :--- | :---: | :---: | :---: |")
    report.append(f"| **Single-choice** ({base_metrics['single_total']} câu) | Accuracy | {base_metrics['single_accuracy']*100:.2f}% | {ablat_metrics['single_accuracy']*100:.2f}% | {(ablat_metrics['single_accuracy']-base_metrics['single_accuracy'])*100:+.2f}% |")
    report.append(f"| **Multiple-choice** ({base_metrics['multiple_total']} câu) | Accuracy (Exact Match) | {base_metrics['multiple_accuracy_exact']*100:.2f}% | {ablat_metrics['multiple_accuracy_exact']*100:.2f}% | {(ablat_metrics['multiple_accuracy_exact']-base_metrics['multiple_accuracy_exact'])*100:+.2f}% |")
    report.append(f"| | Accuracy (Partial Match) | {base_metrics['multiple_accuracy_partial']*100:.2f}% | {ablat_metrics['multiple_accuracy_partial']*100:.2f}% | {(ablat_metrics['multiple_accuracy_partial']-base_metrics['multiple_accuracy_partial'])*100:+.2f}% |")
    
    # By Difficulty comparison
    report.append("\n## 3. Phân tích theo độ khó câu hỏi (Difficulty)")
    base_diff_stats = base_data.get("by_difficulty", {})
    ablat_diff_stats = ablat_data.get("by_difficulty", {})
    report.append("| Mức độ khó | Số câu | Baseline Accuracy | Ablation Accuracy | Thay đổi |")
    report.append("| :---: | :---: | :---: | :---: | :---: |")
    for diff in sorted(set(base_diff_stats.keys()) | set(ablat_diff_stats.keys())):
        b_stat = base_diff_stats.get(diff, {"total": 0, "accuracy": 0})
        a_stat = ablat_diff_stats.get(diff, {"total": 0, "accuracy": 0})
        total = b_stat["total"]
        b_acc = b_stat["accuracy"]
        a_acc = a_stat["accuracy"]
        report.append(f"| **Cấp độ {diff}** | {total} | {b_acc*100:.2f}% | {a_acc*100:.2f}% | {(a_acc-b_acc)*100:+.2f}% |")

    report.append("\n## 4. Phân tích chi tiết các thay đổi")
    report.append(f"- **Số câu giữ nguyên kết quả ĐÚNG:** {len(both_correct)}")
    report.append(f"- **Số câu giữ nguyên kết quả SAI:** {len(both_wrong)}")
    report.append(f"- **Số câu CẢI THIỆN (Sai -> Đúng):** {len(improved)}")
    report.append(f"- **Số câu BỊ GIẢM HIỆU NĂNG (Đúng -> Sai):** {len(degraded)}")
    
    if improved:
        report.append("\n### Danh sách câu hỏi CẢI THIỆN (Sai -> Đúng khi bỏ Hard Filter):")
        for idx, item in enumerate(improved):
            report.append(f"{idx+1}. **{item['id']}**: {item['question']}")
            report.append(f"   - Kỳ vọng: `{item['expected']}`")
            report.append(f"   - Baseline dự đoán: `{item['base_pred']}` | Ablation dự đoán: `{item['ablat_pred']}`")
            
    if degraded:
        report.append("\n### Danh sách câu hỏi BỊ GIẢM HIỆU NĂNG (Đúng -> Sai khi bỏ Hard Filter):")
        for idx, item in enumerate(degraded):
            report.append(f"{idx+1}. **{item['id']}**: {item['question']}")
            report.append(f"   - Kỳ vọng: `{item['expected']}`")
            report.append(f"   - Baseline dự đoán: `{item['base_pred']}` | Ablation dự đoán: `{item['ablat_pred']}`")

    # Output to terminal
    print("\n" + "="*80)
    print("SO SÁNH ABLATION STUDY: BỎ HARD FILTER VS BASELINE")
    print("="*80)
    print(f"Số lượng câu so sánh: {len(base_results)}")
    print(f"Baseline Accuracy: {base_metrics['overall_accuracy']*100:.2f}%")
    print(f"Ablation Accuracy: {ablat_metrics['overall_accuracy']*100:.2f}%")
    print(f"Chênh lệch:        {diff_accuracy*100:+.2f}%")
    print(f"Baseline F1-score (Micro): {base_metrics['f1_micro']*100:.2f}%")
    print(f"Ablation F1-score (Micro): {ablat_metrics['f1_micro']*100:.2f}%")
    print(f"Thời gian trung bình / câu: Baseline = {base_avg_time:.2f}s | Ablation = {ablat_avg_time:.2f}s (Thay đổi: {diff_time:+.2f}s)")
    print(f"Số chunks ứng viên TB / câu: Baseline = {base_avg_chunks:.1f} | Ablation = {ablat_avg_chunks:.1f}")
    print("-"*80)
    print(f"Số câu Cải thiện (Sai -> Đúng): {len(improved)}")
    print(f"Số câu Bị giảm (Đúng -> Sai):   {len(degraded)}")
    print("="*80)

    # Save to file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    print(f"Báo cáo chi tiết đã được lưu tại: {args.output}\n")

if __name__ == "__main__":
    main()
