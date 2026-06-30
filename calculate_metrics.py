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

def evaluate_metrics(report_path):
    if not os.path.exists(report_path):
        print(f"Error: Không tìm thấy file báo cáo tại '{report_path}'")
        return
        
    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    results = data.get("results", [])
    if not results:
        print("Không tìm thấy kết quả ('results') nào trong file báo cáo.")
        return
        
    # Lấy nhãn thực tế (expected) và nhãn dự đoán (predicted)
    y_true = [item.get("expected_answer", []) for item in results]
    y_pred = [item.get("predicted_answer", []) for item in results]
    
    # Sử dụng MultiLabelBinarizer để chuyển đổi sang ma trận nhị phân (one-hot)
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    
    # 1. Tính toán các chỉ số chung
    overall_accuracy = accuracy_score(y_true_bin, y_pred_bin)
    
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average='micro', zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average='macro', zero_division=0
    )
    
    # 2. Phân tách câu hỏi Single-choice vs Multiple-choice
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
                
    # Tính accuracy cho từng loại câu hỏi
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

    # In kết quả chi tiết
    print(f"\n================ BÁO CÁO ĐÁNH GIÁ (File: {os.path.basename(report_path)}) ================")
    print(f"Tổng số câu hỏi: {len(results)}")
    print(f"Accuracy (Exact Match): {overall_accuracy*100:.2f}%")
    print(f"Precision (Micro):      {p_micro*100:.2f}%")
    print(f"Recall (Micro):         {r_micro*100:.2f}%")
    print(f"F1-score (Micro):       {f1_micro*100:.2f}%")
    print(f"Precision (Macro):      {p_macro*100:.2f}%")
    print(f"Recall (Macro):         {r_macro*100:.2f}%")
    print(f"F1-score (Macro):       {f1_macro*100:.2f}%")
    
    print("\n--- Phân tích theo định dạng câu hỏi ---")
    print(f"Single-choice (1 đáp án): {len(single_true)} câu")
    print(f"  - Single đúng (Exact Match): {single_correct_count} / {len(single_true)} ({single_accuracy*100:.2f}%)")
    
    print(f"Multiple-choice (nhiều đáp án): {len(multiple_true)} câu")
    print(f"  - Multiple đúng (Exact Match): {multiple_correct_exact_count} / {len(multiple_true)} ({multiple_accuracy_exact*100:.2f}%)")
    print(f"  - Multiple đúng một phần (Partial): {multiple_partial_count} / {len(multiple_true)} ({multiple_accuracy_partial*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tính toán các chỉ số RAG từ file report JSON.")
    parser.add_argument(
        "report_path",
        nargs="?",
        default="results/evaluation_report_20260630_014236.json",
        help="Đường dẫn tới file JSON report (mặc định: results/evaluation_report_20260630_014236.json)"
    )
    args = parser.parse_args()
    evaluate_metrics(args.report_path)
