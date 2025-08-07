import os
import pandas as pd
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.signal import convolve
import matplotlib.pyplot as plt

FOLDER_PATH = r"C:\\Users\\selin\\OneDrive\\Masaüstü\\data_for _cough"
FILE_COUNT = 64
TOLERANCE = 2400

WEIGHTS = {
    "pulmonary": 0.1,
    "stretch": 0.5,
    "accel": 0.3,
    "ambient": 0.1
}

THRESHOLDS = {
    "pulmonary": 0.8,
    "stretch": 0.5,
    "accel": 0.5,
    "ambient": 0.8
}

TOTAL_THRESHOLD = 0.7

def process_signal(signal):
    signal_centered = signal - np.mean(signal)
    squared_signal = signal_centered**2
    kernel = np.ones(2000) / 2000
    envelope = np.sqrt(convolve(squared_signal, kernel, mode='same'))
    threshold = np.mean(envelope) + np.std(envelope) * 0.8
    binary = (envelope > threshold).astype(int)
    eroded = binary_erosion(binary, structure=np.ones(500)).astype(int)
    dilated = binary_dilation(eroded, structure=np.ones(800)).astype(int)
    return dilated

def detect_events(binary_signal, min_duration=500):
    events = []
    start = None
    for i, val in enumerate(binary_signal):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            end = i
            # Olayın süresini kontrol et
            if (end - start) >= min_duration:
                events.append((start, end))
            start = None
    if start is not None:
        end = len(binary_signal)
        if (end - start) >= min_duration:
            events.append((start, end))
    return events
# ... (detect_events fonksiyonu buraya kadar)

def score_window(pulm_seg, stretch_seg, accel_seg, ambient_seg):
    score = 0
    if np.mean(pulm_seg) > THRESHOLDS["pulmonary"]:
        score += WEIGHTS["pulmonary"]
    if np.mean(stretch_seg) > THRESHOLDS["stretch"]:
        score += WEIGHTS["stretch"]
    if np.mean(accel_seg) > THRESHOLDS["accel"]:
        score += WEIGHTS["accel"]
    if np.mean(ambient_seg) > THRESHOLDS["ambient"]:
        score += WEIGHTS["ambient"]
    return score

def merge_close_predictions(predictions, merge_gap=3000):
    if not predictions:
        return []
    merged = [predictions[0]]
    for start, end in predictions[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= merge_gap:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged

def calculate_confusion_by_criteria(true_events, pred_events, tolerance, criteria='tolerance'):
    """
    Belirtilen kriterlere göre kargaşa matrisini hesaplar.
    criteria: 'tolerance' veya 'overlap' olabilir.
    """
    TP = FN = FP = 0
    matched_pred_indices = set()
    
    for ts, te in true_events:
        found_match = False
        for j, (ps, pe) in enumerate(pred_events):
            if j in matched_pred_indices:
                continue

            is_match = False
            if criteria == 'tolerance':
                # Sadece tolerans kriterine göre eşleşme kontrolü
                if abs(ps - ts) <= tolerance:
                    is_match = True
            elif criteria == 'overlap':
                # Sadece örtüşme (overlap) kriterine göre eşleşme kontrolü
                if min(te, pe) > max(ts, ps):
                    is_match = True
            
            if is_match:
                TP += 1
                matched_pred_indices.add(j)
                found_match = True
                break
        
        if not found_match:
            FN += 1
    
    FP = len(pred_events) - len(matched_pred_indices)
    return TP, FN, FP

# Kümülatif sonuçlar için değişkenler
cumulative_TP_tolerance = 0
cumulative_FN_tolerance = 0
cumulative_FP_tolerance = 0

cumulative_TP_overlap = 0
cumulative_FN_overlap = 0
cumulative_FP_overlap = 0

for idx in range(1, FILE_COUNT + 1):
    file_path = os.path.join(FOLDER_PATH, f"{idx}.csv")
    print(f"--- DOSYA {idx} ---")
    try:
        df = pd.read_csv(file_path, header=None)
        
        pulmonary = process_signal(df.iloc[:, 0].astype(float))
        ambient = process_signal(df.iloc[:, 1].astype(float))
        stretch = process_signal(df.iloc[:, 2].astype(float))
        accel = process_signal(df.iloc[:, 3].astype(float))

        length = len(pulmonary)
        window_size = 700
        step = 600
    
        predictions = []
        for start in range(0, length - window_size, step):
            end = start + window_size
            score = score_window(
                pulmonary[start:end], 
                stretch[start:end], 
                accel[start:end], 
                ambient[start:end]
            )
            if score >= TOTAL_THRESHOLD:
                predictions.append((start, end))

        predictions = merge_close_predictions(predictions, merge_gap=3000)

        button_data = (df.iloc[:, 2].astype(int) % 2)
        true_events = detect_events(button_data)
        
        # Tolerans ile sonuçları hesapla (sadece tolerans kontrolü)
        TP_tol, FN_tol, FP_tol = calculate_confusion_by_criteria(true_events, predictions, TOLERANCE, criteria='tolerance')
        precision_tol = TP_tol / (TP_tol + FP_tol) if TP_tol + FP_tol > 0 else 0
        recall_tol = TP_tol / (TP_tol + FN_tol) if TP_tol + FN_tol > 0 else 0
        f1_tol = 2 * (precision_tol * recall_tol) / (precision_tol + recall_tol) if precision_tol + recall_tol > 0 else 0
        
        print(f"Tolerans İçin Sonuçlar:")
        print(f"TP: {TP_tol}, FN: {FN_tol}, FP: {FP_tol}")
        print(f"Precision: {precision_tol:.3f}, Recall: {recall_tol:.3f}, F1: {f1_tol:.3f}\n")
        
        # Overlap ile sonuçları hesapla (sadece örtüşme kontrolü)
        TP_ov, FN_ov, FP_ov = calculate_confusion_by_criteria(true_events, predictions, TOLERANCE, criteria='overlap')
        precision_ov = TP_ov / (TP_ov + FP_ov) if TP_ov + FP_ov > 0 else 0
        recall_ov = TP_ov / (TP_ov + FN_ov) if TP_ov + FN_ov > 0 else 0
        f1_ov = 2 * (precision_ov * recall_ov) / (precision_ov + recall_ov) if precision_ov + recall_ov > 0 else 0
        
        print(f"Overlap İçin Sonuçlar:")
        print(f"TP: {TP_ov}, FN: {FN_ov}, FP: {FP_ov}")
        print(f"Precision: {precision_ov:.3f}, Recall: {recall_ov:.3f}, F1: {f1_ov:.3f}\n")

        # Kümülatif değerleri topla
        cumulative_TP_tolerance += TP_tol
        cumulative_FN_tolerance += FN_tol
        cumulative_FP_tolerance += FP_tol

        cumulative_TP_overlap += TP_ov
        cumulative_FN_overlap += FN_ov
        cumulative_FP_overlap += FP_ov

    except Exception as e:
        print(f"Hata: {e}")

# Kümülatif tolerans sonuçlarını hesapla ve yazdır
precision_tol_cum = cumulative_TP_tolerance / (cumulative_TP_tolerance + cumulative_FP_tolerance) if cumulative_TP_tolerance + cumulative_FP_tolerance > 0 else 0
recall_tol_cum = cumulative_TP_tolerance / (cumulative_TP_tolerance + cumulative_FN_tolerance) if cumulative_TP_tolerance + cumulative_FN_tolerance > 0 else 0
f1_tol_cum = 2 * (precision_tol_cum * recall_tol_cum) / (precision_tol_cum + recall_tol_cum) if precision_tol_cum + recall_tol_cum > 0 else 0

print("=" * 60)
print("Kümülatif Tolerans Sonuçları:")
print(f"Kümülatif Precision: {precision_tol_cum:.3f}, Recall: {recall_tol_cum:.3f}, F1 Score: {f1_tol_cum:.3f}")
print(f"Kümülatif Confusion Matrix (Tolerans):")
print(f"TP: {cumulative_TP_tolerance}, FN: {cumulative_FN_tolerance}, FP: {cumulative_FP_tolerance}")

print("=" * 60)

# Kümülatif overlap sonuçlarını hesapla ve yazdır
precision_ov_cum = cumulative_TP_overlap / (cumulative_TP_overlap + cumulative_FP_overlap) if cumulative_TP_overlap + cumulative_FP_overlap > 0 else 0
recall_ov_cum = cumulative_TP_overlap / (cumulative_TP_overlap + cumulative_FN_overlap) if cumulative_TP_overlap + cumulative_FN_overlap > 0 else 0
f1_ov_cum = 2 * (precision_ov_cum * recall_ov_cum) / (precision_ov_cum + recall_ov_cum) if precision_ov_cum + recall_ov_cum > 0 else 0

print("Kümülatif Overlap Sonuçları:")
print(f"Kümülatif Precision: {precision_ov_cum:.3f}, Recall: {recall_ov_cum:.3f}, F1 Score: {f1_ov_cum:.3f}")
print(f"Kümülatif Confusion Matrix (Overlap):")
print(f"TP: {cumulative_TP_overlap}, FN: {cumulative_FN_overlap}, FP: {cumulative_FP_overlap}")
print("=" * 60)