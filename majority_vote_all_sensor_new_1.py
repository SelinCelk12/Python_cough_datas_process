import os
import pandas as pd
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.signal import convolve, butter, filtfilt

FOLDER_PATH = r"C:\\Users\\selin\\OneDrive\\Masaüstü\\data_for _cough"
FILE_COUNT = 64
TOLERANCE = 2400

THRESHOLDS = {
    "pulmonary": 0.6,
    "stretch": 0.4,
    "accel": 0.4,
    "ambient": 0.6
}
#Sabit parametreler
fs = 4800  # Örnekleme frekansı
cutoff_frequency = 90.0  # Kesim frekansı (Hz)
order = 4  # Filtre derecesi

def butter_lowpass(cutoff, fs, order):
    """
    Butterworth alçak geçiren filtrenin katsayılarını hesaplar.
    """
    nyq = 0.5 * fs  # Nyquist frekansı
    normal_cutoff = cutoff / nyq  # Normalize edilmiş kesim frekansı
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    """
    Alçak geçiren filtreyi sinyale uygular.
    Faz gecikmesini engellemek için filtfilt kullanılır.
    """
    b, a = butter_lowpass(cutoff, fs, order=4)
    y = filtfilt(b, a, data)
    return y

def process_signal(signal):
    signal_centered = signal - np.mean(signal)
    squared_signal = signal_centered**2
    kernel = np.ones(2000) / 2000
    envelope = np.sqrt(convolve(squared_signal, kernel, mode='same'))
    threshold = np.mean(envelope) + np.std(envelope) * 0.9
    binary = (envelope > threshold).astype(int)
    eroded = binary_erosion(binary, structure=np.ones(600)).astype(int)
    dilated = binary_dilation(eroded, structure=np.ones(800)).astype(int)
    return dilated

def detect_events(binary_signal):
    events = []
    start = None
    for i, val in enumerate(binary_signal):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            events.append((start, i))
            start = None
    if start is not None:
        events.append((start, len(binary_signal)))
    return events

# Adaptif eşik kontrolü
def is_active(signal, threshold_key):
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    base_thresh = THRESHOLDS[threshold_key]
    adaptive_thresh = mean_val + 0.5 * std_val
    return mean_val > min(base_thresh, adaptive_thresh)

# Majority voting + ağırlıklı oy sistemi
def majority_voting_weighted(pulm_seg, stretch_seg, accel_seg, ambient_seg):
    pulm_vote = is_active(pulm_seg, "pulmonary")
    stretch_vote = is_active(stretch_seg, "stretch")
    accel_vote = is_active(accel_seg, "accel")
    ambient_vote = is_active(ambient_seg, "ambient")

    votes = 0
    votes += 1 if pulm_vote else 0
    votes += 2 if stretch_vote else 0
    votes += 2 if accel_vote else 0
    votes += 1 if ambient_vote else 0

    return votes >= 4  # Eşik: toplam 6 oy üzerinden 4 yeterli

def merge_close_predictions(predictions, merge_gap=2400):
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

def calculate_confusion_tolerance(true_events, pred_events, tolerance):
    """
    Sadece tolerans kriterine göre kargaşa matrisini hesaplar.
    """
    TP = FN = FP = 0
    matched_pred_indices = set()
    
    for ts, te in true_events:
        found_match = False
        for j, (ps, pe) in enumerate(pred_events):
            if j in matched_pred_indices:
                continue

            if abs(ps - ts) <= tolerance:
                TP += 1
                matched_pred_indices.add(j)
                found_match = True
                break
        
        if not found_match:
            FN += 1
    
    FP = len(pred_events) - len(matched_pred_indices)
    return TP, FN, FP

def calculate_confusion_overlap(true_events, pred_events):
    """
    Sadece örtüşme (overlap) kriterine göre kargaşa matrisini hesaplar.
    """
    TP = FN = FP = 0
    matched_pred_indices = set()
    
    for ts, te in true_events:
        found_match = False
        for j, (ps, pe) in enumerate(pred_events):
            if j in matched_pred_indices:
                continue

            if min(te, pe) > max(ts, ps):
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

window_size = 700
step = 600

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
        predictions = []

        for start in range(0, length - window_size, step):
            end = start + window_size
            if majority_voting_weighted(
                pulmonary[start:end],
                stretch[start:end],
                accel[start:end],
                ambient[start:end]
            ):
                predictions.append((start, end))

        predictions = merge_close_predictions(predictions, merge_gap=2400)

        button_data = (df.iloc[:, 2].astype(int) % 2)
        true_events = detect_events(button_data)
        
        # Tolerans ile sonuçları hesapla
        TP_tol, FN_tol, FP_tol = calculate_confusion_tolerance(true_events, predictions, TOLERANCE)
        precision_tol = TP_tol / (TP_tol + FP_tol) if TP_tol + FP_tol > 0 else 0
        recall_tol = TP_tol / (TP_tol + FN_tol) if TP_tol + FN_tol > 0 else 0
        f1_tol = 2 * (precision_tol * recall_tol) / (precision_tol + recall_tol) if precision_tol + recall_tol > 0 else 0
        
        print(f"Tolerans İçin Sonuçlar:")
        print(f"TP: {TP_tol}, FN: {FN_tol}, FP: {FP_tol}")
        print(f"Precision: {precision_tol:.3f}, Recall: {recall_tol:.3f}, F1: {f1_tol:.3f}\n")
        
        # Overlap ile sonuçları hesapla
        TP_ov, FN_ov, FP_ov = calculate_confusion_overlap(true_events, predictions)
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