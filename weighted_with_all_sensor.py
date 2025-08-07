import os
import pandas as pd
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.signal import convolve

FOLDER_PATH = r"C:\\Users\\selin\\OneDrive\\Masaüstü\\data_for _cough"
FILE_COUNT = 64
TOLERANCE = 4800

WEIGHTS = {
    "pulmonary": 0.1,
    "stretch": 0.5,
    "accel": 0.3,
    "ambient": 0.1
}

THRESHOLDS = {
    "pulmonary": 0.7,
    "stretch": 0.4,
    "accel": 0.4,
    "ambient": 0.7
}

TOTAL_THRESHOLD = 0.7

def process_signal(signal):
    signal_centered = signal - np.mean(signal)
    squared_signal = signal_centered**2
    kernel = np.ones(2000) / 2000
    envelope = np.sqrt(convolve(squared_signal, kernel, mode='same'))
    threshold = np.mean(envelope) + np.std(envelope) * 0.8
    binary = (envelope > threshold).astype(int)
    eroded = binary_erosion(binary, structure=np.ones(400)).astype(int)
    dilated = binary_dilation(eroded, structure=np.ones(1200)).astype(int)
    return dilated

def detect_events(binary_signal, min_duration=100):
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

# Sadece stretch sinyali için adaptif dik düşüşü kontrol eden yeni bir fonksiyon
def is_active_stretch_adaptive(stretch_signal):
    # Sinyalin türevini (değişim oranını) hesapla
    derivative = np.diff(stretch_signal)

    # Türevin ortalamasını ve standart sapmasını hesapla
    mean_der = np.mean(derivative)
    std_der = np.std(derivative)

    # Eşik: Ortalama değerden standart sapmanın belli bir katı kadar düşük olan değerleri arıyoruz
    # Burada 2.5 katsayısı bir başlangıç noktasıdır, verinize göre değiştirebilirsiniz.
    adaptive_steep_threshold = mean_der - 2.5 * std_der

    # Eğer en dik düşüş, adaptif eşikten daha küçükse (daha negatifse) True döndür
    return np.min(derivative) < adaptive_steep_threshold

def score_window(pulm_seg, stretch_seg, accel_seg, ambient_seg):
    score = 0

    # Diğer sensörler için mevcut kontroller
    if np.mean(pulm_seg) > THRESHOLDS["pulmonary"]:
        score += WEIGHTS["pulmonary"]

    # YENİ ADAPTİF MANTIK: Stretch için
    if is_active_stretch_adaptive(stretch_seg): # Yeni eklediğiniz fonksiyonu çağırın
        score += WEIGHTS["stretch"]

    if np.mean(accel_seg) > THRESHOLDS["accel"]:
        score += WEIGHTS["accel"]

    if np.mean(ambient_seg) > THRESHOLDS["ambient"]:
        score += WEIGHTS["ambient"]

    return score
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

def calculate_confusion(true_events, pred_events, tolerance):
    TP = FN = FP = 0
    matched = set()
    for ts, te in true_events:
        found = False
        for i, (ps, pe) in enumerate(pred_events):
            if i in matched: continue
            if abs(ps - ts) <= tolerance or (min(te, pe) > max(ts, ps)):
                TP += 1
                matched.add(i)
                found = True
                break
        if not found:
            FN += 1
    FP = len(pred_events) - len(matched)
    return TP, FN, FP

cumulative_TP = cumulative_FN = cumulative_FP = 0

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

        predictions = merge_close_predictions(predictions, merge_gap=2400)

        button_data = (df.iloc[:, 2].astype(int) % 2)
        true_events = detect_events(button_data)

        print(f"Gerçek olay sayısı: {len(true_events)}")

        TP, FN, FP = calculate_confusion(true_events, predictions, TOLERANCE)
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        print(f"TP: {TP}, FN: {FN}, FP: {FP}")
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}\\n")

        cumulative_TP += TP
        cumulative_FN += FN
        cumulative_FP += FP
    except Exception as e:
        print(f"Hata: {e}")

precision = cumulative_TP / (cumulative_TP + cumulative_FP) if cumulative_TP + cumulative_FP > 0 else 0
recall = cumulative_TP / (cumulative_TP + cumulative_FN) if cumulative_TP + cumulative_FN > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

print("=" * 60)
print(f"Kümülatif Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")

print("=" * 60)
print(f"Kümülatif Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")
print("Kümülatif Confusion Matrix:")
print(f"TP: {cumulative_TP}, FN: {cumulative_FN}, FP: {cumulative_FP}")

