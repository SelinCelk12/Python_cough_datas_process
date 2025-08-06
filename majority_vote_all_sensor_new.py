import os
import pandas as pd
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.signal import convolve,butter,filtfilt

FOLDER_PATH = r"C:\\Users\\selin\\OneDrive\\Masaüstü\\data_for _cough"
FILE_COUNT = 64
TOLERANCE = 4800

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
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  #
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    """
    Alçak geçiren filtreyi sinyale uygular.
    Faz gecikmesini engellemek için filtfilt kullanılır.
    """
    b, a = butter_lowpass(cutoff, fs, order=4)  #
    y = filtfilt(b, a, data)  #
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

# Ana döngü
cumulative_TP = cumulative_FN = cumulative_FP = 0
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
        TP, FN, FP = calculate_confusion(true_events, predictions, TOLERANCE)
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        print(f"TP: {TP}, FN: {FN}, FP: {FP}")
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}\n")

        cumulative_TP += TP
        cumulative_FN += FN
        cumulative_FP += FP
    except Exception as e:
        print(f"Hata: {e}")

# Sonuçların özeti
precision = cumulative_TP / (cumulative_TP + cumulative_FP) if cumulative_TP + cumulative_FP > 0 else 0
recall = cumulative_TP / (cumulative_TP + cumulative_FN) if cumulative_TP + cumulative_FN > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

print("=" * 60)
print(f"Kümülatif Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")
print("Kümülatif Confusion Matrix:")
print(f"TP: {cumulative_TP}, FN: {cumulative_FN}, FP: {cumulative_FP}")
