import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, medfilt, filtfilt # Filtreleme için
from scipy.ndimage import binary_erosion, binary_dilation # Morfolojik işlemler için
from scipy.signal import convolve
from scipy.signal import hilbert

# === Parametreler ===
FOLDER_PATH = r"C:\Users\selin\OneDrive\Masaüstü\data_for _cough"
FILE_COUNT = 64
TOLERANCE = 4800

# === Kümülatif istatistik değişkenleri ===
cumulative_TP = 0
cumulative_FN = 0
cumulative_FP = 0
cumulative_TN = 0  # TN hesaplaması için eklendi

# Dosya başına sonuçları saklamak için
file_results = []

print("=== DOSYA BAZLI CONFUSION MATRIX SONUÇLARI ===\n")

# 🔹 Yeni eklenen yardımcı fonksiyonlar
def process_sensor_signal(signal):
    signal_centered = signal - np.mean(signal)
    squared_signal = signal_centered**2
    window_size = 2000
    kernel = np.ones(window_size) / window_size
    envelope_signal_raw = convolve(squared_signal, kernel, mode='same')
    envelope_signal = np.sqrt(envelope_signal_raw)
    threshold_envelope = np.mean(envelope_signal) + np.std(envelope_signal) * 1.1
    button_data = (envelope_signal > threshold_envelope).astype(int)
    kernel_size_erosion = 240
    kernel_size_dilation = 2400
    eroded = binary_erosion(button_data, structure=np.ones(kernel_size_erosion)).astype(int)
    final_data = binary_dilation(eroded, structure=np.ones(kernel_size_dilation)).astype(int)
    return final_data

def detect_cough_events(button_signal):
    events = []
    previous_state = 0
    current_start = None
    for i in range(len(button_signal)):
        current_state = button_signal[i]
        if previous_state == 0 and current_state == 1:
            current_start = i
        elif previous_state == 1 and current_state == 0:
            if current_start is not None:
                events.append((current_start, i))
                current_start = None
        previous_state = current_state
    return events

def final_cough_decision(pulm_events, stretch_events, accel_events):
    final_events = []
    for ps, pe in pulm_events:
        stretch_overlap = any(max(ps, ss) < min(pe, se) for ss, se in stretch_events)
        accel_overlap = any(max(ps, as_) < min(pe, ae) for as_, ae in accel_events)

        if stretch_overlap and accel_overlap:
            final_events.append((ps, pe))  # Kesin öksürük var
        elif not stretch_overlap:
            continue  # Pulmonary var ama stretch yok → öksürük yok
        else:
            final_events.append((ps, pe))  # Pulmonary+stretch var ama accel yok → öksürük var
    return final_events

def calculate_event_confusion_matrix_with_overlap(true_events, predicted_events, tolerance):
    TP = 0
    FN = 0
    FP = 0
    matched_predicted_indices = set()

    for true_start, true_end in true_events:
        found_match = False
        for i, (pred_start, pred_end) in enumerate(predicted_events):
            intersection_start = max(true_start, pred_start)
            intersection_end = min(true_end, pred_end)
            has_overlap = intersection_end > intersection_start
            start_match = (pred_start >= true_start - tolerance and pred_start <= true_start + tolerance)

            if (start_match or has_overlap) and i not in matched_predicted_indices:
                TP += 1
                found_match = True
                matched_predicted_indices.add(i)
                break

        if not found_match:
            FN += 1

    for i in range(len(predicted_events)):
        if i not in matched_predicted_indices:
            FP += 1

    return TP, FN, FP

# === Dosya döngüsü ===
for file_index in range(1, FILE_COUNT + 1):
    file_path = os.path.join(FOLDER_PATH, f"{file_index}.csv")
    
    print(f"--- DOSYA {file_index}: {file_index}.csv ---")
    
    try:
        # CSV dosyasını okuyun
        pulmonary_df = pd.read_csv(file_path, header=None)
        pulmonary_signal = pulmonary_df.iloc[:, 0]
        
        print(f"✓ Veri yüklendi: {len(pulmonary_signal)} örnek")
        
        # Manuel ayrıştırma
        processed_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                cleaned_line = line.strip().strip('"')
                parts = [part.strip().strip('"') for part in cleaned_line.split(',')]
                
                numeric_parts = []
                for p in parts:
                    try:
                        numeric_parts.append(pd.to_numeric(p))
                    except ValueError:
                        numeric_parts.append(None)
                
                if numeric_parts:
                    processed_data.append(numeric_parts)
        
        # DataFrame'e dönüştür
        column_names = ['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4']
        df = pd.DataFrame(processed_data, columns=column_names)
        
        # Verileri ayıkla
        pulmonary_data = pulmonary_df.iloc[:, 0]
        stretch_sensor_raw = df.iloc[:, 2].astype(int)
        
        # Button verisini çıkar
        button_data = (stretch_sensor_raw % 2)
        stretch_sensor_raw = (stretch_sensor_raw // 2)
        
        # Pulmonary sinyal işleme
        pulmonary_signal_centered = pulmonary_signal - np.mean(pulmonary_signal)
        squared_signal = pulmonary_signal_centered**2
        
        window_size = 2000
        kernel = np.ones(window_size) / window_size
        envelope_signal_raw = convolve(squared_signal, kernel, mode='same')
        envelope_signal = np.sqrt(envelope_signal_raw)
        
        threshold_envelope = np.mean(envelope_signal) + np.std(envelope_signal) * 1.0
        button_data_2 = (envelope_signal > threshold_envelope).astype(int)
        
        # Morfolojik işlemler
        kernel_size_erosion = 240
        kernel_size_dilation = 2400
        eroded_button_data = binary_erosion(button_data_2, structure=np.ones(kernel_size_erosion)).astype(int)
        final_button_data = binary_dilation(eroded_button_data, structure=np.ones(kernel_size_dilation)).astype(int)
        
        # Öksürük olaylarını tespit et
        true_coughs = detect_cough_events(button_data)
        predicted_coughs = detect_cough_events(final_button_data)

        # 🔹 Stretch ve akselerometre sinyallerini işleme
        stretch_signal = df.iloc[:, 2].astype(float)
        accel_signal = df.iloc[:, 3].astype(float)
        stretch_final = process_sensor_signal(stretch_signal)
        accel_final = process_sensor_signal(accel_signal)
        stretch_coughs = detect_cough_events(stretch_final)
        accel_coughs = detect_cough_events(accel_final)

        # 🔹 Yeni karar mantığı
        final_predicted_coughs = final_cough_decision(predicted_coughs, stretch_coughs, accel_coughs)

        # 🔹 Tolerans + çakışma kontrolü ile yeni confusion matrix
        TP, FN, FP = calculate_event_confusion_matrix_with_overlap(true_coughs, final_predicted_coughs, TOLERANCE)
        
        # TN hesaplama
        total_possible_events = len(pulmonary_signal) // 1000
        TN = max(0, total_possible_events - TP - FN - FP)
        
        # Metrikleri hesapla
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Gerçek öksürük sayısı: {len(true_coughs)}")
        print(f"Tahmin edilen öksürük sayısı: {len(final_predicted_coughs)}")
        print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1_score:.3f}")
        print()
        
        # Kümülatif değerlere ekle
        cumulative_TP += TP
        cumulative_FN += FN
        cumulative_FP += FP
        cumulative_TN += TN
        
        # Bu dosyanın sonuçlarını sakla
        file_results.append({
            'file': file_index,
            'TP': TP,
            'FN': FN,
            'FP': FP,
            'TN': TN,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_coughs': len(true_coughs),
            'predicted_coughs': len(final_predicted_coughs)
        })
        
    except FileNotFoundError:
        print(f"❌ Hata: '{file_path}' dosyası bulunamadı.")
        print()
    except Exception as e:
        print(f"❌ Bir hata oluştu: {e}")
        print()

# === KÜMÜLATIF SONUÇLAR ===
print("=" * 60)
print("=== KÜMÜLATIF CONFUSION MATRIX SONUÇLARI ===")
print("=" * 60)

print(f"\nToplam {FILE_COUNT} dosya işlendi.")
print(f"Kümülatif TP: {cumulative_TP}")
print(f"Kümülatif FN: {cumulative_FN}")
print(f"Kümülatif FP: {cumulative_FP}")
print(f"Kümülatif TN: {cumulative_TN}")

cumulative_precision = cumulative_TP / (cumulative_TP + cumulative_FP) if (cumulative_TP + cumulative_FP) > 0 else 0
cumulative_recall = cumulative_TP / (cumulative_TP + cumulative_FN) if (cumulative_TP + cumulative_FN) > 0 else 0
cumulative_f1 = 2 * (cumulative_precision * cumulative_recall) / (cumulative_precision + cumulative_recall) if (cumulative_precision + cumulative_recall) > 0 else 0

print(f"\nKümülatif Precision: {cumulative_precision:.3f}")
print(f"Kümülatif Recall: {cumulative_recall:.3f}")
print(f"Kümülatif F1-Score: {cumulative_f1:.3f}")

print(f"\n--- KÜMÜLATIF CONFUSION MATRIX ---")
print(f"                    Tahmin")
print(f"                Pozitif  Negatif")
print(f"Gerçek Pozitif    {cumulative_TP:>6}   {cumulative_FN:>6}")
print(f"Gerçek Negatif    {cumulative_FP:>6}   {cumulative_TN:>6}")

print(f"\n--- DOSYA BAZLI ÖZET TABLO ---")
print(f"{'Dosya':<5} {'TP':<4} {'FN':<4} {'FP':<4} {'TN':<4} {'Precision':<9} {'Recall':<7} {'F1':<7} {'Gerçek':<6} {'Tahmin':<6}")
print("-" * 70)
for result in file_results:
    print(f"{result['file']:<5} {result['TP']:<4} {result['FN']:<4} {result['FP']:<4} {result['TN']:<4} "
          f"{result['precision']:<9.3f} {result['recall']:<7.3f} {result['f1_score']:<7.3f} "
          f"{result['true_coughs']:<6} {result['predicted_coughs']:<6}")

print("-" * 70)
print(f"{'TOPLAM':<5} {cumulative_TP:<4} {cumulative_FN:<4} {cumulative_FP:<4} {cumulative_TN:<4} "
      f"{cumulative_precision:<9.3f} {cumulative_recall:<7.3f} {cumulative_f1:<7.3f}")

#print(f"\nTolerans değeri: {TOLERANCE} örnek")
print("İşlem tamamlandı!")
