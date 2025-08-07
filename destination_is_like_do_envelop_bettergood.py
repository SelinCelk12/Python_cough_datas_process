import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, medfilt, filtfilt # Filtreleme için
from scipy.ndimage import binary_erosion, binary_dilation # Morfolojik işlemler için
from scipy.signal import convolve
from scipy.signal import hilbert


# Gerçek verinizi yüklemek için bu kısmı kullanın
file_path = r"C:\Users\selin\OneDrive\Masaüstü\data_for _cough\20.csv"

try:
    # CSV dosyasını okuyun. Eğer dosyanızın başlık satırı yoksa 'header=None' kullanın.
    
    pulmonary_df = pd.read_csv(file_path, header=None) # başlık yok ve tek sütun
    pulmonary_signal = pulmonary_df.iloc[:, 0].values # İlk sütunu al

    print(f"Başarıyla veri yüklendi: {file_path}")

    # Sinyalin karakteristiğini değerlendirme
    # Örnek: Sinyalin standart sapması belirli bir eşiğin altında ve
    # belirli bir pencerede (örneğin 10000 örnek) 
    # eşik üzeri pik sayısı az ise filtreleme yap

    # Bu eşikler ve pencere boyutları veri setinize göre ayarlanmalıdır!
    std_threshold = 100 # Sinyalin standart sapması için bir eşik
    peak_count_threshold = 5 # Belirli bir pencerede izin verilen maksimum pik sayısı
    evaluation_window = 10000 # Sinyali değerlendirmek için pencere boyutu

    #Sinyali parçalara bölerek değerlendirme (veya tüm sinyali değerlendirme)
    # Basitlik için tüm sinyali değerlendirelim:

    current_std = np.std(pulmonary_signal)

    # Geçici bir eşik kullanarak pikleri sayın (bu, öksürük eşiğinden farklı olabilir)
    temp_envelope = np.abs(hilbert(pulmonary_signal))
    temp_threshold = np.mean(temp_envelope) + np.std(temp_envelope) * 3.0 # Daha düşük bir eşik
    temp_button_data = (temp_envelope > temp_threshold).astype(int)

    # Basit bir pik sayacı (1'den 0'a geçişleri sayarak)
    peak_counts = np.sum(np.diff(temp_button_data) == -1) 

    pulmonary_signal_to_process = pulmonary_signal

    current_signal_std = np.std(pulmonary_signal)

    # Geçici zarf ve pik sayımı için
    temp_analytic_signal = hilbert(pulmonary_signal)
    temp_envelope = np.abs(temp_analytic_signal)
    temp_peak_detection_threshold = np.mean(temp_envelope) + np.std(temp_envelope) * 3.0 # Bu çarpanı da ayarlayabilirsiniz
    temp_peak_binary = (temp_envelope > temp_peak_detection_threshold).astype(int)
    temp_eroded_peaks = binary_erosion(temp_peak_binary, structure=np.ones(50)).astype(int) # Küçük bir kernel
    num_detected_peaks = np.sum(np.diff(temp_eroded_peaks) == 1)

    should_filter = (current_signal_std < std_threshold) and \
                (num_detected_peaks < peak_count_threshold)

    # DEBUG: Koşul değerlerini yazdır (BU ÇIKTILARI KONTROL EDİN!)
    print(f"\nDEBUG Koşullu Filtreleme Kontrolü:")
    print(f"  current_signal_std: {current_signal_std:.2f} (Eşik: {std_threshold})")
    print(f"  num_detected_peaks: {num_detected_peaks} (Eşik: {peak_count_threshold})")
    print(f"  should_filter: {should_filter}")

    pulmonary_signal_processed = pulmonary_signal # Varsayılan olarak orijinal sinyal

    if should_filter:
        print("Koşullar karşılandı: Sinyal filtreleniyor.")
        # Butterworth filtreleme kodu buraya gelecek
        Fs = 4800 # Örnekleme frekansı (Gerçek değerinizle değiştirin!)
        nyquist = 0.5 * Fs
        lowcut = 20  # Hz
        highcut = 200 # Hz
        low = lowcut / nyquist
        high = highcut / nyquist
        order = 4 
        b, a = butter(order, [low, high], btype='band')
        pulmonary_signal_to_process = filtfilt(b, a, pulmonary_signal)
    else:
        print("Koşullar karşılanmadı: Sinyal filtrelenmiyor.")

    # Zarf hesaplaması artık filtrelenmiş veya orijinal sinyal üzerinden yapılacak
    analytic_signal = hilbert(pulmonary_signal_to_process) 
    envelope_signal = np.abs(analytic_signal) 
    # Filtrelenmiş sinyal üzerinde zarf hesaplama
  

# ... (Histerezis ve Morfolojik işlemler buradan sonra devam edecek) ...
    
    # Zaman noktalarını gerçek veri boyutuna göre ayarla
    time_points = np.arange(0, len(pulmonary_signal))

except FileNotFoundError:
    print(f"Hata: '{file_path}' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")

# --- Buradan itibaren önceki kodunuzun devamı gelir(manuel ayrıştırma)---
try:
    processed_data = []
    # Dosyayı satır satır okuyacağız
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Her satırı önce baştaki ve sondaki tırnak işaretlerinden arındırın
            # ve boşlukları temizleyin
            cleaned_line = line.strip().strip('"')

            # Şimdi bu temizlenmiş satırı virgülle bölün
            # split() metodu, stringi bir ayırıcıya göre listeye böler
            # Her elemanın baştaki/sondaki tırnakları ve boşlukları temizlensin
            parts = [part.strip().strip('"') for part in cleaned_line.split(',')]

            # Eğer sayılara dönüştürmek isterseniz, burada yapabilirsiniz.
            # Boş değerler varsa hata vermemesi için pd.to_numeric kullanmak daha sağlamdır.
            numeric_parts = []
            for p in parts:
                try:
                    numeric_parts.append(pd.to_numeric(p))
                except ValueError:
                    # Sayıya dönüşemeyenleri None veya NaN olarak bırakabilirsiniz
                    numeric_parts.append(None) # Veya float('nan')

            if numeric_parts: # Boş satırları atlamak için
                processed_data.append(numeric_parts)

    # İşlenmiş veriyi DataFrame'e dönüştürün
    # Sütun adlarını Colab çıktınıza göre belirttik
    column_names = ['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4']
    df = pd.DataFrame(processed_data, columns=column_names)
  

    print("--- DataFrame'in İlk 5 Satırı (MANUEL AYRIŞTIRMA İLE KESİN ÇÖZÜM) ---")
    print(df.head()) # <<< BU ÇIKTIYI GÖRMEK İSTİYORUZ!
    print("\n--- DataFrame'in Sütunları ve Tipleri ---")
    print(df.info()) # <<< BU ÇIKTIYI GÖRMEK İSTİYORUZ!


    # Şimdi ilk sütunu (0. indeksli) seçelim
    pulmonary_full_column = df.iloc[:, 0]
    print(f"\n--- Seçilen İlk Sütunun ('{df.columns[0]}') İlk 5 Değeri ---")
    print(pulmonary_full_column.head())
    print(f"pulmonary_full_column veri tipi: {type(pulmonary_full_column)}")
    print(f"pulmonary_full_column dtype: {pulmonary_full_column.dtype}")

except FileNotFoundError:
    print(f"Hata: {file_path} dosyası bulunamadı. Dosya yolunu kontrol edin.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")


# Sütunları çekelim 
pulmonary_data = df.iloc[:, 0]    # 1. Sütun (A)
#ambient_data = df.iloc[:, 1]      # 2. Sütun (B)
stretch_sensor_raw = df.iloc[:, 2] # 3. Sütun (C) - Ham 12-bit veri
#accelerometer_z_data = df.iloc[:, 3] # 4. Sütun (D)

# ----------- Stretch Sensor Verisinin İşlenmesi -----------
# stretch_sensor_raw'ın sayısal bir tip olduğundan emin olalım.
stretch_sensor_raw = df.iloc[:, 2].astype(int)

# Sadece LSB'yi atarak kalan değeri alıyoruz.
print(stretch_sensor_raw.dtype)


print("\nStretch Sensor (İşlenmiş) Verisinin İlk 5 Değeri:")
#print(stretch_sensor_processed.head())
print("\nButton Verisinin İlk 5 Değeri:")
# Buton verisi kodunuzda zaten doğru tanımlıydı:
button_data = (stretch_sensor_raw % 2)
print(button_data)
stretch_sensor_raw = (stretch_sensor_raw //2)
print(stretch_sensor_raw)


# ---Sinyal İşleme Adımları (Hareketli Ortalama ile)---
pulmonary_signal_centered = pulmonary_signal - np.mean(pulmonary_signal)
squared_signal = pulmonary_signal_centered**2


window_size = 1500
kernel = np.ones(window_size) / window_size
envelope_signal_raw = convolve(squared_signal, kernel, mode='same')
envelope_signal = np.sqrt(envelope_signal_raw)


# Mevcut kodunuzda bu satırdan sonra:
threshold_envelope = np.mean(envelope_signal) + np.std(envelope_signal) * 3.0
button_data_2 = (envelope_signal > threshold_envelope).astype(int)

# Bu 3-4 satırı ekleyin:
# Minimum aktivite kontrolü - izole spike'ları filtrele
min_activity_window = 300  # 200 örneklik pencerede
min_activity_threshold = 100  # en az 50 örnek aktif olmalı

activity_check = np.convolve(button_data_2.astype(float), 
                           np.ones(min_activity_window), mode='same')
button_data_2 = (button_data_2 & (activity_check >= min_activity_threshold)).astype(int)

# Sonra mevcut morfolojik işlemler devam eder:
kernel_size_erosion = 1500
kernel_size_dilation = 2400
eroded_button_data = binary_erosion(button_data_2, structure=np.ones(kernel_size_erosion)).astype(int)
final_button_data = binary_dilation(eroded_button_data, structure=np.ones(kernel_size_dilation)).astype(int)


# ... (envelope_signal hesaplandıktan sonra) ...


previous_button_state = 0 # Başlangıçta butonun 0 olduğunu varsayalım

for i in range(len(final_button_data)):
    current_button_state = final_button_data[i] # Mevcut butun değeri

    # 0'dan 1'e geçişi kontrol et (Öksürük Başlangıcı)
    if previous_button_state == 0 and current_button_state == 1:
        print(f"Örnek {i}: Öksürük Algılandı! (Başladı)")

    # 1'den 0'a geçişi kontrol et (Öksürük Bitişi)
    elif previous_button_state == 1 and current_button_state == 0:
        print(f"Örnek {i}: Öksürük Bitti.")
    # Mevcut durumu bir sonraki döngü için kaydet
    #loop for button-like data obtained from pulmonary signal
    
    previous_button_state = current_button_state


previous_button_state = 0 # Başlangıçta butonun 0 olduğunu varsayalım

for i in range(len(button_data)):
    current_button_state = button_data[i] # Mevcut butun değeri

    # 0'dan 1'e geçişi kontrol et (Öksürük Başlangıcı)
    if previous_button_state == 0 and current_button_state == 1:
        print(f"Örnek {i}: Öksürük Algılandı! (Başladı)")

    # 1'den 0'a geçişi kontrol et (Öksürük Bitişi)
    elif previous_button_state == 1 and current_button_state == 0:
        print(f"Örnek {i}: Öksürük Bitti.")

    previous_button_state = current_button_state



x_limit = 100000
y_limit_pulmonary = (0, 4000) # Pulmonary ve Ambient için
y_limit_stretch_sensor = (0, 4000)
y_limit_button = (-0.1, 1.1) # Buton 0-1 arasında olduğu için biraz boşluk bırakalım

plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(time_points, pulmonary_signal, color='blue', alpha=0.8)
plt.title('1. Orijinal Pulmoner Sinyal')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Değer (Amplitüd)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, len(pulmonary_signal))

plt.subplot(4, 1, 2)
plt.plot(time_points, envelope_signal, color='green')
plt.axhline(y=threshold_envelope, color='red', linestyle='--', label=f'Zarf Eşiği: {threshold_envelope:.2f}')
plt.title('2. Zarf Sinyali (Kare Alma + Hareketli Ortalama) ve Eşik')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Zarf Genliği')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, len(pulmonary_signal))
plt.legend()

plt.subplot(4, 1, 3)
plt.step(time_points, final_button_data, where='post', color='black')
plt.title('3. Pulmoner Sinyalden Türetilmiş Buton Verisi (Zarf ve Morfolojik İşlemlerle)')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Buton Durumu (0: Kapalı, 1: Açık)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.yticks([0, 1], ['0 (Kapalı)', '1 (Açık)'])
plt.xlim(0, len(pulmonary_signal))
plt.ylim(-0.1, 1.1)

plt.subplot(4, 1, 4)
plt.step(range(len(button_data)), button_data, color='black', linewidth=1.5)
plt.title('5. Button Verisi (Stretch Sensörden Çıkarılan)')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Buton Durumu (0: Kapalı, 1: Açık)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.yticks([0, 1], ['0 (Kapalı)', '1 (Açık)'])
plt.xlim(0, len(pulmonary_signal))
plt.ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()

print(f"Hareketli Ortalama Pencere Boyutu: {window_size} örnek")
print(f"Zarf Eşiği: {threshold_envelope:.2f}")
print(f"Morfolojik Erozyon Kernel Boyutu: {kernel_size_erosion}")
print(f"Morfolojik Dilasyon Kernel Boyutu: {kernel_size_dilation}")
print(f"Buton verisi oluşturuldu. 1'lerin oranı: {np.sum(final_button_data) / len(final_button_data):.2f}")