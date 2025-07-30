import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, medfilt, filtfilt # Filtreleme için
from scipy.ndimage import binary_erosion, binary_dilation # Morfolojik işlemler için
from scipy.signal import convolve
from scipy.signal import hilbert


# Gerçek verinizi yüklemek için bu kısmı kullanın
file_path = r"C:\Users\selin\OneDrive\Masaüstü\data_for _cough\36.csv"

try:
    # CSV dosyasını okuyun. Eğer dosyanızın başlık satırı yoksa 'header=None' kullanın.
    
    ambient_df = pd.read_csv(file_path, header=None) # başlık yok ve tek sütun
    ambient_signal = ambient_df.iloc[:, 1].values # İlk sütunu al

    print(f"Başarıyla veri yüklendi: {file_path}")


    # Zarf hesaplaması artık filtrelenmiş veya orijinal sinyal üzerinden yapılacak
    analytic_signal = hilbert(ambient_signal) 
    envelope_signal = np.abs(analytic_signal) 
    # Filtrelenmiş sinyal üzerinde zarf hesaplama
  

# ... (Histerezis ve Morfolojik işlemler buradan sonra devam edecek) ...
    
    # Zaman noktalarını gerçek veri boyutuna göre ayarla
    time_points = np.arange(0, len(ambient_signal))


except FileNotFoundError:
    print(f"Hata: '{file_path}' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")

# --- Buradan itibaren önceki kodunuzun devamı gelir(manuel ayrıştırma)---


# --- Alçak Geçiren Filtre Uygulaması ---

# Filtreleme için parametreler
fs = 4800 # Örnekleme frekansı (Hz) - KENDİ VERİNİZİN FS DEĞERİYLE DEĞİŞTİRİN!
cutoff_frequency = 400.0 # Kesim frekansı (Hz) - Optimizasyon gerekecek!
order = 4 # Filtre derecesi (genellikle 2-4 arası başlanır)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs # Nyquist frekansı
    normal_cutoff = cutoff / nyq # Normalize edilmiş kesim frekansı
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data) # Faz gecikmesini engellemek için filtfilt kullanılır
    return y

# Pulmoner sinyali alçak geçiren filtrele
ambient_signal_lowpass_filtered = butter_lowpass_filter(ambient_signal, cutoff_frequency, fs, order=order)
print(f"Alçak Geçiren Filtre uygulandı (Kesim Frekansı: {cutoff_frequency} Hz, Derece: {order}).")

# --- Görselleştirmeler ---
# Orijinal sinyal ve filtrelenmiş sinyali karşılaştıralım.

plt.figure(figsize=(14, 6))

# Orijinal Sinyal
plt.subplot(2, 1, 1)
plt.plot(time_points, ambient_signal, color='blue', alpha=0.8, label='Orijinal Sinyal')
plt.title('Orijinal Pulmoner Sinyal')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Değer (Amplitüd)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, len(ambient_signal))
plt.ylim(0, 4000) # Önceki grafiğinizdeki y limiti
plt.legend()

# Alçak Geçiren Filtrelenmiş Sinyal
plt.subplot(2, 1, 2)
plt.plot(time_points, ambient_signal_lowpass_filtered, color='purple', alpha=0.8, label=f'Alçak Geçiren Filtrelenmiş Sinyal (Kesim: {cutoff_frequency} Hz)')
plt.title(f'Alçak Geçiren Filtrelenmiş Sinyal')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Değer (Amplitüd)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, len(ambient_signal))
plt.ylim(0, 4000) # Aynı y limiti
plt.legend()

plt.tight_layout()
plt.show()

# --- Daha sonraki zarf ve öksürük tespiti adımlarında,
# pulmonary_signal yerine pulmonary_signal_lowpass_filtered'ı kullanabilirsiniz.
# Örneğin:
# pulmonary_signal_to_process = pulmonary_signal_lowpass_filtered

# pulmonary_signal_centered = pulmonary_signal_to_process - np.mean(pulmonary_signal_to_process)
# squared_signal = pulmonary_signal_centered**2
# ... (geri kalan kodunuz)

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
    ambient_full_column = df.iloc[:, 1]
    print(f"\n--- Seçilen İlk Sütunun ('{df.columns[0]}') İlk 5 Değeri ---")
    print(ambient_full_column.head())
    print(f"pulmonary_full_column veri tipi: {type(ambient_full_column)}")
    print(f"pulmonary_full_column dtype: {ambient_full_column.dtype}")

except FileNotFoundError:
    print(f"Hata: {file_path} dosyası bulunamadı. Dosya yolunu kontrol edin.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")


# Sütunları çekelim 
pulmonary_data = df.iloc[:, 0]    # 1. Sütun (A)
ambient_data = df.iloc[:, 1]      # 2. Sütun (B)
stretch_sensor_raw = df.iloc[:, 2] # 3. Sütun (C) - Ham 12-bit veri
#accelerometer_z_data = df.iloc[:, 3] # 4. Sütun (D)

# ----------- Stretch Sensor Verisinin İşlenmesi -----------
# stretch_sensor_raw'ın sayısal bir tip olduğundan emin olalım.
stretch_sensor_raw = df.iloc[:, 2].astype(int)

# Sadece LSB'yi atarak kalan değeri alıyoruz.
# print(stretch_sensor_raw.dtype)


# print("\nStretch Sensor (İşlenmiş) Verisinin İlk 5 Değeri:")
# #print(stretch_sensor_processed.head())
# print("\nButton Verisinin İlk 5 Değeri:")
# Buton verisi kodunuzda zaten doğru tanımlıydı:
button_data = (stretch_sensor_raw % 2)
# print(button_data)
stretch_sensor_raw = (stretch_sensor_raw //2)
# print(stretch_sensor_raw)

# Sinyalin karakteristiğini değerlendirme
# Örnek: Sinyalin standart sapması belirli bir eşiğin altında ve
# belirli bir pencerede (örneğin 10000 örnek) 
# eşik üzeri pik sayısı az ise filtreleme yap

# Filtreleme için parametreler (ÖRNEK DEĞERLERDİR, Kendi verinize göre ayarlamalısınız)
# fs: Örnekleme frekansı (örneğin 1000 Hz = 1000 örnek/saniye)
# lowcut: Düşük kesim frekansı (örneğin 50 Hz)
# highcut: Yüksek kesim frekansı (örneğin 500 Hz)
# Bu frekanslar öksürük sesinin frekans aralığına göre belirlenmelidir.
# Genellikle öksürük sesleri 100 Hz - 1000 Hz aralığında incelenir, ancak kullanılan sensöre göre değişebilir.
# Sensörünüzün çalışma frekans aralığını ve öksürüğün karakteristik frekanslarını araştırın.

fs = 1000  # Örnekleme frekansını buraya girin (örneğin 1000 Hz)
lowcut = 50.0  # Alt kesim frekansı (Hz)
highcut = 400.0 # Üst kesim frekansı (Hz)
order = 4      # Filtre derecesi

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data) # filtfilt faz gecikmesini ortadan kaldırır
    return y

# Pulmoner sinyali filtrele
if 'ambient_signal' in locals(): # pulmonary_signal'ın tanımlı olduğundan emin olun
    ambient_signal_filtered = butter_bandpass_filter(ambient_signal, lowcut, highcut, fs, order=order)
    print("Pulmoner sinyal başarıyla filtrelendi.")
    # Zarf hesaplaması ve diğer işlemler için pulmonary_signal_filtered'ı kullanın
    ambient_signal_centered_signal_centered = ambient_signal_filtered - np.mean(ambient_signal_filtered)
    # ... devam eden zarf hesaplaması ve diğer adımlar
else:
    print("pulmonary_signal tanımlı değil, lütfen veri yükleme kısmını kontrol edin.")

# ---Sinyal İşleme Adımları (Hareketli Ortalama ile)---
ambient_signal_centered = ambient_signal - np.mean(ambient_signal)
squared_signal = ambient_signal_centered**2


window_size = 1500
kernel = np.ones(window_size) / window_size
envelope_signal_raw = convolve(squared_signal, kernel, mode='same')
envelope_signal = np.sqrt(envelope_signal_raw)


# Mevcut kodunuzda bu satırdan sonra:
threshold_envelope = np.mean(envelope_signal) + np.std(envelope_signal) * 1.2
button_data_2 = (envelope_signal > threshold_envelope).astype(int)


# Sonra mevcut morfolojik işlemler devam eder:
kernel_size_erosion = 500
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
plt.plot(time_points, ambient_signal_lowpass_filtered , color='blue', alpha=0.8)
plt.title('1. Orijinal Ambient Sinyal')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Değer (Amplitüd)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, len(ambient_signal))

plt.subplot(4, 1, 2)
plt.plot(time_points, envelope_signal, color='green')
plt.axhline(y=threshold_envelope, color='red', linestyle='--', label=f'Zarf Eşiği: {threshold_envelope:.2f}')
plt.title('2. Zarf Sinyali (Kare Alma + Hareketli Ortalama) ve Eşik')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Zarf Genliği')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, len(ambient_signal))
plt.legend()

plt.subplot(4, 1, 3)
plt.step(time_points, final_button_data, where='post', color='black')
plt.title('3. Pulmoner Sinyalden Türetilmiş Buton Verisi (Zarf ve Morfolojik İşlemlerle)')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Buton Durumu (0: Kapalı, 1: Açık)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.yticks([0, 1], ['0 (Kapalı)', '1 (Açık)'])
plt.xlim(0, len(ambient_signal))
plt.ylim(-0.1, 1.1)

plt.subplot(4, 1, 4)
plt.step(range(len(button_data)), button_data, color='black', linewidth=1.5)
plt.title('5. Button Verisi (Stretch Sensörden Çıkarılan)')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Buton Durumu (0: Kapalı, 1: Açık)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.yticks([0, 1], ['0 (Kapalı)', '1 (Açık)'])
plt.xlim(0, len(ambient_signal))
plt.ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()

print(f"Hareketli Ortalama Pencere Boyutu: {window_size} örnek")
print(f"Zarf Eşiği: {threshold_envelope:.2f}")
print(f"Morfolojik Erozyon Kernel Boyutu: {kernel_size_erosion}")
print(f"Morfolojik Dilasyon Kernel Boyutu: {kernel_size_dilation}")
print(f"Buton verisi oluşturuldu. 1'lerin oranı: {np.sum(final_button_data) / len(final_button_data):.2f}")