import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, medfilt, filtfilt # Filtreleme için
from scipy.ndimage import binary_erosion, binary_dilation # Morfolojik işlemler için
from scipy.signal import convolve, freqz
from scipy.signal import hilbert


# Gerçek verinizi yüklemek için bu kısmı kullanın
file_path = r"C:\Users\selin\OneDrive\Masaüstü\data_for _cough\8.csv"

try:
    # CSV dosyasını okuyun. Eğer dosyanızın başlık satırı yoksa 'header=None' kullanın.
    
    stretch_df = pd.read_csv(file_path, header=None) # başlık yok ve tek sütun
    stretch_signal = stretch_df.iloc[:, 2].values # İlk sütunu al

    print(f"Başarıyla veri yüklendi: {file_path}")
    analytic_signal = hilbert(stretch_signal)
    envelope_signal = np.abs(analytic_signal) # Zarf, analitik sinyalin mutlak değeri

# ... (Histerezis ve Morfolojik işlemler buradan sonra devam edecek) ...
    
    # Zaman noktalarını gerçek veri boyutuna göre ayarla
    time_points = np.arange(0, len(stretch_signal))

except FileNotFoundError:
    print(f"Hata: '{file_path}' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")

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

def plot_spectrograms(original_signal, filtered_signal, fs, cutoff_frequency):
    """
    Orijinal ve filtrelenmiş sinyallerin spektrogramlarını çizen fonksiyon.
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Orijinal Sinyalin Spektrogramı
    axes[0].set_title('Orijinal Sinyal Spektrogramı')
    axes[0].specgram(original_signal, Fs=fs, NFFT=1024, noverlap=512, cmap='viridis')
    axes[0].set_xlabel('Zaman (s)')
    axes[0].set_ylabel('Frekans (Hz)')
    axes[0].set_ylim(0, fs / 2)
    axes[0].axhline(y=cutoff_frequency, color='red', linestyle='--', label=f'Kesim Frekansı: {cutoff_frequency} Hz')
    axes[0].legend()
    
    # Filtrelenmiş Sinyalin Spektrogramı
    axes[1].set_title('Filtrelenmiş Sinyal Spektrogramı')
    axes[1].specgram(filtered_signal, Fs=fs, NFFT=1024, noverlap=512, cmap='viridis')
    axes[1].set_xlabel('Zaman (s)')
    axes[1].set_ylabel('Frekans (Hz)')
    axes[1].set_ylim(0, fs / 2)
    axes[1].axhline(y=cutoff_frequency, color='red', linestyle='--', label=f'Kesim Frekansı: {cutoff_frequency} Hz')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# Örnek Kullanım:
# Sadece bir dosya için spektrogramı çizdirelim.
# `weighted_with_all_sensor.py` dosyasındaki ilk dosyayı kullanabiliriz.


try:
    df = pd.read_csv(file_path, header=None)
    # Burada hangi sensörün verisini kullanacağınızı belirleyin
    # Örneğin, pulmoner sensör için 0. sütunu seçelim
    original_signal = df.iloc[:, 2].astype(float)

    # Filtreleme parametreleri
    fs = 4800  # Örnekleme frekansı
    cutoff_frequency = 50.0 # Kesim frekansı
    order = 4 # Filtre derecesi

    # Orijinal sinyali filtreleme
    filtered_signal = butter_lowpass_filter(original_signal, cutoff_frequency, fs, order)

    # Spektrogramları çizdirme
    plot_spectrograms(original_signal, filtered_signal, fs, cutoff_frequency)

except FileNotFoundError:
    print(f"Hata: Dosya bulunamadı - {file_path}")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
 


# Görselleştirme fonksiyonu
def visualize_filter_effect(original_signal, filtered_signal, cutoff_frequency, order):
    """
    Orijinal ve filtrelenmiş sinyali karşılaştıran bir grafik çizer.
    """
    time_points = np.arange(len(original_signal))

    plt.figure(figsize=(14, 6))

    # Orijinal sinyal grafiği
    plt.subplot(2, 1, 1)
    plt.plot(time_points, original_signal, color='blue', alpha=0.8, label='Orijinal Sinyal')
    plt.title('Orijinal Pulmoner Sinyal')
    plt.xlabel('Zaman (Örnek Numarası)')
    plt.ylabel('Değer (Amplitüd)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, len(original_signal))
    plt.ylim(-0.1, 1.1)  # Eğer sinyaliniz 0-1 aralığında ise bu limitleri kullanabilirsiniz
    plt.legend()

    # Filtrelenmiş sinyal grafiği
    plt.subplot(2, 1, 2)
    plt.plot(time_points, filtered_signal, color='red', alpha=0.8, label='Filtrelenmiş Sinyal')
    plt.title(f'Alçak Geçiren Filtre Uygulandı (Kesim Frekansı: {cutoff_frequency} Hz, Derece: {order})')
    plt.xlabel('Zaman (Örnek Numarası)')
    plt.ylabel('Değer (Amplitüd)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, len(filtered_signal))
    plt.ylim(-0.1, 1.1)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_filter_frequency_response(b, a, fs):
    """
    Filtre katsayılarını (b, a) kullanarak filtrenin frekans tepkisini çizer.
    """
    w, h = freqz(b, a, worN=8000, fs=fs)
    
    # Frekans tepkisini genlik (dB) v
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    ax1.set_title('Butterworth Alçak Geçiren Filtre Frekans Tepkisi')
    ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Genlik [dB]', color='b')
    ax1.set_xlabel('Frekans (Hz)')
    ax1.grid()
    ax1.axvline(cutoff_frequency, color='r', linestyle='--', label=f'Kesim Frekansı: {cutoff_frequency} Hz')
    ax1.axhline(-3, color='g', linestyle='--', label='-3 dB Noktası')
    ax1.tick_params(axis='y', colors='b')
    ax1.set_ylim(-60, 5)
    ax1.set_xlim(0, 20)
    ax1.legend()
    
    # ax2 = ax1.twinx()
    # ax2.plot(w, np.unwrap(np.angle(h)) * 180 / np.pi, 'g')
    # ax2.set_ylabel('Faz [Derece]', color='g')
    # ax2.tick_params(axis='y', colors='g')
    
    plt.tight_layout()
    plt.show()

# Kullanım:
# Filtre katsayılarını hesapla
b, a = butter_lowpass(cutoff_frequency, fs, order)

# Frekans tepkisini çizdir
plot_filter_frequency_response(b, a, fs)

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
ambient_data = df.iloc[:, 1]      # 2. Sütun (B)
stretch_sensor_raw = df.iloc[:, 2] # 3. Sütun (C) - Ham 12-bit veri
accelerometer_z_data = df.iloc[:, 3] # 4. Sütun (D)

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
stretch_signal_centered = stretch_signal - np.mean(stretch_signal)
squared_signal = stretch_signal_centered**2

window_size = 1500
kernel = np.ones(window_size) / window_size
envelope_signal_raw = convolve(squared_signal, kernel, mode='same')
envelope_signal = np.sqrt(envelope_signal_raw)

threshold_envelope = np.mean(envelope_signal) + np.std(envelope_signal) * 0.5
button_data_2 = (envelope_signal > threshold_envelope).astype(int)


kernel_size_erosion = 500
kernel_size_dilation =800
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



# --- Görselleştirmeler (önceki kodlara benzer) ---
x_limit = 100000
# Her grafik için ayrı dikey eksen (y ekseni) sınırları
y_limit_pulmonary = (0, 4000) # Pulmonary ve Ambient için
y_limit_stretch_sensor = (0, 4000)
y_limit_button = (-0.1, 1.1) # Buton 0-1 arasında olduğu için biraz boşluk bırakalım

plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(time_points, stretch_signal, color='blue', alpha=0.8)
plt.title('1. Orijinal Pulmoner Sinyal')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Değer (Amplitüd)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, len(stretch_signal))

plt.subplot(4, 1, 2)
plt.plot(time_points, stretch_signal, color='green')
plt.axhline(y=threshold_envelope, color='red', linestyle='--', label=f'Zarf Eşiği: {threshold_envelope:.2f}')
plt.title('2. Zarf Sinyali (Kare Alma + Hareketli Ortalama) ve Eşik')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Zarf Genliği')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, len(stretch_signal))
plt.legend()

plt.subplot(4, 1, 3)
plt.step(time_points, final_button_data, where='post', color='black')
plt.title('3. Pulmoner Sinyalden Türetilmiş Buton Verisi (Zarf ve Morfolojik İşlemlerle)')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Buton Durumu (0: Kapalı, 1: Açık)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.yticks([0, 1], ['0 (Kapalı)', '1 (Açık)'])
plt.xlim(0, len(stretch_signal))
plt.ylim(-0.1, 1.1)

plt.subplot(4, 1, 4)
plt.step(range(len(button_data)), button_data, color='black', linewidth=1.5)
plt.title('5. Button Verisi (Stretch Sensörden Çıkarılan)')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Buton Durumu (0: Kapalı, 1: Açık)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.yticks([0, 1], ['0 (Kapalı)', '1 (Açık)'])
plt.xlim(0, len(stretch_signal))
plt.ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()

print(f"Hareketli Ortalama Pencere Boyutu: {window_size} örnek")
print(f"Zarf Eşiği: {threshold_envelope:.2f}")
print(f"Morfolojik Erozyon Kernel Boyutu: {kernel_size_erosion}")
print(f"Morfolojik Dilasyon Kernel Boyutu: {kernel_size_dilation}")

print(f"Buton verisi oluşturuldu. 1'lerin oranı: {np.sum(final_button_data) / len(final_button_data):.2f}")

