import pandas as pd
import matplotlib.pyplot as plt
import csv

file_path = r"C:\Users\selin\Downloads\20250602_001_standup_overClothes.csv"


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


#--------- Oksuruk Kontrolu ----------------
previous_button_state = 0 # Başlangıçta butonun 0 olduğunu varsayalım

for i in range(len(button_data)):
    current_button_state = button_data[i] # Mevcut butun değeri

    # 0'dan 1'e geçişi kontrol et (Öksürük Başlangıcı)
    if previous_button_state == 0 and current_button_state == 1:
        print(f"Örnek {i}: Öksürük Algılandı! (Başladı)")

    # 1'den 0'a geçişi kontrol et (Öksürük Bitişi)
    elif previous_button_state == 1 and current_button_state == 0:
        print(f"Örnek {i}: Öksürük Bitti.")

    # Mevcut durumu bir sonraki döngü için kaydet
    
    previous_button_state = current_button_state


# ----------- Grafik Parametreleri -----------
# Ortak yatay eksen (x ekseni) sınırı: 0'dan 100000'e kadar
x_limit = 100000
# Her grafik için ayrı dikey eksen (y ekseni) sınırları
y_limit_pulmonary = (0, 4000) # Pulmonary ve Ambient için
y_limit_ambient = (0, 4000)
y_limit_stretch_sensor = (0, 4000)
y_limit_accelerometer_z = (0,4000)
y_limit_button = (-0.1, 1.1) # Buton 0-1 arasında olduğu için biraz boşluk bırakalım

# ----------- 1. Grafik: Pulmonary (A Sütunu) -----------
plt.figure(figsize=(25, 6)) # Geniş bir figür boyutu
plt.plot(pulmonary_data, color='blue', linewidth=0.7)
plt.title('1. Pulmonary Sinyali (A Sütunu)')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Değer (Amplitüd)')
plt.xlim(0, x_limit) # Yatay eksen sınırı
plt.ylim(y_limit_pulmonary[0], y_limit_pulmonary[1]) # Dikey eksen sınırı
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# ----------- 2. Grafik: Ambient (B Sütunu) -----------
plt.figure(figsize=(25, 6))
plt.plot(ambient_data, color='red', linewidth=0.7)
plt.title('2. Ambient Sinyali (B Sütunu)')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Değer (Amplitüd)')
plt.xlim(0, x_limit) # Yatay eksen sınırı
plt.ylim(y_limit_ambient[0], y_limit_ambient[1]) # Dikey eksen sınırı
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# ----------- 3. Grafik: Stretch Sensor (C Sütunu - İşlenmiş Sensör Verisi) -----------
plt.figure(figsize=(25, 6))
plt.plot(stretch_sensor_raw, color='cyan', linewidth=0.7)
plt.title('3. Stretch Sensor Sinyali (C Sütunu - İşlenmiş Sensör Verisi)')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Değer')
plt.xlim(0, x_limit) # Yatay eksen sınırı
plt.ylim(y_limit_stretch_sensor[0], y_limit_stretch_sensor[1]) # Dikey eksen sınırı
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# ----------- 4. Grafik: Accelerometer Z-Axis (D Sütunu) -----------
plt.figure(figsize=(25, 6))
plt.plot(accelerometer_z_data, color='green', linewidth=0.7)
plt.title('4. Accelerometer Z-Axis Sinyali (D Sütunu)')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Değer')
plt.xlim(0, x_limit) # Yatay eksen sınırı
plt.ylim(y_limit_accelerometer_z[0], y_limit_accelerometer_z[1]) # Dikey eksen sınırı
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# ----------- 5. Grafik: Button Verisi (Stretch Sensörden Çıkarılan) -----------
plt.figure(figsize=(25, 4)) # Buton verisi için biraz daha kısa bir figür
plt.step(range(len(button_data)), button_data, color='black', linewidth=1.5) # Step plot
plt.title('5. Button Verisi (Stretch Sensörden Çıkarılan)')
plt.xlabel('Zaman (Örnek Numarası)')
plt.ylabel('Buton Durumu (0: Basılmadı, 1: Basıldı)')
plt.xlim(0, x_limit) # Yatay eksen sınırı
plt.ylim(y_limit_button[0], y_limit_button[1]) # Dikey eksen sınırı
plt.yticks([0, 1]) # Y ekseninde sadece 0 ve 1 göstermek için
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()