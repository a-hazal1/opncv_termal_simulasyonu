import cv2  
import numpy as np  

def simulate_bw_thermal(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("resim yüklenemedi!")  # resim bulunamazsa uyarı ver
        return

    # ------------------------
    # 1. HSV ile yeşil alan tespiti
    # ------------------------
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Yeşil ton aralığı
    lower_green = np.array([40, 40, 40])  # H değerini biraz yükselttik
    upper_green = np.array([80, 255, 255])       # üst sınır (H, S, V)

    mask = cv2.inRange(hsv, lower_green, upper_green)  # Yeşil bölgelerin maskesi

    # ------------------------
    # 2. Termal işlem sadece yeşil olmayan bölgelerde uygulanacak
    # ------------------------
    # Yeşil olmayan alanları al
    non_green = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))

    # Gri tonlamaya çevir
    gray_image = cv2.cvtColor(non_green, cv2.COLOR_BGR2GRAY)     

    # CLAHE ile lokal kontrast artırma (daha yumuşak ayar)
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8))     
    gray_clahe = clahe.apply(gray_image)

    # Gamma düzeltmesi (daha düşük değer, 1.2 - 1.5 arası)
    gamma = 2.2
    gamma_corrected = np.array(255 * (gray_clahe / 255.0) ** (1/gamma), dtype='uint8')

    # Normalize et (0-255 arası değerler)
    gray_normalized = cv2.normalize(gamma_corrected, None, 0, 255, cv2.NORM_MINMAX)    

    # Renkleri ters çevir (insanlar beyaz, arka plan siyah)
    thermal_bw = cv2.bitwise_not(gray_normalized)

    # ------------------------
    # 3. Yeşil bölgeleri tamamen siyah yap
    # ------------------------
    thermal_bw[mask > 0] = 0  # Yeşil bölgeleri karart

    return thermal_bw 

def barrel_distortion(image, k=0.09):
    h, w = image.shape[:2]
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    r = np.sqrt(x**2 + y**2)
    x_distorted = x * (1 + k * r**2)
    y_distorted = y * (1 + k * r**2)
    map_x = ((x_distorted + 1) * w - 1) / 2
    map_y = ((y_distorted + 1) * h - 1) / 2
    distorted_img = cv2.remap(
        image, 
        map_x.astype(np.float32), 
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_REPLICATE
    )
    return distorted_img

def zoom_and_crop(image, zoom_factor=1.0):
    h, w = image.shape[:2]
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2
    cropped = image[start_y:start_y+new_h, start_x:start_x+new_w]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
    return resized 

# ---------------------------
# Ana iş akışı
# ---------------------------
image_path = "dron_2024-05-22_21-20-43-632821.jpeg"  # işlenecek resmin yolu
thermal_frame = simulate_bw_thermal(image_path)
distorted = barrel_distortion(thermal_frame, k=0.09)  # k küçültüldü, yuvarlaklık azaldı
zoomed = zoom_and_crop(distorted, zoom_factor=1.2)

# Sonuç zaten grayscale olduğu için tekrar dönüştürmeye gerek yok
final_bw = zoomed  

if final_bw is not None:
    cv2.imwrite("bw_3.png", final_bw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
