import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage

# Resim yükleme 
image = cv2.imread("C:\\Users\\Lenovo\\Desktop\\para.jpg")
assert image is not None, "Dosya okunamadi, os.path.exists() ile kontrol edin"

# pyrMeanShiftFiltering uygula
filtered = cv2.pyrMeanShiftFiltering(image, sp=17, sr=36)

# Gri tonlamaya çevir
gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# Eşikleme uygula
ret, thresh = cv2.threshold(gray, 550, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morfolojik işlemlerle gürültü temizleme
kernel = np.ones((5 , 5), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
# Arka planı bul
sure_bg = cv2.dilate(opening, kernel, iterations=5)

#uzaklik donusumu
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3) #en yakin arka plan pikseline olan uzakligi hesaplar
                                                            #nesne kenarlarinin vurgulanmasina yardimci olur
                           #esik deger nesne ve arka plani ayirmak icin kullanilir
ret, sure_fg = cv2.threshold(dist_transform, 0.27 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)  #veri tipini 8 bit olarak degistirme
unknown = cv2.subtract(sure_bg, sure_fg) #arkaplan ve onplani cikararak bilinmeyen bolgeyi bulur
#bilinmeyen bolge beyaz,diger bolgeler siyah olur

#etiketleme 
ret,markers= cv2.connectedComponents(sure_fg) #ret değişkeni etiketleme işlemi sonucunda kaç ayrı nesnenin tespit edildiğini belirtir
#etiketi bir arttir
markers=markers + 1  #watershed algoritmasinin dogru calismasi icin gerekli

#herhangi bir bolgeyi isaretleme
markers[unknown==255]=0

# Watershed algoritmasi uygula
labels = watershed(-dist_transform, markers, mask=opening)

# Kenarları bul
contours, _ = cv2.findContours(labels.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


# Etiketleri belirle
num_coins = len(np.unique(labels)) - 1 

# Resme sayıyı yaz
cv2.putText(image, f'Toplam para sayisi: {num_coins}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# Numaraları ekle
for label in np.unique(labels):
    if label == 0:
        continue
    mask = np.zeros(gray.shape, dtype="uint8") #paranin etrafindaki alani belirlemek icin maske uygula
    mask[markers == label] = 255      #markers array'inde, etiketi label olan bölgelerin piksellerini beyaz olarak işaretle
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # maske üzerindeki konturlari bul
    cnts = imutils.grab_contours(cnts)    #opencv surumlerindeki uyumsuzluklar 
    c = max(cnts, key=cv2.contourArea)    #konturlar arasinda en buyuk olani sec 
    ((x, y), r) = cv2.minEnclosingCircle(c)
    cv2.circle(image, (int(x), int(y)), int(r)+13, (0, 255, 0), 2)
    cv2.putText(image, f'#{label-1}', (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Sonucu göster
cv2.imshow('sonuc', image)
cv2.waitKey(0)
cv2.destroyAllWindows()