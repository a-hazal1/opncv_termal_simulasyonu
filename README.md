# opncv_termal_simulasyonu
Bu proje, bir drone tarafından çekilmiş hava görüntülerini işleyerek siyah-beyaz bir termal görüntü simülasyonu üretir.
Ek olarak, görüntüye CLAHE, gamma düzeltmesi gibi işlemler uygulanır.
Son olarak balıkgözü (barrel distortion) efekti ve zoom uygulanarak gerçek termal kamera görünümü simüle edilir.

##Gereksinimler
Bu proje aşağıdaki Python kütüphanelerine ihtiyaç duyar:
pip install opencv-python numpy
