# 0007 - Tokenizer Dosyasinin Drive'a Kalici Olarak Kaydedilmesi

**Durum:** Kabul edildi

## Baglam

v3 egitimi (epoch 8, val_loss=2.60) tamamlandiktan sonra Colab oturumu
sifirlandi. Tokenizer dosyasi (`tokenizer/tokenizer_v3.json`) `.gitignore`
ile GitHub'a hic yuklenmiyordu ve sadece Colab'in gecici diskinde
(`/content/`) duruyordu - oturum sifirlaninca kayboldu.

Cozum icin veri tekrar indirilip tokenizer YENIDEN egitildi. Ancak
`fetch_dataset.py`, Hugging Face'ten streaming ile veri cektigi icin
her calistirmada BIREBIR AYNI ornekleri getirdigi garanti degil (farkli
calistirmalarda "Compute merges" sayisi 10951 vs 10876 gibi farkli
cikti - bu, alinan verinin ufak farkliliklar icerdigini gosteriyor).

Sonuc: yeniden egitilen tokenizer, checkpoint'i egitirken kullanilan
ORIJINAL tokenizer ile FARKLI kelime->numara eslesmesine sahip oldu.
Uretim testinde model tamamen anlamsiz cikti verdi - bu modelin kotu
oldugu anlamina gelmiyordu, YANLIS tokenizer ile "okunuyordu" (id 42
eskiden "def" kelimesine karsilik geliyorken, yeni tokenizer'da
tamamen farkli bir token'a karsilik gelebilir).

## Karar

Tokenizer dosyasi artik HER EGITIMDEN ONCE ve SONRA Google Drive'a da
kaydedilecek - checkpoint'lerle AYNI klasore
(`nexus_checkpoints/tokenizer_v3.json`). Boylece bir model + tokenizer
cifti HER ZAMAN birlikte, kalici sekilde saklanmis olur, oturum
sifirlansa bile ikisi arasindaki eslesme kaybolmaz.

## Sonuclar

- (+) Bir dahaki oturum kaybinda, checkpoint + tokenizer ciftini
  Drive'dan dogrudan geri yuklemek mumkun olacak, tokenizer'i
  yeniden uretmeye (ve olasi uyumsuzluga) gerek kalmayacak.
- (-) v3 epoch 8 checkpoint'i, tokenizer'i kaybolmus oldugu icin
  ARTIK KULLANILAMAZ HALDE - text uretimi icin guvenilir degil.
  v3 egitiminin SIFIRDAN, bu YENI (ve artik Drive'a kaydedilecek)
  tokenizer ile tekrar baslatilmasi gerekiyor.
- Genel ders: bir modelin agirliklari ile onu egitmek icin kullanilan
  tokenizer, AYRILMAZ bir cift olarak dusunulmeli ve HER ZAMAN
  birlikte, ayni kalicilik seviyesinde saklanmali.
