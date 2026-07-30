# 0013 - KV-Cache Eklenmesi (Uretim Hizlandirma)

**Durum:** Kabul edildi

## Baglam

ADR-0002'de bilinen bir eksiklik olarak not edilmisti: `generate()`
fonksiyonu her yeni token icin TUM diziyi (prompt + su ana kadar
uretilenler) yeniden isliyordu - bu, uretim uzadikca giderek
yavaslayan, verimsiz bir yontemdi. v4'un basariya ulasmasi
(ADR-0012) sonrasi, bu artik ele alinmasi gereken bir sonraki adimdi.

Onemli tasarim kısıtı: KV-cache eklemek, v5'in (daha buyuk veri/model)
gerektirdigi SIFIRDAN egitimden BAGIMSIZ olarak, MEVCUT v4
checkpoint'ini (`model_v4_best.pt`) BOZMADAN yapilabilmeliydi - cunku
KV-cache sadece attention hesaplamasinin runtime davranisini
degistirir, model agirliklarinin SEKLINI/SAYISINI degistirmez.

## Karar

`models/model_v1.py` guncellendi:

1. `CausalSelfAttention.forward()` artik opsiyonel `kv_cache`
   parametresi aliyor. Cache verilmisse, yeni hesaplanan key/value'lar
   gecmis (past) key/value'larla birlestiriliyor; `is_causal` sadece
   cache YOKKEN (egitim/prefill) `True` oluyor - tek-token decode
   sirasinda ekstra maskeye gerek yok (yeni token dogasi geregi sadece
   kendinden onceki her seyi gorebilir).
2. `KodLLM_v1.forward()` artik `kv_caches` parametresi aliyor ve
   3'lu deger donduruyor: `(logits, loss, new_kv_caches)`. Egitim
   davranisi (kv_caches=None ile cagrildiginda) ONCEKIYLE BIREBIR
   AYNI - sadece cagiran kodun 3 deger ile unpack etmesi gerekiyor.
3. `train.py`'daki iki `model(...)` cagrisi, yeni 3'lu donus degerine
   gore guncellendi (`_, loss, _ = model(...)`).
4. `generate()` fonksiyonu, prompt'u bir kerede isleyip (prefill) ilk
   cache'i olusturuyor, sonrasinda HER ADIMDA SADECE yeni token'i
   isliyor (tum diziyi degil).
5. `model_v1.py`'nin sonuna, KV-cache'li adim-adim uretimin,
   cache'siz tam-gecis ile MATEMATIKSEL OLARAK AYNI sonucu verdigini
   dogrulayan bir sanity check eklendi.

## Sonuclar

- (+) Mevcut `model_v4_best.pt` checkpoint'i, agirliklarin sekli
  degismedigi icin bu yeni kodla SORUNSUZ yuklenebiliyor - hicbir
  yeniden egitim gerekmedi.
- (+) Uzun uretimlerde (`max_new_tokens` buyudukce) belirgin hizlanma
  bekleniyor - her adimda O(T) yerine O(1) yeni token isleniyor.
- (-) Kod karmasikligi bir miktar artti (forward artik 3 deger
  donduruyor, kv_caches yonetimi eklendi) - ama bu, performans
  kazancina deger bir odun olarak degerlendirildi.
- Bu degisiklik v5 (RoPE + daha buyuk veri/model) ONCESINDE,
  BAGIMSIZ olarak yapildi - boylece v5'in "sifirdan egitim gerektiren"
  degisiklikleriyle (RoPE) karismadan, ayri ayri test edilebildi.
