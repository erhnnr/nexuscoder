# Architecture Decision Records (ADR)

Bu klasor, NexusCoder projesinde alinan onemli teknik kararlari kaydeder.
Her ADR: neden o karar alindi, hangi alternatifler dusunuldu, sonucu ne oldu.

Amac: "neden boyle yaptik?" sorusunun cevabinin hafizada degil, yazida
durmasi. Ilerledikce yeni kararlar icin yeni numarali dosyalar eklenir
(0006, 0007, ...). Eski bir karar degisirse, eski ADR SILINMEZ, "Durum"
alani "degistirildi -> 000X" olarak guncellenir ve yeni bir ADR eklenir.

## Indeks

| No | Baslik | Durum |
|----|--------|-------|
| [0001](0001-causal-mask-zorunlulugu.md) | Causal attention mask zorunlulugu | Kabul edildi |
| [0002](0002-v1-ogrenilen-pozisyon-embedding.md) | v1'de RoPE yerine ogrenilen pozisyon embedding | Kabul edildi |
| [0003](0003-model-boyutu-veri-oranı.md) | Model boyutunun veri miktarina gore kucultulmesi | Kabul edildi |
| [0004](0004-veri-kaynagi-secimi.md) | Veri kaynagi: bigcode/the-stack-smol | Kabul edildi |
| [0005](0005-dil-kapsami-daraltma.md) | Dil kapsaminin Python/JS/C ile sinirlandirilmasi | Kabul edildi |
| [0006](0006-dil-karisikligi-duzeltmesi.md) | Egitim verisinin sadece Ingilizce/kod kaynaklarina daraltilmasi | Kabul edildi |
