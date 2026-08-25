# CT-Kılavuzlu PET Kesit İnterpolasyonu — Uçtan Uca SoA Pipeline

**Problem tanımı (tek cümle):** Tam çözünürlüklü CT + seyrek PET kesitleri girdi olarak alınır; eksik ara PET kesitleri, CT'nin anatomik prior'ı kılavuzluğunda sentezlenir (Problem 3b).

**Tracer kararı: PSMA birincil.** Prostat kanseri taramaları klinikte PSMA PET ile yapılır ve Ankara verisi de PSMA olacaktır; geliştirme PSMA verisi üzerinde yürütülür, FDG yalnızca genellenebilirlik deneyi olarak kullanılır. PSMA'nın FDG'den kritik farkları: fizyolojik tutulum paterni farklıdır (tükürük bezleri, böbrekler, mesane çok parlak; beyin/kalp sönük) ve lezyon SUV değerleri çok daha yüksek olabilir (kemik/nodal metastazlarda SUV 30–100+). Bu farklar normalizasyon ve değerlendirme adımlarına aşağıda işlenmiştir.

**Matematiksel formülasyon:**
- Girdi: `PET_sparse ∈ R^(H×W×(D/k))` (k = seyrekleştirme faktörü, örn. 2–4) + `CT_full ∈ R^(H×W×D)`
- Çıktı: `PET_dense ∈ R^(H×W×D)`
- Kısıt (measurement consistency): sentezlenen ince kesitlerin kalınlık-ortalaması, ölçülen kalın kesiti geri vermelidir.

---

## Aşama 0 — Veri Temini ve Organizasyon

| # | Kaynak | Tracer | Rol | Boyut önerisi |
|---|---|---|---|---|
| 1 | TCIA **PSMA-PET-CT-Lesions** (autoPET III Münih kohortu) — cancerimagingarchive.net/collection/psma-pet-ct-lesions | PSMA | **Ana eğitim/geliştirme seti** (prostat, Ankara verisiyle aynı popülasyon ve tracer) | 60–100 hasta (dev aşaması) |
| 2 | **DEEP-PSMA** (Zenodo 15281784) | PSMA **+** FDG (aynı hastada ikisi) | Hızlı başlangıç (NIfTI hazır, ilk zip 4.2 GB) + ikincil PSMA seti + **cross-tracer ablasyonu** | 100 hasta, ~24 GB |
| 3 | TCIA **FDG-PET-CT-Lesions** (autoPET) | FDG | Genellenebilirlik bölümü ("yöntem PSMA'ya özgü değil") | 30–50 hasta |
| 4 | Kaggle **Lung Cancer CT-PET Subset (DICOM)** | FDG | DICOM pipeline smoke test (opsiyonel; #2'nin ilk zip'i de bu işi görür) | Tamamı (küçük) |
| 5 | Ankara Nükleer Tıp verisi | PSMA | Nihai fine-tuning + harici doğrulama | Geldiğinde |

**Not:** DEEP-PSMA doğrudan NIfTI + SUV eşikleri + TTB kontürleriyle geldiği için 1. haftadaki smoke test'i bununla yapmak DICOM dönüşümünü beklemeden model tarafına başlamanızı sağlar; DICOM→NIfTI hattını paralelde #1 üzerinde kurarsınız.

**Klasör yapısı (BIDS-vari):**
```
data/
  raw/            # dokunulmaz DICOM
  nifti/          # sub-XXXX/ct.nii.gz, pet_suv.nii.gz, seg.nii.gz
  processed/      # resample + crop + normalize edilmiş
  splits/         # train/val/test hasta listeleri (JSON)
```

---

## Aşama 1 — Veri Temizleme ve Ön İşleme

### 1.1 DICOM → NIfTI + SUV dönüşümü
- Araç: `dcm2niix` (geometri için güvenilir) + SUV için autoPET'in resmi `lab-midas/TCIA_processing` scriptleri.
- PET voksel değerleri **SUVbw**'ye çevrilir (enjekte aktivite, decay düzeltmesi, hasta kilosu → hepsi DICOM tag'lerinden).
- **QC kontrolü (tracer'a göre):** Karaciğer SUVmean makul fizyolojik aralıkta olmalı — FDG için ≈ 1.5–3.0, Ga-68 PSMA için ≈ 4–8 civarı (F-18 PSMA-1007'de karaciğer daha da parlaktır; kohortun tracer'ını metadata'dan doğrulayın). Aralık dışıysa SUV dönüşümü hatalıdır — bu tek kontrol, ileride yaşayacağınız sessiz hataların %80'ini yakalar. Ek PSMA kontrolü: böbrekler ve mesane görüntünün en parlak yapıları olmalı; değilse bir şeyler ters demektir.

### 1.2 Geometrik hizalama ve yeniden örnekleme
- autoPET/PSMA verisinde PET ve CT aynı seansta çekildiği için hizalıdır; **deformable registration GEREKMEZ.** Sadece grid eşleme yapılır.
- Ortak grid: PET'in in-plane çözünürlüğü esas alınır (örn. 2×2 mm), CT bu gride **B-spline** ile resample edilir. Z ekseninde her iki modalite ortak ince aralığa (örn. 2 mm veya 3 mm) getirilir.
- CT'yi PET gridine indirmek (tersi değil) doğru tercihtir: PET zaten düşük çözünürlüklü sinyal; CT'yi yükseltmek yapay bilgi üretir.

### 1.3 Temizleme
- **Vücut maskesi:** CT'de HU > −500 eşiği + morfolojik kapama + en büyük bağlı bileşen → yatak/masa ve kollar dışarı atılır. (Alternatif: `TotalSegmentator` ile body maskesi.)
- **Alan kırpma:** Tüm vücut yerine **pelvis + iliak bölge** kırpılabilir (prostat odaklı proje). TotalSegmentator ile `hip`/`sacrum`/`prostate` etiketlerinden otomatik bounding box çıkar → hesap yükü ~5× azalır.
- **Kesit sayısı / oryantasyon QC:** LPS/RAS tutarlılığı (`nibabel` ile affine kontrolü), NaN/negatif SUV taraması, aşırı kısa seriler (< 40 kesit) dışlanır.

### 1.4 Normalizasyon
- **CT:** HU clip [−1000, +1000] → [−1, 1] lineer ölçekleme. (Yumuşak doku odaklı ikinci kanal olarak [−200, +300] pencere de eklenebilir — lezyon sınırları için faydalı.)
- **PET (PSMA'ya göre güncellendi):** FDG'deki alışıldık [0, 15] clip PSMA için **yanlıştır** — PSMA lezyonları ve fizyolojik yapılar (böbrek, mesane) SUV 30–100+ olabilir, dar clip lezyon tepe değerlerini ezer. İki geçerli seçenek: **(a)** geniş clip [0, 50] → [0, 1] lineer, veya **(b) önerilen: log dönüşümü** `log(1 + SUV) / log(1 + 50)` — parlak lezyonların dinamik aralığını korurken düşük-SUV dokuların kontrastını da ezmez. Değerlendirme metrikleri her zaman **ters dönüşümle SUV uzayında** hesaplanır. **Hasta-bazlı min-max KULLANMAYIN** — SUV mutlak anlamlıdır, hasta bazlı normalizasyon SUVmax bias metriğinizi anlamsızlaştırır.
- **Mesane komşuluğu uyarısı (PSMA'ya özgü):** Mesane, PSMA'da aşırı parlaktır ve prostata bitişiktir. Kesit interpolasyonunda mesane parlaklığının komşu kesitlere "sızması" (spill-over halüsinasyonu) beklenen bir hata modudur — değerlendirmede mesane-komşusu lezyonları ayrı bir alt grup olarak raporlayın (bkz. Aşama 6).

### 1.5 Hasta-düzeyi split
- %70 train / %10 val / %20 test — **hasta bazında**, asla kesit bazında değil (aksi hâlde leakage ile şişirilmiş sonuç alırsınız, jüri/hakem bunu ilk sorar).
- Split'i JSON'a sabitleyin, seed'i raporlayın. Lezyonlu/lezyonsuz hastaları stratifiye edin.

---

## Aşama 2 — Degradasyon Simülasyonu (LR-HR Çifti Üretimi)

Gerçek "seyrek PET" veriniz yok; simüle edeceksiniz. **Bu adımın fiziği yanlış olursa tüm sonuçlar geçersizdir.**

- **Doğru model: kesit ortalaması (slice averaging), decimation DEĞİL.** Kalın kesitli PET, ince kesitlerin z-ekseni boyunca ortalamasına (yaklaşık dikdörtgen/Gauss kesit profili) karşılık gelir. k komşu ince kesidi ortalayıp bir kalın kesit üretin.
- Faktörler: **k = 2 (ana deney), k = 3 ve k = 4 (zorluk merdiveni).**
- İsteğe bağlı gerçekçilik: ortalamadan sonra hafif Poisson-benzeri gürültü ekleyin (PET gürültüsü sinyal-bağımlıdır).
- Test setinde degradasyon **bir kez** üretilip dondurulur (her epoch'ta rastgele değil) — tekrarlanabilirlik için.

---

## Aşama 3 — Model Merdiveni (Basit → SoA)

Literatürdeki kritik bulgu: kesit interpolasyonunda **problem formülasyonu (hangi komşu kesitlerin girdi verildiği) mimariden çok daha belirleyicidir** — doğru formülasyonla ~%58 iyileşme, mimari değişiklikleriyle < %1. Bu yüzden merdivenin her basamağında aynı formülasyonu sabitleyin, sonra formülasyonu ayrıca ablate edin.

### Basamak 0 — Klasik taban çizgisi (1. hafta)
Trikübik / B-spline z-interpolasyonu. Tüm metrik altyapınızı bununla test edin. Her raporda yer alacak.

### Basamak 1 — 2.5D Rezidüel U-Net (2–4. hafta)
- **Girdi:** hedef kesidin iki yanındaki 2'şer PET kesidi (4 kanal) + hedef konuma karşılık gelen CT kesidi ve komşuları (3 kanal) → 7 kanallı 2D girdi.
- **Çıktı:** ara PET kesidi (rezidüel olarak: trikübik tahmin + ağın öğrendiği düzeltme).
- Mimari: MONAI `BasicUNet` veya `AttentionUnet`, ~5–20M parametre. Tek GPU'da rahat eğitilir.
- Bu basamak muhtemelen sonuçlarınızın %90'ını verecek — küçümsemeyin.

### Basamak 2 — Cross-Modal Attention U-Net (ANA KATKI, 5–9. hafta)
Projenin özgün kısmı: CT'yi kanal olarak yapıştırmak yerine **ayrı bir CT encoder** + PET decoder'a **cross-attention** ile enjeksiyon:
- İki encoder: PET-encoder (seyrek PET slab'ı), CT-encoder (hizalı CT slab'ı).
- Decoder'ın her ölçeğinde: PET özellikleri *query*, CT özellikleri *key/value* olan multi-head cross-attention blokları (SwinUNETR-tarzı pencereli attention bellek için şart).
- Gerekçe: erken füzyon (kanal birleştirme) CT'nin keskin kenar bilgisini derin katmanlara taşıyamaz; cross-attention, PET'in "nereye bakacağını" CT anatomisinden öğrenir.
- **Ablasyon zaten hazır:** Basamak 1 (erken füzyon) vs Basamak 2 (attention füzyon) vs CT'siz varyant → tezin ana tablosu.

### Basamak 3 — Ölçek-bağımsız / üretken uzantı (opsiyonel, 10.+ hafta)
İki yoldan **birini** seçin (ikisini birden değil):
- **(a) INR / arbitrary-scale:** Decoder'ı koordinat-koşullu implicit fonksiyona çevirin (LIIF/ArSSR tarzı) → tek model k=2,3,4'ü ve kesirli faktörleri destekler. Deterministik, kararlı, tez için güvenli.
- **(b) Koşullu diffusion (residual DDPM):** Basamak 2'nin çıktısını başlangıç alan, rezidüeli üreten hafif bir latent/pixel diffusion. Görsel keskinlik kazandırır ama SUV sadakati riske girer ve eğitim + inference maliyeti büyüktür. Ancak Basamak 2 bitip zaman kalırsa.

### Her basamakta zorunlu: Measurement Consistency
İnference sonrası (veya loss içinde) sentezlenen ince kesitlerin k'lı ortalaması, girdideki ölçülmüş kalın keside **projekte edilir** (data-fidelity düzeltmesi: `PET_pred ← PET_pred + upsample(PET_measured − avg_k(PET_pred))`). Bu, SUV bias'ını yapısal olarak sınırlar ve "modelimiz ölçümle tutarlı" savunmasını bedavaya verir.

---

## Aşama 4 — Kayıp Fonksiyonları

- **Ana kayıp: L1 + MS-SSIM** (ör. `0.84·MS-SSIM + 0.16·L1`). Yakın tarihli kontrollü karşılaştırmalarda kesit interpolasyonu için en dengeli profil bu kombinasyon; saf SSIM ailesinde eğitim kararsızlığı raporlanmıştır — MS-SSIM'i L1 ile birlikte kullanın.
- **Lezyon-ağırlıklı terim:** Segmentasyon maskesi mevcut (autoPET/PSMA anotasyonlu!) → lezyon vokselerinde L1'i 5–10× ağırlıklandırın. Az sayıda lezyon vokseli, aksi hâlde aggregate loss içinde kaybolur.
- **Gradyan/kenar kaybı (opsiyonel):** z-yönlü Sobel farkı — kesitler arası süreklilik için.
- **Adversarial kayıp: KULLANMAYIN** (ilk sürümde). GAN'lar SUV değerlerinde halüsinasyon riskini artırır; klinik savunulabilirliği zayıflatır.

---

## Aşama 5 — Eğitim Protokolü

| Bileşen | Öneri |
|---|---|
| Çerçeve | PyTorch + **MONAI** (transform/loader/metrik hazır) + Lightning |
| Patch | 2.5D: 256×256×slab; 3D denerseniz 128×128×32 |
| Batch / Optim | 8–16 (AMP ile), AdamW, lr=2e-4, cosine decay, 200–300 epoch |
| Augmentasyon | Rastgele flip (L-R), ±10° in-plane rotasyon, hafif intensite jitter (CT'ye HU shift ±20). **Z-ekseni augmentasyonu YAPMAYIN** — problem z'de tanımlı. |
| Donanım | Tek 12–24 GB GPU yeter (Basamak 1–2). Colab Pro / Kaggle GPU / üniversite kümesi |
| Tekrarlanabilirlik | Seed sabitleme, config'lerin YAML'da versiyonlanması, W&B veya MLflow loglama |

---

## Aşama 6 — Değerlendirme Protokolü (tezin bel kemiği)

**Aggregate metrikler tek başına YETMEZ.** Rapor şablonu:

1. **Global:** PSNR, SSIM, NRMSE (vücut maskesi içinde; boş hava dahil edilirse metrikler yapay şişer).
2. **Lezyon-stratifiye (asıl tablo):** Her anotasyonlu lezyon için:
   - **SUVmax bias:** `(SUVmax_pred − SUVmax_gt)/SUVmax_gt` — ortalama + %95 CI, Bland-Altman grafiği
   - SUVmean ve lezyon hacmi (izokontur %40 SUVmax) korunumu
   - Küçük lezyon (< 1 mL) alt grubu ayrı raporlanır — interpolasyonun en çok sildiği şey budur
   - **Mesane-komşusu lezyon alt grubu** (PSMA'ya özgü): mesane maskesine < 2 cm mesafedeki lezyonlar ayrı raporlanır — spill-over halüsinasyonunun test edildiği yer
3. **Görev-bazlı:** Hazır bir lezyon segmentasyon modeli (autoPET baseline nnU-Net) GT-PET vs sentez-PET üzerinde çalıştırılır → Dice/detection farkı = "downstream klinik görev korunuyor mu?"
4. **Görsel:** Koronal/sagital MPR karşılaştırmaları (stair-step artefaktının kaybolması), en kötü 5 vaka analizi (failure modes bölümü jüriye güven verir).
5. **İstatistik:** Wilcoxon signed-rank (hasta bazında eşleştirilmiş), Bonferroni düzeltmesi.

**Karşılaştırma matrisi (literatür baseline'ları dahil — bkz. Aşama 8):**

| Model | Tür | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| Trikübik | klasik | ✓ | ✓ | ✓ |
| RIFE / FILM (pretrained) | video interp. | ✓ | — | — |
| SAINT | literatür (CT interp.) | ✓ | ✓ | ✓ |
| I³Net | literatür (SoA, TMI'24) | ✓ | ✓ | ✓ |
| 2.5D U-Net (CT'siz) | bizim merdiven | ✓ | ✓ | ✓ |
| 2.5D U-Net (CT erken füzyon) | bizim merdiven | ✓ | ✓ | ✓ |
| **Cross-attention (bizim)** | **ana katkı** | ✓ | ✓ | ✓ |
| + measurement consistency | bizim | ✓ | ✓ | ✓ |

---

## Aşama 7 — Ablasyonlar (jüri/hakem soruları için hazır cephane)

1. CT var/yok (ana hipotez testi)
2. Füzyon tipi: erken (kanal) vs cross-attention
3. Girdi formülasyonu: 1-1 komşu vs 2-2 komşu vs 3-3 komşu PET kesidi (literatürdeki "%58" bulgusunu kendi verinizde doğrulayın)
4. Kayıp bileşenleri: L1 vs L1+MS-SSIM vs +lezyon ağırlığı
5. Measurement consistency var/yok → SUVmax bias tablosuna etkisi
6. **Cross-tracer genellenebilirlik (bedava ablasyon):** PSMA üzerinde eğit → DEEP-PSMA'daki *aynı hastaların* FDG taramalarında test et. Aynı hastada iki tracer olduğu için hasta-düzeyi karıştırıcı olmadan temiz bir domain-shift deneyi — tezde ayrı bir alt bölüm eder

---

## Aşama 8 — Piyasa Araştırması: Literatür Baseline'ları ve Test Planına Entegrasyonu

Benzer problemlerde (through-plane interpolasyon / kesit sentezi / cross-modal kılavuz) yayınlanmış, **açık kodlu** yöntemler. Hepsi test planına dahildir; hakemin "yayınlanmış yöntemlerle karşılaştırdınız mı?" sorusunun cevabı bu tablodur.

### 8.1 Doğrudan rakipler (aynı problem, farklı modalite)

| Yöntem | Venue | Ne yapar | Kod | Bizim testteki rolü | Uyarlama eforu |
|---|---|---|---|---|---|
| **SAINT** | CVPR 2020 | Sagital+koronal marjinal SR + füzyon ile CT kesit sentezi; keyfi tamsayı faktör | `github.com/cpeng93/SAINT` | Klasik öğrenmeli baseline; CT'siz çalışır → CT kılavuzunun katkısını gösteren kontrast | Orta: PET verisiyle yeniden eğitim, veri loader değişimi |
| **I³Net** | IEEE TMI 2024 | Inter-intra-slice interpolasyon + frekans-domain öğrenme; CT/MR kesit sentezinde güncel SoA | `github.com/eeeric-code/I3Net` | **En güçlü yayınlanmış rakip** — bunu geçmek makalenin ana tablosunu taşır | Orta: aynı degradasyon protokolümüzle PET'te eğit |
| **MSR-Fusion (Peng'19)** | MICCAI dönemi | 2D marjinal SR + eksenel refine | (SAINT'in öncülü; SAINT yeter) | Related work'te anılır, implementasyon gerekmez | — |

### 8.2 Komşu alanlardan transfer (pretrained, eğitimsiz test)

| Yöntem | Alan | Kod | Rolü |
|---|---|---|---|
| **RIFE** | Video frame interpolation (ECCV'22) | `github.com/hzwer/ECCV2022-RIFE` | Sıfır-eğitimle çalıştırılan hazır baseline; PET kesit çiftlerine "iki kare arası kare üret" olarak uygulanır. Kesit interpolasyon literatüründe standart karşılaştırma hâline geldi |
| **FILM** | Video frame interpolation (Google) | `github.com/google-research/frame-interpolation` | RIFE ile aynı rol; birinden biri yeterli, ikisi varsa daha iyi |
| **SMORE** | Self-supervised MRI through-plane SR | SMORE v3+ (JHU sürümü) | Opsiyonel: "harici HR veri gerektirmeyen" alternatif paradigma olarak tartışma bölümüne malzeme |

### 8.3 Kavramsal akrabalar (related work bölümü için, implementasyon yok)

- **CT→PET çeviri:** CPDM (koşullu diffusion, büyük eşlenik set) — "CT tek başına PET'i belirleyemez" argümanımızın literatür dayanağı; seyrek-PET+CT kurgumuzun neden daha savunulabilir olduğunu anlatırken kullanılır.
- **MRI-kılavuzlu PET SR/denoise ailesi:** cross-modal kılavuz fikrinin öncülleri — "biz MRI yerine ko-akiz CT kullanıyoruz" boşluk cümlesinin referans tabanı.
- **DEEP-PSMA challenge leaderboard yöntemleri:** lezyon segmentasyon/TTB metodolojisi — Aşama 6'daki görev-bazlı değerlendirmede (downstream segmentasyon korunumu) hazır model kaynağı.
- **2.5D-SRCNN (EJNMMI Physics 2025):** PET'e özgü SR + SUV kantifikasyon değerlendirmesi — SUV-odaklı değerlendirme protokolümüzün emsali.

### 8.4 Test planına entegrasyon sırası

1. **RIFE/FILM (1 gün):** pretrained ağırlıkla inference-only; PET kesitleri 3 kanala kopyalanıp [0,1]'e ölçeklenerek beslenir. Kod yazımı minimal, tabloya ilk literatür satırını hemen ekler.
2. **SAINT (3–5 gün):** repo'daki eğitim döngüsü bizim PSMA loader'ımıza bağlanır; kendi degradasyon protokolümüzle (**averaging!**) yeniden eğitilir — orijinal decimation protokolüyle DEĞİL, aksi hâlde karşılaştırma adil olmaz.
3. **I³Net (3–5 gün):** aynı şekilde; SAINT ile aynı loader'ı paylaşır.
4. Tüm baseline'lar **aynı split, aynı degradasyon, aynı metrik koduyla** koşulur. "Adil karşılaştırma" cümlesini makalede kurabilmenin tek yolu budur.

---

## Aşama 9 — Makale Ön Hazırlığı: Claude Code Çalışma Sözleşmesi

Bu bölüm, implementasyonu Claude Code ile adım adım yürütürken makale hazırlığının **yan ürün olarak** birikmesi için repo'ya konacak kuralları tanımlar. Repo köküne `CLAUDE.md` olarak yerleştirin; Claude Code her oturumda bunu okur ve uyar.

### 9.1 `CLAUDE.md`

Çalışma kuralları repo kökündeki `CLAUDE.md` dosyasındadır — **tek kaynak orasıdır.** Bu doküman
kuralların kopyasını tutmaz; kural değişikliği doğrudan `CLAUDE.md`'ye yazılır.

**Numaralandırma (belirsizlik giderildi):** Bu dokümandaki "Aşama" (pipeline, 0–9) ve "Basamak"
(model merdiveni, 0–3) numaraları yalnızca anlatım içindir. Repo'nun tek geçerli numaralandırması
`docs/phases/` altındaki faz numaralarıdır (0–6); config önekleri (`p<N>_`) ve
`phase-X-complete` etiketleri bunu kullanır. Eşleme:

| Repo fazı | Bu dokümandaki karşılığı |
|---|---|
| `phase-0-setup` | Aşama 0 |
| `phase-1-preprocessing` | Aşama 1 |
| `phase-2-degradation` | Aşama 2 |
| `phase-3-baselines` | Aşama 8 |
| `phase-4-unet` | Basamak 0–1 |
| `phase-5-crossattention` | Basamak 2–3 |
| `phase-6-evaluation` | Aşama 6–7 |

### 9.2 Deney → makale bölümü eşlemesi

| Pipeline çıktısı | Makale bölümü | Otomatik üretilen varlık |
|---|---|---|
| Aşama 1–2 (ön işleme + degradasyon) | Methods: Data | Hasta demografisi tablosu, degradasyon şeması figürü |
| Basamak 0–2 + Aşama 8 baseline'ları | Results: ana tablo | `main_results.tex` (model × k × metrik) |
| Aşama 6 lezyon-stratifiye | Results: klinik değerlendirme | SUVmax Bland-Altman figürü, lezyon alt grup tablosu |
| Aşama 7 ablasyonları | Results: ablation | `ablation.tex` |
| Cross-tracer deneyi (PSMA→FDG) | Results: generalization | Cross-tracer tablosu |
| En kötü 5 vaka | Discussion: limitations | Failure-case figür paneli |
| Ankara verisi | Results: external validation | Harici doğrulama tablosu |

### 9.3 Yazım takvimi kancaları

- **Basamak 1 biter bitmez** Methods'un veri+ön işleme kısmı yazılır (taze hafızayla; en sık ertelenen ve en çok detay kaybedilen bölüm budur).
- **Ana tablo ilk dolduğunda** (baseline'lar + bizim model, k=2) Introduction'ın katkı paragrafı yazılır — üç katkı: (i) CT-kılavuzlu PET through-plane interpolasyonunun ilk sistematik çalışması, (ii) measurement-consistency ile yapısal SUV sadakati, (iii) lezyon-düzeyi klinik değerlendirme protokolü.
- **Hedef venue merdiveni:** SPIE Medical Imaging / IEEE ISBI / MICCAI workshop (ana plan) → EJNMMI Physics veya Medical Physics (Ankara harici doğrulaması güçlüyse dergi sürümü).

---

## Araç Zinciri Özeti

`dcm2niix` · `nibabel` / `SimpleITK` · `TotalSegmentator` · `MONAI` · PyTorch Lightning · Weights & Biases · `pandas`+`seaborn` (metrik analizi) · autoPET resmi dönüşüm scriptleri (`lab-midas/TCIA_processing`)

## Kırmızı Bayrak Kontrol Listesi

- [ ] Karaciğer SUVmean fizyolojik aralık dışında (FDG: 1.5–3.0, Ga-68 PSMA: ~4–8) → SUV dönüşümü bozuk
- [ ] PSMA verisinde dar SUV clip ([0,15] gibi) → lezyon tepe değerleri ezilir, SUVmax bias yapay iyi görünür
- [ ] Kesit bazlı split → leakage, tüm sonuçlar geçersiz
- [ ] Decimation ile degradasyon → fiziksel olarak yanlış, averaging kullan
- [ ] Hasta bazlı PET normalizasyonu → SUV metrikleri anlamsız
- [ ] Test degradasyonu her koşuda farklı → tekrarlanamaz sonuç
- [ ] Sadece PSNR raporlama → lezyon-düzeyi analiz olmadan klinik iddia yok