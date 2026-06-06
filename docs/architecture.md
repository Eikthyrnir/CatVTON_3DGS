# Architektura systemu

## Przegląd

System CatVTON + 3DGS realizuje pipeline **multi-view virtual try-on z rekonstrukcją 3D**. Na wejściu system otrzymuje wideo przedstawiające kamerę obracającą się wokół osoby oraz zdjęcia ubrania, a na wyjściu generuje model 3D w reprezentacji Gaussian Splatting, przedstawiający osobę w nowym ubraniu, umożliwiający renderowanie z dowolnego kąta.

## Diagram pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT                                            │
│  [Wideo osoby] + [Zdjęcia ubrania (front/back/side)]                   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ETAP 1: Preprocessing                                                   │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────────────┐   │
│  │ FFmpeg        │  │ DensePose        │  │ MediaPipe 3D /          │   │
│  │ (video→frames)│  │ (view classify)  │  │ YOLOv8+DepthAnything    │   │
│  └──────┬───────┘  └────────┬─────────┘  │ (true front detection)  │   │
│         │                    │             └────────────┬────────────┘   │
│         ▼                    ▼                          ▼                │
│  [Klatki 2fps]     [front/back/side]        [Best front/back frame]     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ETAP 2: Multi-View Try-On (per frame)                                   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────┐                │
│  │ Reference Pass (front view):                         │                │
│  │   CatVTON + ReferenceAttentionProcessor              │                │
│  │   → stores K/V features in attention bank            │                │
│  └──────────────────────┬──────────────────────────────┘                │
│                          │                                               │
│  ┌─────────────────────────────────────────────────────┐                │
│  │ Generation Pass (side/back views):                   │                │
│  │   CatVTON + injected reference K/V                   │                │
│  │   → consistent garment appearance                    │                │
│  └──────────────────────┬──────────────────────────────┘                │
│                          │                                               │
│  3-Phase Pipeline (per frame):                                           │
│  ┌───────────┐    ┌──────────────┐    ┌─────────────┐                  │
│  │Phase 1:   │───▶│Phase 2:      │───▶│Phase 3:     │                  │
│  │Coarse     │    │Composite     │    │Refine       │                  │
│  │(wide mask)│    │(paste-back)  │    │(tight mask) │                  │
│  └───────────┘    └──────────────┘    └──────┬──────┘                  │
│                                               │                          │
│  Mask refinement: SCHP / SegFormer            │                          │
└───────────────────────────────────────────────┬─────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ETAP 3: 3D Reconstruction                                               │
│                                                                          │
│  ┌──────────────┐    ┌───────────────────┐    ┌────────────────────┐   │
│  │ Background   │───▶│ COLMAP            │───▶│ 3D Gaussian        │   │
│  │ Removal      │    │ (camera poses)    │    │ Splatting (30k it) │   │
│  │ (SCHP mask)  │    │                   │    │                    │   │
│  └──────────────┘    └───────────────────┘    └─────────┬──────────┘   │
└─────────────────────────────────────────────────────────┬───────────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ETAP 4 (Opcjonalny): SDS Optimization                                  │
│                                                                         │
│  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐    │
│  │ Dataset           │───▶│ Expert LoRA      │───▶│ SDS Loss        │   │
│  │ Partitioning      │    │ Training         │    │ (per-view)      │   │
│  │ (COLMAP angles)   │    │ (front/side/back)│    │                 │   │
│  └──────────────────┘    └──────────────────┘    └────────┬────────┘   ││                                                            │            │
│                                              ┌─────────────▼──────────┐ │
│                                              │ SDS-augmented 3DGS     │ │
│                                              │                        │ │
│                                              └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                            │
│  [Model 3DGS (.ply)] - renderowalny z dowolnego kąta                    │
│  [Try-on images]     - 2D wyniki per frame                              │
│  [Expert LoRAs]      - wytrenowane adaptery per view                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Główne komponenty

### 1. View Detection
- **DensePose** - wstępna klasyfikacja front/back/side na podstawie segmentacji tułowia
- **MediaPipe 3D Pose** - precyzyjna detekcja najlepszego widoku frontalnego przez minimalizację różnicy głębokości Z ramion
- **YOLOv8 + Depth Anything V2** - alternatywna metoda (2D pose + monocular depth estimation)

### 2. Reference Attention
- Custom `ReferenceAttentionProcessor` - modyfikacja self-attention w UNet up_blocks
- Podczas reference pass: zapisuje K/V features z frontalnego try-on
- Podczas generation pass: wstrzykuje reference K/V do attention computation
- Efekt: spójność ubrania między widokami bez dodatkowego treningu

### 3. Try-On Pipeline
- **Phase 1 (Coarse)**: CatVTON z szeroką maską agnostyczną - najlepsza wierność ubrania, ale regeneruje ciało
- **Phase 2 (Composite)**: Wklejenie ubrania z Phase 1 na oryginalne ciało - przywraca anatomię
- **Phase 3 (Refine)**: Ponowny try-on na composicie z ciasną maską - harmonizuje szwy
- Mask refinement: SCHP (human parsing) lub SegFormer (clothes segmentation)

### 4. COLMAP Utilities
- Parsowanie binary COLMAP output (images.bin)
- Obliczanie azymutu kamery (kąt obrotu)
- Anchoring kąta do frontalnego widoku
- Partycjonowanie dataset na front/side/back

### 5. Expert LoRA Training
- RealFill-style training z random masks
- Osobny LoRA per widok (front, side, back)
- Trained on inpainting UNet (SD 1.5 or SD 2.0)
- Cel: memoryzacja wyglądu ubrania z konkretnego kąta

### 6. SDS Loss
- Score Distillation Sampling z per-view expert switching
- Multi-adapter PeftModel (front/side/back loaded simultaneously)
- Automatyczne przełączanie adaptera na podstawie kąta kamery
- Integracja z 3DGS training loop

## Technologie

| Komponent | Technologia |
|-----------|-------------|
| Try-On Model | CatVTON (Stable Diffusion Inpainting + custom attention) |
| 3D Reconstruction | 3D Gaussian Splatting (camenduru fork) |
| Human Parsing | SCHP (Self-Correction for Human Parsing) |
| Pose Estimation | DensePose, MediaPipe, YOLOv8 |
| Depth Estimation | Depth Anything V2 |
| Clothes Segmentation | SegFormer B2 Clothes |
| LoRA Training | PEFT + Accelerate |
| Camera Estimation | COLMAP |
| VAE | sd-vae-ft-mse |

## Środowisko uruchomieniowe

- **Google Colab** (rekomendowane) z GPU runtime (L4 minimum, rekomendowane L4 / A100)
- CUDA 12.x, Python 3.10+
- ~20-30 GB przestrzeni dyskowej na modele (HuggingFace cache, COLMAP, checkpointy 3DGS)
- **Min. 16 G VRAM**

## Zmiany względem pierwotnych założeń

1. **Reference Attention** - oryginalnie CatVTON nie wspiera multi-view consistency. Dodaliśmy custom ReferenceAttentionProcessor inspirowany AnimateDiff/IP-Adapter.
2. **3-Phase Pipeline** - zamiast jednorazowego try-on, wprowadziliśmy coarse→composite→refine dla lepszej jakości.
3. **SDS z Expert LoRA** - inspirowane GS-VTON, ale z uproszczoną implementacją (bez full text encoder LoRA).
4. **Wyniki SDS** - eksperymentalne, nie dają znaczącej poprawy nad vanilla 3DGS w obecnej konfiguracji.


---

## Kluczowe decyzje architektoniczne

### Dlaczego notebook a nie CLI?
- Pipeline jest interaktywny (wizualna inspekcja wyników pośrednich)
- Colab jest naturalnym środowiskiem (dostęp do GPU bez konfiguracji)
- Różne etapy mogą być pomijane lub powtarzane

### Dlaczego Reference Attention zamiast fine-tuning?
- Zero-shot: nie wymaga dodatkowego treningu per garment
- Działa na istniejącym CatVTON bez modyfikacji wag
- Inspirowane AnimateDiff/IP-Adapter approach

### Dlaczego 3-phase pipeline?
- Phase 1 (coarse): CatVTON z szeroką maską daje najlepszą wierność ubrania, ale zmienia ciało
- Phase 2 (composite): przywraca oryginalne ciało
- Phase 3 (refine): harmonizuje szwy między oryginalnym ciałem a nowym ubraniem

### Dlaczego SCHP zamiast SegFormer jako domyślny mask refiner?
- SCHP jest już częścią CatVTON (nie wymaga dodatkowego modelu)
- Daje porównywalne wyniki przy mniejszym zużyciu pamięci
- SegFormer jest opcjonalną alternatywą (lepsza segmentacja, ale osobny model)

---

## Modyfikacja parametrów

Plik `configs/pipeline_config.yaml` zawiera wszystkie domyślne parametry. Note: The Colab notebook currently hardcodes these values. W notebooku można je nadpisać bezpośrednio.

### Najważniejsze parametry do tuningu

| Parametr | Domyślnie | Efekt |
|----------|-----------|-------|
| `guidance_scale` | 2.5 | Wyższy = bardziej wierny prompt, niższy = bardziej naturalny |
| `coarse_steps` / `fine_steps` | 50 | Więcej = lepsza jakość, wolniej |
| `mask_dilate_px` | 14 | Większy = mniej artefaktów na granicach, ale może ukryć detale |
| `injection_steps` | range(5, 45) | Zakres kroków z wstrzykiwaniem reference features |
| `lora.rank` | 8 | Wyższy = większa pojemność LoRA (ryzyko overfitting) |
| `sds.weight` | 0.1 | Waga SDS vs standard loss w 3DGS |

---