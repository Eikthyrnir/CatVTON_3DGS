# Instrukcja użytkownika

## Szybki start

### 1. Otwórz notebook w Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eikthyrnir/CatVTON_3DGS/blob/main/CatVTON_3DGS_pipeline.ipynb)

### 2. Wybierz GPU runtime
- Menu: `Runtime` -> `Change runtime type` -> `A100` (lub L4)

### 3. Uruchom komórki sekwencyjnie
Notebook jest podzielony na sekcje. Każda sekcja ma nagłówek Markdown opisujący co robi.

---

## Użycie z własnymi danymi

### Wymagania dotyczące wideo
- Osoba nieruchoma
- Kamera obracająca się wokół osoby
- Dobrze oświetlone, jednolite tło
- Pełna postać widoczna 
- Czas trwania: 10-20 sekund
- Format: MP4, MOV, lub inny obsługiwany przez FFmpeg

### Wymagania dotyczące zdjęć ubrań
- Zdjęcie ubrania na białym/jednolitym tle LUB na modelu
- Rozdzielczość min. 512x512
- Widoki: front (wymagany), back, side
- Format: JPG, PNG, WEBP

### Jak podmienić dane w notebooku

1. **Wideo**: Zmień ścieżkę w sekcji "Split video into images":
```python
# Zamień tę ścieżkę na swoje wideo:
video_path = "/content/gaussian-splatting/input/YOUR_VIDEO.MOV"
```

2. **Ubrania**: Zmień ścieżki w sekcji "Init front/back cloth images":
```python
front_garment_path = "/path/to/your/garment_front.jpg"
back_garment_path  = "/path/to/your/garment_back.jpg"
side_garment_path  = "/path/to/your/garment_side.jpg"  # opcjonalny (zamiast można użyć front_garment_path)
```

3. **Parametry**:
```python
target_size = (768, 1024)  # rozdzielczość przetwarzania
NUM_INFERENCE_STEPS_CAT_VTON = 50  # kroki diffusion (więcej = wolniej ale lepiej)
```

---

## Sekcje notebooka

### Instalacja (komórki 1-8)
Klonuje CatVTON i nasze repo, instaluje zależności. **Wymaga restartu runtime po instalacji torch**.

### Inicjalizacja modeli (komórki 9-14)
Ładuje CatVTON pipeline, AutoMasker, modele depth/pose. Zajmuje ~2-3 min (pobieranie modeli z HuggingFace).

### Detekcja widoków (sekcja "Select Reference Front/Back images")
Automatycznie wykrywa najlepszy widok frontalny i tylny. Wynik: `front_person_path` i `back_person_path`.

### Try-on (sekcja "Front/Back/Side split Try-on per frame")
Generuje try-on dla każdego widoku. **Najdłuższy etap** (~30 min na A100). Można pominąć pobierając gotowe wyniki.

### 3DGS Training (sekcja "Run 3DGS training")
Trenuje model 3D Gaussian Splatting. Wymaga COLMAP (kompilacja ~10 min) lub gotowych danych COLMAP.

### Expert LoRAs + SDS (opcjonalne)
Zaawansowana optymalizacja 3DGS. Wymaga treningu LoRAs (~30-60 min) + SDS training (~40 min).

---

## Pominięcie długich etapów

### Pominięcie try-on (użyj gotowych wyników)
Przejdź do sekcji "Download try-on results from Google Drive" i uruchom komórkę z `gdown`.

### Pominięcie COLMAP
Przejdź do sekcji "Load COLMAP from Google Drive".

### Pominięcie LoRA training
Przejdź do sekcji "Download trained LoRAs".

---

## Wizualizacja wyników

### Wyniki try-on
W notebooku wyświetlane automatycznie (`show_multiple_images`).

### Model 3DGS
Wynikowy plik `.ply` (w folderze output) można otworzyć w:
- [SuperSplat](https://playcanvas.com/supersplat/editor) (przeglądarka)
- [3DGS Viewer](https://github.com/graphdeco-inria/gaussian-splatting#interactive-viewers) (lokalnie za pomocą repozytorium)

### Loss curves
Funkcja `show_3DGS_loss_curve(output_dir)` wizualizuje krzywe strat z TensorBoard.

---

## Rozwiązywanie problemów

| Problem | Rozwiązanie |
|---------|-------------|
| CUDA out of memory | Zmniejsz `target_size` lub użyj karty z więksym VRAM |
| Różowy overlay na try-on | VAE fix jest już zastosowany (sd-vae-ft-mse) |
| COLMAP nie znajduje punktów | Sprawdź jakość wideo (ostre klatki, ruch kamery/osoby) |
| Import errors po instalacji | Zrestartuj runtime (Runtime -> Restart runtime) |
