# CatVTON + 3D Gaussian Splatting: Multi-View Virtual Try-On Pipeline

## Cel projektu

Pipeline do wirtualnego przymierzania ubrań (Virtual Try-On) z rekonstrukcją 3D. Projekt łączy model **CatVTON** (2D try-on oparty na diffusion) z **3D Gaussian Splatting** (3DGS), aby na podstawie jednego wideo osoby wygenerować spójny 3D model z nowym ubraniem.

**Kluczowe rezultaty:**
- Multi-view try-on z utrzymaniem spójności ubrania między widokami (front/back/side) dzięki custom Reference Attention Processor
- 3-fazowy pipeline try-on: coarse -> composite -> refine
- Automatyczna detekcja widoku frontalnego (MediaPipe 3D / YOLOv8 + Depth Anything V2)
- Trening Expert LoRAs (per view) + Score Distillation Sampling (SDS) do optymalizacji 3DGS

## Wymagania

### Sprzętowe
- **GPU**: NVIDIA GPU z min. 16 GB VRAM (testowano na A100, L4)
- **RAM**: ~20-30 GB
- **Disk**: ~15 GB na modele i dane

### Programowe
- **Platforma**: Google Colab (rekomendowane) lub Linux z CUDA 12.4
- **Python**: 3.10+
- **CUDA**: 12.4
- **Zewnętrzne repozytoria** (klonowane automatycznie):
  - [CatVTON](https://github.com/Zheng-Chong/CatVTON) - model 2D try-on
  - [gaussian-splatting](https://github.com/camenduru/gaussian-splatting) - fork 3DGS dla Colab
- **Modele HuggingFace** (pobierane automatycznie):
  - `runwayml/stable-diffusion-inpainting`
  - `zhengchong/CatVTON`
  - `stabilityai/sd-vae-ft-mse`
  - `depth-anything/Depth-Anything-V2-Small-hf`
  - `mattmdjaga/segformer_b2_clothes`

## Instalacja i konfiguracja

### Google Colab

1. Otwórz notebook w Colab:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eikthyrnir/CatVTON_3DGS/blob/main/src/CatVTON_3DGS_pipeline.ipynb) 

2. Wybierz runtime z GPU (A100 lub L4)
   - **IMPORTANT!** chociaż teoretycznie po wielu wprowadzonych przez nas optymizacjach pipeline powinien być uruchomialny na Colab T4 GPU, 
   - W razie potrzeby możemy udostępnić dostęp do PRO wersji COLAB z dostępem do mocniejszych GPU
   - W tej sprawie prosimy pisać bezpośrednio do Ignacy Byshniou (Teams / e-mail: ihnbys@st.amu.edu.pl)
3. Uruchom komórki sekwencyjnie - notebook automatycznie:
   - Klonuje wymagane repozytoria
   - Instaluje zależności
   - Pobiera modele i dane

## Dokumentacja

- [Architektura systemu](docs/architecture.md) - komponenty, przepływ danych, diagram pipeline
- [Instrukcja użytkownika](docs/usage.md) - jak korzystać z pipeline

## Uruchomienie demonstracji

### Pełny pipeline (Colab)

Otwórz `CatVTON_3DGS_pipeline.ipynb` i uruchamiaj komórki sekwencyjnie. Notebook jest podzielony na sekcje:

1. **Instalacja** - klonowanie repozytoriów, instalacja zależności
2. **Inicjalizacja modeli** - CatVTON, AutoMasker, Depth models
3. **Detekcja widoków** - automatyczne wykrycie front/back/side
4. **Try-on per view** - 3-fazowy try-on na każdym widoku
5. **3DGS training** - rekonstrukcja 3D z wyników try-on
6. **Expert LoRAs + SDS** - opcjonalna optymalizacja 3DGS

### Szybka demonstracja (bez pełnego treningu 3DGS)


Sekcja "Download try-on results from Google Drive" w notebooku pozwala pominąć etap generowania try-on i przejść od razu do 3DGS.

### Orientacyjny czas uruchomienia
| Etap | Czas (A100) | Czas (L4) |
|------|-------------|-----------|
| Instalacja + modele | ~5 min      | ~5 min |
| Detekcja widoków (30 klatek) | ~2 min      | ~5 min |
| Try-on wszystkich widoków | ~30 min     | ~90 min |
| 3DGS training (30k iter) | ~15 min     | ~45 min |
| Expert LoRA training (3x1000 steps) | ~30-60 min  | ~60 min |
| SDS-augmented 3DGS (35k iter) | ~40 min     | ~2h |

## Oczekiwany wynik

Po poprawnym uruchomieniu:

1. **Try-on**: W katalogu output pojawią się obrazy z przymierzonym ubraniem dla każdego widoku (front/back/side). Wizualnie ubranie powinno być spójne między widokami.

2. **3DGS**: Wytrenowany model 3D Gaussian Splatting (plik `.ply` w folderze output) renderujący osobę w nowym ubraniu z dowolnego kąta. Model można przeglądać w [SuperSplat](https://playcanvas.com/supersplat/editor) lub dedykowanym viewerze 3DGS.

3. **Loss curves**: Malejące krzywe strat (L1, D-SSIM) widoczne w TensorBoard lub w notebooku.


## Dane wejściowe
- **Wideo osoby**: krótkie (10–15 s) nagranie osoby stojącej w miejscu, wykonane nieruchomą kamerą, podczas gdy kamera obraca się wokół niej o około 360 stopni.
- **Zdjęcia ubrania**: zdjęcia ubrania (przód, tył, opcjonalnie bok) - np. ze sklepu internetowego

## Reprodukcja i weryfikacja wyników

### Poziom 1: Uruchomienie
Notebook uruchamia się w Colab z GPU. Po instalacji wszystkie komórki powinny wykonać się bez błędów.

### Poziom 2: Demonstracja
- Sekcja "Try-on: coarse -> composite -> refine" pokazuje wizualne wyniki pipeline
- Widoczne porównanie: osoba oryginalna -> maska -> wynik try-on
- Sekcja "Run 3DGS training" produkuje model 3D

### Poziom 3: Weryfikacja wyników
- **Metryki**: Masked PSNR obliczany w ostatniej sekcji notebooka (porównanie renderu 3DGS z GT)
- **Loss curves**: `show_3DGS_loss_curve()` wizualizuje krzywe L1/D-SSIM z TensorBoard logów

Pełna reprodukcja od zera (try-on + 3DGS + LoRA + SDS) zajmuje ~2-3h na A100.

## Struktura repozytorium

```
├── configs/                     # konfiguracja pipeline
│   └── pipeline_config.yaml     # parametry domyślne
├── docs/                        # dokumentacja
│   ├── architecture.md
│   ├── usage.md
├── src/                         # moduły Python (importowane przez notebook)
│   ├── CatVTON_3DGS_pipeline.ipynb  # główny notebook (Colab demo)
├── README.md                    # główny plik README
├── requirements.txt             # zależności Python
```

## Ograniczenia

1. **Wymagania GPU**: Pipeline wymaga GPU z CUDA. Bez GPU nie uruchomi się żaden etap generatywny.
2. **Czas obliczeń**: Pełny pipeline (try-on + 3DGS + LoRA + SDS) to ~2-3h nawet na A100.
3. **Jakość try-on na widokach bocznych**: CatVTON był trenowany głównie na widokach frontalnych - jakość na bokach/plecach jest niższa. Reference Attention Processor łagodzi ten problem, ale nie eliminuje go całkowicie.
4. **COLMAP na Colab**: Wymaga kompilacji z GPU support (~10 min). Alternatywnie można użyć gotowych COLMAP danych z Google Drive.
5. **Zależność od zewnętrznych repozytoriów**: CatVTON i gaussian-splatting są klonowane w runtime.
6. **SDS Loss**: Implementacja eksperymentalna - wyniki SDS-augmented 3DGS są porównywalne z vanilla 3DGS w obecnej konfiguracji. Wymaga dalszego tuningu hiperparametrów.
7. **Dane**: Wideo testowe (vadim.MOV) i ubrania są hostowane na Google Drive - wymagany dostęp do internetu.
