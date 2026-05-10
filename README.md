# mnist-pytorch-pipeline

Kompletan PyTorch trening pipeline za klasifikaciju MNIST cifara.  
Demonstracija profesionalnog ML projekta — dataset, trening, regularizacija, checkpointing, evaluacija i analiza grešaka.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cozakapri/mnist-pytorch-pipeline/blob/main/mnist_pipeline.ipynb)
&nbsp;
![Python](https://img.shields.io/badge/python-3.11+-blue)
&nbsp;
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-orange)

---

## Brzi start

### Lokalno (preporučeno ako imaš NVIDIA GPU)

```bash
git clone https://github.com/cozakapri/mnist-pytorch-pipeline.git
cd mnist-pytorch-pipeline
uv sync
jupyter notebook mnist_pipeline.ipynb
```

### Colab + lokalna evaluacija (preporučeno bez GPU-a)

1. Klikni **Open in Colab** dugme iznad
2. Pokreni prvu ćeliju — setup se izvršava automatski
3. Pokreni trening na besplatnom GPU-u
4. Na kraju treninga pokreni **download ćeliju** — preuzima `checkpoints.zip`
5. Raspakuj u lokalni `checkpoints/` folder
6. Nastavi evaluaciju i analizu grešaka lokalno u VSCode-u

---

## Zašto Colab za trening?

PyTorch zauzima ~2 GB i instalira se jednom sa `uv sync`. Problem nije instalacija — problem je **brzina treninga bez GPU-a**.

| | CPU (bez GPU) | GPU (Colab / NVIDIA) |
| --- | --- | --- |
| MLP trening | veoma sporo | ~4–5 min |
| CNN trening | veoma sporo | ~6–7 min |
| **Evaluacija i inferencija** | **brzo** | brzo |

Trening radi na Colabu. Sve ostalo — pregledanje koda, evaluacija, inferencija — radi lokalno normalno.

---

## Struktura projekta

```text
mnist-pytorch-pipeline/
│
├── mnist_pipeline.ipynb   ← glavni notebook — pokreni ovde
│
├── src/
│   ├── config.py          ← svi hiperparametri na jednom mestu
│   ├── dataset.py         ← učitavanje, podela train/val/test, vizualizacija
│   ├── model.py           ← MLP i CNN arhitekture
│   ├── train.py           ← trening petlja, early stopping, checkpointing
│   └── evaluate.py        ← metrike, matrica konfuzije, analiza grešaka
│
├── checkpoints/           ← čuvaju se automatski tokom treninga  [gitignored]
├── data/                  ← MNIST se preuzima automatski          [gitignored]
│
├── pyproject.toml         ← zavisnosti projekta
└── uv.lock                ← zaključane verzije paketa
```

---

## Šta notebook pokriva

| Sekcija | Sadržaj |
| --- | --- |
| 0. Okruženje | Importi, device, seed — i automatski Colab setup |
| 1. Dataset | Učitavanje, podela train/val/test, vizualizacija uzoraka i klasa |
| 2. Baseline | Majority class — referentna tačka pre ikakvog treninga |
| 3. MLP bez regularizacije | Demonstracija overfitting-a |
| 4. MLP sa regularizacijom | Dropout + weight decay, poređenje krivih |
| 5. CNN | Konvoluciona mreža, weight sharing, prostorni obrasci |
| 6. Checkpointing | Čuvanje best/last stanja, učitavanje za evaluaciju |
| 7. Evaluacija | Test accuracy, confusion matrix, per-class report |
| 8. Analiza grešaka | Vizualizacija grešaka, top konfuzije između klasa |
| 9. Finalno poređenje | Accuracy / vreme treninga / broj parametara sva 3 modela |

---

## Kako adaptirati za sopstveni projekat

Struktura pipeline-a ostaje ista. Menja se:

- `src/dataset.py` → tvoja funkcija za učitavanje podataka
- `src/model.py` → tvoja arhitektura
- `CrossEntropyLoss` → loss koji odgovara zadatku
- metrike → one koje imaju smisla za tvoj problem
