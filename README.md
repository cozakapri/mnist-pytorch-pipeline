# MNIST Pipeline

Kompletan PyTorch trening pipeline — demonstracija profesionalnog ML projekta.

## Struktura projekta

```text
mnist_pipeline.ipynb   ← glavni notebook, pokrenuti ovde
src/
  config.py            ← hiperparametri i konfiguracija
  dataset.py           ← učitavanje MNIST-a, vizualizacija, baseline
  model.py             ← MLP i CNN arhitekture
  train.py             ← trening petlja, early stopping, checkpointing
  evaluate.py          ← metrike, matrica konfuzije, analiza grešaka
checkpoints/           ← čuvaju se automatski tokom treninga (gitignored)
data/                  ← MNIST se preuzima automatski (gitignored)
```

## Pokretanje

```bash
# Instaliraj zavisnosti (preporučuje se uv)
uv sync

# Pokreni Jupyter
jupyter notebook mnist_pipeline.ipynb
```

## Šta notebook pokriva

| Korak                  | Sadržaj                                                              |
|------------------------|----------------------------------------------------------------------|
| Dataset                | Učitavanje, podela train/val/test, vizualizacija, class distribution |
| Baseline               | Majority class — referentna tačka za poređenje                       |
| MLP bez regularizacije | Demonstracija overfitting-a                                          |
| MLP sa regularizacijom | Dropout + weight decay                                               |
| CNN                    | Konvoluciona mreža, weight sharing, prostorni obrasci                |
| Checkpointing          | Čuvanje i učitavanje best/last stanja                                |
| Evaluacija             | Test accuracy, confusion matrix, per-class report                    |
| Analiza grešaka        | Vizualizacija pogrešnih predikcija, top konfuzije                    |

## Zavisnosti

- Python 3.13+
- PyTorch, torchvision
- scikit-learn (metrike)
- matplotlib, numpy
