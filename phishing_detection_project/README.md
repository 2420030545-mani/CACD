# Intelligent AI Framework for Real-Time Phishing Website Detection

A working implementation of the framework described in your paper *"Intelligent
Artificial Intelligence Framework for Real-Time Detection and Classification of
Phishing Websites."* It follows the same pipeline: dataset → preprocessing →
feature extraction → model training/comparison → real-time inference with an
explainable risk score, served through a Flask web app.

```
Dataset Collection → Preprocessing → Feature Extraction → Feature Selection →
Model Training → Evaluation → Best Model Selection → Real-Time URL Input →
Classification → Risk Assessment → Alert
```

---

## 1. What's in this package

```
phishing_detection_project/
├── requirements.txt
├── check_url.py              ← command-line URL checker
├── src/
│   ├── feature_extraction.py ← extracts 28 URL/domain/SSL/redirect/entropy/punycode features
│   ├── generate_dataset.py   ← builds a ready-to-use synthetic training set
│   ├── train_model.py        ← trains & compares DT / RF / LR / SVM / XGBoost
│   └── predict.py            ← real-time classification + risk scoring
├── data/
│   └── phishing_dataset.csv  ← generated training data (created by you, see below)
├── models/                   ← trained model + scaler get saved here
└── webapp/
    ├── app.py                ← Flask web application
    ├── templates/index.html
    └── static/style.css
```

## 2. Requirements

* Python 3.9+
* pip

Install dependencies:

```bash
pip install -r requirements.txt
```

`xgboost` and `python-whois` are optional — the project runs fully without
them (it just trains 4 models instead of 5, and skips the live domain-age
check). Install them for the complete feature set described in the paper.

## 3. How to run the whole project (step by step)

Run these commands **in order**, from the project's root folder
(`phishing_detection_project/`):

### Step 1 — Generate the training dataset
```bash
python src/generate_dataset.py
```
This creates `data/phishing_dataset.csv` (2,400 labeled URL samples, balanced
between legitimate and phishing). It's built entirely offline so the project
works with **zero setup and zero internet access**. See section 5 below for
how to swap in the real UCI PhiUSIIL dataset (235,795 URLs) used in the paper.

### Step 2 — Train and compare the models
```bash
python src/train_model.py
```
This trains Decision Tree, Random Forest, Logistic Regression, SVM (and
XGBoost, if installed), prints a comparison table of accuracy / precision /
recall / F1-score / false-positive rate, and saves the best-performing model
to `models/best_model.joblib`.

Example output:
```
  DecisionTree           acc=0.99  f1=0.99  prec=0.99  rec=0.99  fpr=0.01  (0.02s)
  RandomForest           acc=0.99  f1=0.99  prec=0.99  rec=0.99  fpr=0.01  (0.25s)
  LogisticRegression     acc=0.97  f1=0.97  prec=0.96  rec=0.98  fpr=0.03  (0.01s)
  SVM                    acc=0.98  f1=0.98  prec=0.98  rec=0.98  fpr=0.02  (0.04s)

Best model selected: RandomForest (F1=0.99)
```

### Step 3 — Try it from the command line (optional)
```bash
python check_url.py "http://192.168.1.5/paypal-verify/login.php"
python check_url.py "https://www.github.com"
```
Add `--offline` to skip the live SSL/WHOIS/redirect network checks (useful on
a machine without internet access):
```bash
python check_url.py "http://example.com" --offline
```

### Step 4 — Launch the real-time web app
```bash
python webapp/app.py
```
Then open your browser at:
```
http://127.0.0.1:5000
```
Paste any URL into the box and click **Analyze URL** to get an instant
legitimate / suspicious / phishing verdict, a risk-score bar, and the list of
indicators that drove the decision (the explainable-AI output described in
the paper).

Press `Ctrl+C` in the terminal to stop the server.

## 4. Configuration

* **Disable live network checks** (SSL handshake / WHOIS / redirect
  following) if the deployment machine has no internet access, or you simply
  want faster responses using lexical/structural features only:
  ```bash
  # macOS/Linux
  NETWORK_FEATURES=0 python webapp/app.py
  # Windows (PowerShell)
  $env:NETWORK_FEATURES="0"; python webapp/app.py
  ```
* **Change the port**:
  ```bash
  PORT=8080 python webapp/app.py
  ```

## 5. Using the real PhiUSIIL dataset instead of the synthetic one

The synthetic dataset in Step 1 is designed purely so the whole pipeline runs
immediately with no downloads. For results that match the paper's reported
87–99% accuracy range on real-world data:

1. Download the UCI PhiUSIIL Phishing URL Dataset (235,795 URLs, 54 features):
   https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset
2. Save the CSV anywhere on your machine.
3. Train on it directly:
   ```bash
   python src/train_model.py --real-csv "/path/to/PhiUSIIL_Phishing_URL_Dataset.csv" --label-col label
   ```
   (If the dataset's label column has a different name — PhiUSIIL calls it
   `label`, 1 = legitimate / 0 = phishing in the original release — check the
   column and pass it with `--label-col`, and flip labels with a quick
   `df['label'] = 1 - df['label']` if needed so 1 = phishing throughout this
   project.)

The script automatically keeps only numeric feature columns, so it will use
whatever overlapping/derived features it finds — for full fidelity to your
paper's 54-feature set you can extend `FEATURE_NAMES` in
`src/feature_extraction.py` to mirror PhiUSIIL's exact columns.

## 6. Re-training later (continuous improvement, Section IV-F of the paper)

To simulate periodic retraining as new phishing samples are collected, just
regenerate/append to `data/phishing_dataset.csv` (or point at an updated real
dataset) and re-run:
```bash
python src/train_model.py
```
The new model silently replaces the old one in `models/`.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'flask'` (etc.) | Run `pip install -r requirements.txt` |
| `RuntimeError: No trained model found` | Run Steps 1 and 2 before Step 3/4 |
| Web app loads but every URL times out | You're offline — run with `NETWORK_FEATURES=0` |
| `xgboost` fails to install | It's optional — the project works fine without it |

## 8. Project team

Harshini S R · P. Priyanka · Mani Charan Chava — Guide: Ms. Vinisha
Dept. of Computer Science and Engineering
