"""Data and reference results for the foundation labs on hypothesis testing and classifier evaluation.

    pip install numpy pandas scipy scikit-learn
    python scripts/ml-datasets/prepare_stats.py

hypothesis-testing-explainer/data/stats.json
  sleep    Cushny & Peebles (1905), the data "Student" used in 1908 to introduce the t-test: extra hours of sleep of 10
           patients under two drugs (paired design). Public domain.
  mpg      Auto MPG (UCI, Quinlan 1993): fuel economy of Japanese vs European cars (two independent samples).
  titanic  Titanic passengers (seaborn-data mirror of the Encyclopedia Titanica-derived dataset): survival by sex
           and by class (two proportions; chi-square on a contingency table).
  Reference statistics and p-values come from scipy.stats so the lab's own maths can be tested against them.
classifier-metrics-explainer/data/metrics.json
  Breast Cancer Wisconsin (Diagnostic) (UCI, Street et al. 1993; bundled with scikit-learn): 569 tumours, 30 features.
  Logistic regression scores on a stratified held-out 30% split, plus sklearn reference ROC-AUC and PR-AUC.
"""
import json, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / '.cache/ml-datasets'
TITANIC = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
MPG = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv'


def fetch(name, url):
    path = RAW / name
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    print(path.relative_to(ROOT), f'{path.stat().st_size / 1024:.1f} KB')


def r(v, d=6):
    return float(np.round(v, d))


def hypothesis():
    drug1 = [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0]
    drug2 = [1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4]
    paired = stats.ttest_rel(drug2, drug1)
    mpg = pd.read_csv(fetch('mpg.csv', MPG)).dropna(subset=['mpg'])
    japan, europe = mpg[mpg.origin == 'japan'].mpg.values, mpg[mpg.origin == 'europe'].mpg.values
    welch = stats.ttest_ind(japan, europe, equal_var=False)
    student = stats.ttest_ind(japan, europe, equal_var=True)
    t = pd.read_csv(fetch('titanic.csv', TITANIC))
    by_sex = t.groupby('sex').survived.agg(['sum', 'count'])
    f, m = by_sex.loc['female'], by_sex.loc['male']
    p_pool = (f['sum'] + m['sum']) / (f['count'] + m['count'])
    z = (f['sum'] / f['count'] - m['sum'] / m['count']) / np.sqrt(p_pool * (1 - p_pool) * (1 / f['count'] + 1 / m['count']))
    table = pd.crosstab(t.pclass, t.survived)
    chi2, p_chi, dof, expected = stats.chi2_contingency(table)
    write(ROOT / 'tools/kien-thuc/hypothesis-testing-explainer/data/stats.json', {
        'sleep': {'source': 'Cushny & Peebles (1905), dữ liệu “Student” (1908) dùng để giới thiệu kiểm định t: số giờ ngủ thêm của 10 bệnh nhân với hai loại thuốc.',
                  'drug1': drug1, 'drug2': drug2, 't': r(paired.statistic), 'p': r(paired.pvalue, 8), 'df': 9},
        'mpg': {'source': 'Auto MPG (UCI, Quinlan 1993): mức tiêu hao nhiên liệu (dặm/gallon) của xe Nhật và xe châu Âu.',
                'japan': japan.round(1).tolist(), 'europe': europe.round(1).tolist(),
                'welch': {'t': r(welch.statistic), 'p': r(welch.pvalue, 8)}, 'student': {'t': r(student.statistic), 'p': r(student.pvalue, 8)}},
        'titanic': {'source': 'Hành khách tàu Titanic (bản seaborn-data, 891 người): sống sót theo giới tính và hạng vé.',
                    'female': [int(f['sum']), int(f['count'])], 'male': [int(m['sum']), int(m['count'])], 'z': r(z), 'pz': float(f'{2 * stats.norm.sf(abs(z)):.4e}'),
                    'classTable': table.values.tolist(), 'chi2': r(chi2), 'pChi2': float(f'{p_chi:.4e}'), 'dof': int(dof), 'expected': np.round(expected, 2).tolist()},
    })


def metrics():
    d = load_breast_cancer()
    # Positive class = malignant (label 0 in scikit-learn), the case a screening test must catch.
    y = (d.target == 0).astype(int)
    x, xt, yy, yt = train_test_split(d.data, y, test_size=0.3, random_state=0, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.05)).fit(x, yy)
    p = model.predict_proba(xt)[:, 1]
    write(ROOT / 'tools/kien-thuc/classifier-metrics-explainer/data/metrics.json', {
        'source': 'Breast Cancer Wisconsin (Diagnostic) (UCI, Street và cộng sự 1993; bản đi kèm scikit-learn): 569 khối u, 30 đặc trưng đo từ ảnh tế bào.',
        'positive': 'ác tính', 'negative': 'lành tính', 'scores': np.round(p, 5).tolist(), 'labels': yt.tolist(),
        'rocAuc': r(roc_auc_score(yt, p)), 'prAuc': r(average_precision_score(yt, p)), 'trainSize': int(len(yy)), 'testSize': int(len(yt)),
    })


if __name__ == '__main__':
    hypothesis()
    metrics()
