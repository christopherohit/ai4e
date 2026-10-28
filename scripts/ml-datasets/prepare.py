"""Download the public datasets used by the foundational ML labs, compute reference numbers with scikit-learn and write
compact JSON files next to the tools that use them.

    pip install numpy pandas scikit-learn
    python scripts/ml-datasets/prepare.py [--raw DIR]

Sources (all public, attribution in each JSON and in the articles):
  Auto MPG            UCI ML Repository (Quinlan 1993), CSV mirror from the seaborn-data repository.
  California Housing  StatLib (Pace & Barry 1997), CSV mirror from the handson-ml2 repository.
  SMS Spam Collection UCI ML Repository (Almeida & Gómez Hidalgo 2011), TSV mirror from justmarkham/pycon-2016-tutorial.
  Iris, Diabetes      bundled with scikit-learn (Fisher 1936; Efron et al. 2004).
Raw SMS texts are not copied into the repository: only word counts and aggregate metrics are kept.
"""
import argparse, json, re, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_iris
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, lasso_path
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
URLS = {
    'mpg.csv': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv',
    'housing.csv': 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv',
    'sms.tsv': 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv',
}
TOKEN = r'[a-z0-9£]+'


def fetch(raw: Path, name: str) -> Path:
    path = raw / name
    if not path.exists():
        raw.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URLS[name], path)
    return path


def write(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    print(f'{path.relative_to(ROOT)}: {path.stat().st_size / 1024:.1f} KB')


def r(value, digits=4):
    return float(np.round(value, digits))


def auto_mpg(raw: Path):
    df = pd.read_csv(fetch(raw, 'mpg.csv')).dropna(subset=['horsepower']).reset_index(drop=True)
    x, y = df.weight.values / 1000, df.mpg.values
    fit = LinearRegression().fit(x[:, None], y)
    hp = df.horsepower.values
    deg2 = np.polyfit(hp, y, 2)
    write(ROOT / 'tools/_shared/glass-lab/data/auto-mpg.json', {
        'source': 'Auto MPG, UCI Machine Learning Repository (Quinlan 1993); 392 xe có đủ mã lực.',
        'name': df.name.str.title().tolist(), 'mpg': y.round(1).tolist(), 'weight': df.weight.astype(int).tolist(),
        'horsepower': hp.astype(int).tolist(), 'year': (1900 + df.model_year).astype(int).tolist(), 'origin': df.origin.tolist(),
        'reference': {
            'weightFit': {'slope': r(fit.coef_[0]), 'intercept': r(fit.intercept_), 'r2': r(fit.score(x[:, None], y)), 'mse': r(np.mean((fit.predict(x[:, None]) - y) ** 2))},
            'horsepowerQuadratic': [r(v, 8) for v in deg2],
        },
    })


def housing(raw: Path):
    df = pd.read_csv(fetch(raw, 'housing.csv'))
    x, y = df.median_income.values, df.median_house_value.values / 1e5
    fit = LinearRegression().fit(x[:, None], y)
    write(ROOT / 'tools/kien-thuc/mini-batch-sgd-explainer/data/california-housing.json', {
        'source': 'California Housing, StatLib (Pace & Barry 1997), điều tra dân số Mỹ 1990; 20 640 khu dân cư.',
        'income': np.round(x * 1e4).astype(int).tolist(), 'value': np.round(df.median_house_value.values / 100).astype(int).tolist(),
        'units': {'income': 'đô la / năm (thu nhập trung vị, chia 10 000 khi dùng)', 'value': 'trăm đô la (giá nhà trung vị)'},
        'reference': {'slope': r(fit.coef_[0]), 'intercept': r(fit.intercept_), 'mse': r(np.mean((fit.predict(x[:, None]) - y) ** 2)), 'capped': r(np.mean(df.median_house_value >= 500000))},
    })


def sms(raw: Path):
    df = pd.read_csv(fetch(raw, 'sms.tsv'), sep='\t', header=None, names=['label', 'text'])
    y = (df.label == 'spam').astype(int).values
    train_text, test_text, y_train, y_test = train_test_split(df.text.values, y, test_size=0.2, stratify=y, random_state=42)
    counter = CountVectorizer(token_pattern=TOKEN, lowercase=True, min_df=2)
    Xtr = counter.fit_transform(train_text)
    vocab = counter.get_feature_names_out()
    spam_counts = np.asarray(Xtr[y_train == 1].sum(0)).ravel()
    ham_counts = np.asarray(Xtr[y_train == 0].sum(0)).ravel()
    Xte = counter.transform(test_text)

    def scores(alpha):
        p_spam = (spam_counts + alpha) / (spam_counts.sum() + alpha * len(vocab))
        p_ham = (ham_counts + alpha) / (ham_counts.sum() + alpha * len(vocab))
        prior = np.log([np.mean(y_train == 0), np.mean(y_train == 1)])
        log = np.column_stack([Xte @ np.log(p_ham) + prior[0], Xte @ np.log(p_spam) + prior[1]])
        top = log.max(1, keepdims=True)
        prob = np.exp(log - top); prob /= prob.sum(1, keepdims=True)
        return prob[:, 1]

    alphas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    metrics = []
    for alpha in alphas:
        p = scores(alpha)
        pred = (p >= 0.5).astype(int)
        sk = MultinomialNB(alpha=alpha).fit(Xtr, y_train).predict(Xte)
        assert (sk == pred).all(), 'own Naive Bayes must match scikit-learn'
        tp = int(((pred == 1) & (y_test == 1)).sum()); fp = int(((pred == 1) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum()); tn = int(((pred == 0) & (y_test == 0)).sum())
        metrics.append({'alpha': alpha, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'accuracy': r((tp + tn) / len(pred)), 'precision': r(tp / (tp + fp)), 'recall': r(tp / (tp + fn))})
    p = scores(1)
    edges = [0, 1e-6, 1e-3, 0.1, 0.5, 0.9, 0.999, 1 - 1e-6, 1.0000001]
    bins = [{'from': edges[i], 'to': min(1, edges[i + 1]), 'spam': int(((p >= edges[i]) & (p < edges[i + 1]) & (y_test == 1)).sum()), 'ham': int(((p >= edges[i]) & (p < edges[i + 1]) & (y_test == 0)).sum())} for i in range(len(edges) - 1)]
    order = np.argsort(-(spam_counts + ham_counts))
    write(ROOT / 'tools/kien-thuc/naive-bayes-explainer/data/sms-spam.json', {
        'source': 'SMS Spam Collection v.1, UCI Machine Learning Repository (Almeida & Gómez Hidalgo 2011). Chỉ lưu số đếm từ, không lưu nội dung tin nhắn.',
        'token': TOKEN, 'train': {'spam': int((y_train == 1).sum()), 'ham': int((y_train == 0).sum())}, 'test': {'spam': int((y_test == 1).sum()), 'ham': int((y_test == 0).sum())},
        'vocab': [[vocab[i], int(spam_counts[i]), int(ham_counts[i])] for i in order],
        'metrics': metrics, 'calibration': bins,
        'extreme': r(np.mean((p < 1e-3) | (p > 1 - 1e-3))),
    })


def iris():
    data = load_iris()
    X, t = data.data, data.target
    def svm(cols, mask, positive, C, kernel='linear', gamma='scale'):
        Z = X[mask][:, cols]; Z = (Z - Z.mean(0)) / Z.std(0)
        yy = np.where(t[mask] == positive, 1, -1)
        m = SVC(C=C, kernel=kernel, gamma=gamma).fit(Z, yy)
        out = {'accuracy': r(m.score(Z, yy)), 'supportVectors': int(m.n_support_.sum())}
        if kernel == 'linear': out['margin'] = r(2 / np.linalg.norm(m.coef_[0]))
        return out
    write(ROOT / 'tools/kien-thuc/svm-explainer/data/iris.json', {
        'source': 'Iris (Fisher 1936; Anderson 1935), 150 bông hoa, 3 loài; bản kèm theo scikit-learn.',
        'features': ['chiều dài đài hoa', 'chiều rộng đài hoa', 'chiều dài cánh hoa', 'chiều rộng cánh hoa'], 'unit': 'cm',
        'species': ['setosa', 'versicolor', 'virginica'], 'X': X.round(1).tolist(), 'y': t.tolist(),
        'reference': {
            'separable': svm([0, 1], t < 2, 1, 1e4), 'overlapC1': svm([2, 3], t > 0, 2, 1), 'overlapHard': svm([2, 3], t > 0, 2, 1e4),
            'ringLinear': svm([2, 3], t >= 0, 1, 1), 'ringRbf': svm([2, 3], t >= 0, 1, 1, 'rbf', 0.5),
        },
    })


def diabetes():
    data = load_diabetes(scaled=False)
    X, y = data.data, data.target
    Z = (X - X.mean(0)) / X.std(0); yc = y - y.mean()
    alphas = np.logspace(-2, 2, 41)
    alphas, coefs, _ = lasso_path(Z, yc, alphas=alphas)  # returned in decreasing order, matching coefs
    ols = LinearRegression().fit(Z, y)
    write(ROOT / 'tools/kien-thuc/regularization-explainer/data/diabetes.json', {
        'source': 'Diabetes (Efron, Hastie, Johnstone & Tibshirani 2004, “Least Angle Regression”); 442 bệnh nhân, 10 chỉ số ban đầu; bản kèm theo scikit-learn.',
        'features': list(data.feature_names), 'X': X.round(4).tolist(), 'y': y.astype(int).tolist(),
        'reference': {'ols': [r(v) for v in ols.coef_], 'lassoAlphas': [r(a, 6) for a in alphas], 'lassoPath': [[r(v) for v in c] for c in coefs.T], 'corrS1S2': r(np.corrcoef(X[:, 4], X[:, 5])[0, 1])},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=ROOT / '.cache/ml-datasets')
    args = parser.parse_args()
    auto_mpg(args.raw); housing(args.raw); sms(args.raw); iris(); diabetes()
