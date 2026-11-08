
"""Препроцесинг UCI Bank Marketing для mid-term проєкту."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split   # НОВЕ
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

UNKNOWN_COLS = ['job', 'marital', 'education', 'default', 'housing', 'loan']

CAT_COLS = ['job', 'education', 'poutcome',
            'age_bin', 'campaign_bin', 'previous_bin']

BIN_COLS = ['pdays_known', 'has_unknown', 'is_single',
            'is_monday', 'default_clean', 'is_cellular']

NUM_COLS = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
            'euribor3m', 'nr.employed']

INPUT_COLS = CAT_COLS + BIN_COLS + NUM_COLS


def add_time_axis(df: pd.DataFrame) -> pd.DataFrame:
    """Номер блоку місяця (1..26) і рік за порядком рядків у файлі.
    Викликати на сирому файлі до будь-якого перемішування."""
    df = df.copy()
    df['period'] = (df['month'] != df['month'].shift()).cumsum()
    df['year'] = np.where(df['period'] <= 7, 2008,
                          np.where(df['period'] <= 17, 2009, 2010))
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless ознаки: кожна залежить тільки від свого рядка.
    Працює і на нових даних без колонки 'y'."""
    df = df.copy()
    if 'y' in df.columns:
        df['target'] = (df['y'] == 'yes').astype(int)

    df['age_bin'] = pd.cut(df['age'], bins=[-np.inf, 25, 30, 40, 50, 60, 70, np.inf],
                           labels=['<=25', '26-30', '31-40', '41-50', '51-60', '61-70', '71+'])
    df['campaign_bin'] = pd.cut(df['campaign'], bins=[-np.inf, 1, 2, 5, np.inf],
                                labels=['1', '2', '3-5', '6+'])
    df['previous_bin'] = pd.cut(df['previous'], bins=[-np.inf, 0, 1, 2, np.inf],
                                labels=['0', '1', '2', '3+'])

    df['pdays_known']   = (df['pdays'] < 999).astype(int)
    df['has_unknown']   = ((df[UNKNOWN_COLS] == 'unknown').sum(axis=1) > 0).astype(int)
    df['is_single']     = (df['marital'] == 'single').astype(int)
    df['is_monday']     = (df['day_of_week'] == 'mon').astype(int)
    df['default_clean'] = (df['default'] == 'no').astype(int)
    df['is_cellular']   = (df['contact'] == 'cellular').astype(int)
    df['education']     = df['education'].replace('illiterate', 'basic.4y')
    return df


def split_random(df: pd.DataFrame, test_size: float = 0.2,          # НОВЕ
                 random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Випадкове стратифіковане розбиття на train і val.
    Підбір гіперпараметрів далі через CV всередині train."""
    train_df, val_df = train_test_split(df, test_size=test_size,
                                        random_state=random_state,
                                        stratify=df['target'])
    return train_df.copy(), val_df.copy()

def split_time(df: pd.DataFrame,
               last_train_period: int = 17) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Часове розбиття: train 2008 і 2009 (блоки 1..17), test 2010 (блоки 18..26).
    Без перемішування і без stratify."""
    train_df = df[df['period'] <= last_train_period]
    test_df  = df[df['period'] >  last_train_period]
    return train_df.copy(), test_df.copy()

def fit_clip_bounds(train_df: pd.DataFrame, cols: list[str]) -> dict:
    """Межі обрізання з train: {колонка: (min, max)}."""
    return {c: (train_df[c].min(), train_df[c].max()) for c in cols}


def apply_clip(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Обрізати колонки до меж train. Жодного fit усередині."""
    df = df.copy()
    for c, (lo, hi) in bounds.items():
        df[c] = df[c].clip(lo, hi)
    return df


def fit_preprocessor(train_df: pd.DataFrame, kind: str = 'linear') -> dict:
    """Навчити препроцесор ТІЛЬКИ на train.
    kind='linear': обрізання + one-hot + масштабування числових (логістична, kNN).
    kind='tree':   one-hot, числові без змін (дерева)."""
    assert kind in ('linear', 'tree'), kind
    train_df = train_df[INPUT_COLS].copy()

    if kind == 'linear':
        bounds = fit_clip_bounds(train_df, NUM_COLS)
        train_df = apply_clip(train_df, bounds)
        num_step = StandardScaler()
    else:
        bounds = None
        num_step = 'passthrough'

    ct = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
            ('bin', 'passthrough', BIN_COLS),
            ('num', num_step, NUM_COLS)],
        verbose_feature_names_out=False)
    ct.set_output(transform='pandas')
    ct.fit(train_df)
    return {'bounds': bounds, 'ct': ct}


def apply_preprocessor(df: pd.DataFrame, prep: dict) -> pd.DataFrame:
    """Застосувати навчений препроцесор. Жодного fit усередині."""
    df = df[INPUT_COLS].copy()
    if prep['bounds'] is not None:
        df = apply_clip(df, prep['bounds'])
    return prep['ct'].transform(df)


def fit_category_dtypes(train_df: pd.DataFrame, cols: list[str] = CAT_COLS) -> dict:
    """Списки категорій ТІЛЬКИ з train: {колонка: CategoricalDtype}."""
    return {c: pd.CategoricalDtype(categories=sorted(train_df[c].astype(str).unique()))
            for c in cols}


def apply_category_dtypes(df: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
    """Матриця для бустингу: категорії з train, решта без змін.
    Рівень, якого не було в train, стає NaN."""
    df = df[INPUT_COLS].copy()
    for c, dt in dtypes.items():
        df[c] = df[c].astype(str).astype(dt)
    return df


def fit_prep(train_df: pd.DataFrame, kind: str) -> dict:
    """Навчити препроцесор потрібного виду ТІЛЬКИ на train."""
    if kind == 'boost':
        return {'kind': kind, 'dtypes': fit_category_dtypes(train_df)}
    return {'kind': kind, **fit_preprocessor(train_df, kind)}


def apply_prep(df: pd.DataFrame, prep: dict) -> pd.DataFrame:
    """Застосувати навчений препроцесор. Жодного fit усередині."""
    if prep['kind'] == 'boost':
        return apply_category_dtypes(df, prep['dtypes'])
    return apply_preprocessor(df, prep)


def preprocess_data(raw_df: pd.DataFrame, kind: str = 'linear', split: str = 'random',
                    random_state: int = 42, dedup_train: bool = False) -> dict:
    """Повний цикл: ознаки, розбиття, fit на train, матриці train і val.
    kind: 'linear' | 'tree' | 'boost'; split: 'random' | 'time'."""
    df = build_features(add_time_axis(raw_df))

    if split == 'random':
        train_df, val_df = split_random(df, random_state=random_state)
    else:
        train_df, val_df = split_time(df)

    if dedup_train:
        train_df = train_df.drop_duplicates(subset=list(raw_df.columns))

    prep = fit_prep(train_df, kind)
    X_train = apply_prep(train_df, prep)
    X_val   = apply_prep(val_df, prep)

    return {'X_train': X_train, 'train_targets': train_df['target'],
            'X_val': X_val, 'val_targets': val_df['target'],
            'prep': prep, 'input_cols': list(X_train.columns)}


def preprocess_new_data(new_df: pd.DataFrame, prep: dict, input_cols: list[str]) -> pd.DataFrame:
    """Нові сирі дані: ознаки і навчений препроцесор. Жодного fit.
    Колонка 'y' не потрібна."""
    X = apply_prep(build_features(new_df), prep)
    assert list(X.columns) == input_cols, 'колонки не збігаються з train'
    return X


BUSINESS_SHARE = 1000 / 41188   # 1 000 дзвінків на всю базу, 2.43 %


def lift_at_share(y_true, proba, share: float = BUSINESS_SHARE) -> float:
    """Частка позитивів серед топ-share рядків за ймовірністю, поділена на частку в усіх."""
    y = np.asarray(y_true)
    k = max(1, int(round(len(y) * share)))
    top = np.argsort(-np.asarray(proba))[:k]
    return y[top].mean() / y.mean()


def evaluate(y_true, proba, share: float = BUSINESS_SHARE) -> dict:
    """Метрики ранжування. proba = predict_proba(X)[:, 1], ніколи не predict(X)."""
    base = np.mean(y_true)
    ap = average_precision_score(y_true, proba)
    return {'base_rate': base, 'ap': ap, 'ap_lift': ap / base,
            'roc_auc': roc_auc_score(y_true, proba),
            'lift_top': lift_at_share(y_true, proba, share)}
