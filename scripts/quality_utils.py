"""
Quality utilities for Russian Financial News dataset.
Used across all data quality notebooks.
"""
import re
import ast
import json
import pandas as pd
import numpy as np
from pathlib import Path


# ─── Constants ────────────────────────────────────────────────────────────────

EMOJI_PATTERN = re.compile(
    '[\U00010000-\U0010ffff'
    '\U0001F600-\U0001F64F'
    '\U0001F300-\U0001F5FF'
    '\U0001F680-\U0001F6FF'
    '\U0001F1E0-\U0001F1FF'
    '\u2600-\u26FF'
    '\u2700-\u27BF]',
    flags=re.UNICODE
)

SPECIAL_CHARS_PATTERN = re.compile(r'[⚡️🔥💥⭐🌟✅❌⚠️📊📈📉💰🏦]')


# ─── Issue detection ──────────────────────────────────────────────────────────

def detect_missing(df: pd.DataFrame) -> dict:
    """Detect missing values per column."""
    nulls = df.isnull().sum()
    result = {}
    for col in df.columns:
        n = int(nulls[col])
        if n > 0:
            result[col] = {
                'count': n,
                'pct': round(n / len(df) * 100, 2),
                'severity': 'critical' if n / len(df) > 0.3 else
                            'high' if n / len(df) > 0.05 else 'medium'
            }
    return result


def detect_duplicates(df: pd.DataFrame) -> dict:
    """Detect duplicate rows by title and by title+body."""
    no_title_mask = df['title'] == 'no title'
    dup_title_body = int(df[~no_title_mask].duplicated(subset=['title', 'body']).sum())
    no_title_count = int(no_title_mask.sum())
    no_title_unique_bodies = int(df[no_title_mask]['body'].nunique())

    return {
        'exact_duplicates': {
            'count': dup_title_body,
            'pct': round(dup_title_body / len(df) * 100, 2),
            'severity': 'high' if dup_title_body > 0 else 'ok'
        },
        'no_title_entries': {
            'count': no_title_count,
            'pct': round(no_title_count / len(df) * 100, 2),
            'unique_bodies': no_title_unique_bodies,
            'severity': 'medium',
            'note': 'Telegram-style posts from rdv/t_invest/t_analytic — no title by design'
        }
    }


def detect_text_issues(df: pd.DataFrame) -> dict:
    """Detect text quality issues in title and body."""
    emoji_titles = int(df['title'].apply(lambda x: bool(EMOJI_PATTERN.search(str(x)))).sum())
    short_body = int((df['body'].str.len() < 20).sum())
    empty_body = int((df['body'].str.strip() == '').sum())

    return {
        'emoji_in_title': {
            'count': emoji_titles,
            'pct': round(emoji_titles / len(df) * 100, 2),
            'severity': 'low'
        },
        'short_body': {
            'count': short_body,
            'pct': round(short_body / len(df) * 100, 2),
            'severity': 'medium' if short_body > 10 else 'low',
            'threshold': '< 20 chars'
        },
        'empty_body': {
            'count': empty_body,
            'pct': round(empty_body / len(df) * 100, 2),
            'severity': 'high' if empty_body > 0 else 'ok'
        }
    }


def detect_label_issues(df: pd.DataFrame, target_col: str = 'positive_market_impact') -> dict:
    """Detect issues with target variable."""
    null_labels = int(df[target_col].isna().sum())
    if null_labels == 0:
        balance = df[target_col].value_counts(normalize=True).to_dict()
        minority_pct = min(balance.values()) * 100
        return {
            'missing_labels': {'count': 0, 'severity': 'ok'},
            'class_balance': {
                'distribution': {str(k): round(v, 4) for k, v in balance.items()},
                'minority_pct': round(minority_pct, 1),
                'severity': 'high' if minority_pct < 20 else
                            'medium' if minority_pct < 35 else 'ok'
            }
        }
    else:
        labeled = df[df[target_col].notna()]
        balance = labeled[target_col].value_counts(normalize=True).to_dict()
        minority_pct = min(balance.values()) * 100
        return {
            'missing_labels': {
                'count': null_labels,
                'pct': round(null_labels / len(df) * 100, 2),
                'severity': 'medium'
            },
            'class_balance': {
                'distribution': {str(k): round(v, 4) for k, v in balance.items()},
                'minority_pct': round(minority_pct, 1),
                'severity': 'high' if minority_pct < 20 else
                            'medium' if minority_pct < 35 else 'ok'
            }
        }


def detect_content_issues(df: pd.DataFrame) -> dict:
    """Detect content-level issues (advertising, noise categories)."""
    if 'article_type' not in df.columns:
        return {}
    type_counts = df['article_type'].value_counts(dropna=False).to_dict()
    advertising = int(df['article_type'].eq('advertising').sum())
    return {
        'advertising_articles': {
            'count': advertising,
            'pct': round(advertising / len(df) * 100, 2),
            'severity': 'medium',
            'note': 'May add noise to classification task'
        },
        'article_type_distribution': {str(k): int(v) for k, v in type_counts.items()}
    }


def detect_schema_issues(df: pd.DataFrame) -> dict:
    """Detect schema/type issues."""
    issues = {}
    # sectors and tickers stored as strings
    if 'sectors' in df.columns:
        non_list = df['sectors'].dropna().apply(
            lambda x: not isinstance(x, list) and not (isinstance(x, str) and x.startswith('['))
        ).sum()
        issues['sectors_as_string'] = {
            'count': int(non_list),
            'severity': 'low',
            'note': 'Stored as string repr of list, needs ast.literal_eval'
        }
    return issues


# ─── Quality score ─────────────────────────────────────────────────────────────

SEVERITY_PENALTIES = {'critical': 25, 'high': 10, 'medium': 5, 'low': 1, 'ok': 0}


def compute_quality_score(issues: dict) -> int:
    """Compute overall quality score 0-100 from issues dict."""
    total_penalty = 0

    def recurse(d):
        nonlocal total_penalty
        if isinstance(d, dict):
            if 'severity' in d:
                total_penalty += SEVERITY_PENALTIES.get(d['severity'], 0)
            else:
                for v in d.values():
                    recurse(v)

    recurse(issues)
    return max(0, 100 - total_penalty)


def run_full_detection(df: pd.DataFrame, target_col: str = 'positive_market_impact') -> dict:
    """Run all detection checks and return full report."""
    issues = {
        'missing': detect_missing(df),
        'duplicates': detect_duplicates(df),
        'text': detect_text_issues(df),
        'labels': detect_label_issues(df, target_col),
        'content': detect_content_issues(df),
        'schema': detect_schema_issues(df),
    }
    score = compute_quality_score(issues)
    return {
        'dataset': {
            'rows': len(df),
            'cols': len(df.columns),
            'columns': list(df.columns)
        },
        'quality_score': score,
        'issues': issues
    }


# ─── Fix utilities ────────────────────────────────────────────────────────────

def fix_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows (title+body), keep first."""
    mask = df['title'] != 'no title'
    dups = df[mask].duplicated(subset=['title', 'body'], keep='first')
    dup_idx = df[mask][dups].index
    return df.drop(index=dup_idx).reset_index(drop=True)


def fix_short_body(df: pd.DataFrame, min_len: int = 20) -> pd.DataFrame:
    """Remove rows where body is too short."""
    return df[df['body'].str.len() >= min_len].reset_index(drop=True)


def fix_emoji_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Remove emoji from titles."""
    df = df.copy()
    df['title'] = df['title'].apply(lambda x: EMOJI_PATTERN.sub('', str(x)).strip())
    return df


def fix_advertising(df: pd.DataFrame) -> pd.DataFrame:
    """Remove advertising articles."""
    return df[df['article_type'] != 'advertising'].reset_index(drop=True)


def parse_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse sectors and tickers columns from string to list."""
    df = df.copy()
    for col in ['sectors', 'tickers']:
        if col in df.columns:
            def safe_parse(x):
                if pd.isna(x):
                    return []
                if isinstance(x, list):
                    return x
                try:
                    result = ast.literal_eval(str(x))
                    return result if isinstance(result, list) else []
                except Exception:
                    return []
            df[col] = df[col].apply(safe_parse)
    return df


def fix_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date column to datetime."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df


# ─── Reporting ─────────────────────────────────────────────────────────────────

def save_report(report: dict, path: str = 'data/reports/quality_report.json'):
    """Save quality report to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f'Report saved: {path}')


def load_report(path: str = 'data/reports/quality_report.json') -> dict:
    """Load quality report from JSON."""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def print_report_summary(report: dict):
    """Print human-readable summary of quality report."""
    print(f"Quality Score: {report['quality_score']}/100")
    print(f"Dataset: {report['dataset']['rows']} rows × {report['dataset']['cols']} cols")
    print()
    issues = report['issues']

    def print_issues(d, indent=0):
        for k, v in d.items():
            if isinstance(v, dict) and 'severity' in v:
                sev = v['severity']
                icon = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢', 'ok': '✅'}.get(sev, '⚪')
                count = v.get('count', '')
                pct = v.get('pct', '')
                note = v.get('note', '')
                count_str = f"{count} ({pct}%)" if pct else str(count)
                print(f"{'  '*indent}{icon} {k}: {count_str}" + (f" — {note}" if note else ''))
            elif isinstance(v, dict):
                print(f"{'  '*indent}[ {k} ]")
                print_issues(v, indent+1)

    print_issues(issues)
