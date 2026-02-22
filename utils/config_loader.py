import os

import yaml


DEFAULT_CONFIG = {
    "scoring": {
        "use_semantic": True,
        "weight_semantic": 0.4,
        "weight_tfidf": 0.3,
        "use_bm25": True,
        "weight_bm25": 0.2,
        "weight_skill_ratio": 0.3,
        "top_k_highlight_sentences": 3,
    },
    "app": {
        "language": "english",
        "enable_audit_log": True,
        "audit_log_dir": "logs",
    },
    "ui": {
        "show_processing_time": True,
    },
}


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads YAML configuration with sensible defaults.
    """
    if not os.path.exists(config_path):
        return DEFAULT_CONFIG

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
    except Exception:
        return DEFAULT_CONFIG

    # Merge user config over defaults (shallow merge is enough here)
    cfg = DEFAULT_CONFIG.copy()
    for section, values in user_cfg.items():
        if isinstance(values, dict) and section in cfg:
            merged = cfg[section].copy()
            merged.update(values)
            cfg[section] = merged
        else:
            cfg[section] = values
    return cfg

