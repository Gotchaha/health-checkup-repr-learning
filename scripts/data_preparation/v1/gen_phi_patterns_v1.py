# scripts/data_preparation/v1/gen_phi_patterns_v1.py
"""
Generate PHI patterns (v1) for de-identification.

Changes vs previous version:
- Clean YAML structure with top-level "meta" and "patterns" sections.
- No pseudo-comments-as-keys.
- Tighter, precision-first regexes; boundaries and hyphen families unified.
- Configurable log directory with timestamped log filename.
"""

import os
import re
import sys
import yaml
import argparse
import logging
from datetime import datetime


# ------------------------- Logging ------------------------- #
def setup_logging(log_dir: str) -> logging.Logger:
    """Configure logging to file (timestamped) and console."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"phi_patterns_gen_{ts}.log")

    logger = logging.getLogger("phi_patterns_v1")
    logger.setLevel(logging.INFO)

    # Clear existing handlers if re-initialized
    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Logging initialized")
    logger.info("Log file: %s", logfile)
    return logger


# ------------------------- Patterns ------------------------- #
def build_patterns(logger: logging.Logger) -> dict:
    """Build finalized pattern lists (precision-first)."""
    # Prefecture alternation (no duplicates)
    prefectures = (
        "東京都|北海道|青森県|岩手県|宮城県|秋田県|山形県|福島県|茨城県|栃木県|群馬県|埼玉県|千葉県|神奈川県|"
        "新潟県|富山県|石川県|福井県|山梨県|長野県|岐阜県|静岡県|愛知県|三重県|滋賀県|京都府|大阪府|兵庫県|"
        "奈良県|和歌山県|鳥取県|島根県|岡山県|広島県|山口県|徳島県|香川県|愛媛県|高知県|福岡県|佐賀県|"
        "長崎県|熊本県|大分県|宮崎県|鹿児島県|沖縄県"
    )

    # Common JP surnames (Top ~20; precision-focused)
    jp_surnames = (
        "佐藤|鈴木|高橋|田中|伊藤|渡辺|山本|中村|小林|加藤|吉田|山田|佐々木|山口|松本|井上|木村|林|清水|(?:斎藤|斉藤)"
    )

    # Hyphen family (half/full width, various dashes)
    H = r"[-−‐–—－]"

    patterns = {
        "postal": [
            # JP postal code; optional "〒" and optional hyphen; digit boundaries
            rf"(?<![0-9０-９])〒?\s*[0-9０-９]{{3}}(?:{H}?\s*[0-9０-９]{{4}})(?![0-9０-９])"
        ],
        "email": [
            # Case-insensitive email
            r"(?i)[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}"
        ],
        "phone": [
            # Mobile 070/080/090
            rf"(?<![0-9０-９])0[5789]0(?:{H}?\d{{4}}){{2}}(?![0-9０-９])",
            # 050 IP phone
            rf"(?<![0-9０-９])050{H}?\d{{4}}{H}?\d{{4}}(?![0-9０-９])",
            # Toll-free 0120 / 0800
            rf"(?<![0-9０-９])0120{H}?\d{{3}}{H}?\d{{3}}(?![0-9０-９])",
            rf"(?<![0-9０-９])0800{H}?\d{{3}}{H}?\d{{3}}(?![0-9０-９])",
            # (0X)XXXX-XXXX
            rf"(?<![0-9０-９])\(0\d{{1,4}}\)\d{{1,4}}{H}?\d{{4}}(?![0-9０-９])",
            # General landline
            rf"(?<![0-9０-９])0\d{{1,4}}{H}?\d{{1,4}}{H}?\d{{4}}(?![0-9０-９])",
            # International +81 with optional (0)
            r"(?<!\d)\+81[-\s]?(?:\(0\))?\d{1,4}[-\s]?\d{1,4}[-\s]?\d{4}(?!\d)",
        ],
        "id": [
            # Context-anchored token (6–20 chars), must contain at least one ASCII letter; allow digits (incl. fullwidth), hyphen, underscore
            r"(?i)(?:ID|ｉｄ|ＩＤ|番号|患者ID|受付番号|会員番号)[:：]?\s*(?=[A-Z])[A-Z0-9０-９][A-Z0-9０-９_\-－]{5,19}(?![A-Za-z0-9０-９])",
            # Context-anchored segmented token: 2–5 segments of 2–8 alnum chars separated by hyphens; must contain at least one ASCII letter
            r"(?i)(?:ID|ｉｄ|ＩＤ|番号|患者ID|受付番号|会員番号)[:：]?\s*(?=[A-Z])(?:[A-Z0-9０-９]{2,8}(?:[-－][A-Z0-9０-９]{2,8}){1,4})(?![A-Za-z0-9０-９])",
        ],
        "name": [
            # Common JP surnames + 1–2 JP chars; avoid immediate hospital terms
            rf"(?:{jp_surnames})[一-龥々〆〇ヶぁ-んァ-ヶー]{{1,2}}(?!(?:病院|医院))",
            # Surname + honorific (precision-focused)
            rf"(?:{jp_surnames})(?:さん|様|氏|先生)",
        ],
        "facility": [
            # Strong-tail facility names; limited local context; avoid \S* swallowing
            r"[一-龥々〆〇ヶぁ-んァ-ヶーA-Za-z0-9０-９・\-（）()]{0,20}(?:大学病院|総合病院|医療センター|クリニック|医院)",
            # Generic words: require 2–6 JP chars before; strict boundaries
            r"(?<![一-龥々ぁ-んァ-ヶA-Za-z0-9０-９])[一-龥々〆〇ヶぁ-んァ-ヶー]{2,6}(?:病院|センター)(?![一-龥々ぁ-んァ-ヶA-Za-z0-9０-９])",
        ],
        "address": [
            # Prefecture + (city/ward/town/village); cap middle chunk to 0–20 non-space/non-punct general chars
            rf"(?:{prefectures})[^\s、，。]{{0,20}}(?:市|区|町|村)",
            # County (郡) + town/village
            r"[一-龥々〆〇ヶぁ-んァ-ヶー]{1,8}郡[一-龥々〆〇ヶぁ-んァ-ヶー]{1,12}(?:町|村)",
            # 丁目/番地/号 variations
            r"[0-9０-９]+丁目(?:[0-9０-９]+番地?(?:の[0-9０-９]+)?)?[0-9０-９]*号?",
            # Numeric block pattern like 1-2-3 or 12-34-56-7, preceded by JP char; do not trail into more digits
            rf"(?<=[一-龥々〆〇ヶぁ-んァ-ヶー])[0-9０-９]{{1,3}}(?:{H}[0-9０-９]{{1,3}}){{1,3}}(?![0-9０-９])",
        ],
    }

    # Basic validation: compile all regexes once to fail fast if malformed
    total = 0
    for cat, plist in patterns.items():
        if not isinstance(plist, list) or not plist:
            logger.warning("Category %s has empty or invalid list", cat)
            continue
        for i, pat in enumerate(plist):
            try:
                re.compile(pat)
            except re.error as e:
                logger.error("Regex compile error in category '%s' index %d: %s", cat, i, e)
                raise
            total += 1

    logger.info("Built %d categories with %d total regex patterns", len(patterns), total)
    return patterns


# ------------------------- YAML I/O ------------------------- #
def build_yaml_document(patterns: dict) -> dict:
    """Assemble final YAML document with meta + patterns sections."""
    meta = {
        "version": "v1",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generator": "scripts/data_preparation/v1/gen_phi_patterns_v1.py",
        "notes": [
            "Precision-first regexes with explicit boundaries and hyphen families.",
            "No pseudo-comment keys; clean 'meta' and 'patterns' sections.",
            "Facility rules prefer strong-tail forms; generic forms are boundary-restricted.",
            "Name rules use top JP surnames; English names intentionally excluded in v1.",
            "ID rules are context-anchored and letter-required to avoid numeric collisions.",
        ],
    }
    return {"meta": meta, "patterns": patterns}


def save_yaml(doc: dict, output_path: str, logger: logging.Logger) -> None:
    """Save YAML with stable key order and Unicode preserved."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(doc, f, allow_unicode=True, sort_keys=False)
    logger.info("Saved patterns YAML to: %s", output_path)


# ------------------------- CLI ------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PHI regex patterns YAML (v1) for de-identification"
    )
    parser.add_argument(
        "--output",
        default="config/cleaning/v1/deidentification/phi_patterns.yaml",
        help="Output path for the patterns YAML",
    )
    parser.add_argument(
        "--logdir",
        default="outputs/audit/v1/deidentification/",
        help="Directory to store logs (log filename is timestamped)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.logdir)
    logger.info("Starting PHI patterns generation (v1)")

    try:
        patterns = build_patterns(logger)
        doc = build_yaml_document(patterns)
        save_yaml(doc, args.output, logger)

        # Summary
        cat_count = len(patterns)
        pat_count = sum(len(v) for v in patterns.values())
        print("\nPHI Patterns Generation Summary (v1)")
        print(f"Categories: {cat_count}")
        print(f"Total regex patterns: {pat_count}")
        print(f"Output: {args.output}")
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        sys.exit(1)

    logger.info("PHI patterns generation complete")


if __name__ == "__main__":
    main()
