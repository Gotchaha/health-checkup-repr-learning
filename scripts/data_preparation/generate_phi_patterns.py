# scripts/data_preparation/generate_phi_patterns.py
"""
Generate PHI patterns YAML file for text de-identification.
Creates a configuration file with regex patterns for detecting PHI in text.
"""
import os
import argparse
import yaml
import logging
import sys
from datetime import datetime

def setup_logging():
    """Configure logging to file and console"""
    os.makedirs("outputs/audit/phi_removal_logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("outputs/audit/phi_removal_logs/phi_patterns_generation.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("phi_patterns")

def generate_phi_patterns():
    """
    Generate PHI detection patterns for Japanese medical text.
    
    Returns:
        Dictionary of patterns by category
    """
    logger = setup_logging()
    logger.info("Generating PHI patterns")
    
    # Prefecture list for address pattern
    prefectures = "東京都|大阪府|京都府|北海道|青森県|岩手県|宮城県|秋田県|山形県|福島県|茨城県|栃木県|群馬県|埼玉県|千葉県|神奈川県|新潟県|富山県|石川県|福井県|山梨県|長野県|岐阜県|静岡県|愛知県|三重県|滋賀県|京都府|大阪府|兵庫県|奈良県|和歌山県|鳥取県|島根県|岡山県|広島県|山口県|徳島県|香川県|愛媛県|高知県|福岡県|佐賀県|長崎県|熊本県|大分県|宮崎県|鹿児島県|沖縄県"
    
    # Define patterns for different PHI categories
    patterns = {
        # Phone number patterns (Japanese format)
        "phone": [
            # Standard Japanese phone numbers
            r"0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{4}",
            # International format with Japan country code
            r"\+81[-\s]?\d{1,4}[-\s]?\d{1,4}[-\s]?\d{4}"
        ],
        
        # Email patterns
        "email": [
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        ],
        
        # Postal code patterns (Japanese format)
        "postal": [
            r"〒?\s*\d{3}[-－]\d{4}"  # Japanese postal code
        ],
        
        # Address patterns (Japanese prefectures and cities) - REVISED PATTERN
        "address": [
            # Prefecture pattern
            f"({prefectures})(\\S{{0,15}})(市|区|町|村)(\\S{{0,15}})([0-9０-９]+丁目)?",
            # Street address patterns
            r"([0-9０-９]+丁目[0-9０-９]+番地?[0-9０-９]*号?)"
        ],
        
        # Person name patterns (Japanese and Western)
        "name": [
            # Western name pattern (First Last)
            r"[A-Z][a-z]+ [A-Z][a-z]+",
            # Common Japanese surnames followed by given names
            r"(山田|佐藤|鈴木|田中|高橋|伊藤|渡辺|中村|小林|加藤)(太郎|一郎|二郎|花子|裕子|恵子|健太|翔|拓也|洋子|直樹|大輔)"
        ],
        
        # Medical facility patterns
        "facility": [
            r"[^\s]*(大学病院|総合病院|クリニック|医療センター|医院)[^\s]*",
            r"[^\s]*(病院|病棟|センター|クリニック|医院)[^\s]*"
        ],
        
        # ID number patterns
        "id": [
            r"[A-Z]\d{8}",  # Generic ID pattern
            r"\d{4}[-]\d{4}[-]\d{4}"  # Hyphenated ID pattern
        ]
    }
    
    logger.info(f"Generated patterns for {len(patterns)} PHI categories")
    
    # Add comments/documentation to the patterns
    pattern_docs = {
        "phone": "# Phone number patterns (Japanese format)",
        "email": "# Email address patterns",
        "postal": "# Japanese postal code patterns",
        "address": "# Japanese address patterns including prefectures and detailed addresses",
        "name": "# Person name patterns (both Western and Japanese formats)",
        "facility": "# Medical facility and institution name patterns",
        "id": "# ID number patterns that might appear in medical text"
    }
    
    # Create documented patterns with header
    documented_patterns = {
        "# PHI Patterns for Japanese Medical Text": "",
        "# Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"): "",
        "# Use these patterns to identify and replace PHI in free text": "",
        "# Each pattern will be replaced with <CATEGORY> tag (e.g., <PHONE>, <EMAIL>)": "",
        "# ": ""
    }
    
    # Add each pattern category with its documentation
    for category, pattern_list in patterns.items():
        documented_patterns[pattern_docs[category]] = ""
        documented_patterns[category] = pattern_list
        
    return documented_patterns

def save_patterns(patterns, output_path):
    """Save patterns to YAML file"""
    logger = setup_logging()
    logger.info(f"Saving patterns to {output_path}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(patterns, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Successfully saved patterns to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save patterns: {str(e)}")
        raise

def main():
    """Main entry point for PHI patterns generation"""
    parser = argparse.ArgumentParser(description='Generate PHI patterns for text de-identification')
    parser.add_argument('--output', default='config/cleaning/phi_patterns.yaml',
                        help='Output path for patterns file')
    
    args = parser.parse_args()
    
    # Generate patterns
    logger = setup_logging()
    logger.info("Starting PHI patterns generation process")
    patterns = generate_phi_patterns()
    
    # Save patterns
    save_patterns(patterns, args.output)
    
    logger.info("PHI patterns generation complete")
    
    # Print summary
    category_count = sum(1 for k in patterns.keys() if not k.startswith('#'))
    total_patterns = sum(len(v) for k, v in patterns.items() if isinstance(v, list))
    
    print("\nPHI Patterns Generation Summary:")
    print(f"Generated patterns for {category_count} PHI categories")
    print(f"Total number of patterns: {total_patterns}")
    print(f"Patterns saved to: {args.output}")

if __name__ == "__main__":
    main()