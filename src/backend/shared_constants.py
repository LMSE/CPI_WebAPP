from pathlib import Path

secret = (Path(__file__).parent / 'secret.txt').read_text().strip()