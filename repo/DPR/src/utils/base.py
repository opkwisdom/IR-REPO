import os

def line_count(filename: str) -> int:
    """Counts the number of lines in a file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())