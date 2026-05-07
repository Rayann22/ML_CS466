import re
import os

STOPWORDS = {
    "the", "a", "an",
    "is", "are", "was", "were", "be", "been", "being",
    "am", "to", "of", "in", "on", "at", "for", "from",
    "with", "as", "by", "this", "that", "these", "those",
    "it", "its", "he", "she", "they", "we", "you", "i",
    "his", "her", "their", "our", "my", "your",
    "and"
}

# IMPORTANT: do NOT remove "not", "no", "never", "but"
# because they are important for sentiment.

def break_grammar(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)

def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for line in lines:
            new_line = break_grammar(line)
            if new_line:
                out.write(new_line + "\n")

process_file("data/rt-polarity.pos", "data/imdb_grammar.pos")
process_file("data/rt-polarity.neg", "data/imdb_grammar.neg")

print("Done.")
print("Created grammar-perturbed dataset in data/grammar/")
