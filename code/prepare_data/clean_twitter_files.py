import re

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)  # remove links
    text = re.sub(r"@\w+", "", text)            # remove usernames
    text = re.sub(r"\s+", " ", text)            # remove extra spaces
    return text.strip()

def clean_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    with open(output_path, "w", encoding="utf-8") as out:
        for line in lines:
            cleaned = clean_text(line)
            if cleaned:
                out.write(cleaned + "\n")

clean_file("data2/twitter.pos", "data2/twitter_clean.pos")
clean_file("data2/twitter.neg", "data2/twitter_clean.neg")

print("Done cleaning Twitter files.")
print("Created:")
print("data2/twitter_clean.pos")
print("data2/twitter_clean.neg")
