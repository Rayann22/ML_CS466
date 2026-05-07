import csv

input_file = "data/sentiment140/training.1600000.processed.noemoticon.csv"

pos_file = "data/sentiment140/twitter.pos"
neg_file = "data/sentiment140/twitter.neg"

max_each = 500   # keep small for now

pos_count = 0
neg_count = 0

with open(input_file, "r", encoding="latin-1") as f, \
     open(pos_file, "w", encoding="utf-8") as pos_out, \
     open(neg_file, "w", encoding="utf-8") as neg_out:

    reader = csv.reader(f)

    for row in reader:
        label = row[0]
        text = row[5]

        if label == "4" and pos_count < max_each:
            pos_out.write(text + "\n")
            pos_count += 1

        elif label == "0" and neg_count < max_each:
            neg_out.write(text + "\n")
            neg_count += 1

        if pos_count >= max_each and neg_count >= max_each:
            break

print("DONE")
print("Positive:", pos_count)
print("Negative:", neg_count)
