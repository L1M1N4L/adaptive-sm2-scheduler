from datasets import load_dataset
import pandas as pd

# Stream the dataset instead of loading into memory
ds = load_dataset("open-spaced-repetition/FSRS-Anki-20k", split="train", streaming=True)

batch_size = 100000   # adjust if needed
batch = []
file_number = 1

for i, row in enumerate(ds):
    batch.append(row)

    # Every 100k rows, save to a CSV chunk
    if len(batch) >= batch_size:
        df = pd.DataFrame(batch)
        df.to_csv(f"fsrs_anki_20k_part_{file_number}.csv", index=False)
        print(f"Saved part {file_number}")
        file_number += 1
        batch = []

# Save remaining rows
if batch:
    df = pd.DataFrame(batch)
    df.to_csv(f"fsrs_anki_20k_part_{file_number}.csv", index=False)
    print(f"Saved part {file_number}")
