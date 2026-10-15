#
# Building the bilingual tokenizer BPE (Arabic + English) for the JAPARTI LLM from scratch.
# This script is the main entry point for building the tokenizer.
#======================================================================

from pathlib import Path
import sys
import pandas as pd
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    decoders,
    trainers,
    normalizers,
)

from cleaning import build_normalizer, normalize_text

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
#============================
# Configurations
#============================

current_dir = Path(__file__).parent

clean_paths = [
    current_dir / "output" / "phase1_train_cleaned.parquet",
    current_dir / "output" / "phase2_train_cleaned.parquet",
]

output_path = current_dir/'tokenizer.json'

vocab_size = 32000
min_frequency = 2

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]",
                  "[AR]", "[EN]",
                  "[SYS]", "[USER]", "[ASST]"]

#============================
# Build the tokenizer
#============================
def build_tokenizer():

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    tokenizer.normalizer = build_normalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="always")
    tokenizer.decoder       = decoders.Metaspace(prepend_scheme="always")

    return tokenizer

#===============================
# train BPE tokenizer
#===============================
def corpus_iterator(df, batch_size=1000):
    texts = df["text"].tolist()
    for i in range(0, len(texts), batch_size):
        yield texts[i: i + batch_size]

def train(tokenizer, df):
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        corpus_iterator(df),
        trainer=trainer,
        length=len(df),
    )

    return tokenizer


#===============================
# Main function
#===============================
def main():

    print("STEP 1: LOAD CLEANED CORPUS")

    frames = []
    for path in clean_paths:
        frame = pd.read_parquet(path)
        print(f"  {path.name}: {len(frame)}")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    print("STEP 2: BUILD PIPELINE + TRAIN BPE")
    tokenizer = build_tokenizer()
    tokenizer = train(tokenizer=tokenizer, df=df)

    print("SANITY CHECK: encode / decode bilingual examples")
    examples = [
        "جُمْهُورِيَّةُ مِصْرَ الْعَرَبِيَّة عاصمتها القاهرة",
        "أحمد إبراهيم آدم وُلد في مصر",
        "The United States and the United Nations signed the agreement",
        "machine learning models require large amounts of training data",
    ]

    for text in examples:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded_text = tokenizer.decode(encoded.ids)

        print(f"\n  input  : {text}")
        print(f"  tokens : {encoded.tokens}")
        print(f"  ids    : {encoded.ids}")

        print(f"  decoded: {decoded_text}")

        print("="*30)

    tokenizer.save(str(output_path))

if __name__ == "__main__":
    main()        
        

