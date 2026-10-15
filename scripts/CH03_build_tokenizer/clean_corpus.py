#======================================
# Download, filter and clean the corpus
#======================================

# step 1: download the corpus
# step 2: type A cleaning: remove empty lines, remove lines with only punctuation, remove lines with non-ASCII characters
#===============================    

from importlib.resources import path
from pathlib import Path
import os
from huggingface_hub import hf_hub_download
from datasets import load_dataset
import pandas as pd
from pathlib import Path

from cleaning import (prepare_document, normalize_text, clean_dataframe)

REPO ='bakrianoo/jabarti-llm-dataset'
chunk_size = 20

Hug_fac={
    'phase1_train': 'pretrain/phase1_train-00000-of-00001.parquet',
    'phase1_valid': 'pretrain/phase1_eval-00000-of-00001.parquet',

    'phase2_train': 'pretrain/phase2_train-00000-of-00001.parquet',
    'phase2_valid': 'pretrain/phase2_eval-00000-of-00001.parquet',

    'ft_train': 'finetune/train-00000-of-00001.parquet',
    'ft_valid': 'finetune/eval-00000-of-00001.parquet'

}

train_split = {'phase1_train','phase2_train'}
eval_split = {'phase1_valid','phase2_valid'}

out_dir = Path(__file__).parent/'output'


def download_and_filter():

    # Iterate through the dataset files
    for name, filename in Hug_fac.items():
        print(f"Downloading {name} ...")

        # Download the Parquet file from Hugging Face
        local = hf_hub_download(
            repo_id=REPO,
            filename=filename,
            repo_type="dataset"
        )
        print(f"Downloaded {name} to {local}")

        # Load the downloaded Parquet file into a DataFrame
        df = pd.read_parquet(local)

        # Apply filtering only to training splits
        if name in train_split:
            before_count = len(df)

            # Sort chunks and keep the first N chunks per article
            df = (
                df.sort_values(["article_id", "chunk_index"])
                .groupby("article_id")
                .head(chunk_size)
                .reset_index(drop=True)
            )

            after_count = len(df)
            print(f"Filtered {name}: {before_count} -> {after_count} rows")

        # Define the output path for the processed dataset
        output_file = out_dir / f"{name}.parquet"

        # Save the processed DataFrame as a Parquet file
        df.to_parquet(output_file, index=False)

def clean_phase(phase, input, output) :
    print(f"Cleaning phase: {phase} ...")

    df = pd.read_parquet(input)

    print(f'source file: {input} , shape: {df.shape}')
    print(f'row : {df.shape[0]} , column : {df.shape[1]}')

    clean_df = clean_dataframe(df)

    removed_rows = df.shape[0] - clean_df.shape[0]
    print(f'cleaned file: {output} , shape: {clean_df.shape}')

    clean_df.to_parquet(output, index=False)



def main():
    download_and_filter()

    for name in train_split:
        input_file = out_dir / f"{name}.parquet"
        output_file = out_dir / f"{name}_cleaned.parquet"

        clean_phase(name, input_file, output_file)
    for name in eval_split:
        input_file = out_dir / f"{name}.parquet"
        output_file = out_dir / f"{name}_cleaned.parquet"

        clean_phase(name, input_file, output_file)
        
if __name__ == "__main__":
    main()