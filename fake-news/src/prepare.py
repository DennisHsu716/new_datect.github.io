
import argparse, pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    df = df.dropna(subset=["text","label"]).reset_index(drop=True)
    df.to_json(args.out, orient="records", lines=True)
    print("saved jsonl:", args.out, "n=", len(df))
