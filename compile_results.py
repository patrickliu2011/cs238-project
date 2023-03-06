import os
import pandas as pd

from argparse import ArgumentParser

def main(args):
    rows = []
    for exp in sorted(os.listdir(args.results_dir)):
        print("Experiment: ", exp)
        exp_folder = os.path.join(args.results_dir, exp)
        for file in sorted(os.listdir(exp_folder)):
            if file.endswith(".csv"):
                print("File: ", file)
                df = pd.read_csv(os.path.join(exp_folder, file))
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                metrics = df.mean(axis=0).to_dict()
                print(metrics)
                row = {"exp": exp, "epoch": file[:-4]}
                row.update(metrics)
                rows.append(row)
        print()
    results = pd.DataFrame(rows)
    results.to_csv(args.output_file, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="eval_results",
                        help="Directory containing experiment results")
    parser.add_argument("--output-file", type=str, default="results.csv",
                        help="File to save results to")
    args = parser.parse_args()
    main(args)