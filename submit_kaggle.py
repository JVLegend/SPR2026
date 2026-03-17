"""
submit_kaggle.py
Envia submissions para o Kaggle a partir dos arquivos gerados em submissions/.

Uso:
    python submit_kaggle.py                  # envia TODOS os CSVs em submissions/
    python submit_kaggle.py --exp EXP-01     # envia so o de um experimento
    python submit_kaggle.py --file submissions/submission_EXP-01.csv
    python submit_kaggle.py --best           # envia o de maior cv_f1 no results.tsv
"""

import argparse
import os
import subprocess
import sys

import pandas as pd

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SUB_DIR     = os.path.join(BASE_DIR, "submissions")
RESULTS_TSV = os.path.join(BASE_DIR, "experiments", "results.tsv")
COMPETITION = "spr-2026-mammography-report-classification"


def submit(filepath, message):
    print(f"\n  Enviando: {filepath}")
    print(f"  Mensagem: {message}")
    result = subprocess.run(
        [
            "kaggle", "competitions", "submit",
            "-c", COMPETITION,
            "-f", filepath,
            "-m", message,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"  OK: {result.stdout.strip()}")
    else:
        print(f"  ERRO: {result.stderr.strip()}")
    return result.returncode == 0


def get_message(exp_id):
    """Busca cv_f1 e notes do results.tsv para montar mensagem descritiva."""
    if os.path.exists(RESULTS_TSV):
        try:
            df = pd.read_csv(RESULTS_TSV, sep="\t")
            row = df[df["exp_id"] == exp_id]
            if not row.empty:
                r = row.iloc[0]
                return f"{exp_id} cv_f1={r.get('cv_f1', '?')} | {r.get('notes', '')}"
        except Exception:
            pass
    return exp_id


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--exp",  default=None, help="Ex: EXP-01 ou EXP-01,EXP-03")
    group.add_argument("--file", default=None, help="Caminho direto para o CSV")
    group.add_argument("--best", action="store_true", help="Envia o experimento com maior cv_f1")
    args = parser.parse_args()

    if args.file:
        # Envia arquivo especifico
        exp_id = os.path.basename(args.file).replace("submission_", "").replace(".csv", "")
        submit(args.file, get_message(exp_id))

    elif args.exp:
        # Envia lista de experimentos
        for exp_id in [e.strip() for e in args.exp.split(",")]:
            path = os.path.join(SUB_DIR, f"submission_{exp_id}.csv")
            if os.path.exists(path):
                submit(path, get_message(exp_id))
            else:
                print(f"  Arquivo nao encontrado: {path}")

    elif args.best:
        # Envia o de maior cv_f1
        if not os.path.exists(RESULTS_TSV):
            print("results.tsv nao encontrado.")
            sys.exit(1)
        df = pd.read_csv(RESULTS_TSV, sep="\t")
        df = df[df["status"] == "keep"].sort_values("cv_f1", ascending=False)
        if df.empty:
            print("Nenhum experimento com status 'keep' encontrado.")
            sys.exit(1)
        best = df.iloc[0]
        exp_id = best["exp_id"]
        path = os.path.join(SUB_DIR, f"submission_{exp_id}.csv")
        if os.path.exists(path):
            submit(path, get_message(exp_id))
        else:
            print(f"Arquivo nao encontrado: {path}")

    else:
        # Envia todos os CSVs em submissions/
        files = sorted(f for f in os.listdir(SUB_DIR) if f.endswith(".csv"))
        if not files:
            print("Nenhum CSV encontrado em submissions/")
            sys.exit(1)
        print(f"Encontrados {len(files)} arquivo(s):\n")
        for f in files:
            print(f"  {f}")

        confirm = input("\nEnviar todos? (s/n): ").strip().lower()
        if confirm != "s":
            print("Cancelado.")
            sys.exit(0)

        for f in files:
            exp_id = f.replace("submission_", "").replace(".csv", "")
            submit(os.path.join(SUB_DIR, f), get_message(exp_id))


if __name__ == "__main__":
    main()
