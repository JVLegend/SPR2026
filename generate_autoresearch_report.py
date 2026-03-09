"""
generate_autoresearch_report.py

Roda APÓS o loop autoresearch (Fases 1 e 2).
Lê o results.tsv, analisa padrões e gera um relatório markdown.

Uso:
    python generate_autoresearch_report.py
    python generate_autoresearch_report.py --results results.tsv --output experiments/autoresearch_report.md

O report gerado vai para o GitHub. Checkpoints ficam locais.
"""

import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path


def generate_report(results_tsv: str = "results.tsv",
                    output_md: str = "experiments/autoresearch_report.md") -> str:

    df = pd.read_csv(results_tsv, sep="\t")

    keep     = df[df["status"] == "keep"].copy()
    discard  = df[df["status"] == "discard"].copy()
    crashed  = df[df["status"] == "crash"].copy()

    # Sort by cv_f1 desc
    if "cv_f1" in keep.columns:
        keep_sorted = keep.nlargest(5, "cv_f1")
    else:
        keep_sorted = keep.head(5)

    best = keep_sorted.iloc[0] if len(keep_sorted) > 0 else None

    # ── Markdown ──────────────────────────────────────────────────────────────
    lines = [
        f"# Autoresearch Report — SPR 2026 Mammography",
        f"",
        f"> Gerado automaticamente por `generate_autoresearch_report.py`  ",
        f"> Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"---",
        f"",
        f"## Resumo Geral",
        f"",
        f"| Métrica | Valor |",
        f"|---|---|",
        f"| Total de experimentos | {len(df)} |",
        f"| Mantidos (keep) | {len(keep)} |",
        f"| Descartados (discard) | {len(discard)} |",
        f"| Crashes | {len(crashed)} |",
        f"| Taxa de sucesso | {len(keep)/max(len(df),1)*100:.1f}% |",
    ]

    if best is not None:
        lines += [
            f"| **Melhor cv_f1** | **{best.get('cv_f1', 'N/A')}** |",
        ]

    lines += ["", "---", "", "## Top 5 Experimentos", ""]

    if len(keep_sorted) > 0:
        cols = [c for c in ["exp_id", "model", "cv_f1", "vram_gb", "time_min", "notes"] if c in keep_sorted.columns]
        lines.append(keep_sorted[cols].to_markdown(index=False))
    else:
        lines.append("_Nenhum experimento com status 'keep' encontrado._")

    lines += ["", "---", "", "## Melhor Configuração Encontrada", ""]

    if best is not None:
        for col in ["cv_f1", "model", "notes", "commit"]:
            if col in best.index:
                lines.append(f"- **{col}**: {best[col]}")
    else:
        lines.append("_Nenhum resultado positivo ainda._")

    lines += [
        "",
        "---",
        "",
        "## O Que Funcionou (padrões nos 'keep')",
        "",
    ]
    for _, row in keep_sorted.iterrows():
        note = row.get("notes", "")
        f1   = row.get("cv_f1", "")
        lines.append(f"- `cv_f1={f1}` — {note}")

    lines += [
        "",
        "## O Que Não Funcionou (padrões nos 'discard')",
        "",
    ]
    for _, row in discard.head(10).iterrows():
        note = row.get("notes", "")
        f1   = row.get("cv_f1", "")
        lines.append(f"- `cv_f1={f1}` — {note}")

    lines += [
        "",
        "---",
        "",
        "## Próximos Passos Recomendados",
        "",
        "1. Submeter melhor checkpoint para Kaggle e checar score público",
        "2. Rodar Fase 2 do autoresearch com foco nos hiperparâmetros do top-1",
        "3. Ensemble dos top-3 modelos — otimizar pesos com Nelder-Mead no val set",
        "4. Analisar F1 por classe — identificar onde ainda estamos perdendo",
        "5. Investigar se classes 5/6 melhoraram com as técnicas aplicadas",
        "",
        "---",
        "",
        "## Checkpoints (ficam na máquina local — NÃO subir no GitHub)",
        "",
        "```",
        "autoresearch/checkpoints/    ← .pt / .safetensors ficam aqui",
        "```",
        "",
        "Para compartilhar o melhor modelo entre o time:",
        "- Usar Google Drive ou `scp` para transferir `best_model.pt`",
        "- Incluir o `config.json` e o script de inferência correspondente",
        "",
        "---",
        "",
        "_Report gerado por `generate_autoresearch_report.py` — Time: RUMO AO TOPO_",
    ]

    output_path = Path(output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Report salvo em: {output_path}")
    print(f"Para commitar: git add {output_md} experiments/results.tsv && git commit -m 'autoresearch report'")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.tsv", help="Path para o results.tsv do autoresearch")
    parser.add_argument("--output", default="experiments/autoresearch_report.md", help="Path do report de saída")
    args = parser.parse_args()

    generate_report(results_tsv=args.results, output_md=args.output)
