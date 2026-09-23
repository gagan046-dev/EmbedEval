import pandas as pd
import json
import io

def results_to_csv(raw_results: list[dict]) -> bytes:
    rows = []
    for r in raw_results:
        row = {k: v for k, v in r.items() if k != "metrics"}
        row.update(r.get("metrics", {}))
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")

def matrix_to_csv(benchmark_matrix_df: pd.DataFrame) -> bytes:
    return benchmark_matrix_df.to_csv().encode("utf-8")

def results_to_json(raw_results: list[dict], benchmark_matrix: dict) -> bytes:
    payload = {
        "benchmark_matrix": benchmark_matrix,
        "raw_results": raw_results,
    }
    return json.dumps(payload, indent=2).encode("utf-8")
