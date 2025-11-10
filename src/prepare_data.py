# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, os, re, sys
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _to_float(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("$","").replace(",","")
    try:
        return float(s)
    except:
        try:
            return float(re.sub(r"[^0-9\.\-]+","", s))
        except:
            return np.nan

def read_apple_db_quarterly():
    p = DATA_DIR / "Apple_DB.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    rename = {"Date":"date","Net_income":"profit","NetSales_Total":"revenue","OperatingExpase_SGA":"marketing_cost"}
    df = df.rename(columns=rename)
    for c in ["revenue","marketing_cost","profit"]:
        if c in df.columns:
            df[c] = df[c].map(_to_float)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    keep = [c for c in ["date","revenue","marketing_cost","profit"] if c in df.columns]
    return df[keep].dropna(subset=["profit"])

def read_accumulated_statement():
    p = DATA_DIR / "Data for Training - Apple DB - Accumulated Statement.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "Date" not in df.columns:
        return None
    def _row_vals(rowname):
        if rowname not in df["Date"].values: return None
        row = df[df["Date"]==rowname].drop(columns=[c for c in ["Year","Date","Quarter"] if c in df.columns]).iloc[0]
        return row.astype(str).apply(_to_float)
    # lấy dãy cột mốc (ví dụ 9.2023 ...)
    hdr = df[df["Date"]=="Date"].drop(columns=[c for c in ["Year","Date","Quarter"] if c in df.columns]).iloc[0]
    # parse quarter-end
    qts = pd.to_datetime(hdr.astype(str).str.replace(" ","").str.replace(".","-").str.replace("/","-"), errors="coerce")
    qts = pd.PeriodIndex(qts.dt.to_period("Q"), freq="Q-DEC").to_timestamp("Q")
    revenue = _row_vals("NetSales_Total")
    sga     = _row_vals("OperatingExpase_SGA")
    profit  = _row_vals("Net_income")
    if revenue is None or profit is None: return None
    out = pd.DataFrame({"date": qts.values,
                        "revenue": revenue.values,
                        "marketing_cost": (sga.values if sga is not None else np.nan),
                        "profit": profit.values})
    out = out.dropna(subset=["date"]).sort_values("date")
    return out

def main():
    a = read_apple_db_quarterly()
    b = read_accumulated_statement()
    if a is None and b is None:
        sys.exit("❌ Không tìm thấy data/Apple_DB.csv hoặc Data for Training - Apple DB - Accumulated Statement.csv")
    if a is None:
        model = b.copy()
    elif b is None:
        model = a.copy()
    else:
        model = pd.merge(b, a, on="date", how="outer", suffixes=("_b",""))
        for col in ["revenue","marketing_cost","profit"]:
            if col in model.columns and f"{col}_b" in model.columns:
                model[col] = model[col].combine_first(model[f"{col}_b"])
                model.drop(columns=[f"{col}_b"], inplace=True)
        model = model.sort_values("date")
    q = model[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    q["cpi"] = 100 + np.arange(len(q))*0.5
    model = model.merge(q, on="date", how="left")
    model.to_csv(OUT_DIR / "processed_quarterly.csv", index=False)
    model[["date","profit"]].to_csv(OUT_DIR / "quarterly_profit.csv", index=False)
    print("✅ out/processed_quarterly.csv, out/quarterly_profit.csv")

if __name__ == "__main__":
    main()
