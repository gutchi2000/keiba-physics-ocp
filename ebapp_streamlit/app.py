# -*- coding: utf-8 -*-
import io
from math import pi
import numpy as np
import pandas as pd
import re
import streamlit as st

st.set_page_config(page_title="Keiba Physics OCP (Streamlit)", layout="wide")

# =========================
# 共通ユーティリティ
# =========================

TRACK_TABLE = {"東京":"tokyo","中山":"nakayama","京都":"kyoto","阪神":"hanshin",
               "中京":"chukyo","新潟":"niigata","札幌":"sapporo",
               "函館":"hakodate","福島":"fukushima","小倉":"kokura"}

def to_seconds(val):
    """'1:12.3' / '1\'12\"3' / '72.3' を秒に。空は NaN。"""
    if val is None or (isinstance(val, float) and pd.isna(val)): return np.nan
    s = str(val).strip()
    if s == "": return np.nan
    m = re.match(r"^\s*(\d+):(\d+(?:\.\d+)?)\s*$", s)  # m:ss.s
    if m: return int(m.group(1))*60 + float(m.group(2))
    m2 = re.match(r"^\s*(\d+)[\'’](\d+)(?:[\"”](\d))?\s*$", s)  # 1'12"3
    if m2:
        minutes = int(m2.group(1)); seconds = int(m2.group(2))
        tenth = m2.group(3)
        return minutes*60 + seconds + (float(tenth)/10.0 if tenth else 0.0)
    try: return float(s)
    except: return np.nan

def find_col(df: pd.DataFrame, *names_or_regex):
    """列名を柔軟に発見（完全一致→正規表現）。"""
    # 完全一致
    for n in names_or_regex:
        if isinstance(n, str) and n in df.columns: return n
    # 正規表現
    for pat in names_or_regex:
        p = re.compile(str(pat), flags=re.IGNORECASE)
        for c in df.columns:
            if p.search(str(c)): return c
    return None

def norm_track_from_text(x):
    if pd.isna(x): return None
    s = str(x)
    for jp, slug in TRACK_TABLE.items():
        if jp in s: return slug
    return None

def norm_surface(x):
    s = "" if pd.isna(x) else str(x)
    if "芝" in s: return "turf"
    if "ダ" in s: return "dirt"
    if s.lower() in ("turf","dirt"): return s.lower()
    return None

def zscore(series: pd.Series):
    mu = np.nanmean(series)
    sd = np.nanstd(series, ddof=0)
    return (series - mu) / (sd if (sd and sd>0) else np.nan)

def dft_4(last4):
    """4点系列（古→新）にDFT。返り値: mean, trend(+なら加速), var, k1_amp, k2_amp"""
    arr = np.array(last4, dtype=float)
    if np.any(np.isnan(arr)): return (np.nan,)*5
    dc = arr.mean()
    trend = -(arr[-1]-arr[0])/3.0       # ＋なら加速（良い）
    var = float(np.var(arr))
    N = 4
    k1 = abs(np.sum(arr * np.exp(-2j*np.pi*1*np.arange(N)/N))) / N
    k2 = abs(np.sum(arr * np.exp(-2j*np.pi*2*np.arange(N)/N))) / N
    return float(dc), float(trend), float(var), float(k1), float(k2)

# =========================
# 調教 → 特徴量（別物で処理）
# =========================

def build_hill_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    date_col = find_col(df, "年月日","日付","調教日","追切日","date")
    name_col = find_col(df, "馬名","horse_name","馬")
    fac_col  = find_col(df, "施設","トレセン","美浦|栗東","所属","場所","場")

    lap_cols = [find_col(df, f"Lap{i}", f"ラップ{i}", f"区間{i}", fr"^Lap\s*{i}$") for i in range(1,5)]
    lap_cols = [c for c in lap_cols if c]
    time_cols = [find_col(df, f"Time{i}", fr"^Time\s*{i}$") for i in range(1,5)]
    time_cols = [c for c in time_cols if c]

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    out["horse_name"] = df[name_col].astype(str).str.strip() if name_col else ""
    if fac_col:
        fac = df[fac_col].astype(str)
        out["facility"] = np.where(fac.str.contains("栗東"), "栗東", np.where(fac.str.contains("美浦"), "美浦", ""))
    else:
        out["facility"] = ""

    # split1..4（古→新）
    if lap_cols:
        for i,c in enumerate(lap_cols[:4]):
            out[f"split{i+1}_s"] = df[c].map(to_seconds)
    else:
        # Time1..4 = 4F,3F,2F,1F の累積とみなす
        t = [df.get(find_col(df, f"Time{i}", fr"^Time\s*{i}$")) for i in range(1,5)]
        t = [col.map(to_seconds) if col is not None else pd.Series([np.nan]*len(df)) for col in t]
        out["split1_s"] = t[0]-t[1]
        out["split2_s"] = t[1]-t[2]
        out["split3_s"] = t[2]-t[3]
        out["split4_s"] = t[3]

    out["time1F_s"] = out["split4_s"]
    out["time4F_s"] = out[["split1_s","split2_s","split3_s","split4_s"]].sum(axis=1, skipna=False)

    feat = out.apply(lambda r: pd.Series(
        dft_4([r["split1_s"],r["split2_s"],r["split3_s"],r["split4_s"]]),
        index=["dc_mean_s","trend_accel","var","dft_k1_amp","dft_k2_amp"]), axis=1)
    out = pd.concat([out, feat], axis=1)

    out["z_time4F"] = out.groupby("facility")["time4F_s"].transform(zscore)
    out["z_time1F"] = out.groupby("facility")["time1F_s"].transform(zscore)
    out["z_trend"]  = out.groupby("facility")["trend_accel"].transform(zscore)
    out["date_str"] = out["date"].dt.strftime("%Y-%m-%d")
    return out

def build_wood_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    date_col = find_col(df, "年月日","日付","調教日","追切日","date")
    name_col = find_col(df, "馬名","horse_name","馬")
    fac_col  = find_col(df, "施設","トレセン","美浦|栗東","所属","場所","場")
    dir_col  = find_col(df, "回り","方向","向き")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    out["horse_name"] = df[name_col].astype(str).str.strip() if name_col else ""
    if fac_col:
        fac = df[fac_col].astype(str)
        out["facility"] = np.where(fac.str.contains("栗東"), "栗東", np.where(fac.str.contains("美浦"), "美浦", ""))
    else:
        out["facility"] = ""
    if dir_col:
        d = df[dir_col].astype(str)
        out["direction"] = np.where(d.str.contains("右"), "right", np.where(d.str.contains("左"), "left", ""))
    else:
        out["direction"] = ""

    # 1F..10F → split1F..split10F
    for i in range(1, 10+1):
        c = find_col(df, f"{i}F", f"{i}Ｆ", rf"^{i}\s*F$", rf"^{i}\s*Ｆ$")
        out[f"split{i}F_s"] = df[c].map(to_seconds) if c else np.nan

    out["time1F_s"] = out["split1F_s"]
    out["time4F_s"] = out[[f"split{i}F_s" for i in [4,3,2,1]]].sum(axis=1, skipna=False)

    feat = out.apply(lambda r: pd.Series(
        dft_4([r.get("split4F_s"),r.get("split3F_s"),r.get("split2F_s"),r.get("split1F_s")]),
        index=["dc_mean_s","trend_accel","var","dft_k1_amp","dft_k2_amp"]), axis=1)
    out = pd.concat([out, feat], axis=1)

    out["z_time4F"] = out.groupby("facility")["time4F_s"].transform(zscore)
    out["z_time1F"] = out.groupby("facility")["time1F_s"].transform(zscore)
    out["z_trend"]  = out.groupby("facility")["trend_accel"].transform(zscore)
    out["date_str"] = out["date"].dt.strftime("%Y-%m-%d")
    return out

# =========================
# 過去走（レース）読み込み
# =========================

def load_races_from_df(df: pd.DataFrame) -> pd.DataFrame:
    def pick(*names):
        for n in names:
            if n in df.columns: return n
        return None
    date_col = pick("日付","date")
    track_col= pick("競馬場","場名","コース名","track")
    name_col = pick("馬名","horse_name","馬")
    surface_col = pick("芝・ダ","surface")
    dist_col = pick("距離","distance_m")
    rail_col = pick("コース区分","rail_code")
    heads_col= pick("頭数","heads")
    frame_col= pick("枠番","枠","frame_no")
    horse_no_col= pick("馬番","horse_no")
    time_col = pick("走破タイム","time_total_s")
    last3f_col = pick("上がり３Fタイム","上がり3F","Ave-3F","last3f_s")

    r = pd.DataFrame({
        "race_id": df.get(pick("レースID","race_id")),
        "date": pd.to_datetime(df.get(date_col), errors="coerce"),
        "track": df.get(track_col).map(lambda x: TRACK_TABLE.get(str(x), norm_track_from_text(x))) if track_col else None,
        "surface": df.get(surface_col).map(norm_surface) if surface_col else None,
        "distance_m": df.get(dist_col).map(lambda x: int(str(x)) if pd.notna(x) and str(x).isdigit() else np.nan) if dist_col else np.nan,
        "rail_code": df.get(rail_col),
        "heads": df.get(heads_col).map(lambda x: int(str(x)) if pd.notna(x) and str(x).isdigit() else np.nan) if heads_col else np.nan,
        "frame_no": df.get(frame_col).map(lambda x: int(str(x)) if pd.notna(x) and str(x).isdigit() else np.nan) if frame_col else np.nan,
        "horse_no": df.get(horse_no_col).map(lambda x: int(str(x)) if pd.notna(x) and str(x).isdigit() else np.nan) if horse_no_col else np.nan,
        "horse_name": df.get(name_col).astype(str).str.strip() if name_col else "",
        "time_total_s": df.get(time_col).map(to_seconds) if time_col else np.nan,
        "last3f_s": df.get(last3f_col).map(to_seconds) if last3f_col else np.nan,
    })
    r["date_str"] = r["date"].dt.strftime("%Y-%m-%d")
    return r

# =========================
# 物理（外々ロス Δs）
# =========================

TOKYO_WIDTH_MIN = {"A":31, "B":28, "C":25, "D":22}

def phi_and_share(track, surface, distance_m):
    # 最低限ルール（拡張はYAML読み込みで）
    if track=="tokyo" and surface=="turf" and distance_m in (1400,1600,1800):
        return pi, 0.30
    if track=="nakayama" and surface=="turf" and distance_m==2000:
        return 2*pi, 0.45
    return 2*pi, 0.45

def delta_s_row(row, alpha=0.15):
    track, surface, rail_code, heads, frame_no, D = row["track"], row["surface"], str(row["rail_code"]), row["heads"], row["frame_no"], row["distance_m"]
    if pd.isna(heads) or pd.isna(frame_no) or pd.isna(D): 
        return pd.Series({"delta_s_m": np.nan, "d_turn_m": np.nan, "w_eff_m": np.nan, "R_eff_m": np.nan})
    phi, turn_share = phi_and_share(track, surface, D)
    if track=="tokyo" and surface=="turf" and rail_code in TOKYO_WIDTH_MIN:
        w_min = TOKYO_WIDTH_MIN[rail_code]
    else:
        w_min = 20.0
    w_eff = max(w_min - 4.0, 8.0)
    q = (frame_no - 0.5) / heads
    d_turn = alpha * w_eff * q
    delta_s = phi * d_turn
    s_turn = turn_share * D
    R_eff = s_turn / phi
    return pd.Series({"delta_s_m": delta_s, "d_turn_m": d_turn, "w_eff_m": w_eff, "R_eff_m": R_eff})

# =========================
# スコア計算
# =========================

DEFAULT_WEIGHTS = dict(
    w_delta_sec=1.0,
    w_h4f_z=0.8, w_h1f_z=0.5, w_htrend_z=0.6, w_hk1_z=0.2, w_hk2_z=0.2,
    w_w4f_z=0.8, w_w1f_z=0.5, w_wtrend_z=0.6, w_wk1_z=0.2, w_wk2_z=0.2,
)

def attach_latest(base: pd.DataFrame, feat: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if feat is None or feat.empty:
        for c in ["time4F_s","time1F_s","dc_mean_s","trend_accel","var","dft_k1_amp","dft_k2_amp","z_time4F","z_time1F","z_trend"]:
            base[f"{prefix}_{c}"] = np.nan
        return base
    feat = feat.copy()
    feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
    feat["horse_name"] = feat["horse_name"].astype(str).stripped if "stripped" in dir(str) else feat["horse_name"].astype(str).str.strip()

    merged = base.merge(feat, on="horse_name", how="left", suffixes=("","_wk"))
    merged = merged[(merged["date_wk"].notna()) & (merged["date_wk"] <= merged["date"])]
    merged.sort_values(["race_id","horse_no","date_wk"], inplace=True)
    latest = merged.groupby(["race_id","horse_no"], as_index=False).tail(1)

    keep = ["race_id","horse_no","time4F_s","time1F_s","dc_mean_s","trend_accel","var","dft_k1_amp","dft_k2_amp","z_time4F","z_time1F","z_trend"]
    for c in keep[2:]:
        latest.rename(columns={c: f"{prefix}_{c}"}, inplace=True)
    out = base.merge(latest[["race_id","horse_no"]+[f"{prefix}_{c}" for c in keep[2:]]], on=["race_id","horse_no"], how="left")
    return out

def score_df(r: pd.DataFrame, W: dict) -> pd.DataFrame:
    # 物理
    phys = r.apply(delta_s_row, axis=1)
    r = pd.concat([r, phys], axis=1)
    v_ref = np.where(r["surface"].eq("dirt"), 16.0, 17.0)   # m/s
    r["delta_s_sec"] = r["delta_s_m"] / v_ref

    # DFT振幅の全体z
    for base in ["hill","wood"]:
        for col in ["dft_k1_amp","dft_k2_amp"]:
            c, zc = f"{base}_{col}", f"{base}_{col}_z"
            mu, sd = r[c].mean(skipna=True), r[c].std(skipna=True)
            r[zc] = (r[c]-mu)/(sd if sd and sd>0 else np.nan)

    # 欠損は z 系のみ 0（中立）
    for col in [
        "hill_z_time4F","hill_z_time1F","hill_z_trend","hill_dft_k1_amp_z","hill_dft_k2_amp_z",
        "wood_z_time4F","wood_z_time1F","wood_z_trend","wood_dft_k1_amp_z","wood_dft_k2_amp_z"
    ]:
        if col in r.columns: r[col] = r[col].fillna(0.0)

    r["score"] = (
        -W["w_delta_sec"]*r["delta_s_sec"].fillna(0.0)
        +W["w_h4f_z"]*(-r.get("hill_z_time4F",0.0))
        +W["w_h1f_z"]*(-r.get("hill_z_time1F",0.0))
        +W["w_htrend_z"]*( r.get("hill_z_trend",0.0))
        +W["w_hk1_z"]*(-r.get("hill_dft_k1_amp_z",0.0))
        +W["w_hk2_z"]*(-r.get("hill_dft_k2_amp_z",0.0))
        +W["w_w4f_z"]*(-r.get("wood_z_time4F",0.0))
        +W["w_w1f_z"]*(-r.get("wood_z_time1F",0.0))
        +W["w_wtrend_z"]*( r.get("wood_z_trend",0.0))
        +W["w_wk1_z"]*(-r.get("wood_dft_k1_amp_z",0.0))
        +W["w_wk2_z"]*(-r.get("wood_dft_k2_amp_z",0.0))
    )
    r["rank"] = r.groupby("race_id")["score"].rank(ascending=False, method="min")
    return r

# =========================
# UI
# =========================

st.title("Keiba Physics OCP – Streamlit")
st.caption("Excel(.xlsx) をアップロードして、物理×調教（坂路/ウッド別）でスコアを出します。")

col_u1, col_u2, col_u3 = st.columns(3)
with col_u1:
    races_file = st.file_uploader("過去走Excel（必須）", type=["xlsx"], key="races")
with col_u2:
    hill_file  = st.file_uploader("坂路Excel（任意）", type=["xlsx"], key="hill")
with col_u3:
    wood_file  = st.file_uploader("ウッドExcel（任意）", type=["xlsx"], key="wood")

st.sidebar.header("重み（上級者向け）")
W = DEFAULT_WEIGHTS.copy()
W["w_delta_sec"] = st.sidebar.slider("外々ロス（秒）", 0.0, 2.0, W["w_delta_sec"], 0.1)
st.sidebar.subheader("坂路")
W["w_h4f_z"] = st.sidebar.slider("坂路 z_time4F", 0.0, 2.0, W["w_h4f_z"], 0.1)
W["w_h1f_z"] = st.sidebar.slider("坂路 z_time1F", 0.0, 2.0, W["w_h1f_z"], 0.1)
W["w_htrend_z"] = st.sidebar.slider("坂路 z_trend", 0.0, 2.0, W["w_htrend_z"], 0.1)
W["w_hk1_z"] = st.sidebar.slider("坂路 DFT k1(z)", 0.0, 1.0, W["w_hk1_z"], 0.1)
W["w_hk2_z"] = st.sidebar.slider("坂路 DFT k2(z)", 0.0, 1.0, W["w_hk2_z"], 0.1)
st.sidebar.subheader("ウッド")
W["w_w4f_z"] = st.sidebar.slider("ウッド z_time4F", 0.0, 2.0, W["w_w4f_z"], 0.1)
W["w_w1f_z"] = st.sidebar.slider("ウッド z_time1F", 0.0, 2.0, W["w_w1f_z"], 0.1)
W["w_wtrend_z"] = st.sidebar.slider("ウッド z_trend", 0.0, 2.0, W["w_wtrend_z"], 0.1)
W["w_wk1_z"] = st.sidebar.slider("ウッド DFT k1(z)", 0.0, 1.0, W["w_wk1_z"], 0.1)
W["w_wk2_z"] = st.sidebar.slider("ウッド DFT k2(z)", 0.0, 1.0, W["w_wk2_z"], 0.1)

if st.button("スコア計算 ▶"):
    if not races_file:
        st.error("過去走Excelをアップしてください。")
        st.stop()

    # --- 過去走 ---
    races_df = pd.read_excel(races_file)
    races = load_races_from_df(races_df)

    # --- 調教（任意） ---
    hill_feat = None
    wood_feat = None
    if hill_file:
        hill_raw = pd.read_excel(hill_file)
        hill_feat = build_hill_features_from_df(hill_raw)
    if wood_file:
        wood_raw = pd.read_excel(wood_file)
        wood_feat = build_wood_features_from_df(wood_raw)

    # レース×調教（最新1本ずつ付与）
    base = races.copy()
    base = attach_latest(base, hill_feat if hill_feat is not None else pd.DataFrame(), "hill")
    base = attach_latest(base, wood_feat if wood_feat is not None else pd.DataFrame(), "wood")

    # スコア
    details = score_df(base, W)

    # 表示（scores）
    show_cols = [
        "race_id","date_str","track","surface","distance_m","rail_code",
        "frame_no","horse_no","horse_name","score","rank",
        "delta_s_m","delta_s_sec",
        "hill_time4F_s","hill_time1F_s","hill_z_time4F","hill_z_time1F","hill_z_trend",
        "wood_time4F_s","wood_time1F_s","wood_z_time4F","wood_z_time1F","wood_z_trend",
    ]
    for c in show_cols:
        if c not in details.columns: details[c] = np.nan
    scores = details[show_cols].sort_values(["race_id","rank","horse_no"])

    st.success("計算完了！下に結果とダウンロードを用意しました。")
    st.dataframe(scores, use_container_width=True)

    # ダウンロード（race_scores.xlsx）
    with io.BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine="openpyxl") as w:
            scores.to_excel(w, sheet_name="scores", index=False)
            details.to_excel(w, sheet_name="details", index=False)
            pd.DataFrame(W.items(), columns=["weight_key","value"]).to_excel(w, sheet_name="weights", index=False)
            if hill_feat is not None:
                hill_feat.to_excel(w, sheet_name="hill_features", index=False)
            if wood_feat is not None:
                wood_feat.to_excel(w, sheet_name="wood_features", index=False)
        st.download_button(
            label="結果Excelをダウンロード（race_scores.xlsx）",
            data=buffer.getvalue(),
            file_name="race_scores.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
