# -*- coding: utf-8 -*-
import base64, io, re
from math import pi
import numpy as np
import pandas as pd

from dash import Dash, html, dcc, dash_table, Input, Output, State
from dash.dcc import send_bytes

# =========================
# 共通ユーティリティ
# =========================
TRACK_TABLE = {"東京":"tokyo","中山":"nakayama","京都":"kyoto","阪神":"hanshin",
               "中京":"chukyo","新潟":"niigata","札幌":"sapporo",
               "函館":"hakodate","福島":"fukushima","小倉":"kokura"}

DEFAULT_W = dict(
    w_delta_sec=1.0,
    w_h4f_z=0.8, w_h1f_z=0.5, w_htrend_z=0.6, w_hk1_z=0.2, w_hk2_z=0.2,
    w_w4f_z=0.8, w_w1f_z=0.5, w_wtrend_z=0.6, w_wk1_z=0.2, w_wk2_z=0.2,
)

def to_seconds(val):
    if val is None or (isinstance(val, float) and pd.isna(val)): return np.nan
    s = str(val).strip()
    if s == "": return np.nan
    m = re.match(r"^\s*(\d+):(\d+(?:\.\d+)?)\s*$", s)  # m:ss.s
    if m: return int(m.group(1))*60 + float(m.group(2))
    m2 = re.match(r"^\s*(\d+)[\'’](\d+)(?:[\"”](\d))?\s*$", s)  # 1'12"3
    if m2:
        minutes, seconds = int(m2.group(1)), int(m2.group(2))
        tenth = m2.group(3)
        return minutes*60 + seconds + (float(tenth)/10.0 if tenth else 0.0)
    try: return float(s)
    except: return np.nan

def find_col(df, *names_or_regex):
    for n in names_or_regex:
        if isinstance(n, str) and n in df.columns: return n
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

def zscore(series):
    mu = np.nanmean(series)
    sd = np.nanstd(series, ddof=0)
    return (series - mu) / (sd if (sd and sd>0) else np.nan)

def dft_4(last4):
    arr = np.array(last4, dtype=float)
    if np.any(np.isnan(arr)): return (np.nan,)*5
    dc = arr.mean()
    trend = -(arr[-1]-arr[0])/3.0      # ＋なら加速
    var = float(np.var(arr))
    N = 4
    k1 = abs(np.sum(arr * np.exp(-2j*np.pi*1*np.arange(N)/N))) / N
    k2 = abs(np.sum(arr * np.exp(-2j*np.pi*2*np.arange(N)/N))) / N
    return float(dc), float(trend), float(var), float(k1), float(k2)

# =========================
# 調教：坂路/ウッド（別物で処理）
# =========================
def build_hill_features_from_df(df):
    date_col = find_col(df, "年月日","日付","調教日","追切日","date")
    name_col = find_col(df, "馬名","horse_name","馬")
    fac_col  = find_col(df, "施設","トレセン","美浦|栗東","所属","場所","場")

    lap_cols  = [find_col(df, f"Lap{i}", f"ラップ{i}", f"区間{i}", fr"^Lap\s*{i}$") for i in range(1,5)]
    lap_cols  = [c for c in lap_cols if c]
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

    if lap_cols:
        for i,c in enumerate(lap_cols[:4]):
            out[f"split{i+1}_s"] = df[c].map(to_seconds)
    else:
        t = [df.get(find_col(df, f"Time{i}", fr"^Time\s*{i}$")) for i in range(1,5)]
        t = [col.map(to_seconds) if col is not None else pd.Series([np.nan]*len(df)) for col in t]
        out["split1_s"] = t[0]-t[1]; out["split2_s"] = t[1]-t[2]
        out["split3_s"] = t[2]-t[3]; out["split4_s"] = t[3]

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

def build_wood_features_from_df(df):
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

    for i in range(1, 11):
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
# 過去走（レース） & 物理
# =========================
TOKYO_WIDTH_MIN = {"A":31, "B":28, "C":25, "D":22}

def load_races_from_df(df):
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
        "distance_m": df.get(dist_col).apply(lambda x: int(str(x)) if pd.notna(x) and str(x).isdigit() else np.nan) if dist_col else np.nan,
        "rail_code": df.get(rail_col),
        "heads": df.get(heads_col).apply(lambda x: int(str(x)) if pd.notna(x) and str(x).isdigit() else np.nan) if heads_col else np.nan,
        "frame_no": df.get(frame_col).apply(lambda x: int(str(x)) if pd.notna(x) and str(x).isdigit() else np.nan) if frame_col else np.nan,
        "horse_no": df.get(horse_no_col).apply(lambda x: int(str(x)) if pd.notna(x) and str(x).isdigit() else np.nan) if horse_no_col else np.nan,
        "horse_name": df.get(name_col).astype(str).str.strip() if name_col else "",
        "time_total_s": df.get(time_col).map(to_seconds) if time_col else np.nan,
        "last3f_s": df.get(last3f_col).map(to_seconds) if last3f_col else np.nan,
    })
    r["date_str"] = r["date"].dt.strftime("%Y-%m-%d")
    return r

def phi_and_share(track, surface, distance_m):
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

def attach_latest(base, feat, prefix):
    if feat is None or feat.empty:
        for c in ["time4F_s","time1F_s","dc_mean_s","trend_accel","var","dft_k1_amp","dft_k2_amp","z_time4F","z_time1F","z_trend"]:
            base[f"{prefix}_{c}"] = np.nan
        return base
    feat = feat.copy()
    feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
    feat["horse_name"] = feat["horse_name"].astype(str).str.strip()
    merged = base.merge(feat, on="horse_name", how="left", suffixes=("","_wk"))
    merged = merged[(merged["date_wk"].notna()) & (merged["date_wk"] <= merged["date"])]
    merged.sort_values(["race_id","horse_no","date_wk"], inplace=True)
    latest = merged.groupby(["race_id","horse_no"], as_index=False).tail(1)
    keep = ["race_id","horse_no","time4F_s","time1F_s","dc_mean_s","trend_accel","var","dft_k1_amp","dft_k2_amp","z_time4F","z_time1F","z_trend"]
    for c in keep[2:]:
        latest.rename(columns={c: f"{prefix}_{c}"}, inplace=True)
    out = base.merge(latest[["race_id","horse_no"]+[f"{prefix}_{c}" for c in keep[2:]]], on=["race_id","horse_no"], how="left")
    return out

def score_with_weights(r, W):
    phys = r.apply(delta_s_row, axis=1)
    r = pd.concat([r, phys], axis=1)
    v_ref = np.where(r["surface"].eq("dirt"), 16.0, 17.0)
    r["delta_s_sec"] = r["delta_s_m"] / v_ref

    for base in ["hill","wood"]:
        for col in ["dft_k1_amp","dft_k2_amp"]:
            c, zc = f"{base}_{col}", f"{base}_{col}_z"
            mu, sd = r[c].mean(skipna=True), r[c].std(skipna=True)
            r[zc] = (r[c]-mu)/(sd if sd and sd>0 else np.nan)

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

def parse_upload(contents):
    if contents is None:
        return None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_excel(io.BytesIO(decoded))

# =========================
# Dash アプリ
# =========================
app = Dash(__name__)
server = app.server  # for gunicorn

app.layout = html.Div([
    html.H2("Keiba Physics OCP – Dash"),
    html.P("Excel(.xlsx) をアップして『物理×調教（坂路/ウッド別）』でスコア化します。"),

    html.Div([
        html.Div([
            html.Label("過去走Excel（必須）"),
            dcc.Upload(id="up-races", children=html.Div(["ここにドラッグ or クリックして選択"]),
                       multiple=False, style={"border":"1px dashed #888","padding":"10px","cursor":"pointer"})
        ], style={"flex":"1","margin":"4px"}),
        html.Div([
            html.Label("坂路Excel（任意）"),
            dcc.Upload(id="up-hill", children=html.Div(["任意：坂路"]),
                       multiple=False, style={"border":"1px dashed #888","padding":"10px","cursor":"pointer"})
        ], style={"flex":"1","margin":"4px"}),
        html.Div([
            html.Label("ウッドExcel（任意）"),
            dcc.Upload(id="up-wood", children=html.Div(["任意：ウッド"]),
                       multiple=False, style={"border":"1px dashed #888","padding":"10px","cursor":"pointer"})
        ], style={"flex":"1","margin":"4px"}),
    ], style={"display":"flex"}),

    html.Details([
        html.Summary("重み（上級者向け）"),
        html.Div([
            html.Div(["外々ロス（秒）", dcc.Slider(0,2,0.1, value=DEFAULT_W["w_delta_sec"], id="w-delta")], style={"margin":"8px 0"}),
            html.H4("坂路"),
            dcc.Slider(0,2,0.1, value=DEFAULT_W["w_h4f_z"], id="w-h4"),  html.Div("z_time4F"),
            dcc.Slider(0,2,0.1, value=DEFAULT_W["w_h1f_z"], id="w-h1"),  html.Div("z_time1F"),
            dcc.Slider(0,2,0.1, value=DEFAULT_W["w_htrend_z"], id="w-ht"),html.Div("z_trend"),
            dcc.Slider(0,1,0.1, value=DEFAULT_W["w_hk1_z"], id="w-hk1"), html.Div("DFT k1(z)"),
            dcc.Slider(0,1,0.1, value=DEFAULT_W["w_hk2_z"], id="w-hk2"), html.Div("DFT k2(z)"),
            html.H4("ウッド"),
            dcc.Slider(0,2,0.1, value=DEFAULT_W["w_w4f_z"], id="w-w4"),  html.Div("z_time4F"),
            dcc.Slider(0,2,0.1, value=DEFAULT_W["w_w1f_z"], id="w-w1"),  html.Div("z_time1F"),
            dcc.Slider(0,2,0.1, value=DEFAULT_W["w_wtrend_z"], id="w-wt"),html.Div("z_trend"),
            dcc.Slider(0,1,0.1, value=DEFAULT_W["w_wk1_z"], id="w-wk1"), html.Div("DFT k1(z)"),
            dcc.Slider(0,1,0.1, value=DEFAULT_W["w_wk2_z"], id="w-wk2"), html.Div("DFT k2(z)"),
        ])
    ]),

    html.Button("スコア計算 ▶", id="btn-run", n_clicks=0, style={"margin":"8px 0"}),
    html.Div(id="msg"),
    dash_table.DataTable(id="tbl",
        page_size=20, sort_action="native", filter_action="native",
        style_table={"overflowX":"auto"}, style_cell={"fontFamily":"monospace","fontSize":"12px"}),
    html.Br(),
    html.Button("結果Excelをダウンロード", id="btn-dl", n_clicks=0),
    dcc.Store(id="store-scores"),  # JSON
    dcc.Store(id="store-details"),
    dcc.Download(id="download-xlsx"),
])

# ---- 計算コールバック ----
@app.callback(
    Output("msg","children"),
    Output("tbl","columns"),
    Output("tbl","data"),
    Output("store-scores","data"),
    Output("store-details","data"),
    Input("btn-run","n_clicks"),
    State("up-races","contents"),
    State("up-hill","contents"),
    State("up-wood","contents"),
    State("w-delta","value"), State("w-h4","value"), State("w-h1","value"),
    State("w-ht","value"), State("w-hk1","value"), State("w-hk2","value"),
    State("w-w4","value"), State("w-w1","value"), State("w-wt","value"),
    State("w-wk1","value"), State("w-wk2","value"),
    prevent_initial_call=True
)
def run(n_clicks, races_cts, hill_cts, wood_cts,
        w_delta, w_h4, w_h1, w_ht, w_hk1, w_hk2, w_w4, w_w1, w_wt, w_wk1, w_wk2):
    if not races_cts:
        return ("過去走Excelをアップしてください。", [], [], None, None)

    races_df = parse_upload(races_cts)
    base = load_races_from_df(races_df)

    hill_feat = None
    if hill_cts:
        hill_df = parse_upload(hill_cts)
        hill_feat = build_hill_features_from_df(hill_df)
        base = attach_latest(base, hill_feat, "hill")
    else:
        base = attach_latest(base, pd.DataFrame(), "hill")

    wood_feat = None
    if wood_cts:
        wood_df = parse_upload(wood_cts)
        wood_feat = build_wood_features_from_df(wood_df)
        base = attach_latest(base, wood_feat, "wood")
    else:
        base = attach_latest(base, pd.DataFrame(), "wood")

    W = dict(
        w_delta_sec=w_delta,
        w_h4f_z=w_h4, w_h1f_z=w_h1, w_htrend_z=w_ht, w_hk1_z=w_hk1, w_hk2_z=w_hk2,
        w_w4f_z=w_w4, w_w1f_z=w_w1, w_wtrend_z=w_wt, w_wk1_z=w_wk1, w_wk2_z=w_wk2,
    )
    details = score_with_weights(base, W)

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
    cols = [{"name": c, "id": c} for c in scores.columns]
    data = scores.round(4).to_dict("records")

    return ("計算完了！", cols, data,
            scores.to_json(date_format="iso", orient="split"),
            details.to_json(date_format="iso", orient="split"))

# ---- ダウンロード ----
@app.callback(
    Output("download-xlsx","data"),
    Input("btn-dl","n_clicks"),
    State("store-scores","data"),
    State("store-details","data"),
    prevent_initial_call=True
)
def download(n_clicks, j_scores, j_details):
    if not j_scores or not j_details:
        return None
    scores = pd.read_json(j_scores, orient="split")
    details = pd.read_json(j_details, orient="split")

    def writer(buf):
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            scores.to_excel(w, sheet_name="scores", index=False)
            details.to_excel(w, sheet_name="details", index=False)
            pd.DataFrame(DEFAULT_W.items(), columns=["weight_key","value"]).to_excel(w, sheet_name="weights", index=False)

    return send_bytes(writer, "race_scores.xlsx")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
