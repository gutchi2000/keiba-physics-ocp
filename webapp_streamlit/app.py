# -*- coding: utf-8 -*-
# 物理版スペクトル競馬論 β1
# 単体で実行できる Streamlit アプリ（完成形）
# 使い方：
#   1) `pip install streamlit pandas numpy scipy openpyxl`
#   2) `streamlit run keiba_physics_app.py`
#   3) サンプル/テンプレxlsxをダウンロード→必要列を埋めてアップロード
#
# 物理モデルの要点：
#   - ラップ列のスペクトル解析（rFFT）→ 低/中/高周波帯のバンドパワー
#   - 「場のポテンシャル」V：距離・馬場・枠・脚質×展開（ペース場）・斤量 等の不利を集約
#   - 「運動エネルギー」K：近走指数、終い(上がり3F)、調教指数 等の能力側
#   - L = K - V をベースに総合作用 S を構成し、Boltzmann重み p ∝ exp(-S/T) で勝率化
#   - T（温度）= レースのカオス度（頭数・能力分散）で自動算出
#   - Gumbel-top‑k でのPlackett–Luceサンプリングにより 複勝・連対 近似確率をモンテカルロで厳密に近い形で推定
#   - 買い目：◎○▲△△×を自動付与し、単勝/複勝/ワイド/三連複(一頭軸)を予算配分
#
# 必須列（テンプレに含む）：
#   馬名, 枠, 番, 脚質(逃げ/先行/差し/追込), 斤量, 年齢, 距離(m), 想定馬場(良/稍/重/不),
#   近走指数(スピード), 上がり3F(s), 調教指数, 枠順バイアス(±), 馬場適性_良, 馬場適性_稍, 馬場適性_重, 馬場適性_不,
#   距離適性中央値(m), 距離幅(m), LapTimes（JSON文字列; 200m or 400m毎のラップ秒 例: "[12.5,12.1, ...]"）
#   ※LapTimesが無い場合でも動作します（スペクトル項は代替推定）

import io
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import get_window

# ========= ユーティリティ =========

def _to_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_json_list(s):
    """文字列 s をJSONとして読み、float配列に変換。失敗はNone。"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    if isinstance(s, (list, tuple, np.ndarray)):
        try:
            return [float(v) for v in s]
        except Exception:
            return None
    try:
        arr = json.loads(str(s))
        return [float(v) for v in arr]
    except Exception:
        return None


# ========= スペクトル解析 =========

def fft_band_powers(lap_list, bands=(
    (1, 2),   # 低周波：スタミナ/粘り傾向
    (3, 4),   # 中周波：持続力
    (5, 999)  # 高周波：加減速/ちぐはぐ
)):
    """ラップ(秒/区間)配列から rFFT パワーを算出し、指定バンドで集約。
    長さ < 6 の場合はハニング窓でゼロ埋め（長さ8）して近似。
    返値: dict { 'low':x, 'mid':y, 'high':z, 'spec_sum':sum }
    """
    if lap_list is None:
        return dict(low=np.nan, mid=np.nan, high=np.nan, spec_sum=np.nan)

    x = np.array(lap_list, dtype=float)
    n0 = len(x)
    if n0 < 6:
        # 短すぎる場合は線形補間→8点にリサンプル
        if n0 <= 1:
            return dict(low=np.nan, mid=np.nan, high=np.nan, spec_sum=np.nan)
        xi = np.linspace(0, n0-1, num=8)
        xp = np.arange(n0)
        x = np.interp(xi, xp, x)
    n = len(x)
    x = (x - x.mean())  # DC除去
    win = get_window('hann', n)
    X = np.fft.rfft(x * win)
    P = (np.abs(X)**2) / np.sum(win**2)

    # 周波数インデックス（0,1,2,...）; 0はDCなので除外
    idx = np.arange(len(P))
    # バンド集約
    def band_power(i1, i2):
        i1 = max(i1, 1)
        i2 = min(i2, len(P)-1)
        if i2 < i1:
            return 0.0
        return float(P[i1:i2+1].sum())

    low = band_power(*bands[0])
    mid = band_power(*bands[1])
    high = band_power(*bands[2])
    return dict(low=low, mid=mid, high=high, spec_sum=float(P[1:].sum()))


# ========= 展開（ペース場） =========

def expected_pace_profile(styles, n_segments=12, early_weight=1.0):
    """脚質分布から簡易的な想定ペース（相対秒/区間）を合成。
    逃げ・先行が多いほど前半が速く、終いが失速しやすい形を作る。
    返値: list 長さ n_segments（小さいほど速い）。
    """
    styles = [s or '' for s in styles]
    n = len(styles)
    if n == 0:
        return [1.0 for _ in range(n_segments)]
    nige = sum(1 for s in styles if '逃' in s)
    senko = sum(1 for s in styles if '先' in s)
    sashi = sum(1 for s in styles if '差' in s)
    oikomi = sum(1 for s in styles if '追' in s)

    early_pressure = (nige + 0.6*senko) / max(n,1)
    late_pressure  = (sashi + oikomi) / max(n,1)

    # 速い=小、遅い=大 の相対秒を生成
    base = np.ones(n_segments)
    trend = np.linspace(-early_pressure, late_pressure, n_segments)
    shape = base + 0.35 * trend  # 前傾→後半やや遅
    # 正規化（平均=1.0）
    shape = shape / shape.mean()
    return shape.tolist()


def spectral_alignment(lap_list, pace_profile):
    """馬のスペクトルと想定ペースのスペクトルのコサイン類似度（0~1に射影）"""
    if lap_list is None or pace_profile is None:
        return np.nan
    # ラップは「秒」なので、速い=小さい。ペースprofileも相対秒。
    def spec(v):
        v = np.array(v, dtype=float)
        v = (v - v.mean())
        S = np.abs(np.fft.rfft(v * get_window('hann', len(v))))
        return S[1:]  # DC除去

    s1 = spec(lap_list)
    # pace側は逆相関（速い=小）なので -center してからFFT
    s2 = spec(pace_profile)
    # コサイン類似度
    if np.linalg.norm(s1) == 0 or np.linalg.norm(s2) == 0:
        return np.nan
    cos = float(np.dot(s1, s2) / (np.linalg.norm(s1)*np.linalg.norm(s2)))
    return 0.5*(cos+1.0)  # [-1,1]→[0,1]


# ========= 物理スコア（L=K-V, Sの最小化） =========

def compute_physics_scores(df):
    df = df.copy()

    # 想定ペース（場）
    pace_prof = expected_pace_profile(df['脚質'].fillna('').tolist(), n_segments=12)

    # スペクトル特徴
    spec_low = []
    spec_mid = []
    spec_high = []
    spec_sum = []
    align = []

    for _, r in df.iterrows():
        lap = safe_json_list(r.get('LapTimes', None))
        bp = fft_band_powers(lap)
        spec_low.append(bp['low'])
        spec_mid.append(bp['mid'])
        spec_high.append(bp['high'])
        spec_sum.append(bp['spec_sum'])
        align.append(spectral_alignment(lap, pace_prof))

    df['SPEC_low'] = spec_low
    df['SPEC_mid'] = spec_mid
    df['SPEC_high'] = spec_high
    df['SPEC_sum'] = spec_sum
    df['SPEC_align'] = align

    # 代替：ラップが無い場合のSPEC_alignを上がり3Fと脚質から近似
    # 追い込み×上がり優秀→高アライン、逃げで上がり平凡→低アライン（前傾想定）
    no_align = df['SPEC_align'].isna()
    if no_align.any():
        z_agari = -zscore(df['上がり3F(s)'])  # 速いほど良い→符号反転
        style_map = df['脚質'].fillna('').map(lambda s: 0.2 if '逃' in s else (0.5 if '先' in s else (0.8 if '差' in s else 1.0)))
        approx = np.clip(0.5 + 0.3*z_agari + 0.2*style_map, 0, 1)
        df.loc[no_align, 'SPEC_align'] = approx[no_align]

    # 能力側（K）
    # ・近走指数(スピード)、終い(上がり3F)、調教指数を正規化して統合
    z_speed = zscore(df['近走指数(スピード)'])
    z_agari = -zscore(df['上がり3F(s)'])  # 小さいほど良い
    z_chokyo = zscore(df['調教指数'])

    K = 0.55*z_speed + 0.30*z_agari + 0.15*z_chokyo

    # 斤量→有効質量（kg）
    # ベース=56kg として (斤量-56)*重み をVに加算
    mass_pen = 0.02*(df['斤量'].fillna(56.0) - 56.0)

    # 距離適性：中央値±幅 から目標距離とのずれ
    dist = df['距離(m)'].fillna(df['距離適性中央値(m)'])
    dist0 = df['距離適性中央値(m)'].fillna(dist.median())
    dw = df['距離幅(m)'].replace({0: np.nan}).fillna(400.0)
    dist_mis = ((dist - dist0).abs() / dw).clip(lower=0)

    # 馬場適性：想定馬場に対する適性スコア（1.0が基準、低いほど不利）
    baba = df['想定馬場'].fillna('良')
    def baba_score(row):
        m = {
            '良': row.get('馬場適性_良', np.nan),
            '稍': row.get('馬場適性_稍', np.nan),
            '重': row.get('馬場適性_重', np.nan),
            '不': row.get('馬場適性_不', np.nan),
        }
        val = m.get(str(row.get('想定馬場','良')), np.nan)
        try:
            return float(val)
        except Exception:
            return np.nan

    baba_sc = df.apply(baba_score, axis=1)
    # 1.0 が中立。<1 不利、>1 有利 → Vに (1 - baba_sc)
    ground_pen = (1.0 - baba_sc.fillna(1.0))

    # 枠順バイアス（+有利, -不利）をVにマイナスで反映
    draw_bias = df['枠順バイアス(±)'].fillna(0.0)

    # スタイル×展開のミスマッチ（前傾で差し有利/後傾で先行有利）をスペクトル整合度と併用
    style_pref = df['脚質'].fillna('').map(lambda s: 0.0 if '逃' in s else (0.1 if '先' in s else (0.2 if '差' in s else 0.25)))
    pace_mismatch = (0.25 - df['SPEC_align'].fillna(0.5)) + style_pref

    # ポテンシャル V の合成
    V = 0.60*dist_mis + 0.40*ground_pen + 0.50*mass_pen - 0.20*draw_bias + 0.60*pace_mismatch

    # 物理スコア：L=K-V を標準化
    L = K - V
    zL = zscore(L)

    # 作用S（小さいほど良い）を S = -zL と定義
    S = -zL

    # カオス温度 T：頭数と能力分散で決定
    n = len(df)
    spread = np.std(z_speed.fillna(0.0))
    T = 0.85 + 0.10*(n/18.0) + 0.15*spread

    # Boltzmann重みで勝率
    w = np.exp(-S / max(T, 1e-6))
    p1 = w / w.sum()

    # Gumbel-top-k サンプリングで 複勝(=top3)、連対(=top2) を近似
    topk = gumbel_topk_rates(w.values, k=3, n_sim=20000, seed=42)
    p_top1 = p1.values
    p_top2 = topk['top2']
    p_top3 = topk['top3']

    # 評価印：勝率とLから
    order = np.argsort(-p_top1)
    marks = ['◎','○','▲','△','△','×']
    mark_col = ['']*n
    for i, idx in enumerate(order[:len(marks)]):
        mark_col[idx] = marks[i]

    df_out = df.copy()
    df_out['L_score'] = L
    df_out['PhysicsZ'] = zL
    df_out['S_action'] = S
    df_out['勝率(top1)'] = p_top1
    df_out['連対率(top2)'] = p_top2
    df_out['複勝率(top3)'] = p_top3
    df_out['印'] = mark_col

    # 並び替え
    df_out = df_out.sort_values(['勝率(top1)','連対率(top2)','複勝率(top3)'], ascending=False).reset_index(drop=True)

    return df_out, dict(T=T, pace_profile=pace_prof)


def zscore(series):
    s = pd.to_numeric(series, errors='coerce')
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s*0
    return (s - mu) / sd


def gumbel_topk_rates(weights, k=3, n_sim=20000, seed=42):
    """Gumbel-max トリックで Plackett–Luce ランダム順序を生成し、topk包含率を推定。
    weights: 正の重み（np.array）
    返値: dict {'top1':p1_i, 'top2':p2_i, 'top3':p3_i}
    """
    rng = np.random.default_rng(seed)
    w = np.array(weights, dtype=float)
    w = np.maximum(w, 1e-12)
    logw = np.log(w)
    n = len(w)
    c_top = np.zeros((3, n), dtype=float)
    # ベクトル化しつつバッチで実行
    batch = 2000
    remaining = n_sim
    while remaining > 0:
        b = min(batch, remaining)
        g = rng.gumbel(size=(b, n))
        y = logw + g
        # argsort降順
        idx = np.argsort(-y, axis=1)
        for r in range(min(k,3)):
            pick = idx[:, r]
            # r<=1 なら top2、r<=2 なら top3
            for i in range(b):
                c_top[0, idx[i,0]] += 1  # top1
                c_top[1, idx[i,0]] += 1  # top2 includes top1
                c_top[1, idx[i,1]] += 1
                c_top[2, idx[i,0]] += 1  # top3 includes top1,2
                c_top[2, idx[i,1]] += 1
                c_top[2, idx[i,2]] += 1
            break  # 1回で集計済み
        remaining -= b
    p1 = c_top[0] / n_sim
    p2 = c_top[1] / n_sim
    p3 = c_top[2] / n_sim
    return {'top1': p1, 'top2': p2, 'top3': p3}


# ========= 買い目生成 =========

def generate_bets(df_ranked, total_budget=5000, modes=None):
    """簡易自動買い目（ユーザー方針に沿う）
    - 単勝：◎, ○ を推奨
    - 複勝：◎, ○ を推奨
    - ワイド：◎ - （○,▲,△,△,×）の流し
    - 三連複：一頭軸（◎）- 相手5頭(○,▲,△,△,×)
    予算配分：勝率に比例＋最低100円、端数調整
    """
    if modes is None:
        modes = dict(tansho=True, fukusho=True, wide=True, sanrenpuku=True)

    rows = df_ranked.copy()
    # 上位6頭
    sub = rows.head(6).reset_index(drop=True)
    # 印→index
    mark_to_idx = {sub.loc[i, '印']: i for i in range(len(sub))}
    idx_star = mark_to_idx.get('◎', 0)

    p1 = sub['勝率(top1)'].fillna(0.0).values
    names = sub['馬名'].tolist()

    # 予算按分のヘルパ
    def alloc(amounts, budget):
        amounts = np.array(amounts, dtype=float)
        if amounts.sum() <= 0:
            amounts = np.ones_like(amounts)
        w = amounts / amounts.sum()
        allocs = np.floor(w * budget / 100) * 100  # 100円単位
        # 端数は上から追加
        rest = int(budget - allocs.sum())
        i = 0
        while rest >= 100 and i < len(allocs):
            allocs[i] += 100
            rest -= 100
            i += 1
        return allocs.astype(int)

    tickets = []

    # 単勝
    if modes.get('tansho', True):
        cand = [i for i, m in enumerate(sub['印']) if m in ('◎','○')]
        probs = [p1[i] for i in cand]
        budget = int(total_budget * 0.30)
        allocs = alloc(probs, budget)
        for a, i in zip(allocs, cand):
            if a > 0:
                tickets.append(dict(type='単勝', combo=f"{names[i]}", yen=int(a)))

    # 複勝
    if modes.get('fukusho', True):
        # 複勝率で按分
        p3 = sub['複勝率(top3)'].fillna(0.0).values
        cand = [i for i, m in enumerate(sub['印']) if m in ('◎','○')]
        probs = [p3[i] for i in cand]
        budget = int(total_budget * 0.30)
        allocs = alloc(probs, budget)
        for a, i in zip(allocs, cand):
            if a > 0:
                tickets.append(dict(type='複勝', combo=f"{names[i]}", yen=int(a)))

    # ワイド：◎ - 相手5頭
    if modes.get('wide', True):
        allies = [i for i in range(len(sub)) if i != idx_star]
        budget = int(total_budget * 0.20)
        probs = [p1[i] + sub.loc[i,'複勝率(top3)'] for i in allies]
        allocs = alloc(probs, budget)
        for a, i in zip(allocs, allies):
            if a > 0:
                tickets.append(dict(type='ワイド', combo=f"{names[idx_star]} - {names[i]}", yen=int(a)))

    # 三連複 一頭軸：◎ - 相手5頭
    if modes.get('sanrenpuku', True):
        allies = [i for i in range(len(sub)) if i != idx_star]
        # 組合せ (5C2=10点)
        pairs = []
        for i in range(len(allies)):
            for j in range(i+1, len(allies)):
                pairs.append((allies[i], allies[j]))
        budget = int(total_budget * 0.20)
        # 簡易な重み：相手2頭の複勝率積
        p3 = sub['複勝率(top3)'].fillna(0.0).values
        probs = [p3[i]*p3[j] for (i,j) in pairs]
        allocs = alloc(probs, budget)
        for a, (i,j) in zip(allocs, pairs):
            if a > 0:
                tickets.append(dict(type='三連複', combo=f"{names[idx_star]} - {names[i]} - {names[j]}", yen=int(a)))

    return pd.DataFrame(tickets)


# ========= Streamlit UI =========

st.set_page_config(page_title='物理版スペクトル競馬論 β1', layout='wide')

st.title('物理版スペクトル競馬論 β1')
st.caption('Physics-informed Spectral Racing Model / 完成版（単体動作）')

with st.sidebar:
    st.header('設定')
    total_budget = st.number_input('総予算（円）', min_value=0, max_value=200000, value=5000, step=100)
    col1, col2 = st.columns(2)
    with col1:
        use_tansho = st.checkbox('単勝', value=True)
        use_wide = st.checkbox('ワイド', value=True)
    with col2:
        use_fukusho = st.checkbox('複勝', value=True)
        use_sanrenpuku = st.checkbox('三連複(一頭軸)', value=True)

    modes = dict(tansho=use_tansho, fukusho=use_fukusho, wide=use_wide, sanrenpuku=use_sanrenpuku)

    st.divider()
    st.subheader('入力Excel (.xlsx)')
    up = st.file_uploader('テンプレに沿ったExcelを選択', type=['xlsx'])

    if st.button('テンプレートをダウンロード/表示'):
        tpl = make_template_dataframe()
        st.session_state['template_df'] = tpl

if 'template_df' in st.session_state:
    st.subheader('テンプレート（例）')
    st.dataframe(st.session_state['template_df'], use_container_width=True)
    st.markdown('※ これをExcelに保存して編集後、左のアップローダから読み込んでください。')

# メイン処理
if up is not None:
    try:
        df_in = pd.read_excel(up)
    except Exception as e:
        st.error(f'Excelの読み込みに失敗: {e}')
        st.stop()

    # 必須列チェック
    required = ['馬名','枠','番','脚質','斤量','年齢','距離(m)','想定馬場',
                '近走指数(スピード)','上がり3F(s)','調教指数','枠順バイアス(±)',
                '馬場適性_良','馬場適性_稍','馬場適性_重','馬場適性_不',
                '距離適性中央値(m)','距離幅(m)','LapTimes']
    missing = [c for c in required if c not in df_in.columns]
    if missing:
        st.warning('必須列が不足しています: ' + ', '.join(missing))

    # 型整形
    for c in ['斤量','年齢','距離(m)','近走指数(スピード)','上がり3F(s)','調教指数','枠順バイアス(±)',
              '馬場適性_良','馬場適性_稍','馬場適性_重','馬場適性_不',
              '距離適性中央値(m)','距離幅(m)']:
        if c in df_in.columns:
            df_in[c] = pd.to_numeric(df_in[c], errors='coerce')

    df_ranked, meta = compute_physics_scores(df_in)

    st.subheader('物理スコア & 確率推定')
    show_cols = ['印','枠','番','馬名','脚質','斤量','年齢','距離(m)','想定馬場',
                 '近走指数(スピード)','上がり3F(s)','調教指数','枠順バイアス(±)',
                 'SPEC_align','L_score','PhysicsZ','S_action','勝率(top1)','連対率(top2)','複勝率(top3)']
    st.dataframe(df_ranked[show_cols], height=480, use_container_width=True)

    st.caption(f"温度T={meta['T']:.3f} / 展開プロファイル（相対秒/区間）: {np.round(meta['pace_profile'],3).tolist()}")

    bets = generate_bets(df_ranked, total_budget=total_budget, modes=modes)
    st.subheader('自動買い目（方針デフォルト）')
    if len(bets) == 0:
        st.info('選択した券種と予算では買い目が生成されません。')
    else:
        st.dataframe(bets, use_container_width=True)
        st.write('合計：', int(bets['yen'].sum()), '円')

else:
    st.info('左の「テンプレートをダウンロード/表示」で例を確認できます。Excelを用意したらアップロードしてください。')


# ========= テンプレ生成 =========

def make_template_dataframe():
    data = [
        {
            '馬名':'サンプルA','枠':1,'番':1,'脚質':'先行','斤量':56,'年齢':4,'距離(m)':1800,'想定馬場':'良',
            '近走指数(スピード)':72,'上がり3F(s)':34.5,'調教指数':65,'枠順バイアス(±)':0.1,
            '馬場適性_良':1.00,'馬場適性_稍':0.98,'馬場適性_重':0.95,'馬場適性_不':0.90,
            '距離適性中央値(m)':1800,'距離幅(m)':300,
            'LapTimes':'[12.7,12.3,12.2,12.1,12.0,11.9,11.8,11.9,12.0,12.2,12.5,12.7]'
        },
        {
            '馬名':'サンプルB','枠':7,'番':12,'脚質':'差し','斤量':55,'年齢':5,'距離(m)':1800,'想定馬場':'良',
            '近走指数(スピード)':70,'上がり3F(s)':33.8,'調教指数':60,'枠順バイアス(±)':-0.05,
            '馬場適性_良':1.02,'馬場適性_稍':1.00,'馬場適性_重':0.92,'馬場適性_不':0.88,
            '距離適性中央値(m)':2000,'距離幅(m)':400,
            'LapTimes':'[13.0,12.8,12.6,12.4,12.3,12.1,12.0,12.1,12.2,12.3,12.1,11.8]'
        }
    ]
    return pd.DataFrame(data)
