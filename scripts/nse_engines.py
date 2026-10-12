"""
scripts/nse_engines.py  --  VERBATIM extract from github.com/srikanthgodavarthy/nse-scanner
(app_fib.py @ fecae26).  Standalone: numpy + pandas only, no Streamlit import.

Lifted unchanged (generated with ast.get_source_segment, not retyped):
    MODE_CFG, _ema_np, _count_squeeze_bars,
    compute_emerging_score            -> "EM"  = Emerging Momentum (NOT expectation-maximization)
    compute_preconfirmation_accumulation -> "PCA" = Pre-Confirmation Accumulation (NOT principal component analysis)

Both read a DataFrame with capitalised columns Open/High/Low/Close/Volume and
look only at the LAST bar backwards -- i.e. they are causal if you hand them
bars up to the decision bar.  Sector momentum is 0 here (the original enriches
it later in run_scan), so EmScore tops out at 90, not 100.
"""
import numpy as np
import pandas as pd

MODE_CFG = {
    "Intraday":   dict(period="5d",  interval="5m",  ema_fast=9,  ema_slow=21,
                       atr_mult=1.5, atr_wide=3.0, atr_max=1.0,
                       mom1_th=2, mom3_th=5, mom6_th=8, score_th=65, rsi_len=14,
                       htf_period="3mo", htf_interval="15m", validity_hours=4,
                       yf_period="5d",  yf_interval="5m",
                       live_interval="1m", hist_min_bars=60),
    "Swing":      dict(period="1y",  interval="1d",  ema_fast=50, ema_slow=200,
                       atr_mult=2.5, atr_wide=4.0, atr_max=1.5,
                       mom1_th=3, mom3_th=7, mom6_th=10, score_th=70, rsi_len=21,
                       htf_period="2y", htf_interval="1wk", validity_hours=72,
                       yf_period="1y",  yf_interval="1d",
                       live_interval="1d", hist_min_bars=50),
    "Positional": dict(period="2y",  interval="1d",  ema_fast=50, ema_slow=200,
                       atr_mult=3.5, atr_wide=5.0, atr_max=1.5,
                       mom1_th=5, mom3_th=10, mom6_th=15, score_th=70, rsi_len=21,
                       htf_period="5y", htf_interval="1wk", validity_hours=240,
                       yf_period="2y",  yf_interval="1d",
                       live_interval="1d", hist_min_bars=50),
}

def _ema_np(arr: np.ndarray, span: int) -> np.ndarray:
    """EMA on 1-D array."""
    alpha  = 2.0 / (span + 1)
    result = np.empty_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result

def _count_squeeze_bars(df: pd.DataFrame, period: int = 20,
                         max_lookback: int = 40) -> int:
    """Count consecutive bars currently in Keltner/BB squeeze (from most recent bar back)."""
    n = len(df)
    if n < period + 2:
        return 0
    cl = df["Close"].values.astype(np.float64)
    hi = df["High"].values.astype(np.float64)
    lo = df["Low"].values.astype(np.float64)
    count = 0
    for offset in range(min(max_lookback, n - period)):
        end = n - offset
        if end < period:
            break
        cl_w = cl[end - period:end]
        hi_w = hi[end - period:end]
        lo_w = lo[end - period:end]
        mid   = _ema_np(cl_w, period)[-1]
        dev2  = _ema_np((cl_w - _ema_np(cl_w, period)) ** 2, period)[-1]
        std   = float(np.sqrt(max(dev2, 0)))
        bb_hi = mid + 2.0 * std;  bb_lo = mid - 2.0 * std
        prev_cl_w = np.roll(cl_w, 1); prev_cl_w[0] = cl_w[0]
        tr_w   = np.maximum(hi_w - lo_w,
                 np.maximum(np.abs(hi_w - prev_cl_w), np.abs(lo_w - prev_cl_w)))
        atr_k  = float(_ema_np(tr_w, period)[-1])
        kc_hi  = mid + 1.5 * atr_k;  kc_lo = mid - 1.5 * atr_k
        if bb_hi <= kc_hi and bb_lo >= kc_lo:
            count += 1
        else:
            break
    return count

def compute_emerging_score(
    df: pd.DataFrame,
    mode: str,
    nifty_close: pd.Series,
    rs_rank: int = 50,
) -> dict:
    """
    Emerging Momentum Score (0–100) — surfaces stocks BEFORE they become obvious.

    Components contributing to score (v19.1 — post orthogonality fix):
    1. ATR Compression      20 — volatility coiling toward a breakout     [was 15]
    2. RVOL Acceleration    20 — volume building quietly                  [was 15]
    3. EMA Convergence      20 — fast/slow EMAs tightening                [was 15]
    4. Squeeze Pressure     20 — BB-inside-KC consecutive bars            [was 15]
    5. Sector Momentum      10 — sector tailwind (enriched in run_scan)
    6. Opening Range Exp.   10 — price expanding beyond consolidation     [was 15]

    Metadata only (NOT included in score — RS engine owns this):
    * RS Acceleration       — stored in EmRSAccel for UI display only

    Labels: IGNITING ≥65 · BUILDING ≥50 · COILING ≥35 · LATENT ≥20 · QUIET <20
    """
    out = dict(
        EmScore=0.0, EmLabel="QUIET",
        EmRSAccel=0.0, EmATRCompress=0.0, EmRVolAccel=0.0,
        EmEMAConv=0.0, EmSqzPressure=0.0, EmSectorMom=0.0, EmORExpansion=0.0,
    )
    try:
        if df is None or len(df) < 40:
            return out
        cl  = df["Close"].values.astype(np.float64)
        hi  = df["High"].values.astype(np.float64)
        lo  = df["Low"].values.astype(np.float64)
        vol = df["Volume"].values.astype(np.float64)
        n   = len(cl)
        cfg = MODE_CFG[mode]
        ef_span = cfg["ema_fast"]
        es_span = cfg["ema_slow"]

        # ── 1. RS ACCELERATION — DIAGNOSTIC METADATA ONLY (v19.1 V1-FIX)
        # ORTHOGONALITY: RS engine owns "leadership" (relative strength vs market).
        # EmScore owns ONLY energy-coil mechanics (compression, squeeze, EMA convergence,
        # RVOL). RS acceleration is computed here for UI metadata and audit tracing
        # but its point value is EXCLUDED from the EmScore total to prevent
        # double-counting when Readiness aggregates EmScore + RSLeaderScore.
        rs_pts = 0.0   # <-- intentionally zeroed; see _ORTHOGONALITY_RULES V1
        rs_pts_diagnostic = 0.0   # diagnostic value stored in EmRSAccel below
        try:
            nifty = (nifty_close.values.astype(np.float64)
                     if nifty_close is not None and len(nifty_close) >= 20
                     else None)
            if nifty is not None:
                def _rs(bars):
                    if n < bars + 1 or len(nifty) < bars + 1:
                        return 0.0
                    s = (cl[-1] - cl[-bars]) / (cl[-bars] + 1e-10) * 100
                    m = (nifty[-1] - nifty[-bars]) / (nifty[-bars] + 1e-10) * 100
                    return s - m
                rs5, rs10, rs20 = _rs(5), _rs(10), _rs(20)
                if rs5 > rs10 > 0:
                    rs_pts_diagnostic = min(15.0, (rs5 - rs10) * 2.5 + 5)
                elif rs5 > 0 and rs5 > rs20 * 0.5:
                    rs_pts_diagnostic = min(8.0, rs5 * 0.6)
                if rs_rank >= 70 and rs5 > 0:
                    rs_pts_diagnostic = min(15.0, rs_pts_diagnostic + 3)
        except Exception:
            pass

        # ── 2. ATR COMPRESSION (0–20 pts) [rescaled from 15 after V1-FIX]
        # Volatility contracting → coiling energy before expansion
        atr_pts = 0.0
        try:
            if n >= 25:
                prev_cl = np.roll(cl, 1); prev_cl[0] = cl[0]
                tr_arr  = np.maximum(hi - lo, np.maximum(
                          np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
                atr5_now  = float(_ema_np(tr_arr, 5)[-1])
                atr20_now = float(_ema_np(tr_arr, 20)[-1])
                ratio_now = atr5_now / (atr20_now + 1e-10)
                if   ratio_now < 0.65: atr_pts = 20.0
                elif ratio_now < 0.75: atr_pts = 16.0
                elif ratio_now < 0.85: atr_pts = 11.0
                elif ratio_now < 0.95: atr_pts = 5.0
                # Bonus: actively compressing (trend in ratio)
                if n > 15:
                    atr5_5  = float(_ema_np(tr_arr[:-5], 5)[-1])
                    atr20_5 = float(_ema_np(tr_arr[:-5], 20)[-1])
                    if ratio_now < atr5_5 / (atr20_5 + 1e-10) - 0.05:
                        atr_pts = min(20.0, atr_pts + 4.0)
        except Exception:
            pass

        # ── 3. RVOL ACCELERATION (0–20 pts) [rescaled from 15 after V1-FIX] ───
        # Volume building quietly across successive windows
        rvol_pts = 0.0
        try:
            if n >= 20:
                avg_vol = float(np.mean(vol[-21:-1])) if n >= 22 else float(np.mean(vol[:-1]))
                if avg_vol > 0:
                    v_now   = float(np.mean(vol[-3:]))
                    v_5ago  = float(np.mean(vol[-8:-5]))   if n >= 8  else avg_vol
                    v_10ago = float(np.mean(vol[-13:-10])) if n >= 13 else avg_vol
                    r_now   = v_now   / avg_vol
                    r_5ago  = v_5ago  / avg_vol
                    r_10ago = v_10ago / avg_vol
                    if r_now > r_5ago > r_10ago and r_now > 0.8:
                        rvol_pts = min(20.0, (r_now - r_10ago) * 20)
                    elif r_now > r_5ago and r_now > 0.9:
                        rvol_pts = min(12.0, (r_now - r_5ago) * 15)
                    elif 0.4 < r_now < 0.75:
                        rvol_pts = 6.0   # quiet dry-up = stealth accumulation
        except Exception:
            pass

        # ── 4. EMA CONVERGENCE (0–20 pts) [rescaled from 15 after V1-FIX]
        # Fast + slow EMAs tightening → coiling, decision point approaching
        conv_pts = 0.0
        try:
            if n >= es_span + 5:
                ef_arr = _ema_np(cl, ef_span)
                es_arr = _ema_np(cl, es_span)
                dist_now   = abs(ef_arr[-1]  - es_arr[-1])
                dist_5ago  = abs(ef_arr[-6]  - es_arr[-6])  if n >= 6  else dist_now
                dist_10ago = abs(ef_arr[-11] - es_arr[-11]) if n >= 11 else dist_now
                c_last     = cl[-1] + 1e-10
                dpct_now   = dist_now   / c_last * 100
                dpct_5ago  = dist_5ago  / c_last * 100
                dpct_10ago = dist_10ago / c_last * 100
                converging = dpct_now < dpct_5ago
                if   converging and dpct_now < 0.5:  conv_pts = 20.0
                elif converging and dpct_now < 1.0:  conv_pts = 15.0
                elif converging and dpct_now < 2.0:  conv_pts = 10.0
                elif converging:                     conv_pts =  5.0
                elif dpct_now < dpct_10ago * 0.65:   conv_pts =  8.0
                # Bonus: bullish convergence (fast still > slow)
                if ef_arr[-1] > es_arr[-1] and converging:
                    conv_pts = min(20.0, conv_pts + 4.0)
        except Exception:
            pass

        # ── 5. SQUEEZE PRESSURE (0–20 pts) [rescaled from 15 after V1-FIX]
        # Consecutive bars in BB/KC squeeze → stored kinetic energy
        sqz_pts = 0.0
        try:
            csq = _count_squeeze_bars(df, period=20, max_lookback=40)
            if   csq >= 20: sqz_pts = 20.0
            elif csq >= 15: sqz_pts = 16.0
            elif csq >= 10: sqz_pts = 12.0
            elif csq >= 5:  sqz_pts =  8.0
            elif csq >= 2:  sqz_pts =  4.0
        except Exception:
            pass

        # ── 6. SECTOR MOMENTUM (0–10 pts) — placeholder enriched in run_scan ─
        sec_pts = 0.0

        # ── 7. OPENING RANGE EXPANSION (0–10 pts) [rescaled from 15 after V1-FIX]
        # Price expanding beyond recent consolidation = first activation signal
        or_pts = 0.0
        try:
            if mode == "Intraday" and n >= 8:
                or_hi = float(np.max(hi[:6]))
                or_lo = float(np.min(lo[:6]))
                or_rng = or_hi - or_lo
                cur = cl[-1]
                if or_rng > 0:
                    exp_pct = (cur - or_hi) / or_rng
                    if   exp_pct > 0.20:  or_pts = 10.0
                    elif exp_pct > 0.05:  or_pts =  7.0
                    elif exp_pct >= 0:    or_pts =  4.0
                    elif exp_pct > -0.15: or_pts =  2.0
            else:
                lb = min(10 if mode == "Swing" else 20, n - 2)
                rng_hi = float(np.max(hi[-lb - 1:-1]))
                rng_lo = float(np.min(lo[-lb - 1:-1]))
                rng_w  = rng_hi - rng_lo
                cur    = cl[-1]
                if rng_w > 0:
                    exp_pct = (cur - rng_hi) / rng_w
                    if   exp_pct > 0.15:  or_pts = 10.0
                    elif exp_pct > 0.02:  or_pts =  7.0
                    elif exp_pct >= 0:    or_pts =  4.0
                    elif cur > rng_lo + rng_w * 0.65: or_pts = 2.0
        except Exception:
            pass

        # ── TOTAL (V1-FIX: rs_pts excluded — RS engine owns it)
        # Max = ATR(20) + RVOL(20) + EMAConv(20) + Squeeze(20) + Sector(10) + OR(10) = 100
        total = round(float(np.clip(
            atr_pts + rvol_pts + conv_pts + sqz_pts + sec_pts + or_pts,
            0, 100)), 1)
        label = ("IGNITING" if total >= 65 else "BUILDING" if total >= 50
                 else "COILING" if total >= 35 else "LATENT" if total >= 20 else "QUIET")
        out.update(
            EmScore       = total,
            EmLabel       = label,
            EmRSAccel     = round(rs_pts_diagnostic, 1),   # metadata only — NOT in score
            EmATRCompress = round(atr_pts,  1),
            EmRVolAccel   = round(rvol_pts, 1),
            EmEMAConv     = round(conv_pts, 1),
            EmSqzPressure = round(sqz_pts,  1),
            EmSectorMom   = 0.0,
            EmORExpansion = round(or_pts,   1),
        )
    except Exception:
        pass
    return out

def compute_preconfirmation_accumulation(
    df: pd.DataFrame,
    mode: str,
) -> dict:
    """
    Pre-Confirmation Accumulation Score (0–100).

    Components contributing to score (v19.1 — post orthogonality fix):
    1. Vol Compression Seq   18 — sequence of tightening ATR/range contractions [was 15]
    2. Hidden Accumulation   18 — down-days low-vol, up-days high-vol            [was 15]
    3. Effort vs Result      18 — big volume barely moving price = supply absorbed[was 15]
    4. Range Contraction     12 — persistent NR bars below historical avg range   [was 10]
    5. Failed Breakdown      18 — wicks below support closed back above = buying  [was 15]
    6. Volume Asymmetry      16 — up-day volume persistently > down-day volume    [was 15]

    Metadata only (NOT included in score — Smart Money engine owns CMF):
    * Relative CMF           — stored in PCACMFRel for UI display only

    Labels: ACCUMULATING ≥65 · BUILDING ≥50 · FORMING ≥35 · WEAK ≥20 · NONE <20
    """
    out = dict(
        PCAScore=0.0, PCALabel="NONE",
        PCACMFRel=0.0, PCAVolCmpSeq=0.0, PCAHiddenAccum=0.0,
        PCAEffortResult=0.0, PCARangeCont=0.0, PCAFailedBrkdn=0.0, PCAVolAsym=0.0,
    )
    try:
        if df is None or len(df) < 30:
            return out

        cl  = df["Close"].values.astype(np.float64)
        hi  = df["High"].values.astype(np.float64)
        lo  = df["Low"].values.astype(np.float64)
        vol = df["Volume"].values.astype(np.float64)
        n   = len(cl)
        op  = (df["Open"].values.astype(np.float64)
               if "Open" in df.columns else cl.copy())

        # ── 1. RELATIVE CMF — DIAGNOSTIC METADATA ONLY (v19.1 V2-FIX)
        # ORTHOGONALITY: Smart Money engine owns "participation quality".
        # CMF (Chaikin Money Flow) is a participation quality metric and is
        # already scored inside compute_smart_money_model() as SMCMFScore (20 pts).
        # Computing it here too and adding it to PCAScore creates double-counting
        # in ReadinessScore. CMF is retained as metadata (PCACMFRel) for UI display
        # and audit tracing but its value is EXCLUDED from PCAScore total.
        cmf_pts            = 0.0   # <-- intentionally zeroed; see _ORTHOGONALITY_RULES V2
        cmf_pts_diagnostic = 0.0   # stored in PCACMFRel below
        try:
            win = min(20, n)
            hlr = np.where((hi[-win:] - lo[-win:]) == 0, 1e-10, hi[-win:] - lo[-win:])
            mfm = ((cl[-win:] - lo[-win:]) - (hi[-win:] - cl[-win:])) / hlr
            cmf_now = float(np.sum(mfm * vol[-win:]) / (np.sum(vol[-win:]) + 1e-10))
            if n >= 40:
                hlr_old = np.where(
                    (hi[-40:-20] - lo[-40:-20]) == 0, 1e-10,
                    hi[-40:-20] - lo[-40:-20]
                )
                mfm_old = ((cl[-40:-20] - lo[-40:-20]) -
                           (hi[-40:-20] - cl[-40:-20])) / hlr_old
                cmf_old = float(
                    np.sum(mfm_old * vol[-40:-20]) / (np.sum(vol[-40:-20]) + 1e-10)
                )
                delta = cmf_now - cmf_old
                if   cmf_now > 0.15 and delta > 0:        cmf_pts_diagnostic = 15.0
                elif cmf_now > 0.05 and delta > 0:        cmf_pts_diagnostic = 11.0
                elif cmf_now > 0    and delta > 0.05:     cmf_pts_diagnostic = 8.0
                elif cmf_now > 0:                          cmf_pts_diagnostic = 4.0
                elif cmf_now > -0.05 and delta > 0.10:   cmf_pts_diagnostic = 3.0
            else:
                if   cmf_now > 0.15: cmf_pts_diagnostic = 10.0
                elif cmf_now > 0.05: cmf_pts_diagnostic = 6.0
                elif cmf_now > 0:    cmf_pts_diagnostic = 3.0
        except Exception:
            pass

        # ── 2. VOLATILITY COMPRESSION SEQUENCING (0–18 pts) [rescaled V2-FIX] ─
        # Consecutive narrowing ranges AND multi-window avg sequence tightening
        vc_pts = 0.0
        try:
            if n >= 10:
                ranges = hi - lo
                consec = 0
                for k in range(1, min(15, n - 1)):
                    if ranges[-k] < ranges[-(k + 1)]:
                        consec += 1
                    else:
                        break
                avg5  = float(np.mean(ranges[-5:]))  if n >= 5  else float(np.mean(ranges))
                avg10 = float(np.mean(ranges[-10:])) if n >= 10 else avg5
                avg20 = float(np.mean(ranges[-20:])) if n >= 20 else avg10
                sequenced = avg5 < avg10 < avg20

                if   consec >= 8 and sequenced: vc_pts = 18.0
                elif consec >= 5 and sequenced: vc_pts = 14.0
                elif consec >= 3:               vc_pts = 10.0
                elif consec >= 2 and sequenced: vc_pts = 6.0
                elif sequenced:                 vc_pts = 3.0
        except Exception:
            pass

        # ── 3. HIDDEN ACCUMULATION (0–18 pts) [rescaled from 15 after V2-FIX] ─
        # Up-day vol >> down-day vol even while price is flat/slightly negative
        ha_pts = 0.0
        try:
            lb = min(20, n - 1)
            up_vols   = [float(vol[i]) for i in range(-lb, 0) if cl[i] > cl[i - 1]]
            down_vols = [float(vol[i]) for i in range(-lb, 0) if cl[i] <= cl[i - 1]]
            if up_vols and down_vols:
                avg_up   = float(np.mean(up_vols))
                avg_down = float(np.mean(down_vols))
                ratio    = avg_up / (avg_down + 1e-10)
                price_chg = (float(cl[-1]) - float(cl[-lb])) / (float(cl[-lb]) + 1e-10) * 100

                if   ratio >= 2.5:                         ha_pts = 18.0
                elif ratio >= 2.0:                         ha_pts = 14.0
                elif ratio >= 1.5:                         ha_pts = 10.0
                elif ratio >= 1.2:                         ha_pts =  7.0
                elif ratio >= 1.05 and price_chg < 2.0:   ha_pts =  3.0
        except Exception:
            pass

        # ── 4. EFFORT VS RESULT ANOMALIES (0–18 pts) [rescaled from 15 after V2-FIX]
        # High volume + tiny body = supply absorbed (Wyckoff spring fingerprint)
        evr_pts = 0.0
        try:
            lb_evr   = min(10, n - 1)
            avg_vol_ = float(np.mean(vol[-21:-1])) if n >= 22 else float(np.mean(vol[:-1]))
            avg_rng_ = (float(np.mean(hi[-21:-1] - lo[-21:-1]))
                        if n >= 22 else float(np.mean(hi - lo)))
            if avg_vol_ > 0 and avg_rng_ > 0:
                anomalies = 0
                for k in range(1, lb_evr + 1):
                    v_k    = float(vol[-k])
                    body_k = abs(float(cl[-k]) - float(op[-k]))
                    cl_k   = float(cl[-k])
                    op_k   = float(op[-k])
                    hi_k   = float(hi[-k])
                    lo_k   = float(lo[-k])
                    mid_k  = (hi_k + lo_k) / 2
                    if v_k > avg_vol_ * 1.3 and body_k < avg_rng_ * 0.30:
                        anomalies += 1
                    elif (v_k > avg_vol_ * 1.5 and cl_k < op_k
                          and cl_k > mid_k):
                        anomalies += 1

                if   anomalies >= 5: evr_pts = 18.0
                elif anomalies >= 3: evr_pts = 13.0
                elif anomalies >= 2: evr_pts =  9.0
                elif anomalies >= 1: evr_pts =  5.0
        except Exception:
            pass

        # ── 5. RANGE CONTRACTION PERSISTENCE (0–12 pts) [rescaled from 10 V2-FIX]
        # How many of last 10 bars have range below 85% of 20-bar historical avg
        rc_pts = 0.0
        try:
            if n >= 12:
                ranges_ = hi - lo
                avg_rng20 = float(np.mean(ranges_[-21:-1])) if n >= 22 else float(np.mean(ranges_[:-1]))
                if avg_rng20 > 0:
                    nr_count = int(np.sum(ranges_[-10:] < avg_rng20 * 0.85))
                    if   nr_count >= 9: rc_pts = 12.0
                    elif nr_count >= 7: rc_pts = 10.0
                    elif nr_count >= 5: rc_pts =  7.0
                    elif nr_count >= 3: rc_pts =  5.0
                    elif nr_count >= 2: rc_pts =  2.0
        except Exception:
            pass

        # ── 6. FAILED BREAKDOWN ABSORPTION (0–18 pts) [rescaled from 15 after V2-FIX]
        # Bars where lows pierced recent support but closed back above = buying
        fba_pts = 0.0
        try:
            if n >= 15:
                lb_fba  = min(10, n - 5)
                support = float(np.min(lo[-lb_fba - 5:-5]))
                fba_cnt = 0
                for k in range(1, lb_fba + 1):
                    lo_k = float(lo[-k]); cl_k = float(cl[-k])
                    if lo_k < support * 0.997 and cl_k >= support * 0.999:
                        fba_cnt += 1

                if   fba_cnt >= 4: fba_pts = 18.0
                elif fba_cnt >= 3: fba_pts = 14.0
                elif fba_cnt >= 2: fba_pts = 10.0
                elif fba_cnt >= 1: fba_pts =  6.0

                if fba_cnt >= 1 and float(cl[-1]) > support:
                    rng_now   = float(hi[-1] - lo[-1])
                    avg_rng_b = float(np.mean(hi[-20:] - lo[-20:])) if n >= 20 else rng_now
                    if avg_rng_b > 0 and rng_now < avg_rng_b * 0.70:
                        fba_pts = min(18.0, fba_pts + 3.0)
        except Exception:
            pass

        # ── 7. VOLUME ASYMMETRY (0–16 pts) [rescaled from 15 after V2-FIX] ───
        # Up-day vol is consistently AND significantly > down-day vol
        va_pts = 0.0
        try:
            lb_va       = min(30, n - 1)
            up_vols_a   = [float(vol[i]) for i in range(-lb_va, 0) if cl[i] > cl[i - 1]]
            down_vols_a = [float(vol[i]) for i in range(-lb_va, 0) if cl[i] < cl[i - 1]]
            if len(up_vols_a) >= 3 and len(down_vols_a) >= 3:
                median_down  = float(np.median(down_vols_a))
                pct_above    = (sum(1 for v in up_vols_a if v > median_down)
                                / len(up_vols_a))
                mag_ratio    = (float(np.mean(up_vols_a))
                                / (float(np.mean(down_vols_a)) + 1e-10))

                if   pct_above >= 0.80 and mag_ratio >= 1.8: va_pts = 16.0
                elif pct_above >= 0.70 and mag_ratio >= 1.5: va_pts = 13.0
                elif pct_above >= 0.65 and mag_ratio >= 1.3: va_pts = 10.0
                elif pct_above >= 0.55 and mag_ratio >= 1.2: va_pts =  6.0
                elif pct_above >= 0.50 and mag_ratio >= 1.1: va_pts =  3.0
        except Exception:
            pass

        # ── TOTAL (V2-FIX: cmf_pts excluded — Smart Money engine owns CMF) ───
        # Max = VolCmpSeq(18) + HiddenAccum(18) + EffortResult(18)
        #       + RangeCont(12) + FailedBrkdn(18) + VolAsym(16) = 100
        total = round(float(np.clip(
            vc_pts + ha_pts + evr_pts + rc_pts + fba_pts + va_pts,
            0, 100
        )), 1)
        label = (
            "ACCUMULATING" if total >= 65 else
            "BUILDING"     if total >= 50 else
            "FORMING"      if total >= 35 else
            "WEAK"         if total >= 20 else
            "NONE"
        )
        out.update(
            PCAScore        = total,
            PCALabel        = label,
            PCACMFRel       = round(cmf_pts_diagnostic, 1),  # metadata only — NOT in score
            PCAVolCmpSeq    = round(vc_pts,   1),
            PCAHiddenAccum  = round(ha_pts,   1),
            PCAEffortResult = round(evr_pts,  1),
            PCARangeCont    = round(rc_pts,   1),
            PCAFailedBrkdn  = round(fba_pts,  1),
            PCAVolAsym      = round(va_pts,   1),
        )
    except Exception:
        pass
    return out
