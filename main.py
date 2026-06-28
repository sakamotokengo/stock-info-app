from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import yfinance as yf
import feedparser
import requests
from bs4 import BeautifulSoup
from typing import Optional
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def _normalize_yield(v):
    """yfinanceは銘柄によって小数(0.037)または%値(3.7)を返すため統一する"""
    if v is None:
        return None
    return round(v / 100 if v > 1 else v, 6)


def _resolve_ticker(ticker: str) -> str:
    """4桁数字のみなら東証(.T)を補完して返す"""
    if re.fullmatch(r"\d{4}", ticker):
        return ticker + ".T"
    return ticker


app = FastAPI(title="Stock Info Service")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/robots.txt")
def robots():
    return FileResponse("static/robots.txt", media_type="text/plain")


@app.get("/api/stock/{ticker}")
def get_stock(ticker: str):
    try:
        resolved = _resolve_ticker(ticker)
        t = yf.Ticker(resolved)
        info = t.info
        hist = t.history(period="1mo")

        if hist.empty:
            raise HTTPException(
                status_code=404,
                detail=f"銘柄 '{ticker}' が見つかりません。ティッカーを確認してください（日本株は '7203.T' または '7203' の形式）"
            )

        prices = [
            {"date": str(d.date()), "close": round(float(v), 2)}
            for d, v in zip(hist.index, hist["Close"])
        ]

        return {
            "ticker": ticker.upper(),
            "name": info.get("longName") or info.get("shortName", ticker),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "currency": info.get("currency", ""),
            "change_pct": info.get("regularMarketChangePercent"),
            "market_cap": info.get("marketCap"),
            "per": info.get("trailingPE"),
            "pbr": info.get("priceToBook"),
            "eps": info.get("trailingEps"),
            "dividend_yield": _normalize_yield(info.get("dividendYield")),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "summary": info.get("longBusinessSummary", ""),
            "prices": prices,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/financials/{ticker}")
def get_financials(ticker: str):
    try:
        t = yf.Ticker(_resolve_ticker(ticker))
        info = t.info

        def safe(key, fmt=None):
            v = info.get(key)
            if v is None:
                return None
            if fmt == "pct":
                return round(v * 100, 2)
            if fmt == "b":
                return round(v / 1e8, 2)
            return round(v, 2) if isinstance(v, float) else v

        return {
            "ticker": ticker.upper(),
            "revenue": safe("totalRevenue", "b"),
            "net_income": safe("netIncomeToCommon", "b"),
            "operating_margin": safe("operatingMargins", "pct"),
            "profit_margin": safe("profitMargins", "pct"),
            "roe": safe("returnOnEquity", "pct"),
            "roa": safe("returnOnAssets", "pct"),
            "debt_to_equity": safe("debtToEquity"),
            "current_ratio": safe("currentRatio"),
            "quick_ratio": safe("quickRatio"),
            "per": safe("trailingPE"),
            "forward_per": safe("forwardPE"),
            "pbr": safe("priceToBook"),
            "psr": safe("priceToSalesTrailing12Months"),
            "ev_ebitda": safe("enterpriseToEbitda"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


NEWS_FEEDS = [
    ("Reuters JP", "https://feeds.reuters.com/reuters/JPBusinessNews"),
    ("NHK経済", "https://www3.nhk.or.jp/rss/news/cat5.xml"),
    ("Bloomberg JP", "https://feeds.bloomberg.co.jp/bloomberg/news"),
]


@app.get("/api/news")
def get_news(ticker: Optional[str] = None):
    articles = []
    for source, url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:8]:
                title = entry.get("title", "")
                if ticker and ticker.upper() not in title.upper():
                    # include all news if no ticker filter for general news
                    pass
                articles.append({
                    "source": source,
                    "title": title,
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "summary": entry.get("summary", "")[:200],
                })
        except Exception:
            continue

    return {"articles": articles[:30]}


@app.get("/api/analyze/{ticker}")
def analyze_stock(ticker: str):
    """財務指標のルールベーススコアリングで売買判断を返す（APIコスト不要）"""
    try:
        t = yf.Ticker(_resolve_ticker(ticker))
        info = t.info
        hist = t.history(period="3mo")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"株価取得エラー: {e}")

    def g(key):
        return info.get(key)

    score = 0
    reasons = []
    risks = []
    positive_signals = 0
    negative_signals = 0
    total_checked = 0

    def pos(msg):
        nonlocal positive_signals, total_checked
        positive_signals += 1; total_checked += 1
        reasons.append(msg)

    def neg(msg):
        nonlocal negative_signals, total_checked
        negative_signals += 1; total_checked += 1
        risks.append(msg)

    def neu():
        nonlocal total_checked
        total_checked += 1

    # --- 株価トレンド ---
    prices = [float(v) for v in hist["Close"]] if not hist.empty else []
    trend_pct = None
    if len(prices) >= 2:
        trend_pct = (prices[-1] - prices[0]) / prices[0] * 100
    if trend_pct is not None:
        if trend_pct > 5:
            score += 2; pos(f"直近3ヶ月の株価が {trend_pct:+.1f}% 上昇トレンド")
        elif trend_pct < -10:
            score -= 2; neg(f"直近3ヶ月で株価が {trend_pct:.1f}% 下落しており売り圧力が継続")
        else:
            neu(); reasons.append(f"直近3ヶ月の株価変動は {trend_pct:+.1f}% と比較的安定")

    # 移動平均（25日 vs 75日）ゴールデン/デッドクロス
    if len(prices) >= 75:
        ma25 = sum(prices[-25:]) / 25
        ma75 = sum(prices[-75:]) / 75
        if ma25 > ma75:
            score += 1; pos("25日移動平均が75日移動平均を上回り（ゴールデンクロス圏）")
        else:
            score -= 1; neg("25日移動平均が75日移動平均を下回り（デッドクロス圏）")

    # --- バリュエーション ---
    per = g("trailingPE")
    if per is not None:
        if per < 15:
            score += 2; pos(f"PER {per:.1f}倍は割安水準（目安: 15倍未満）")
        elif per < 25:
            score += 1; pos(f"PER {per:.1f}倍は適正範囲内")
        elif per < 40:
            score -= 1; neg(f"PER {per:.1f}倍はやや割高（目安: 25倍超）")
        else:
            score -= 2; neg(f"PER {per:.1f}倍は高水準で割高感が強い")

    pbr = g("priceToBook")
    if pbr is not None:
        if pbr < 1.0:
            score += 2; pos(f"PBR {pbr:.2f}倍は純資産以下で割安")
        elif pbr < 2.5:
            score += 1; pos(f"PBR {pbr:.2f}倍は適正範囲")
        elif pbr > 5:
            score -= 1; neg(f"PBR {pbr:.2f}倍は割高水準")
        else:
            neu()

    # --- 収益性 ---
    roe = g("returnOnEquity")
    if roe is not None:
        roe_pct = roe * 100
        if roe_pct >= 15:
            score += 2; pos(f"ROE {roe_pct:.1f}% は高収益（目安: 15%以上）")
        elif roe_pct >= 8:
            score += 1; pos(f"ROE {roe_pct:.1f}% は平均的な水準")
        else:
            score -= 1; neg(f"ROE {roe_pct:.1f}% は低水準で資本効率に課題")

    op_margin = g("operatingMargins")
    if op_margin is not None:
        op_pct = op_margin * 100
        if op_pct >= 20:
            score += 2; pos(f"営業利益率 {op_pct:.1f}% と高い収益体質")
        elif op_pct >= 10:
            score += 1; pos(f"営業利益率 {op_pct:.1f}% は業界平均的")
        elif op_pct < 0:
            score -= 2; neg(f"営業利益率 {op_pct:.1f}% で営業赤字")
        else:
            neg(f"営業利益率 {op_pct:.1f}% はやや低め")

    # --- 財務健全性 ---
    de = g("debtToEquity")
    if de is not None:
        if de < 50:
            score += 1; pos(f"D/Eレシオ {de:.0f} と財務健全性が高い")
        elif de > 200:
            score -= 2; neg(f"D/Eレシオ {de:.0f} と負債が多く財務リスクあり")
        else:
            neu()

    cr = g("currentRatio")
    if cr is not None:
        if cr >= 2.0:
            score += 1; pos(f"流動比率 {cr:.2f} と短期支払能力が高い")
        elif cr < 1.0:
            score -= 1; neg(f"流動比率 {cr:.2f} で短期的な資金繰りリスクあり")
        else:
            neu()

    # --- 配当 ---
    div = _normalize_yield(g("dividendYield"))
    if div is not None and div > 0:
        div_pct = div * 100
        if div_pct >= 3:
            score += 1; pos(f"配当利回り {div_pct:.2f}% と高配当")
        else:
            neu(); reasons.append(f"配当利回り {div_pct:.2f}%")

    # --- EPS（赤字チェック）---
    eps = g("trailingEps")
    if eps is not None and eps < 0:
        score -= 2; neg(f"EPS {eps:.2f} で最終赤字（収益化できていない）")

    # --- 予想PER vs 実績PERの乖離（増益・減益予想）---
    fwd_per = g("forwardPE")
    if per is not None and fwd_per is not None and per > 0 and fwd_per > 0:
        per_ratio = fwd_per / per
        if per_ratio < 0.7:
            score += 2; pos(f"予想PER {fwd_per:.1f}倍が実績 {per:.1f}倍を大幅に下回り、強い増益期待")
        elif per_ratio > 1.3:
            score -= 1; neg(f"予想PER {fwd_per:.1f}倍が実績 {per:.1f}倍を上回り、減益・利益圧縮の懸念")
        else:
            neu()

    # --- 時価総額（小型株リスク）---
    mkt_cap = g("marketCap")
    currency = g("currency") or ""
    is_small_cap = False
    if mkt_cap is not None:
        cap_b = mkt_cap / 1e8
        if currency == "JPY" and cap_b < 500:
            is_small_cap = True
            neg(f"時価総額 {cap_b:.0f}億円の小型株（流動性・ボラティリティリスク）")
        elif currency != "JPY" and mkt_cap < 3_000_000_000:
            is_small_cap = True
            neg(f"時価総額 {mkt_cap/1e8:.1f}億{currency}の小型株（流動性・ボラティリティリスク）")

    # --- セクター別リスク ---
    sector = g("sector") or ""
    is_high_risk_sector = sector in ("Healthcare", "Biotechnology")
    if is_high_risk_sector:
        neg("バイオテク／ヘルスケアセクター固有のリスク（臨床結果・規制審査）あり")
    elif sector == "Financial Services":
        if de is not None and de > 300:
            neg(f"金融セクターの高レバレッジ（D/E {de:.0f}）は業種特性も考慮が必要")

    # --- 判定 ---
    if score >= 4:
        verdict = "BUY"
    elif score <= -3:
        verdict = "SELL"
    else:
        verdict = "HOLD"

    # --- 確信度（根拠付き）---
    # ポジティブ比率 × スコア絶対値の強さを合算して算出
    signal_ratio = positive_signals / max(total_checked, 1)
    score_strength = min(1.0, abs(score) / 10)
    raw_conf = int((signal_ratio * 0.6 + score_strength * 0.4) * 100)
    confidence = min(85, max(30, raw_conf))
    confidence_basis = (
        f"分析した{total_checked}項目のうち {positive_signals}項目がポジティブ、"
        f"{negative_signals}項目がネガティブ（ポジティブ比率 {signal_ratio*100:.0f}%）。"
        f"スコア合計 {score:+d} の強度と比率を組み合わせて算出。"
        f"確信度は「現在の財務指標がどれだけ判断方向に揃っているか」を示す指標であり、"
        f"将来の株価を予測するものではありません。"
    )

    # --- BUY/SELL時：損切り・目標・ポジションサイズ ---
    trade_plan = None
    current_price = g("currentPrice") or g("regularMarketPrice")
    if verdict in ("BUY", "SELL") and current_price and len(prices) >= 10:
        import statistics
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        vol = statistics.stdev(returns) if len(returns) >= 2 else 0.02
        # 損切り幅 = 2σ（月次ボラ）、最低7%・最大25%
        stop_pct = min(0.25, max(0.07, vol * 2 * (3 ** 0.5)))
        # 目標幅 = リスクリワード比 2:1
        target_pct = stop_pct * 2.0

        if verdict == "BUY":
            stop_price = round(current_price * (1 - stop_pct), 2)
            target_price = round(current_price * (1 + target_pct), 2)
        else:
            stop_price = round(current_price * (1 + stop_pct), 2)
            target_price = round(current_price * (1 - target_pct), 2)

        # ポジションサイズ（総資産比）
        # 基本5%。小型株・高リスクセクターは減額。確信度が低いほど縮小。
        base_pct = 5.0
        if is_small_cap:
            base_pct -= 2.0
        if is_high_risk_sector:
            base_pct -= 1.0
        base_pct = base_pct * (confidence / 85)
        position_pct = round(max(1.0, min(8.0, base_pct)), 1)

        trade_plan = {
            "current_price": current_price,
            "currency": currency,
            "stop_loss": stop_price,
            "stop_loss_pct": round(stop_pct * 100, 1),
            "target_price": target_price,
            "target_pct": round(target_pct * 100, 1),
            "position_size_pct": position_pct,
            "risk_reward_ratio": "1:2",
            "basis": (
                f"損切り幅は直近3ヶ月の日次ボラティリティ（標準偏差）の2倍（{stop_pct*100:.1f}%）を基準に設定。"
                f"目標はリスクリワード比 1:2 で算出（{target_pct*100:.1f}%）。"
                f"推奨保有比率は基本5%から小型株・セクターリスク・確信度を考慮して調整。"
            ),
        }

    if not reasons:
        reasons.append("十分な財務データが取得できませんでした")

    # -------------------------------------------------------
    # マルチバガー候補スコアリング
    # 根拠: Yartseva (2025) "The Alchemy of Multibagger Stocks"
    #        Birmingham City University CAFÉ Working Paper No.33
    # -------------------------------------------------------
    mb_score = 0
    mb_checks = []

    # 1. 小型株（サイズ効果）
    # 論文: 時価総額$250M未満が最高リターン、$2B未満が中型で次点
    if mkt_cap is not None:
        if currency == "JPY":
            cap_b = mkt_cap / 1e8
            if cap_b < 500:
                mb_score += 2
                mb_checks.append(("✅", f"小型株（{cap_b:.0f}億円）— 低ベース効果でマルチバガー候補になりやすい"))
            elif cap_b < 5000:
                mb_score += 1
                mb_checks.append(("🟡", f"中型株（{cap_b:.0f}億円）— やや大きいが可能性あり"))
            else:
                mb_checks.append(("❌", f"大型株（{cap_b:.0f}億円）— マルチバガーには規模が大きすぎる"))
        else:
            cap_usd_b = mkt_cap / 1e9
            if cap_usd_b < 2:
                mb_score += 2
                mb_checks.append(("✅", f"小型株（${cap_usd_b:.1f}B）— サイズ効果でリターン期待"))
            elif cap_usd_b < 10:
                mb_score += 1
                mb_checks.append(("🟡", f"中型株（${cap_usd_b:.1f}B）— やや大きいが可能性あり"))
            else:
                mb_checks.append(("❌", f"大型株（${cap_usd_b:.1f}B）— マルチバガーには規模が大きすぎる"))
    else:
        mb_checks.append(("⬜", "時価総額データなし"))

    # 2. バリュー割安（B/M比）
    # 論文: PBR<1(B/M>1)が理想、開始時中央値PBR=1.1
    #       B/M>0.40(PBR<2.5) + 正の営業利益の組み合わせが有効
    if pbr is not None:
        bm = 1 / pbr if pbr > 0 else 0
        if bm >= 1.0:
            mb_score += 2
            mb_checks.append(("✅", f"PBR {pbr:.2f}倍（B/M={bm:.2f}）— 純資産以下で高バリュー。論文の理想域"))
        elif bm >= 0.4:
            mb_score += 1
            mb_checks.append(("🟡", f"PBR {pbr:.2f}倍（B/M={bm:.2f}）— 論文基準（B/M>0.4）は満たす"))
        else:
            mb_checks.append(("❌", f"PBR {pbr:.2f}倍（B/M={bm:.2f}）— 割高で論文の基準（B/M>0.4）未達"))
    else:
        mb_checks.append(("⬜", "PBRデータなし"))

    # 3. 正の営業利益（必須条件）
    # 論文: 営業赤字の小型株は年率-18%損失。絶対的な除外条件
    if op_margin is not None:
        if op_margin > 0:
            mb_score += 2
            mb_checks.append(("✅", f"営業利益率 {op_margin*100:.1f}%（正）— 論文の必須条件を満たす"))
        else:
            mb_checks.append(("❌", f"営業利益率 {op_margin*100:.1f}%（赤字）— 論文では小型株の営業赤字は最大の除外要因"))
    else:
        mb_checks.append(("⬜", "営業利益率データなし"))

    # 4. フリーキャッシュフロー
    # 論文: FCF利回りが高いことが独立した予測因子
    fcf = g("freeCashflow")
    if fcf is not None and mkt_cap and mkt_cap > 0:
        fcf_yield = fcf / mkt_cap * 100
        if fcf_yield >= 3:
            mb_score += 2
            mb_checks.append(("✅", f"FCF利回り {fcf_yield:.1f}%— 論文が実証した重要因子。高い余剰キャッシュ"))
        elif fcf_yield > 0:
            mb_score += 1
            mb_checks.append(("🟡", f"FCF利回り {fcf_yield:.1f}%（正）— キャッシュ創出はできているが水準は低め"))
        else:
            mb_checks.append(("❌", f"FCF利回り {fcf_yield:.1f}%（負）— キャッシュ消費中。注意が必要"))
    elif fcf is not None:
        if fcf > 0:
            mb_score += 1
            mb_checks.append(("🟡", "FCFは黒字（利回り計算不可）"))
        else:
            mb_checks.append(("❌", "FCFが負（キャッシュ消費中）"))
    else:
        mb_checks.append(("⬜", "FCFデータなし"))

    # 5. PEG比率
    # 論文: 開始時中央値PEG=0.8。成長に対して割安であることが重要
    peg = g("pegRatio")
    if peg is not None and peg > 0:
        if peg < 1.0:
            mb_score += 1
            mb_checks.append(("✅", f"PEG {peg:.2f}（<1.0）— 成長に対して割安。論文の中央値0.8に近い"))
        elif peg < 1.5:
            mb_checks.append(("🟡", f"PEG {peg:.2f}— やや割高だが許容範囲"))
        else:
            mb_checks.append(("❌", f"PEG {peg:.2f}（>1.5）— 成長に対して割高"))
    else:
        # PEGなければ予想PERで代替
        if fwd_per is not None and fwd_per < 15:
            mb_score += 1
            mb_checks.append(("✅", f"予想PER {fwd_per:.1f}倍（PEG非公開）— 論文中央値11.3倍に近い割安水準"))
        else:
            mb_checks.append(("⬜", "PEG・予想PERデータなし、または割高"))

    # 6. 52週安値圏（テクニカル：有利なエントリーポイント）
    # 論文: 株価が12ヶ月安値に近いことがリターン予測因子
    low52 = g("fiftyTwoWeekLow")
    high52 = g("fiftyTwoWeekHigh")
    cur_price = g("currentPrice") or g("regularMarketPrice")
    if low52 and high52 and cur_price and high52 > low52:
        position = (cur_price - low52) / (high52 - low52)
        if position <= 0.33:
            mb_score += 1
            mb_checks.append(("✅", f"52週レンジの下位{position*100:.0f}%— 論文が示す有利なエントリーポイント圏"))
        elif position <= 0.60:
            mb_checks.append(("🟡", f"52週レンジの{position*100:.0f}%— 中間帯。安値圏ではない"))
        else:
            mb_checks.append(("❌", f"52週レンジの上位{(1-position)*100:.0f}%— 高値圏。論文の推奨エントリーではない"))
    else:
        mb_checks.append(("⬜", "52週高値安値データなし"))

    # 7. 積極的な再投資（資産成長）
    # 論文: 資産縮小（保守的投資）は赤フラグ。EBITDA連動の積極投資が重要
    # yfinanceでは前期比は直接取れないため価格トレンドと収益性で代替
    ebitda = g("ebitda")
    ev = g("enterpriseValue")
    if ebitda is not None and ev is not None and ebitda > 0:
        ev_ebitda = ev / ebitda
        if ev_ebitda < 10:
            mb_score += 1
            mb_checks.append(("✅", f"EV/EBITDA {ev_ebitda:.1f}倍— 低水準で割安。成長余地あり"))
        else:
            mb_checks.append(("🟡", f"EV/EBITDA {ev_ebitda:.1f}倍— 標準的水準"))
    else:
        mb_checks.append(("⬜", "EBITDA/EVデータなし"))

    # マルチバガー総合判定（最大11点）
    mb_max = 11
    mb_pct = round(mb_score / mb_max * 100)
    if mb_score >= 8:
        mb_verdict = "高い可能性あり"
        mb_color = "green"
    elif mb_score >= 5:
        mb_verdict = "部分的に合致"
        mb_color = "yellow"
    else:
        mb_verdict = "現時点では低い"
        mb_color = "red"

    multibagger = {
        "score": mb_score,
        "max_score": mb_max,
        "score_pct": mb_pct,
        "verdict": mb_verdict,
        "color": mb_color,
        "checks": mb_checks,
        "paper": "Yartseva (2025) 'The Alchemy of Multibagger Stocks' — Birmingham City University CAFÉ Working Paper No.33",
        "note": "論文はEPS成長単独は統計的に有意でないと実証。「小型×割安×正の営業利益×高FCF×安値圏」の組み合わせが重要。",
    }

    return {
        "verdict": verdict,
        "score": score,
        "confidence": confidence,
        "confidence_basis": confidence_basis,
        "reasons": reasons[:5],
        "risks": risks[:4],
        "trade_plan": trade_plan,
        "multibagger": multibagger,
        "disclaimer": "本分析は財務指標のルールベース判定です。投資判断はご自身の責任で行ってください。",
    }


# -------------------------------------------------------
# スクリーニング用ユニバース
# -------------------------------------------------------
_US_UNIVERSE = [
    # Healthcare / Biotech
    "HRMY","AXSM","SUPN","PRAX","ACAD","ITCI","INVA","EXAS","NTRA","TMDX",
    # Technology
    "TTGT","CRUS","FORM","AMBA","POWI","VIAV","PCTY","AMSWA","DSGX","PLUS",
    # Financials
    "WSBC","LKFN","GBCI","CUBI","NBTB","HTLF","FFIN","SFNC",
    # Consumer
    "BOOT","SHOO","HIBB","CAKE","CBRL","PLCE","CHEF","NXST",
    # Industrials
    "KLIC","ASTE","GMS","AZZ","ERII","HLIO","HCKT","CW",
    # Energy
    "RES","WTTR","NR","PTEN","CEIX","ARCH",
    # Other small/mid
    "MGLN","HAFC","MGRC","STRL","DXPE","IART","UCTT","REXR",
]

_JP_UNIVERSE = [
    # TSEグロース・スタンダード 小型成長株
    "2413.T","3659.T","4477.T","6254.T","3769.T","4423.T","6031.T",
    "2148.T","3092.T","7148.T","4478.T","4485.T","3994.T","6095.T",
    "4480.T","3853.T","4488.T","4565.T","6050.T","7059.T",
    # 中型バリュー
    "4543.T","6723.T","7832.T","4911.T","6367.T","4063.T","8035.T",
]


def _quick_mb_score(ticker: str) -> Optional[dict]:
    """マルチバガー候補スコアのみを高速に算出（スクリーニング用）"""
    try:
        resolved = _resolve_ticker(ticker)
        t = yf.Ticker(resolved)
        info = t.info
        if not info or not info.get("regularMarketPrice") and not info.get("currentPrice"):
            return None

        score = 0
        passed = []
        failed = []

        currency = info.get("currency", "USD")
        mkt_cap = info.get("marketCap")
        pbr = info.get("priceToBook")
        op_margin = info.get("operatingMargins")
        fcf = info.get("freeCashflow")
        fwd_per = info.get("forwardPE")
        peg = info.get("pegRatio")
        low52 = info.get("fiftyTwoWeekLow")
        high52 = info.get("fiftyTwoWeekHigh")
        cur = info.get("currentPrice") or info.get("regularMarketPrice")
        ebitda = info.get("ebitda")
        ev = info.get("enterpriseValue")

        # 1. 小型株
        if mkt_cap:
            if currency == "JPY":
                if mkt_cap / 1e8 < 500: score += 2; passed.append("小型株")
                elif mkt_cap / 1e8 < 5000: score += 1; passed.append("中型株")
                else: failed.append("大型株")
            else:
                if mkt_cap < 2e9: score += 2; passed.append("小型株")
                elif mkt_cap < 10e9: score += 1; passed.append("中型株")
                else: failed.append("大型株")

        # 2. バリュー (PBR)
        if pbr and pbr > 0:
            bm = 1 / pbr
            if bm >= 1.0: score += 2; passed.append(f"PBR割安({pbr:.2f}x)")
            elif bm >= 0.4: score += 1; passed.append(f"PBR適正({pbr:.2f}x)")
            else: failed.append(f"PBR割高({pbr:.2f}x)")

        # 3. 正の営業利益（必須）
        if op_margin is not None:
            if op_margin > 0: score += 2; passed.append(f"営業利益率{op_margin*100:.1f}%")
            else: failed.append("営業赤字"); score -= 1

        # 4. FCF
        if fcf is not None and mkt_cap and mkt_cap > 0:
            fy = fcf / mkt_cap * 100
            if fy >= 3: score += 2; passed.append(f"FCF利回{fy:.1f}%")
            elif fy > 0: score += 1; passed.append(f"FCF正({fy:.1f}%)")
            else: failed.append("FCF負")

        # 5. PEG or 予想PER
        if peg and 0 < peg < 1.0: score += 1; passed.append(f"PEG{peg:.2f}")
        elif fwd_per and fwd_per < 15: score += 1; passed.append(f"予想PER{fwd_per:.1f}x")

        # 6. 52週安値圏
        if low52 and high52 and cur and high52 > low52:
            pos = (cur - low52) / (high52 - low52)
            if pos <= 0.33: score += 1; passed.append(f"安値圏({pos*100:.0f}%)")
            elif pos > 0.67: failed.append(f"高値圏({pos*100:.0f}%)")

        # 7. EV/EBITDA
        if ebitda and ev and ebitda > 0:
            ratio = ev / ebitda
            if ratio < 10: score += 1; passed.append(f"EV/EBITDA{ratio:.1f}x")

        if score < 3:
            return None

        price = cur

        # 直近値動き（1ヶ月）
        try:
            hist = t.history(period="1mo")
            if not hist.empty:
                prices_1m = hist["Close"].tolist()
                chg_1m = (prices_1m[-1] / prices_1m[0] - 1) * 100 if len(prices_1m) > 1 else 0
                vol_daily = hist["Close"].pct_change().std() * 100
                # ボラティリティベース損切・目標
                import math
                stop_pct = min(25, max(7, vol_daily * 2 * math.sqrt(3)))
                target_pct = stop_pct * 2
            else:
                chg_1m = None; stop_pct = None; target_pct = None
        except Exception:
            chg_1m = None; stop_pct = None; target_pct = None

        # 売買シグナル簡易判定
        pos_count = len(passed)
        neg_count = len(failed)
        if score >= 7 and neg_count <= 1:
            signal = "BUY"
        elif score <= 4 or neg_count >= 3:
            signal = "HOLD"
        else:
            signal = "BUY" if pos_count > neg_count * 2 else "HOLD"

        price_str = f"¥{price:,.0f}" if currency == "JPY" else f"${price:.2f}"
        return {
            "ticker": resolved,
            "name": info.get("longName") or info.get("shortName", resolved),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "price": price_str,
            "price_raw": price,
            "currency": currency,
            "mb_score": score,
            "mb_pct": min(100, round(score / 11 * 100)),
            "passed": passed,
            "failed": failed,
            "signal": signal,
            "chg_1m": round(chg_1m, 1) if chg_1m is not None else None,
            "stop_pct": round(stop_pct, 1) if stop_pct else None,
            "target_pct": round(target_pct, 1) if target_pct else None,
            "mkt_cap": mkt_cap,
            "fwd_per": round(fwd_per, 1) if fwd_per else None,
            "pbr": round(pbr, 2) if pbr else None,
            "op_margin": round(op_margin * 100, 1) if op_margin else None,
        }
    except Exception:
        return None


@app.get("/api/screen")
def screen_stocks(market: str = "US"):
    """マルチバガー候補スクリーニング（並列処理）"""
    universe = _JP_UNIVERSE if market == "JP" else _US_UNIVERSE
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_quick_mb_score, t): t for t in universe}
        for f in as_completed(futures):
            r = f.result()
            if r is not None:
                results.append(r)
    results.sort(key=lambda x: x["mb_score"], reverse=True)
    return {"market": market, "scanned": len(universe), "results": results[:20]}


@app.get("/api/compare")
def compare_stocks(tickers: str, market: str = "US"):
    """指定銘柄リストの財務指標を並列取得して比較用に返す"""
    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()][:8]
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_quick_mb_score, t): t for t in ticker_list}
        for f in as_completed(futures):
            r = f.result()
            if r is not None:
                results.append(r)
    results.sort(key=lambda x: x["mb_score"], reverse=True)
    return {"results": results}


@app.get("/api/bbs/{ticker_code}")
def get_bbs(ticker_code: str):
    """Yahoo掲示板から話題を取得（日本株コード向け）"""
    code = re.sub(r"[^0-9]", "", ticker_code)
    if not code:
        return {"posts": [], "message": "日本株コード（数字4桁）を指定してください"}

    url = f"https://finance.yahoo.co.jp/cm/message/{code}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        posts = []
        # Yahoo掲示板の投稿要素を探す
        for item in soup.select(".elmMessageItem, [class*='message']")[:15]:
            text_el = item.select_one("[class*='body'], [class*='text'], p")
            time_el = item.select_one("time, [class*='time'], [class*='date']")
            if text_el:
                posts.append({
                    "text": text_el.get_text(strip=True)[:200],
                    "time": time_el.get_text(strip=True) if time_el else "",
                })

        return {"posts": posts, "url": url}
    except Exception as e:
        return {"posts": [], "error": str(e)}
