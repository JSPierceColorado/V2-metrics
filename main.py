import html
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse


# -----------------------------
# Logging
# -----------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("alpaca-metrics-bot")


# -----------------------------
# Config
# -----------------------------


def getenv_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def getenv_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number; got {raw!r}") from exc


def getenv_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer; got {raw!r}") from exc


def getenv_optional_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw.strip().replace(",", ""))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number; got {raw!r}") from exc


def getenv_csv(name: str, default: str) -> Tuple[str, ...]:
    raw = os.getenv(name, default)
    values = tuple(x.strip().upper() for x in raw.split(",") if x.strip())
    return values or tuple(x.strip().upper() for x in default.split(",") if x.strip())


@dataclass(frozen=True)
class Config:
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_paper: bool
    trading_base_url: str

    refresh_seconds: float
    web_refresh_seconds: int
    dashboard_token: str

    portfolio_history_period: str
    portfolio_history_timeframe: str
    portfolio_history_extended_hours: bool

    activity_after: str
    activity_types: Tuple[str, ...]
    activity_max_pages_per_type: int
    net_deposits_override: Optional[float]

    request_timeout_seconds: float
    request_retries: int
    request_sleep_seconds: float
    rate_limit_sleep_seconds: float
    error_body_max_chars: int


def load_config() -> Config:
    alpaca_api_key = (
        os.getenv("ALPACA_API_KEY")
        or os.getenv("ALPACA_API_KEY_ID")
        or os.getenv("APCA_API_KEY_ID")
        or ""
    ).strip()
    alpaca_secret_key = (
        os.getenv("ALPACA_SECRET_KEY")
        or os.getenv("ALPACA_API_SECRET")
        or os.getenv("ALPACA_API_SECRET_KEY")
        or os.getenv("APCA_API_SECRET_KEY")
        or ""
    ).strip()

    if not alpaca_api_key or not alpaca_secret_key:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")

    alpaca_paper = getenv_bool("ALPACA_PAPER", True)
    default_trading_base_url = (
        "https://paper-api.alpaca.markets" if alpaca_paper else "https://api.alpaca.markets"
    )

    return Config(
        alpaca_api_key=alpaca_api_key,
        alpaca_secret_key=alpaca_secret_key,
        alpaca_paper=alpaca_paper,
        trading_base_url=os.getenv("ALPACA_TRADING_BASE_URL", default_trading_base_url).strip(),
        refresh_seconds=getenv_float("METRICS_REFRESH_SECONDS", 60.0),
        web_refresh_seconds=getenv_int("WEB_REFRESH_SECONDS", 30),
        dashboard_token=os.getenv("DASHBOARD_TOKEN", "").strip(),
        portfolio_history_period=os.getenv("PORTFOLIO_HISTORY_PERIOD", "1M").strip(),
        portfolio_history_timeframe=os.getenv("PORTFOLIO_HISTORY_TIMEFRAME", "1D").strip(),
        portfolio_history_extended_hours=getenv_bool("PORTFOLIO_HISTORY_EXTENDED_HOURS", False),
        activity_after=os.getenv("ACTIVITY_AFTER", "1970-01-01").strip(),
        activity_types=getenv_csv("CASH_FLOW_ACTIVITY_TYPES", "CSD,CSW,TRANS,JNLC,ACATC,ACATS"),
        activity_max_pages_per_type=getenv_int("ACTIVITY_MAX_PAGES_PER_TYPE", 20),
        net_deposits_override=getenv_optional_float("NET_DEPOSITS_OVERRIDE"),
        request_timeout_seconds=getenv_float("REQUEST_TIMEOUT_SECONDS", 12.0),
        request_retries=getenv_int("REQUEST_RETRIES", 3),
        request_sleep_seconds=getenv_float("REQUEST_SLEEP_SECONDS", 0.20),
        rate_limit_sleep_seconds=getenv_float("RATE_LIMIT_SLEEP_SECONDS", 10.0),
        error_body_max_chars=getenv_int("ERROR_BODY_MAX_CHARS", 800),
    )


# -----------------------------
# Shared runtime state
# -----------------------------

app = FastAPI(title="Alpaca Metrics Bot")
_stop_event = threading.Event()
_worker_thread: Optional[threading.Thread] = None
_state_lock = threading.Lock()
_metrics_lock = threading.Lock()
_session = requests.Session()

_state: Dict[str, Any] = {
    "started_at": None,
    "last_refresh_started_at": None,
    "last_refresh_finished_at": None,
    "last_error": None,
    "refresh_count": 0,
    "paper": None,
    "dashboard_protected": False,
}
_metrics_cache: Optional[Dict[str, Any]] = None


# -----------------------------
# Utility helpers
# -----------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_from_timestamp(ts: Any) -> Optional[str]:
    try:
        if ts is None:
            return None
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None


def set_state(**kwargs: Any) -> None:
    with _state_lock:
        _state.update(kwargs)


def get_state_snapshot() -> Dict[str, Any]:
    with _state_lock:
        return dict(_state)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip().replace(",", "")
            if value == "":
                return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def to_optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip().replace(",", "")
            if value == "":
                return None
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def pct(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def fmt_money(value: Optional[float]) -> str:
    if value is None:
        return "—"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value * 100:,.2f}%"


def fmt_num(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}"


def clean_symbol(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def safe_text(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


# -----------------------------
# Alpaca HTTP helpers
# -----------------------------


class HttpStatusError(RuntimeError):
    def __init__(self, method: str, url: str, status_code: int, body: str) -> None:
        self.method = method
        self.url = url
        self.status_code = status_code
        self.body = body
        body_suffix = f" body={body}" if body else ""
        super().__init__(f"{method} {url} failed status={status_code}{body_suffix}")


def request_headers(cfg: Config) -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": cfg.alpaca_api_key,
        "APCA-API-SECRET-KEY": cfg.alpaca_secret_key,
        "Content-Type": "application/json",
    }


def response_body_for_log(resp: requests.Response, cfg: Config) -> str:
    body = (resp.text or "").strip().replace("\n", " ")
    max_chars = max(0, cfg.error_body_max_chars)
    if max_chars <= 0:
        return ""
    if len(body) > max_chars:
        return body[:max_chars] + "...[truncated]"
    return body


def raise_for_status_with_body(resp: requests.Response, cfg: Config, method: str, url: str) -> None:
    if resp.status_code >= 400:
        raise HttpStatusError(method, url, resp.status_code, response_body_for_log(resp, cfg))


def http_get(
    session: requests.Session,
    url: str,
    cfg: Config,
    *,
    params: Optional[Dict[str, Any]] = None,
    tolerate_missing: bool = False,
) -> Any:
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max(1, cfg.request_retries) + 1):
        try:
            resp = session.get(
                url,
                headers=request_headers(cfg),
                params=params,
                timeout=cfg.request_timeout_seconds,
            )
            if tolerate_missing and resp.status_code in {400, 404, 422}:
                log.info("GET tolerated status=%s url=%s body=%s", resp.status_code, url, response_body_for_log(resp, cfg))
                return []
            if resp.status_code == 429:
                msg = f"rate limited status 429: {response_body_for_log(resp, cfg)}"
                if attempt < cfg.request_retries:
                    log.warning("GET rate limited attempt=%s url=%s err=%s; retrying in %.1fs", attempt, url, msg, cfg.rate_limit_sleep_seconds)
                    time.sleep(cfg.rate_limit_sleep_seconds)
                    continue
                raise RuntimeError(msg)
            if 500 <= resp.status_code < 600:
                msg = f"retryable status {resp.status_code}: {response_body_for_log(resp, cfg)}"
                if attempt < cfg.request_retries:
                    backoff = min(2 ** attempt, 30)
                    log.warning("GET failed attempt=%s url=%s err=%s; retrying in %.1fs", attempt, url, msg, backoff)
                    time.sleep(backoff)
                    continue
                raise RuntimeError(msg)
            raise_for_status_with_body(resp, cfg, "GET", url)
            if cfg.request_sleep_seconds > 0:
                time.sleep(cfg.request_sleep_seconds)
            return resp.json() if resp.text else None
        except Exception as exc:
            last_exc = exc
            if attempt < cfg.request_retries:
                backoff = min(2 ** attempt, 30)
                log.warning("GET failed attempt=%s url=%s err=%s; retrying in %.1fs", attempt, url, exc, backoff)
                time.sleep(backoff)
                continue
    raise RuntimeError(f"GET failed after {cfg.request_retries} attempts: {url}: {last_exc}")


def get_account(session: requests.Session, cfg: Config) -> Dict[str, Any]:
    data = http_get(session, f"{cfg.trading_base_url}/v2/account", cfg)
    return data if isinstance(data, dict) else {}


def get_positions(session: requests.Session, cfg: Config) -> List[Dict[str, Any]]:
    data = http_get(session, f"{cfg.trading_base_url}/v2/positions", cfg)
    return data if isinstance(data, list) else []


def get_portfolio_history(session: requests.Session, cfg: Config) -> Dict[str, Any]:
    params = {
        "period": cfg.portfolio_history_period,
        "timeframe": cfg.portfolio_history_timeframe,
        "extended_hours": str(cfg.portfolio_history_extended_hours).lower(),
    }
    data = http_get(session, f"{cfg.trading_base_url}/v2/account/portfolio/history", cfg, params=params)
    return data if isinstance(data, dict) else {}


def get_account_activities_by_type(
    session: requests.Session,
    cfg: Config,
    activity_type: str,
) -> List[Dict[str, Any]]:
    activities: List[Dict[str, Any]] = []
    page_token: Optional[str] = None

    for page in range(max(1, cfg.activity_max_pages_per_type)):
        params: Dict[str, Any] = {
            "after": cfg.activity_after,
            "direction": "asc",
            "page_size": 100,
        }
        if page_token:
            params["page_token"] = page_token

        data = http_get(
            session,
            f"{cfg.trading_base_url}/v2/account/activities/{activity_type}",
            cfg,
            params=params,
            tolerate_missing=True,
        )
        if not isinstance(data, list) or not data:
            break

        items = [x for x in data if isinstance(x, dict)]
        activities.extend(items)
        last_id = items[-1].get("id") if items else None
        if not last_id or len(items) < 100:
            break
        page_token = str(last_id)

    return activities


# -----------------------------
# Metrics calculations
# -----------------------------


def position_metrics(raw_positions: Sequence[Dict[str, Any]], equity: float) -> Dict[str, Any]:
    positions: List[Dict[str, Any]] = []
    total_market_value = 0.0
    total_unrealized_pl = 0.0
    red_positions = 0
    green_positions = 0

    for p in raw_positions:
        symbol = clean_symbol(p.get("symbol"))
        market_value = to_float(p.get("market_value"), 0.0)
        unrealized_pl = to_float(p.get("unrealized_pl"), 0.0)
        unrealized_plpc = to_optional_float(p.get("unrealized_plpc"))
        cost_basis = to_float(p.get("cost_basis"), 0.0)
        qty = to_float(p.get("qty"), 0.0)
        exposure_pct = pct(abs(market_value), equity) if equity else None

        if unrealized_pl < 0:
            red_positions += 1
        elif unrealized_pl > 0:
            green_positions += 1

        total_market_value += market_value
        total_unrealized_pl += unrealized_pl
        positions.append(
            {
                "symbol": symbol,
                "asset_class": p.get("asset_class"),
                "side": p.get("side"),
                "qty": qty,
                "market_value": market_value,
                "cost_basis": cost_basis,
                "current_price": to_optional_float(p.get("current_price")),
                "avg_entry_price": to_optional_float(p.get("avg_entry_price")),
                "unrealized_pl": unrealized_pl,
                "unrealized_plpc": unrealized_plpc,
                "exposure_pct_of_equity": exposure_pct,
            }
        )

    top_by_exposure = sorted(positions, key=lambda x: abs(to_float(x.get("market_value"), 0.0)), reverse=True)[:10]
    losers = [p for p in positions if p.get("unrealized_plpc") is not None]
    largest_loser = min(losers, key=lambda x: to_float(x.get("unrealized_plpc"), 0.0), default=None)
    largest_winner = max(losers, key=lambda x: to_float(x.get("unrealized_plpc"), 0.0), default=None)

    return {
        "positions_count": len(positions),
        "red_positions": red_positions,
        "green_positions": green_positions,
        "flat_positions": len(positions) - red_positions - green_positions,
        "total_market_value": total_market_value,
        "total_unrealized_pl": total_unrealized_pl,
        "top_by_exposure": top_by_exposure,
        "largest_loser": largest_loser,
        "largest_winner": largest_winner,
        "positions": sorted(positions, key=lambda x: to_float(x.get("unrealized_plpc"), 0.0)),
    }


def history_metrics(history: Dict[str, Any]) -> Dict[str, Any]:
    raw_equity = history.get("equity") or []
    raw_ts = history.get("timestamp") or []
    equity_series: List[Tuple[Optional[str], float]] = []

    if isinstance(raw_equity, list):
        for idx, value in enumerate(raw_equity):
            eq = to_optional_float(value)
            if eq is None or eq <= 0:
                continue
            ts = raw_ts[idx] if isinstance(raw_ts, list) and idx < len(raw_ts) else None
            equity_series.append((utc_from_timestamp(ts), eq))

    peak: Optional[float] = None
    max_drawdown: Optional[float] = None
    current_drawdown: Optional[float] = None
    high_water_mark: Optional[float] = None

    for _, eq in equity_series:
        if peak is None or eq > peak:
            peak = eq
        dd = (eq - peak) / peak if peak else None
        if dd is not None:
            if max_drawdown is None or dd < max_drawdown:
                max_drawdown = dd
            current_drawdown = dd
            high_water_mark = peak

    return {
        "history_points": len(equity_series),
        "high_water_mark": high_water_mark,
        "current_drawdown": current_drawdown,
        "max_drawdown": max_drawdown,
        "equity_series": equity_series[-30:],
    }


def first_number_from_activity(activity: Dict[str, Any]) -> Optional[float]:
    for key in ("net_amount", "amount", "cash", "price"):
        value = to_optional_float(activity.get(key))
        if value is not None:
            return value
    return None


def cashflow_from_activity(activity: Dict[str, Any]) -> float:
    activity_type = str(activity.get("activity_type") or activity.get("entry_type") or activity.get("type") or "").upper()
    amount = first_number_from_activity(activity)
    if amount is None:
        return 0.0

    if activity_type in {"CSD"}:
        return abs(amount)
    if activity_type in {"CSW"}:
        return -abs(amount)

    # Journals / transfers / ACAT cash activity are usually signed already. Preserve sign.
    if activity_type in {"TRANS", "JNLC", "JNL", "ACATC", "ACATS"}:
        return amount

    return 0.0


def cash_flow_metrics(session: requests.Session, cfg: Config) -> Dict[str, Any]:
    if cfg.net_deposits_override is not None:
        return {
            "source": "NET_DEPOSITS_OVERRIDE",
            "net_deposits": cfg.net_deposits_override,
            "deposits": max(cfg.net_deposits_override, 0.0),
            "withdrawals": abs(min(cfg.net_deposits_override, 0.0)),
            "activity_count": 0,
            "activity_types_checked": list(cfg.activity_types),
            "notes": "Using NET_DEPOSITS_OVERRIDE from environment variables.",
        }

    deposits = 0.0
    withdrawals = 0.0
    activity_count = 0

    for activity_type in cfg.activity_types:
        try:
            activities = get_account_activities_by_type(session, cfg, activity_type)
        except Exception as exc:
            log.warning("Could not fetch account activities type=%s: %s", activity_type, exc)
            continue

        for activity in activities:
            amount = cashflow_from_activity(activity)
            if amount > 0:
                deposits += amount
            elif amount < 0:
                withdrawals += abs(amount)
            if amount != 0:
                activity_count += 1

    return {
        "source": "account_activities",
        "net_deposits": deposits - withdrawals,
        "deposits": deposits,
        "withdrawals": withdrawals,
        "activity_count": activity_count,
        "activity_types_checked": list(cfg.activity_types),
        "notes": (
            "Calculated from Alpaca account activities. If this looks wrong, set NET_DEPOSITS_OVERRIDE "
            "to your true deposits minus withdrawals."
        ),
    }


def collect_metrics(session: requests.Session, cfg: Config) -> Dict[str, Any]:
    started = time.time()
    set_state(last_refresh_started_at=now_iso(), last_error=None)

    account = get_account(session, cfg)
    positions_raw = get_positions(session, cfg)
    history = get_portfolio_history(session, cfg)
    cashflows = cash_flow_metrics(session, cfg)

    equity = to_float(account.get("equity") or account.get("portfolio_value"), 0.0)
    last_equity = to_optional_float(account.get("last_equity"))
    cash = to_optional_float(account.get("cash"))
    buying_power = to_optional_float(account.get("buying_power"))
    regt_buying_power = to_optional_float(account.get("regt_buying_power"))
    portfolio_value = to_optional_float(account.get("portfolio_value"))
    long_market_value = to_optional_float(account.get("long_market_value"))
    short_market_value = to_optional_float(account.get("short_market_value"))

    positions = position_metrics(positions_raw, equity)
    hist = history_metrics(history)

    net_deposits = to_float(cashflows.get("net_deposits"), 0.0)
    cash_flow_adjusted_pnl: Optional[float]
    cash_flow_adjusted_return: Optional[float]
    if net_deposits > 0:
        cash_flow_adjusted_pnl = equity - net_deposits
        cash_flow_adjusted_return = cash_flow_adjusted_pnl / net_deposits
    else:
        cash_flow_adjusted_pnl = None
        cash_flow_adjusted_return = None

    day_pnl = equity - last_equity if last_equity is not None else None
    day_return = pct(day_pnl, last_equity) if day_pnl is not None and last_equity else None

    metrics: Dict[str, Any] = {
        "generated_at": now_iso(),
        "runtime_seconds": round(time.time() - started, 3),
        "mode": "paper" if cfg.alpaca_paper else "live",
        "account": {
            "status": account.get("status"),
            "currency": account.get("currency"),
            "equity": equity,
            "last_equity": last_equity,
            "day_pnl": day_pnl,
            "day_return": day_return,
            "cash": cash,
            "buying_power": buying_power,
            "regt_buying_power": regt_buying_power,
            "portfolio_value": portfolio_value,
            "long_market_value": long_market_value,
            "short_market_value": short_market_value,
            "initial_margin": to_optional_float(account.get("initial_margin")),
            "maintenance_margin": to_optional_float(account.get("maintenance_margin")),
            "daytrade_count": account.get("daytrade_count"),
            "pattern_day_trader": account.get("pattern_day_trader"),
            "trading_blocked": account.get("trading_blocked"),
            "transfers_blocked": account.get("transfers_blocked"),
            "account_blocked": account.get("account_blocked"),
        },
        "cash_flows": cashflows,
        "profitability": {
            "net_deposits": net_deposits,
            "cash_flow_adjusted_pnl": cash_flow_adjusted_pnl,
            "cash_flow_adjusted_return": cash_flow_adjusted_return,
            "unrealized_pl_open_positions": positions["total_unrealized_pl"],
            "notes": "Cash-flow-adjusted P/L is equity minus net deposits. This is the main scorecard metric.",
        },
        "positions": positions,
        "history": hist,
    }

    largest_loser = positions.get("largest_loser") or {}
    largest_winner = positions.get("largest_winner") or {}
    log.info(
        "Metrics refreshed mode=%s equity=%.2f net_deposits=%.2f cfa_pnl=%s cfa_return=%s positions=%d red=%d green=%d day_pnl=%s largest_loser=%s:%s largest_winner=%s:%s",
        metrics["mode"],
        equity,
        net_deposits,
        fmt_money(cash_flow_adjusted_pnl),
        fmt_pct(cash_flow_adjusted_return),
        positions["positions_count"],
        positions["red_positions"],
        positions["green_positions"],
        fmt_money(day_pnl),
        largest_loser.get("symbol", ""),
        fmt_pct(largest_loser.get("unrealized_plpc")) if largest_loser else "—",
        largest_winner.get("symbol", ""),
        fmt_pct(largest_winner.get("unrealized_plpc")) if largest_winner else "—",
    )

    with _metrics_lock:
        global _metrics_cache
        _metrics_cache = metrics

    current_state = get_state_snapshot()
    set_state(
        last_refresh_finished_at=now_iso(),
        refresh_count=int(current_state.get("refresh_count") or 0) + 1,
        paper=cfg.alpaca_paper,
        dashboard_protected=bool(cfg.dashboard_token),
    )
    return metrics


def get_cached_or_refresh(cfg: Config, *, force: bool = False) -> Dict[str, Any]:
    with _metrics_lock:
        cached = _metrics_cache

    if cached and not force:
        generated = cached.get("generated_at")
        try:
            age = time.time() - datetime.fromisoformat(str(generated)).timestamp()
        except Exception:
            age = cfg.refresh_seconds + 1
        if age <= max(5.0, cfg.refresh_seconds):
            return cached

    return collect_metrics(_session, cfg)


def metrics_loop(cfg: Config) -> None:
    log.info(
        "Metrics service started paper=%s base_url=%s refresh_seconds=%.1f dashboard_protected=%s activity_types=%s",
        cfg.alpaca_paper,
        cfg.trading_base_url,
        cfg.refresh_seconds,
        bool(cfg.dashboard_token),
        ",".join(cfg.activity_types),
    )
    while not _stop_event.is_set():
        try:
            collect_metrics(_session, cfg)
        except Exception as exc:
            log.exception("Metrics refresh error: %s", exc)
            set_state(last_error=str(exc))
        sleep_for = max(5.0, cfg.refresh_seconds)
        _stop_event.wait(timeout=sleep_for)


# -----------------------------
# HTML rendering
# -----------------------------


def metric_card(title: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="card">
      <div class="card-title">{safe_text(title)}</div>
      <div class="card-value">{safe_text(value)}</div>
      <div class="card-sub">{safe_text(sub)}</div>
    </div>
    """


def render_positions_table(rows: Sequence[Dict[str, Any]], limit: int = 25) -> str:
    if not rows:
        return "<p>No open positions.</p>"
    body = []
    for p in rows[:limit]:
        pl = to_float(p.get("unrealized_pl"), 0.0)
        plpc = p.get("unrealized_plpc")
        klass = "neg" if pl < 0 else "pos" if pl > 0 else ""
        body.append(
            "<tr>"
            f"<td>{safe_text(p.get('symbol'))}</td>"
            f"<td>{safe_text(p.get('asset_class') or '')}</td>"
            f"<td class='num'>{fmt_num(p.get('qty'), 4)}</td>"
            f"<td class='num'>{fmt_money(p.get('market_value'))}</td>"
            f"<td class='num {klass}'>{fmt_money(pl)}</td>"
            f"<td class='num {klass}'>{fmt_pct(plpc)}</td>"
            f"<td class='num'>{fmt_pct(p.get('exposure_pct_of_equity'))}</td>"
            "</tr>"
        )
    return """
    <table>
      <thead><tr><th>Symbol</th><th>Class</th><th>Qty</th><th>Market Value</th><th>Unrealized P/L</th><th>Unrealized %</th><th>Exposure</th></tr></thead>
      <tbody>
    """ + "\n".join(body) + "</tbody></table>"


def render_dashboard(metrics: Dict[str, Any], state: Dict[str, Any], cfg: Config) -> str:
    account = metrics.get("account", {})
    profitability = metrics.get("profitability", {})
    cash_flows = metrics.get("cash_flows", {})
    positions = metrics.get("positions", {})
    history = metrics.get("history", {})

    largest_loser = positions.get("largest_loser") or {}
    largest_winner = positions.get("largest_winner") or {}

    warning = ""
    if not cfg.dashboard_token:
        warning = "<div class='warning'>DASHBOARD_TOKEN is not set. This Railway web dashboard is unprotected.</div>"
    if cash_flows.get("source") == "account_activities" and to_float(cash_flows.get("net_deposits"), 0.0) <= 0:
        warning += "<div class='warning'>Net deposits could not be confidently calculated from activities. Set NET_DEPOSITS_OVERRIDE to your true deposits minus withdrawals if this looks wrong.</div>"

    cards = "\n".join(
        [
            metric_card("Equity", fmt_money(account.get("equity")), f"Mode: {metrics.get('mode', 'unknown')}"),
            metric_card("Cash-flow adjusted P/L", fmt_money(profitability.get("cash_flow_adjusted_pnl")), "Equity - net deposits"),
            metric_card("Cash-flow adjusted return", fmt_pct(profitability.get("cash_flow_adjusted_return")), "Main scorecard"),
            metric_card("Net deposits", fmt_money(profitability.get("net_deposits")), f"Source: {cash_flows.get('source')}"),
            metric_card("Day P/L", fmt_money(account.get("day_pnl")), fmt_pct(account.get("day_return"))),
            metric_card("Buying power", fmt_money(account.get("buying_power")), f"Reg-T: {fmt_money(account.get('regt_buying_power'))}"),
            metric_card("Open unrealized P/L", fmt_money(profitability.get("unrealized_pl_open_positions")), f"Positions: {positions.get('positions_count', 0)}"),
            metric_card("Red / Green", f"{positions.get('red_positions', 0)} / {positions.get('green_positions', 0)}", "Open positions"),
            metric_card("Current drawdown", fmt_pct(history.get("current_drawdown")), f"Max: {fmt_pct(history.get('max_drawdown'))}"),
            metric_card("Largest loser", safe_text(largest_loser.get("symbol") or "—"), fmt_pct(largest_loser.get("unrealized_plpc")) if largest_loser else "—"),
            metric_card("Largest winner", safe_text(largest_winner.get("symbol") or "—"), fmt_pct(largest_winner.get("unrealized_plpc")) if largest_winner else "—"),
        ]
    )

    top_exposure = render_positions_table(positions.get("top_by_exposure") or [], limit=10)
    worst_positions = render_positions_table(positions.get("positions") or [], limit=25)

    json_url = "/api/metrics"
    refresh_url = "/refresh"
    token_note = ""
    if cfg.dashboard_token:
        token_note = "Token auth is enabled. Use ?token=... or X-Dashboard-Token for JSON/refresh endpoints."

    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <meta http-equiv="refresh" content="{int(max(10, cfg.web_refresh_seconds))}" />
      <title>Alpaca Metrics Bot</title>
      <style>
        :root {{ color-scheme: dark; }}
        body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: #101114; color: #eeeeee; }}
        header {{ padding: 28px 28px 8px; }}
        h1 {{ margin: 0 0 6px; font-size: 28px; }}
        h2 {{ margin-top: 34px; }}
        .muted {{ color: #a7a7a7; font-size: 14px; }}
        .warning {{ margin: 16px 28px; padding: 12px 14px; border: 1px solid #7c5b00; background: #2a2107; color: #ffd782; border-radius: 12px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; padding: 20px 28px; }}
        .card {{ background: #181a20; border: 1px solid #2b2e37; border-radius: 14px; padding: 16px; }}
        .card-title {{ color: #a7a7a7; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; }}
        .card-value {{ margin-top: 8px; font-size: 24px; font-weight: 700; }}
        .card-sub {{ margin-top: 4px; color: #a7a7a7; font-size: 13px; }}
        main {{ padding: 0 28px 40px; }}
        table {{ width: 100%; border-collapse: collapse; background: #181a20; border: 1px solid #2b2e37; border-radius: 14px; overflow: hidden; }}
        th, td {{ padding: 10px 12px; border-bottom: 1px solid #2b2e37; text-align: left; }}
        th {{ color: #a7a7a7; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
        td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
        .neg {{ color: #ff8a8a; }}
        .pos {{ color: #7ddc91; }}
        .footer {{ margin-top: 28px; color: #a7a7a7; font-size: 13px; }}
        a {{ color: #9cc2ff; }}
      </style>
    </head>
    <body>
      <header>
        <h1>Alpaca Metrics Bot</h1>
        <div class="muted">Generated at {safe_text(metrics.get('generated_at'))}. Last refresh finished {safe_text(state.get('last_refresh_finished_at'))}. Auto-refreshes every {int(max(10, cfg.web_refresh_seconds))} seconds.</div>
      </header>
      {warning}
      <section class="grid">{cards}</section>
      <main>
        <h2>Top exposure</h2>
        {top_exposure}
        <h2>Worst positions first</h2>
        {worst_positions}
        <div class="footer">
          <p>JSON: <a href="{json_url}">{json_url}</a> · Force refresh: <a href="{refresh_url}">{refresh_url}</a></p>
          <p>{safe_text(token_note)}</p>
          <p>Cash-flow adjusted P/L is only as good as the deposit/withdrawal data. Set NET_DEPOSITS_OVERRIDE if Alpaca activities do not match your actual funding history.</p>
        </div>
      </main>
    </body>
    </html>
    """


# -----------------------------
# FastAPI routes
# -----------------------------


CONFIG: Optional[Config] = None


def cfg() -> Config:
    global CONFIG
    if CONFIG is None:
        CONFIG = load_config()
    return CONFIG


def require_dashboard_token(request: Request, config: Config) -> None:
    if not config.dashboard_token:
        return
    provided = request.query_params.get("token") or request.headers.get("X-Dashboard-Token") or ""
    if provided != config.dashboard_token:
        raise HTTPException(status_code=401, detail="Missing or invalid dashboard token")


@app.on_event("startup")
def on_startup() -> None:
    global _worker_thread
    config = cfg()
    set_state(started_at=now_iso(), paper=config.alpaca_paper, dashboard_protected=bool(config.dashboard_token))
    if config.refresh_seconds > 0:
        _worker_thread = threading.Thread(target=metrics_loop, args=(config,), name="metrics-loop", daemon=True)
        _worker_thread.start()
    else:
        log.info("Background metrics loop disabled because METRICS_REFRESH_SECONDS<=0")


@app.on_event("shutdown")
def on_shutdown() -> None:
    _stop_event.set()
    if _worker_thread and _worker_thread.is_alive():
        _worker_thread.join(timeout=5)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    config = cfg()
    require_dashboard_token(request, config)
    metrics = get_cached_or_refresh(config)
    return HTMLResponse(render_dashboard(metrics, get_state_snapshot(), config))


@app.get("/api/metrics")
def api_metrics(request: Request) -> JSONResponse:
    config = cfg()
    require_dashboard_token(request, config)
    metrics = get_cached_or_refresh(config)
    return JSONResponse(metrics)


@app.get("/refresh")
def refresh(request: Request) -> JSONResponse:
    config = cfg()
    require_dashboard_token(request, config)
    metrics = get_cached_or_refresh(config, force=True)
    return JSONResponse({"status": "ok", "metrics": metrics})


@app.get("/status")
def status() -> Dict[str, Any]:
    return get_state_snapshot()


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    state = get_state_snapshot()
    return {
        "status": "ok" if not state.get("last_error") else "degraded",
        "service": "alpaca-metrics-bot",
        **state,
    }


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots_txt() -> str:
    return "User-agent: *\nDisallow: /\n"
