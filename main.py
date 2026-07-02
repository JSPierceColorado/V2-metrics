import html
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

APP_VERSION = "metrics-bot-v6-clean-dashboard-2026-07-02"

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
    cash_flow_activity_debug_limit: int

    trade_activity_after: str
    trade_activity_max_pages: int
    trade_debug_limit: int

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
        # Default to direct external-cash activity types. Avoid TRANS by default because it can overlap
        # with CSD/CSW and cause double-counting on some accounts.
        activity_types=getenv_csv("CASH_FLOW_ACTIVITY_TYPES", "CSD,CSW,ACATC,JNLC"),
        activity_max_pages_per_type=getenv_int("ACTIVITY_MAX_PAGES_PER_TYPE", 100),
        cash_flow_activity_debug_limit=getenv_int("CASH_FLOW_ACTIVITY_DEBUG_LIMIT", 12),
        trade_activity_after=os.getenv("TRADE_ACTIVITY_AFTER", os.getenv("ACTIVITY_AFTER", "1970-01-01")).strip(),
        trade_activity_max_pages=getenv_int("TRADE_ACTIVITY_MAX_PAGES", 100),
        trade_debug_limit=getenv_int("TRADE_DEBUG_LIMIT", 20),
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
    *,
    after: Optional[str] = None,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    activities: List[Dict[str, Any]] = []
    page_token: Optional[str] = None
    effective_after = after if after is not None else cfg.activity_after
    effective_max_pages = max_pages if max_pages is not None else cfg.activity_max_pages_per_type

    for page in range(max(1, effective_max_pages)):
        params: Dict[str, Any] = {
            "after": effective_after,
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
    total_market_value = 0.0
    total_unrealized_pl = 0.0
    red_positions = 0
    green_positions = 0
    largest_loser: Optional[Dict[str, Any]] = None
    largest_winner: Optional[Dict[str, Any]] = None

    for p in raw_positions:
        symbol = clean_symbol(p.get("symbol"))
        market_value = to_float(p.get("market_value"), 0.0)
        unrealized_pl = to_float(p.get("unrealized_pl"), 0.0)
        unrealized_plpc = to_optional_float(p.get("unrealized_plpc"))

        if unrealized_pl < 0:
            red_positions += 1
        elif unrealized_pl > 0:
            green_positions += 1

        total_market_value += market_value
        total_unrealized_pl += unrealized_pl

        if unrealized_plpc is not None:
            compact = {
                "symbol": symbol,
                "unrealized_pl": unrealized_pl,
                "unrealized_plpc": unrealized_plpc,
                "market_value": market_value,
            }
            if largest_loser is None or unrealized_plpc < to_float(largest_loser.get("unrealized_plpc"), 0.0):
                largest_loser = compact
            if largest_winner is None or unrealized_plpc > to_float(largest_winner.get("unrealized_plpc"), 0.0):
                largest_winner = compact

    return {
        "positions_count": len(raw_positions),
        "red_positions": red_positions,
        "green_positions": green_positions,
        "flat_positions": len(raw_positions) - red_positions - green_positions,
        "total_market_value": total_market_value,
        "total_unrealized_pl": total_unrealized_pl,
        "largest_loser": largest_loser,
        "largest_winner": largest_winner,
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
    drawdown_series: List[Tuple[Optional[str], float]] = []

    for ts, eq in equity_series:
        if peak is None or eq > peak:
            peak = eq
        dd = (eq - peak) / peak if peak else None
        if dd is not None:
            drawdown_series.append((ts, dd))
            if max_drawdown is None or dd < max_drawdown:
                max_drawdown = dd
            current_drawdown = dd
            high_water_mark = peak

    return {
        "history_points": len(equity_series),
        "high_water_mark": high_water_mark,
        "current_drawdown": current_drawdown,
        "max_drawdown": max_drawdown,
        "equity_series": equity_series[-60:],
        "drawdown_series": drawdown_series[-60:],
    }


def first_number_from_activity(activity: Dict[str, Any]) -> Optional[float]:
    """Return the best cash amount field for a non-trade activity.

    Do not use price/qty here: those are trade fields and can accidentally turn fills into
    fake cash flows if activity filters are changed later.
    """
    for key in ("net_amount", "amount", "cash", "cash_amount"):
        value = to_optional_float(activity.get(key))
        if value is not None:
            return value
    return None


def activity_type_of(activity: Dict[str, Any]) -> str:
    return str(activity.get("activity_type") or activity.get("entry_type") or activity.get("type") or "").upper()


def activity_date_of(activity: Dict[str, Any]) -> str:
    for key in ("date", "transaction_time", "created_at", "updated_at"):
        raw = activity.get(key)
        if raw:
            return str(raw)
    return ""


def activity_text(activity: Dict[str, Any]) -> str:
    parts = []
    for key in ("activity_type", "entry_type", "type", "description", "status", "note"):
        value = activity.get(key)
        if value is not None:
            parts.append(str(value))
    return " ".join(parts).lower()


def activity_dedupe_key(activity: Dict[str, Any], amount: float) -> str:
    """Stable key for activity dedupe.

    Alpaca activity IDs are preferred. The fallback signature protects the calculation from
    occasional overlapping category/type pulls where the same cash event is returned without
    the same ID.
    """
    activity_id = str(activity.get("id") or "").strip()
    if activity_id:
        return f"id:{activity_id}"
    day = activity_date_of(activity)[:10]
    activity_type = activity_type_of(activity)
    return f"sig:{day}:{activity_type}:{amount:.2f}:{str(activity.get('description') or '')[:40]}"


def classify_cashflow_activity(activity: Dict[str, Any]) -> Tuple[float, str, str]:
    """Return (signed external cash flow, category, note).

    Positive means money came into the account from outside. Negative means money left the
    account. Dividends, interest, fees, fills, and market P/L are intentionally excluded because
    those are investment results, not funding.
    """
    activity_type = activity_type_of(activity)
    amount = first_number_from_activity(activity)
    if amount is None:
        return 0.0, "ignored", "no cash amount field"

    text = activity_text(activity)

    # Direct deposit/withdrawal types are the cleanest source.
    if activity_type == "CSD" or " cash deposit" in text or "ach deposit" in text or "csd" in text:
        return abs(amount), "deposit", "direct deposit activity"
    if activity_type == "CSW" or " cash withdrawal" in text or "ach withdrawal" in text or "csw" in text:
        return -abs(amount), "withdrawal", "direct withdrawal activity"

    # Cash ACATS and cash journals/transfers are usually signed. Preserve sign.
    if activity_type in {"ACATC", "JNLC", "JNL", "TRANS"}:
        if amount > 0:
            return amount, "deposit", f"signed {activity_type} cash activity"
        if amount < 0:
            return amount, "withdrawal", f"signed {activity_type} cash activity"
        return 0.0, "ignored", f"zero {activity_type} amount"

    return 0.0, "ignored", f"ignored activity type {activity_type}"


def cash_flow_metrics(session: requests.Session, cfg: Config) -> Dict[str, Any]:
    if cfg.net_deposits_override is not None:
        net = cfg.net_deposits_override
        return {
            "source": "NET_DEPOSITS_OVERRIDE",
            "net_deposits": net,
            "net_funding": net,
            "deposits": max(net, 0.0),
            "withdrawals": abs(min(net, 0.0)),
            "activity_count": 0,
            "raw_activity_count": 0,
            "duplicate_activity_count": 0,
            "ignored_activity_count": 0,
            "activity_types_checked": list(cfg.activity_types),
            "activity_after": cfg.activity_after,
            "first_cash_flow_date": None,
            "last_cash_flow_date": None,
            "recent_cash_flows": [],
            "notes": "Using NET_DEPOSITS_OVERRIDE from environment variables.",
        }

    deposits = 0.0
    withdrawals = 0.0
    raw_activity_count = 0
    duplicate_activity_count = 0
    ignored_activity_count = 0
    seen: Set[str] = set()
    cashflow_events: List[Dict[str, Any]] = []
    activity_type_counts: Dict[str, int] = {}

    for activity_type in cfg.activity_types:
        try:
            activities = get_account_activities_by_type(session, cfg, activity_type)
        except Exception as exc:
            log.warning("Could not fetch account activities type=%s: %s", activity_type, exc)
            continue

        for activity in activities:
            if not isinstance(activity, dict):
                continue
            raw_activity_count += 1
            signed_amount, category, note = classify_cashflow_activity(activity)
            actual_type = activity_type_of(activity) or activity_type
            activity_type_counts[actual_type] = activity_type_counts.get(actual_type, 0) + 1

            if signed_amount == 0:
                ignored_activity_count += 1
                continue

            key = activity_dedupe_key(activity, signed_amount)
            if key in seen:
                duplicate_activity_count += 1
                continue
            seen.add(key)

            if signed_amount > 0:
                deposits += signed_amount
            else:
                withdrawals += abs(signed_amount)

            cashflow_events.append(
                {
                    "date": activity_date_of(activity),
                    "activity_type": actual_type,
                    "amount": signed_amount,
                    "category": category,
                    "id": activity.get("id"),
                    "note": note,
                }
            )

    cashflow_events.sort(key=lambda x: str(x.get("date") or ""))
    limit = max(0, cfg.cash_flow_activity_debug_limit)

    return {
        "source": "account_activities_deduped",
        "net_deposits": deposits - withdrawals,
        "net_funding": deposits - withdrawals,
        "deposits": deposits,
        "withdrawals": withdrawals,
        "activity_count": len(cashflow_events),
        "raw_activity_count": raw_activity_count,
        "duplicate_activity_count": duplicate_activity_count,
        "ignored_activity_count": ignored_activity_count,
        "activity_type_counts": activity_type_counts,
        "activity_types_checked": list(cfg.activity_types),
        "activity_after": cfg.activity_after,
        "first_cash_flow_date": cashflow_events[0].get("date") if cashflow_events else None,
        "last_cash_flow_date": cashflow_events[-1].get("date") if cashflow_events else None,
        "recent_cash_flows": cashflow_events[-limit:] if limit else [],
        "notes": (
            "External funding is calculated as deposits minus withdrawals from deduped Alpaca cash activities. "
            "It intentionally excludes fills, dividends, interest, fees, and market P/L. If this still does not "
            "match your real funding history, set NET_DEPOSITS_OVERRIDE to your true deposits minus withdrawals."
        ),
    }



def parse_activity_time(activity: Dict[str, Any]) -> Tuple[float, str]:
    for key in ("transaction_time", "date", "created_at", "updated_at"):
        raw = activity.get(key)
        if not raw:
            continue
        text = str(raw)
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp(), dt.isoformat()
        except Exception:
            try:
                dt = datetime.fromisoformat(text[:10]).replace(tzinfo=timezone.utc)
                return dt.timestamp(), dt.isoformat()
            except Exception:
                continue
    return 0.0, ""


def fill_side(activity: Dict[str, Any]) -> str:
    side = str(activity.get("side") or activity.get("order_side") or "").strip().lower()
    if side in {"buy", "sell"}:
        return side
    net_amount = first_number_from_activity(activity)
    # For normal long fills, buys are cash outflows and sells are cash inflows.
    if net_amount is not None:
        if net_amount < 0:
            return "buy"
        if net_amount > 0:
            return "sell"
    text = activity_text(activity)
    if " buy" in f" {text} ":
        return "buy"
    if " sell" in f" {text} ":
        return "sell"
    return ""


def fill_price(activity: Dict[str, Any]) -> Optional[float]:
    for key in ("price", "avg_price", "filled_avg_price", "execution_price"):
        value = to_optional_float(activity.get(key))
        if value is not None and value > 0:
            return value
    return None


def fill_qty(activity: Dict[str, Any]) -> Optional[float]:
    for key in ("qty", "quantity", "cum_qty", "filled_qty"):
        value = to_optional_float(activity.get(key))
        if value is not None and value > 0:
            return value
    return None


def trade_activity_dedupe_key(activity: Dict[str, Any]) -> str:
    activity_id = str(activity.get("id") or "").strip()
    if activity_id:
        return f"id:{activity_id}"
    ts, _ = parse_activity_time(activity)
    symbol = clean_symbol(activity.get("symbol"))
    qty = fill_qty(activity) or 0.0
    price = fill_price(activity) or 0.0
    side = fill_side(activity)
    return f"sig:{ts:.0f}:{symbol}:{side}:{qty:.8f}:{price:.8f}"


def trade_expectancy_metrics(session: requests.Session, cfg: Config) -> Dict[str, Any]:
    """Approximate closed-trade expectancy from Alpaca FILL activities using FIFO lots.

    This is intentionally read-only and conservative. It is best used as a directional
    strategy-health metric, not as tax accounting.
    """
    raw_fills = get_account_activities_by_type(
        session,
        cfg,
        "FILL",
        after=cfg.trade_activity_after,
        max_pages=cfg.trade_activity_max_pages,
    )

    seen: Set[str] = set()
    fills: List[Dict[str, Any]] = []
    duplicate_count = 0
    ignored_count = 0

    for activity in raw_fills:
        if not isinstance(activity, dict):
            ignored_count += 1
            continue
        key = trade_activity_dedupe_key(activity)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)

        symbol = clean_symbol(activity.get("symbol"))
        side = fill_side(activity)
        qty = fill_qty(activity)
        price = fill_price(activity)
        ts, iso = parse_activity_time(activity)
        if not symbol or side not in {"buy", "sell"} or qty is None or price is None:
            ignored_count += 1
            continue
        fills.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "timestamp": ts,
                "time": iso,
                "id": activity.get("id"),
            }
        )

    fills.sort(key=lambda x: (to_float(x.get("timestamp"), 0.0), str(x.get("id") or "")))

    lots_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    closed_trades: List[Dict[str, Any]] = []
    buy_fill_count = 0
    sell_fill_count = 0
    unmatched_sell_qty = 0.0

    for fill in fills:
        symbol = str(fill["symbol"])
        side = str(fill["side"])
        qty = float(fill["qty"])
        price = float(fill["price"])
        lots = lots_by_symbol.setdefault(symbol, [])

        if side == "buy":
            buy_fill_count += 1
            lots.append({"qty": qty, "price": price, "opened_at": fill.get("timestamp"), "opened_time": fill.get("time")})
            continue

        sell_fill_count += 1
        remaining = qty
        realized_pnl = 0.0
        cost_basis = 0.0
        matched_qty = 0.0
        hold_seconds_weighted = 0.0

        while remaining > 1e-9 and lots:
            lot = lots[0]
            lot_qty = float(lot.get("qty") or 0.0)
            if lot_qty <= 1e-9:
                lots.pop(0)
                continue
            matched = min(remaining, lot_qty)
            buy_price = float(lot.get("price") or 0.0)
            basis = matched * buy_price
            pnl = matched * (price - buy_price)
            realized_pnl += pnl
            cost_basis += basis
            matched_qty += matched
            opened_at = to_float(lot.get("opened_at"), 0.0)
            closed_at = to_float(fill.get("timestamp"), 0.0)
            if opened_at > 0 and closed_at >= opened_at:
                hold_seconds_weighted += (closed_at - opened_at) * matched
            lot["qty"] = lot_qty - matched
            remaining -= matched
            if float(lot.get("qty") or 0.0) <= 1e-9:
                lots.pop(0)

        if remaining > 1e-9:
            unmatched_sell_qty += remaining

        if matched_qty > 1e-9:
            realized_pct = realized_pnl / cost_basis if cost_basis > 0 else None
            avg_hold_days = (hold_seconds_weighted / matched_qty / 86400.0) if matched_qty > 0 and hold_seconds_weighted > 0 else None
            closed_trades.append(
                {
                    "symbol": symbol,
                    "closed_at": fill.get("time"),
                    "closed_timestamp": fill.get("timestamp"),
                    "qty": matched_qty,
                    "sell_price": price,
                    "realized_pnl": realized_pnl,
                    "realized_pct": realized_pct,
                    "cost_basis": cost_basis,
                    "avg_hold_days": avg_hold_days,
                }
            )

    closed_count = len(closed_trades)
    wins = [t for t in closed_trades if to_float(t.get("realized_pnl"), 0.0) > 0]
    losses = [t for t in closed_trades if to_float(t.get("realized_pnl"), 0.0) < 0]
    breakevens = closed_count - len(wins) - len(losses)
    gross_profit = sum(to_float(t.get("realized_pnl"), 0.0) for t in wins)
    gross_loss = abs(sum(to_float(t.get("realized_pnl"), 0.0) for t in losses))
    total_realized_pnl = gross_profit - gross_loss
    avg_win = gross_profit / len(wins) if wins else None
    avg_loss = gross_loss / len(losses) if losses else None
    win_rate = len(wins) / closed_count if closed_count else None
    loss_rate = len(losses) / closed_count if closed_count else None
    expectancy = total_realized_pnl / closed_count if closed_count else None
    expectancy_pct = None
    total_cost_basis = sum(to_float(t.get("cost_basis"), 0.0) for t in closed_trades)
    if total_cost_basis > 0 and closed_count:
        expectancy_pct = total_realized_pnl / total_cost_basis
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (None if gross_profit <= 0 else float("inf"))
    payoff_ratio = (avg_win / avg_loss) if avg_win is not None and avg_loss and avg_loss > 0 else None
    avg_hold_days_values = [to_float(t.get("avg_hold_days"), math.nan) for t in closed_trades]
    avg_hold_days_values = [x for x in avg_hold_days_values if not math.isnan(x)]
    avg_hold_days = sum(avg_hold_days_values) / len(avg_hold_days_values) if avg_hold_days_values else None
    open_lots = sum(1 for lots in lots_by_symbol.values() for lot in lots if to_float(lot.get("qty"), 0.0) > 1e-9)
    limit = max(0, cfg.trade_debug_limit)

    return {
        "source": "fifo_from_fill_activities",
        "trade_activity_after": cfg.trade_activity_after,
        "raw_fill_count": len(raw_fills),
        "usable_fill_count": len(fills),
        "duplicate_fill_count": duplicate_count,
        "ignored_fill_count": ignored_count,
        "buy_fill_count": buy_fill_count,
        "sell_fill_count": sell_fill_count,
        "closed_trade_count": closed_count,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "breakeven_trades": breakevens,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "realized_pnl": total_realized_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "expectancy_pct_on_cost_basis": expectancy_pct,
        "profit_factor": profit_factor,
        "payoff_ratio": payoff_ratio,
        "avg_hold_days": avg_hold_days,
        "open_fifo_lots": open_lots,
        "unmatched_sell_qty": unmatched_sell_qty,
        "first_closed_trade_at": closed_trades[0].get("closed_at") if closed_trades else None,
        "last_closed_trade_at": closed_trades[-1].get("closed_at") if closed_trades else None,
        "recent_closed_trades": closed_trades[-limit:] if limit else [],
        "notes": "Expectancy is approximate FIFO matching from Alpaca FILL activities. It is a strategy-health estimate, not tax/accounting truth.",
    }



def epoch_from_isoish(value: Any) -> Optional[float]:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        try:
            dt = datetime.fromisoformat(str(value)[:10]).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return None


def days_between(start: Any, end: Any) -> Optional[float]:
    start_ts = epoch_from_isoish(start)
    end_ts = epoch_from_isoish(end)
    if start_ts is None or end_ts is None:
        return None
    days = (end_ts - start_ts) / 86400.0
    return days if days > 0 else None


def projection_point(label_days: int, equity: float, pnl_per_day: Optional[float]) -> Dict[str, Any]:
    when = datetime.now(timezone.utc) + timedelta(days=label_days)
    projected_pnl = pnl_per_day * label_days if pnl_per_day is not None else None
    return {
        "date": when.isoformat(),
        "days": label_days,
        "projected_pnl": projected_pnl,
        "projected_equity": equity + projected_pnl if projected_pnl is not None else None,
    }


def projection_metrics(
    *,
    equity: float,
    lifetime_pnl: Optional[float],
    cashflows: Dict[str, Any],
    history: Dict[str, Any],
    trades: Dict[str, Any],
) -> Dict[str, Any]:
    """Derived forward-looking run-rate estimates.

    These are not predictions. They simply answer: "If the recent/account/trade run-rate kept going,
    what would that imply?" The output is intentionally explicit about the basis used.
    """
    now_text = now_iso()

    equity_series = history.get("equity_series") or []
    history_pnl_per_day: Optional[float] = None
    history_days: Optional[float] = None
    history_annualized_return: Optional[float] = None
    if isinstance(equity_series, list) and len(equity_series) >= 2:
        first_ts, first_eq = equity_series[0]
        last_ts, last_eq = equity_series[-1]
        first_eq_f = to_float(first_eq, 0.0)
        last_eq_f = to_float(last_eq, 0.0)
        history_days = days_between(first_ts, last_ts)
        if history_days and first_eq_f > 0 and last_eq_f > 0:
            history_pnl_per_day = (last_eq_f - first_eq_f) / history_days
            # Annualized recent equity trend. This can be distorted by deposits/withdrawals inside the window.
            try:
                history_annualized_return = (last_eq_f / first_eq_f) ** (365.0 / history_days) - 1.0
            except Exception:
                history_annualized_return = None

    first_cash_flow_date = cashflows.get("first_cash_flow_date")
    funding_days = days_between(first_cash_flow_date, now_text) if first_cash_flow_date else None
    lifetime_pnl_per_day: Optional[float] = None
    if funding_days and lifetime_pnl is not None:
        lifetime_pnl_per_day = lifetime_pnl / funding_days

    expectancy = to_optional_float(trades.get("expectancy")) if isinstance(trades, dict) else None
    avg_win = to_optional_float(trades.get("avg_win")) if isinstance(trades, dict) else None
    avg_loss = to_optional_float(trades.get("avg_loss")) if isinstance(trades, dict) else None
    win_rate = to_optional_float(trades.get("win_rate")) if isinstance(trades, dict) else None
    closed_count = int(to_float(trades.get("closed_trade_count"), 0.0)) if isinstance(trades, dict) else 0
    trade_days = days_between(trades.get("first_closed_trade_at"), trades.get("last_closed_trade_at")) if isinstance(trades, dict) else None
    closed_trades_per_day: Optional[float] = None
    trade_pnl_per_day: Optional[float] = None
    if trade_days and closed_count > 1:
        # Floor at one day so a burst of same-day closes does not create absurd projections.
        closed_trades_per_day = closed_count / max(1.0, trade_days)
        if expectancy is not None:
            trade_pnl_per_day = expectancy * closed_trades_per_day

    breakeven_win_rate: Optional[float] = None
    edge_gap: Optional[float] = None
    if avg_win is not None and avg_win > 0 and avg_loss is not None and avg_loss > 0:
        breakeven_win_rate = avg_loss / (avg_win + avg_loss)
        if win_rate is not None:
            edge_gap = win_rate - breakeven_win_rate

    # Pick a headline projection basis. Prefer trade expectancy when there are enough closed trades;
    # otherwise use recent equity trend; otherwise use lifetime P/L run-rate.
    basis = "unavailable"
    headline_pnl_per_day: Optional[float] = None
    if trade_pnl_per_day is not None and closed_count >= 5:
        basis = "trade_expectancy_run_rate"
        headline_pnl_per_day = trade_pnl_per_day
    elif history_pnl_per_day is not None and history_days is not None and history_days >= 5:
        basis = "recent_equity_trend"
        headline_pnl_per_day = history_pnl_per_day
    elif lifetime_pnl_per_day is not None and funding_days is not None and funding_days >= 7:
        basis = "lifetime_cash_flow_adjusted_run_rate"
        headline_pnl_per_day = lifetime_pnl_per_day

    high_water_mark = to_optional_float(history.get("high_water_mark"))
    recovery_needed: Optional[float] = None
    recovery_return_needed: Optional[float] = None
    days_to_recover: Optional[float] = None
    if high_water_mark is not None and equity > 0 and high_water_mark > equity:
        recovery_needed = high_water_mark - equity
        recovery_return_needed = high_water_mark / equity - 1.0
        if headline_pnl_per_day is not None and headline_pnl_per_day > 0:
            days_to_recover = recovery_needed / headline_pnl_per_day

    projected_series = [projection_point(0, equity, headline_pnl_per_day)]
    # Use now for the starting point rather than adding zero days in the future.
    projected_series[0]["date"] = now_text
    projected_series[0]["projected_pnl"] = 0.0
    projected_series[0]["projected_equity"] = equity
    projected_series.extend(projection_point(d, equity, headline_pnl_per_day) for d in (30, 60, 90, 180, 365))

    def projected_return(days: int) -> Optional[float]:
        if headline_pnl_per_day is None or equity <= 0:
            return None
        return (headline_pnl_per_day * days) / equity

    return {
        "basis": basis,
        "headline_pnl_per_day": headline_pnl_per_day,
        "projected_30d_pnl": headline_pnl_per_day * 30 if headline_pnl_per_day is not None else None,
        "projected_90d_pnl": headline_pnl_per_day * 90 if headline_pnl_per_day is not None else None,
        "projected_365d_pnl": headline_pnl_per_day * 365 if headline_pnl_per_day is not None else None,
        "projected_30d_return": projected_return(30),
        "projected_90d_return": projected_return(90),
        "projected_365d_return": projected_return(365),
        "projected_equity_series": projected_series,
        "history_pnl_per_day": history_pnl_per_day,
        "history_days": history_days,
        "history_annualized_return": history_annualized_return,
        "lifetime_pnl_per_day": lifetime_pnl_per_day,
        "funding_days": funding_days,
        "trade_pnl_per_day": trade_pnl_per_day,
        "trade_days": trade_days,
        "closed_trades_per_day": closed_trades_per_day,
        "closed_trades_per_week": closed_trades_per_day * 7 if closed_trades_per_day is not None else None,
        "projected_closed_trades_30d": closed_trades_per_day * 30 if closed_trades_per_day is not None else None,
        "breakeven_win_rate": breakeven_win_rate,
        "edge_gap": edge_gap,
        "recovery_needed": recovery_needed,
        "recovery_return_needed": recovery_return_needed,
        "days_to_recover_high_water": days_to_recover,
        "notes": "Projections are run-rate estimates from existing data, not forecasts or guarantees. Cash flows can distort recent equity trend projections.",
    }


def collect_metrics(session: requests.Session, cfg: Config) -> Dict[str, Any]:
    started = time.time()
    set_state(last_refresh_started_at=now_iso(), last_error=None)

    account = get_account(session, cfg)
    positions_raw = get_positions(session, cfg)
    history = get_portfolio_history(session, cfg)
    cashflows = cash_flow_metrics(session, cfg)
    try:
        trades = trade_expectancy_metrics(session, cfg)
    except Exception as exc:
        log.warning("Could not calculate trade expectancy metrics: %s", exc)
        trades = {"source": "error", "error": str(exc), "notes": "Trade expectancy metrics failed; account-level metrics are still available."}

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

    total_deposits = to_float(cashflows.get("deposits"), 0.0)
    total_withdrawals = to_float(cashflows.get("withdrawals"), 0.0)
    net_deposits = to_float(cashflows.get("net_deposits"), 0.0)

    # This is the most robust simple funding-adjusted P/L formula when the account has
    # been withdrawn to zero and later re-funded:
    #   lifetime P/L = current equity + all withdrawals - all deposits
    # It is equivalent to equity - net funding, but the expanded formula is easier to audit.
    lifetime_pnl = equity + total_withdrawals - total_deposits
    cash_flow_adjusted_pnl: Optional[float] = lifetime_pnl
    cash_flow_adjusted_return: Optional[float] = lifetime_pnl / net_deposits if net_deposits > 0 else None
    return_on_total_deposits: Optional[float] = lifetime_pnl / total_deposits if total_deposits > 0 else None

    day_pnl = equity - last_equity if last_equity is not None else None
    day_return = pct(day_pnl, last_equity) if day_pnl is not None and last_equity else None

    initial_margin = to_optional_float(account.get("initial_margin"))
    maintenance_margin = to_optional_float(account.get("maintenance_margin"))
    risk = {
        "cash_pct_of_equity": pct(cash, equity) if cash is not None and equity else None,
        "long_exposure_pct_of_equity": pct(long_market_value, equity) if long_market_value is not None and equity else None,
        "initial_margin_pct_of_equity": pct(initial_margin, equity) if initial_margin is not None and equity else None,
        "maintenance_margin_pct_of_equity": pct(maintenance_margin, equity) if maintenance_margin is not None and equity else None,
        "daytrade_count": account.get("daytrade_count"),
        "pattern_day_trader": account.get("pattern_day_trader"),
        "trading_blocked": account.get("trading_blocked"),
        "transfers_blocked": account.get("transfers_blocked"),
        "account_blocked": account.get("account_blocked"),
    }

    projection = projection_metrics(
        equity=equity,
        lifetime_pnl=lifetime_pnl,
        cashflows=cashflows,
        history=hist,
        trades=trades if isinstance(trades, dict) else {},
    )

    metrics: Dict[str, Any] = {
        "app_version": APP_VERSION,
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
            "initial_margin": initial_margin,
            "maintenance_margin": maintenance_margin,
            "daytrade_count": account.get("daytrade_count"),
            "pattern_day_trader": account.get("pattern_day_trader"),
            "trading_blocked": account.get("trading_blocked"),
            "transfers_blocked": account.get("transfers_blocked"),
            "account_blocked": account.get("account_blocked"),
        },
        "cash_flows": cashflows,
        "profitability": {
            "total_deposits": total_deposits,
            "total_withdrawals": total_withdrawals,
            "net_deposits": net_deposits,
            "net_funding": net_deposits,
            "lifetime_pnl": lifetime_pnl,
            "cash_flow_adjusted_pnl": cash_flow_adjusted_pnl,
            "cash_flow_adjusted_return": cash_flow_adjusted_return,
            "return_on_total_deposits": return_on_total_deposits,
            "unrealized_pl_open_positions": positions["total_unrealized_pl"],
            "notes": "Lifetime P/L is equity plus withdrawals minus deposits. Return on total deposits is usually clearer when the account has had withdrawals/re-funding.",
        },
        "positions": positions,
        "history": hist,
        "trades": trades,
        "risk": risk,
        "projection": projection,
    }

    largest_loser = positions.get("largest_loser") or {}
    largest_winner = positions.get("largest_winner") or {}
    log.info(
        "Metrics refreshed mode=%s equity=%.2f deposits=%.2f withdrawals=%.2f net_funding=%.2f lifetime_pnl=%s return_on_deposits=%s expectancy=%s win_rate=%s closed_trades=%s projection_basis=%s projected_30d=%s positions=%d red=%d green=%d day_pnl=%s largest_loser=%s:%s largest_winner=%s:%s",
        metrics["mode"],
        equity,
        total_deposits,
        total_withdrawals,
        net_deposits,
        fmt_money(lifetime_pnl),
        fmt_pct(return_on_total_deposits),
        fmt_money(trades.get("expectancy")) if isinstance(trades, dict) else "—",
        fmt_pct(trades.get("win_rate")) if isinstance(trades, dict) else "—",
        trades.get("closed_trade_count") if isinstance(trades, dict) else "—",
        projection.get("basis"),
        fmt_money(projection.get("projected_30d_pnl")),
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


def iso_to_epoch(text: Any) -> Optional[float]:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    try:
        seconds = max(0.0, float(seconds))
    except Exception:
        return "—"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} sec"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} min"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:.1f} hr"
    return f"{hours / 24:.1f} days"


def duration_since_iso(text: Any) -> Optional[float]:
    ts = iso_to_epoch(text)
    if ts is None:
        return None
    return max(0.0, time.time() - ts)


def fmt_short_date(text: Any) -> str:
    if not text:
        return ""
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
        return dt.strftime("%b %-d")
    except Exception:
        return str(text)[:10]


def finite_number(value: Any) -> Optional[float]:
    value = to_optional_float(value)
    if value is None or math.isnan(value) or math.isinf(value):
        return None
    return value


def metric_card(title: str, value: str, sub: str = "", help_text: str = "") -> str:
    help_attr = f' data-help="{safe_text(help_text)}"' if help_text else ""
    return f"""
    <div class="card"{help_attr}>
      <div class="card-title">{safe_text(title)}</div>
      <div class="card-value">{safe_text(value)}</div>
      <div class="card-sub">{safe_text(sub)}</div>
    </div>
    """


def line_chart(title: str, points: Sequence[Tuple[Any, Any]], *, value_kind: str = "money", help_text: str = "") -> str:
    clean: List[Tuple[str, float]] = []
    for label, value in points:
        num = finite_number(value)
        if num is None:
            continue
        clean.append((fmt_short_date(label), num))

    if len(clean) < 2:
        return f"""
        <div class="chart-card" data-help="{safe_text(help_text)}">
          <div class="chart-title">{safe_text(title)}</div>
          <div class="empty-chart">Not enough history yet.</div>
        </div>
        """

    width, height = 720, 240
    pad_left, pad_right, pad_top, pad_bottom = 64, 18, 24, 42
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    values = [v for _, v in clean]
    min_v = min(values)
    max_v = max(values)
    if abs(max_v - min_v) < 1e-9:
        max_v += 1
        min_v -= 1
    # Add a tiny vertical buffer so the line does not sit on the border.
    span = max_v - min_v
    min_v -= span * 0.08
    max_v += span * 0.08
    span = max_v - min_v

    coords = []
    n = len(clean)
    for idx, (_, value) in enumerate(clean):
        x = pad_left + (idx / max(1, n - 1)) * plot_w
        y = pad_top + (max_v - value) / span * plot_h
        coords.append((x, y))
    path = " ".join(("M" if idx == 0 else "L") + f" {x:.2f} {y:.2f}" for idx, (x, y) in enumerate(coords))

    def fmt_axis(v: float) -> str:
        if value_kind == "percent":
            return fmt_pct(v)
        return fmt_money(v)

    y_max = safe_text(fmt_axis(max(values)))
    y_min = safe_text(fmt_axis(min(values)))
    first_label = safe_text(clean[0][0])
    last_label = safe_text(clean[-1][0])
    dot_markup = []
    step = max(1, len(coords) // 12)
    for idx, ((x, y), (label, value)) in enumerate(zip(coords, clean)):
        if idx % step != 0 and idx not in {len(coords) - 1}:
            continue
        dot_label = f"{label}: {fmt_axis(value)}"
        dot_markup.append(f'<circle class="chart-dot" cx="{x:.2f}" cy="{y:.2f}" r="3"><title>{safe_text(dot_label)}</title></circle>')

    return f"""
    <div class="chart-card" data-help="{safe_text(help_text)}">
      <div class="chart-title">{safe_text(title)}</div>
      <svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{safe_text(title)}">
        <line class="axis" x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" />
        <line class="axis" x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" />
        <line class="gridline" x1="{pad_left}" y1="{pad_top}" x2="{width - pad_right}" y2="{pad_top}" />
        <text class="axis-label" x="8" y="{pad_top + 5}">{y_max}</text>
        <text class="axis-label" x="8" y="{height - pad_bottom}">{y_min}</text>
        <text class="axis-label" x="{pad_left}" y="{height - 12}">{first_label}</text>
        <text class="axis-label" x="{width - 82}" y="{height - 12}">{last_label}</text>
        <path class="line-path" d="{path}" />
        {''.join(dot_markup)}
      </svg>
    </div>
    """


def bar_chart(title: str, rows: Sequence[Dict[str, Any]], *, help_text: str = "") -> str:
    clean: List[Tuple[str, float]] = []
    for row in rows:
        value = finite_number(row.get("realized_pnl"))
        if value is None:
            continue
        label = f"{row.get('symbol', '')} {fmt_short_date(row.get('closed_at'))}".strip()
        clean.append((label or "trade", value))
    clean = clean[-20:]

    if not clean:
        return f"""
        <div class="chart-card" data-help="{safe_text(help_text)}">
          <div class="chart-title">{safe_text(title)}</div>
          <div class="empty-chart">No closed trades found yet.</div>
        </div>
        """

    width, height = 720, 260
    pad_left, pad_right, pad_top, pad_bottom = 44, 18, 24, 54
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    values = [v for _, v in clean]
    max_abs = max(abs(v) for v in values) or 1.0
    zero_y = pad_top + plot_h / 2
    bar_w = max(8, plot_w / len(clean) * 0.62)
    bars = []
    for idx, (label, value) in enumerate(clean):
        x = pad_left + (idx + 0.5) * plot_w / len(clean) - bar_w / 2
        bar_h = abs(value) / max_abs * (plot_h / 2 - 8)
        if value >= 0:
            y = zero_y - bar_h
            cls = "bar-pos"
        else:
            y = zero_y
            cls = "bar-neg"
        bars.append(f'<rect class="{cls}" x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" rx="4"><title>{safe_text(label)}: {safe_text(fmt_money(value))}</title></rect>')

    return f"""
    <div class="chart-card" data-help="{safe_text(help_text)}">
      <div class="chart-title">{safe_text(title)}</div>
      <svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{safe_text(title)}">
        <line class="axis" x1="{pad_left}" y1="{zero_y:.2f}" x2="{width - pad_right}" y2="{zero_y:.2f}" />
        <text class="axis-label" x="8" y="{pad_top + 8}">{safe_text(fmt_money(max_abs))}</text>
        <text class="axis-label" x="8" y="{height - pad_bottom + 4}">-{safe_text(fmt_money(max_abs))[1:] if fmt_money(max_abs).startswith('$') else safe_text(fmt_money(max_abs))}</text>
        {''.join(bars)}
        <text class="axis-label" x="{pad_left}" y="{height - 16}">Oldest</text>
        <text class="axis-label" x="{width - 72}" y="{height - 16}">Newest</text>
      </svg>
    </div>
    """


def compact_status_flags(account: Dict[str, Any], risk: Dict[str, Any]) -> str:
    flags = []
    if risk.get("pattern_day_trader") is True:
        flags.append("PDT")
    if risk.get("trading_blocked") is True:
        flags.append("Trading blocked")
    if risk.get("transfers_blocked") is True:
        flags.append("Transfers blocked")
    if risk.get("account_blocked") is True:
        flags.append("Account blocked")
    if not flags:
        return "OK"
    return ", ".join(flags)


def render_dashboard(metrics: Dict[str, Any], state: Dict[str, Any], cfg: Config) -> str:
    account = metrics.get("account", {})
    profitability = metrics.get("profitability", {})
    cash_flows = metrics.get("cash_flows", {})
    positions = metrics.get("positions", {})
    history = metrics.get("history", {})
    trades = metrics.get("trades", {})
    risk = metrics.get("risk", {})
    projection = metrics.get("projection", {})

    largest_loser = positions.get("largest_loser") or {}
    largest_winner = positions.get("largest_winner") or {}

    warning = ""
    if not cfg.dashboard_token:
        warning = "<div class='warning'>DASHBOARD_TOKEN is not set. This Railway web dashboard is unprotected.</div>"
    if cash_flows.get("source", "").startswith("account_activities") and to_float(cash_flows.get("deposits"), 0.0) <= 0:
        warning += "<div class='warning'>No deposit activity was found. Check CASH_FLOW_ACTIVITY_TYPES / ACTIVITY_AFTER, or set NET_DEPOSITS_OVERRIDE if the funding number looks wrong.</div>"
    if trades.get("source") == "error":
        warning += f"<div class='warning'>Trade expectancy could not be calculated: {safe_text(trades.get('error'))}</div>"
    elif trades.get("closed_trade_count", 0) == 0:
        warning += "<div class='warning'>No closed trades were found from FILL activities yet, so expectancy metrics are blank. Check TRADE_ACTIVITY_AFTER if this looks wrong.</div>"

    exposure_sub = f"Cash: {fmt_pct(risk.get('cash_pct_of_equity'))} · Long: {fmt_pct(risk.get('long_exposure_pct_of_equity'))}"
    margin_sub = f"Initial: {fmt_pct(risk.get('initial_margin_pct_of_equity'))} · Maint: {fmt_pct(risk.get('maintenance_margin_pct_of_equity'))}"
    cards = "\n".join(
        [
            metric_card("Equity", fmt_money(account.get("equity")), f"Mode: {metrics.get('mode', 'unknown')}", "Current Alpaca account equity / portfolio value."),
            metric_card("Lifetime P/L", fmt_money(profitability.get("lifetime_pnl")), "Equity + withdrawals - deposits", "Cash-flow-adjusted profit/loss. This answers whether the account is above or below all money put in, after withdrawals."),
            metric_card("Return on deposits", fmt_pct(profitability.get("return_on_total_deposits")), "Lifetime P/L / total deposits", "Return measured against all deposits ever put into the account. Better for accounts that were drawn down and refilled."),
            metric_card("Expectancy / trade", fmt_money(trades.get("expectancy")), f"Closed trades: {trades.get('closed_trade_count', 0)}", "Approximate average realized P/L per closed sell event, FIFO-matched from Alpaca FILL activities."),
            metric_card("Win rate", fmt_pct(trades.get("win_rate")), f"Wins/Losses: {trades.get('winning_trades', 0)} / {trades.get('losing_trades', 0)}", "Share of closed trades with positive realized P/L. A high win rate can still lose money if losses are much larger than wins."),
            metric_card("Profit factor", fmt_num(trades.get("profit_factor"), 2), f"Gross profit / gross loss", "Above 1 means closed winners are larger than closed losers in aggregate. Blank means there are no losses yet or not enough data."),
            metric_card("Realized P/L", fmt_money(trades.get("realized_pnl")), f"Since {trades.get('trade_activity_after', 'start')}", "Approximate realized P/L reconstructed from fill activities. This is not tax accounting."),
            metric_card("Avg winner / loser", f"{fmt_money(trades.get('avg_win'))} / {fmt_money(trades.get('avg_loss'))}", f"Avg hold: {fmt_num(trades.get('avg_hold_days'), 1)} days", "Average winner and average absolute loser from closed trades. Compare this with win rate to judge strategy quality."),
            metric_card("Projected 30d P/L", fmt_money(projection.get("projected_30d_pnl")), f"Basis: {projection.get('basis', 'unavailable')}", "Run-rate estimate: what the next 30 days would look like if the selected historical rate continued. This is not a forecast or guarantee."),
            metric_card("Projected 90d return", fmt_pct(projection.get("projected_90d_return")), f"90d P/L: {fmt_money(projection.get('projected_90d_pnl'))}", "Projected 90-day return on current equity using the same run-rate basis as the 30-day projection."),
            metric_card("Trade pace", f"{fmt_num(projection.get('closed_trades_per_week'), 1)} / week", f"30d projected closes: {fmt_num(projection.get('projected_closed_trades_30d'), 1)}", "How frequently the strategy has been closing trades, based on FIFO-matched closed fills."),
            metric_card("Break-even win rate", fmt_pct(projection.get("breakeven_win_rate")), f"Edge gap: {fmt_pct(projection.get('edge_gap'))}", "Win rate needed to break even given the current average winner and average loser. Edge gap is actual win rate minus break-even win rate."),
            metric_card("Recovery needed", fmt_money(projection.get("recovery_needed")), f"Return needed: {fmt_pct(projection.get('recovery_return_needed'))}", "How much the account needs to gain to get back to the recent equity high-water mark."),
            metric_card("Days to recover", fmt_num(projection.get("days_to_recover_high_water"), 1), f"Using projection basis", "Estimated days to regain the recent high-water mark if the selected positive run-rate continued. Blank if there is no drawdown or the run-rate is not positive."),
            metric_card("Net funding", fmt_money(profitability.get("net_funding")), f"Source: {cash_flows.get('source')}", "Deposits minus withdrawals from deduped Alpaca cash-flow activities, or NET_DEPOSITS_OVERRIDE if set."),
            metric_card("Total deposits", fmt_money(profitability.get("total_deposits")), f"Since {cash_flows.get('activity_after', 'start')}", "Total external cash found entering the account."),
            metric_card("Total withdrawals", fmt_money(profitability.get("total_withdrawals")), f"Cash-flow events: {cash_flows.get('activity_count', 0)}", "Total external cash found leaving the account."),
            metric_card("Day P/L", fmt_money(account.get("day_pnl")), fmt_pct(account.get("day_return")), "Current equity minus Alpaca last_equity. Useful for today's move, but not enough to prove strategy edge."),
            metric_card("Drawdown", fmt_pct(history.get("current_drawdown")), f"Max: {fmt_pct(history.get('max_drawdown'))}", "Current and worst drop from the recent equity high-water mark in the portfolio-history window."),
            metric_card("Exposure", fmt_pct(risk.get("long_exposure_pct_of_equity")), exposure_sub, "Long market value divided by equity. Shows how invested the account is without listing top positions."),
            metric_card("Margin use", fmt_pct(risk.get("initial_margin_pct_of_equity")), margin_sub, "Initial and maintenance margin divided by equity. Higher values mean less room for volatility."),
            metric_card("Buying power", fmt_money(account.get("buying_power")), f"Reg-T: {fmt_money(account.get('regt_buying_power'))}", "Alpaca buying power and Reg-T buying power."),
            metric_card("Open unrealized P/L", fmt_money(profitability.get("unrealized_pl_open_positions")), f"Positions: {positions.get('positions_count', 0)}", "Open-position P/L that has not been realized through sells yet."),
            metric_card("Red / Green", f"{positions.get('red_positions', 0)} / {positions.get('green_positions', 0)}", "Open positions", "Count of open positions with negative versus positive unrealized P/L."),
            metric_card("Largest loser / winner", f"{safe_text(largest_loser.get('symbol') or '—')} / {safe_text(largest_winner.get('symbol') or '—')}", f"{fmt_pct(largest_loser.get('unrealized_plpc')) if largest_loser else '—'} / {fmt_pct(largest_winner.get('unrealized_plpc')) if largest_winner else '—'}", "Single worst and best open position by unrealized percentage. Kept as a quick warning signal, not a full positions table."),
        ]
    )

    charts = "\n".join(
        [
            line_chart("Equity trend", history.get("equity_series") or [], value_kind="money", help_text="Recent account equity from Alpaca portfolio history."),
            line_chart("Drawdown trend", history.get("drawdown_series") or [], value_kind="percent", help_text="Drop from the recent equity high-water mark. Lower means deeper drawdown."),
            bar_chart("Recent closed-trade P/L", trades.get("recent_closed_trades") or [], help_text="Approximate realized P/L for the most recent closed trades, FIFO-matched from FILL activities."),
            line_chart("Projected equity path", [(row.get("date"), row.get("projected_equity")) for row in (projection.get("projected_equity_series") or [])], value_kind="money", help_text="Run-rate projection from the selected basis. This visualizes continuation math, not a promise of future returns."),
        ]
    )

    json_url = "/api/metrics"
    refresh_url = "/refresh"
    trades_url = "/api/trades"
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
        :root {{ color-scheme: dark; --bg: #101114; --panel: #181a20; --border: #2b2e37; --muted: #a7a7a7; --text: #eeeeee; --accent: #9cc2ff; --good: #7ddc91; --bad: #ff8a8a; }}
        body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: var(--bg); color: var(--text); }}
        header {{ padding: 28px 28px 8px; }}
        h1 {{ margin: 0 0 6px; font-size: 28px; }}
        h2 {{ margin: 26px 28px 8px; font-size: 20px; }}
        .muted {{ color: var(--muted); font-size: 14px; }}
        .warning {{ margin: 16px 28px; padding: 12px 14px; border: 1px solid #7c5b00; background: #2a2107; color: #ffd782; border-radius: 12px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; padding: 20px 28px; }}
        .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; padding: 20px 28px; }}
        .card, .chart-card {{ position: relative; background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 16px; }}
        .card[data-help]::after, .chart-card[data-help]::after {{ content: attr(data-help); position: absolute; left: 14px; right: 14px; bottom: calc(100% + 8px); background: #07080a; border: 1px solid #3a3e49; color: var(--text); padding: 10px 12px; border-radius: 10px; font-size: 13px; line-height: 1.35; opacity: 0; transform: translateY(4px); transition: opacity .12s ease, transform .12s ease; pointer-events: none; z-index: 10; box-shadow: 0 10px 28px rgba(0,0,0,.35); }}
        .card[data-help]:hover::after, .chart-card[data-help]:hover::after {{ opacity: 1; transform: translateY(0); }}
        .card-title {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: .04em; }}
        .card-value {{ margin-top: 8px; font-size: 24px; font-weight: 700; }}
        .card-sub {{ margin-top: 4px; color: var(--muted); font-size: 13px; }}
        .chart-title {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 8px; }}
        .chart {{ width: 100%; height: auto; display: block; }}
        .axis {{ stroke: #454a55; stroke-width: 1; }}
        .gridline {{ stroke: #2b2e37; stroke-width: 1; }}
        .axis-label {{ fill: var(--muted); font-size: 12px; }}
        .line-path {{ fill: none; stroke: var(--accent); stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }}
        .chart-dot {{ fill: var(--accent); }}
        .bar-pos {{ fill: var(--good); opacity: .86; }}
        .bar-neg {{ fill: var(--bad); opacity: .86; }}
        .empty-chart {{ min-height: 180px; display: grid; place-items: center; color: var(--muted); border: 1px dashed var(--border); border-radius: 12px; }}
        main {{ padding: 0 28px 40px; }}
        .footer {{ margin-top: 28px; color: var(--muted); font-size: 13px; }}
        a {{ color: var(--accent); }}
      </style>
    </head>
    <body>
      <header>
        <h1>Alpaca Metrics Bot</h1>
        <div class="muted">Version: {safe_text(metrics.get('app_version'))}. Generated at {safe_text(metrics.get('generated_at'))}. Last refresh finished {safe_text(state.get('last_refresh_finished_at'))}. Auto-refreshes every {int(max(10, cfg.web_refresh_seconds))} seconds.</div>
      </header>
      {warning}
      <h2>Scoreboard</h2>
      <section class="grid">{cards}</section>
      <h2>Visuals</h2>
      <section class="charts">{charts}</section>
      <main>
        <div class="footer">
          <p>JSON: <a href="{json_url}">{json_url}</a> · Trades: <a href="{trades_url}">{trades_url}</a> · Force refresh: <a href="{refresh_url}">{refresh_url}</a></p>
          <p>{safe_text(token_note)}</p>
          <p>Lifetime P/L uses: current equity + total withdrawals - total deposits. The funding calculation is deduped and excludes fills, dividends, interest, fees, and market P/L.</p>
          <p>Expectancy is approximate FIFO matching from Alpaca FILL activities since {safe_text(trades.get('trade_activity_after', 'start'))}. It is meant for strategy health, not tax accounting.</p>
          <p>Projected returns are simple run-rate estimates, not guarantees. Projection basis: {safe_text(projection.get('basis', 'unavailable'))}. Cash flows can distort recent equity-trend projections.</p>
          <p>Cash-flow activity types checked: {safe_text(','.join(cash_flows.get('activity_types_checked') or []))}. Raw activities: {safe_text(cash_flows.get('raw_activity_count'))}; duplicates skipped: {safe_text(cash_flows.get('duplicate_activity_count'))}; ignored: {safe_text(cash_flows.get('ignored_activity_count'))}.</p>
          <p>Set NET_DEPOSITS_OVERRIDE only if Alpaca activities still do not match your actual funding history. Use TRADE_ACTIVITY_AFTER if expectancy pulls too much old fill history.</p>
          <p>Version: {safe_text(metrics.get('app_version'))}.</p>
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




@app.get("/api/cashflows")
def api_cashflows(request: Request) -> JSONResponse:
    config = cfg()
    require_dashboard_token(request, config)
    metrics = get_cached_or_refresh(config)
    return JSONResponse(metrics.get("cash_flows", {}))


@app.get("/api/trades")
def api_trades(request: Request) -> JSONResponse:
    config = cfg()
    require_dashboard_token(request, config)
    metrics = get_cached_or_refresh(config)
    return JSONResponse(metrics.get("trades", {}))


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
        "version": APP_VERSION,
        **state,
    }


@app.get("/version")
def version() -> Dict[str, str]:
    return {"service": "alpaca-metrics-bot", "version": APP_VERSION}


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots_txt() -> str:
    return "User-agent: *\nDisallow: /\n"
