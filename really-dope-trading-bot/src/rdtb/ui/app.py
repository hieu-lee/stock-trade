from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from rdtb.config import get_default_config
from rdtb.portfolio.transactions import replay_transactions
from rdtb.service.pipeline import build_holdings_template, generate_daily_decisions, train_system
from rdtb.utils import read_json

st.set_page_config(page_title="Really Dope Trading Bot", layout="wide")

config = get_default_config(Path(__file__).resolve().parents[3])


def _load_symbol_catalog() -> pd.DataFrame:
    catalog = pd.DataFrame({"symbol": list(config.symbols)})
    if config.company_metadata_path.exists():
        metadata = pd.read_parquet(config.company_metadata_path).copy()
        metadata["sector"] = metadata.get("industry_level3", pd.Series(index=metadata.index, dtype=object)).fillna(
            metadata.get("industry_level2", pd.Series(index=metadata.index, dtype=object))
        )
        catalog = catalog.merge(metadata[["symbol", "sector"]], on="symbol", how="left")
    else:
        catalog["sector"] = ""
    catalog["sector"] = catalog["sector"].fillna("").astype(str)
    catalog["label"] = catalog.apply(
        lambda row: f"{row['symbol']} · {row['sector']}" if row["sector"].strip() else str(row["symbol"]),
        axis=1,
    )
    return catalog.sort_values("symbol").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _load_trading_calendar_and_latest_date(project_root: str) -> tuple[list[pd.Timestamp], pd.Timestamp]:
    prices_path = Path(project_root) / "data" / "raw" / "prices.parquet"
    if prices_path.exists():
        dates = pd.to_datetime(pd.read_parquet(prices_path, columns=["date"])["date"]).drop_duplicates().sort_values()
        calendar = [pd.Timestamp(date).normalize() for date in dates.tolist()]
        latest = calendar[-1] if calendar else pd.Timestamp.today().normalize()
        return calendar, latest
    latest = pd.Timestamp.today().normalize()
    calendar = [pd.Timestamp(date).normalize() for date in pd.bdate_range(end=latest, periods=260)]
    return calendar, latest


def _normalize_holdings(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return build_holdings_template(config)
    holdings = frame.copy()
    for column in ["symbol", "buy_date"]:
        if column not in holdings.columns:
            holdings[column] = ""
    for column in ["quantity", "sellable_quantity", "avg_cost"]:
        if column not in holdings.columns:
            holdings[column] = 0
    holdings["symbol"] = holdings["symbol"].fillna("").astype(str).str.upper().str.strip()
    holdings = holdings.loc[holdings["symbol"] != ""].reset_index(drop=True)
    holdings["quantity"] = pd.to_numeric(holdings["quantity"], errors="coerce").fillna(0).astype(int)
    holdings["sellable_quantity"] = pd.to_numeric(holdings["sellable_quantity"], errors="coerce").fillna(0).astype(int)
    holdings["avg_cost"] = pd.to_numeric(holdings["avg_cost"], errors="coerce").fillna(0.0)
    holdings["buy_date"] = holdings["buy_date"].fillna("").astype(str).str.strip()
    holdings["sellable_quantity"] = holdings[["quantity", "sellable_quantity"]].min(axis=1)
    holdings = holdings.loc[holdings["quantity"] > 0].reset_index(drop=True)
    return holdings.sort_values("symbol").reset_index(drop=True)


def _upsert_holding(
    holdings: pd.DataFrame,
    symbol: str,
    quantity: int,
    sellable_quantity: int,
    avg_cost: float,
    buy_date: str,
) -> pd.DataFrame:
    frame = _normalize_holdings(holdings)
    updated = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "quantity": int(quantity),
                "sellable_quantity": int(min(quantity, sellable_quantity)),
                "avg_cost": float(avg_cost),
                "buy_date": buy_date.strip(),
            }
        ]
    )
    frame = frame.loc[frame["symbol"] != symbol].reset_index(drop=True)
    frame = pd.concat([frame, updated], ignore_index=True)
    return _normalize_holdings(frame)


def _remove_holding(holdings: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return _normalize_holdings(holdings.loc[holdings["symbol"] != symbol])


def _empty_transactions() -> pd.DataFrame:
    return pd.DataFrame(columns=["transaction_id", "date", "symbol", "action", "quantity", "price"])


def _normalize_transactions(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_transactions()
    transactions = frame.copy()
    for column in ["transaction_id", "date", "symbol", "action", "quantity", "price"]:
        if column not in transactions.columns:
            transactions[column] = ""
    transactions["transaction_id"] = pd.to_numeric(transactions["transaction_id"], errors="coerce")
    if transactions["transaction_id"].isna().all():
        transactions["transaction_id"] = range(1, len(transactions) + 1)
    transactions["transaction_id"] = transactions["transaction_id"].fillna(0).astype(int)
    transactions["date"] = pd.to_datetime(transactions["date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    transactions["symbol"] = transactions["symbol"].fillna("").astype(str).str.upper().str.strip()
    transactions["action"] = transactions["action"].fillna("").astype(str).str.upper().str.strip()
    transactions["quantity"] = pd.to_numeric(transactions["quantity"], errors="coerce").fillna(0).astype(int)
    transactions["price"] = pd.to_numeric(transactions["price"], errors="coerce").fillna(0.0)
    transactions = transactions.loc[
        transactions["symbol"].ne("")
        & transactions["action"].isin(["BUY", "SELL"])
        & (transactions["quantity"] > 0)
        & (transactions["price"] > 0)
        & transactions["date"].ne("")
    ].copy()
    return transactions.sort_values(["date", "transaction_id"]).reset_index(drop=True)


def _append_transaction(
    transactions: pd.DataFrame,
    date: str,
    symbol: str,
    action: str,
    quantity: int,
    price: float,
) -> pd.DataFrame:
    frame = _normalize_transactions(transactions)
    next_id = int(frame["transaction_id"].max()) + 1 if not frame.empty else 1
    row = pd.DataFrame(
        [
            {
                "transaction_id": next_id,
                "date": date,
                "symbol": symbol,
                "action": action,
                "quantity": int(quantity),
                "price": float(price),
            }
        ]
    )
    return _normalize_transactions(pd.concat([frame, row], ignore_index=True))


def _remove_transaction(transactions: pd.DataFrame, transaction_id: int) -> pd.DataFrame:
    frame = _normalize_transactions(transactions)
    return frame.loc[frame["transaction_id"] != transaction_id].reset_index(drop=True)


def _transaction_label(row: pd.Series) -> str:
    return f"#{int(row['transaction_id'])} · {row['date']} · {row['action']} {row['symbol']} · {int(row['quantity'])} @ {float(row['price']):,.2f}"


def _build_transaction_preview(transactions: pd.DataFrame, starting_cash: float):
    calendar, latest_market_date = _load_trading_calendar_and_latest_date(str(config.project_dir))
    if transactions.empty:
        return None, latest_market_date
    preview = replay_transactions(
        transactions=transactions,
        starting_cash=float(starting_cash),
        config=config,
        as_of_date=latest_market_date,
        trading_calendar=calendar,
    )
    return preview, latest_market_date


def _sync_saved_recommendation() -> None:
    if not config.latest_decision_path.exists():
        return
    latest_mtime = config.latest_decision_path.stat().st_mtime
    current_mtime = st.session_state.get("rdtb_result_mtime")
    if current_mtime == latest_mtime:
        return
    st.session_state.rdtb_result = read_json(config.latest_decision_path)
    st.session_state.rdtb_result_mtime = latest_mtime


def _clear_displayed_recommendation() -> None:
    st.session_state.pop("rdtb_result", None)
    st.session_state.pop("rdtb_result_mtime", None)


st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(24,32,53,0.96), rgba(14,98,81,0.92));
        color: white;
        margin-bottom: 1rem;
    }
    .card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        background: rgba(17,25,40,0.75);
    }
    .rule-chip {
        display: inline-block;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        margin: 0.25rem 0.35rem 0.25rem 0;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1 style="margin: 0;">Really Dope Trading Bot</h1>
      <p style="margin-top: 0.5rem;">
        VN60 daily decision engine with richer fundamentals, flow-aware scoring,
        settlement-aware execution, and one-click recommendations for the next session.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

summary = read_json(config.training_summary_path) if config.training_summary_path.exists() else None
symbol_catalog = _load_symbol_catalog()

if "rdtb_holdings" not in st.session_state:
    st.session_state.rdtb_holdings = build_holdings_template(config)
st.session_state.rdtb_holdings = _normalize_holdings(st.session_state.rdtb_holdings)
if "rdtb_transactions" not in st.session_state:
    st.session_state.rdtb_transactions = _empty_transactions()
st.session_state.rdtb_transactions = _normalize_transactions(st.session_state.rdtb_transactions)
_sync_saved_recommendation()

metrics_cols = st.columns(5)
if summary:
    metrics = summary["final_metrics"]
    metrics_cols[0].metric("Final Test Annualized", f"{metrics['annualized_return']:.2%}")
    metrics_cols[1].metric("Final Test Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    metrics_cols[2].metric("2024 Return", f"{metrics['yearly_returns'].get('2024', 0.0):.2%}")
    metrics_cols[3].metric("2025 Return", f"{metrics['yearly_returns'].get('2025', 0.0):.2%}")
    metrics_cols[4].metric("Deployable", "Yes" if summary["deployable"] else "No")
else:
    for column, label in zip(
        metrics_cols,
        ["Final Test Annualized", "Final Test Max Drawdown", "2024 Return", "2025 Return", "Deployable"],
    ):
        column.metric(label, "N/A")

if summary and summary.get("selection_note"):
    st.caption(summary["selection_note"])

top_left, top_right = st.columns([1.15, 0.85])

with top_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Train Or Refresh")
    refresh_train = st.checkbox("Refresh market data before training", value=False)
    if st.button("Run strict training pipeline", width="stretch"):
        with st.spinner("Training the VN60 system..."):
            summary = train_system(config=config, refresh_data=refresh_train)
        _clear_displayed_recommendation()
        st.success("Training pipeline completed.")
        st.info("Run `Get daily decisions` to refresh the table with the newly trained model.")
        st.json(summary)
    st.markdown("</div>", unsafe_allow_html=True)

with top_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live Rules")
    st.markdown(
        """
        <span class="rule-chip">Buy fee 0.03%</span>
        <span class="rule-chip">Sell fee 0.13%</span>
        <span class="rule-chip">Buy settles T+3</span>
        <span class="rule-chip">Sell cash settles T+2</span>
        <span class="rule-chip">Unsettled shares are not sellable</span>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Use the search form below to add positions fast. If some shares are still settling, set `sellable_quantity` below `quantity`.")
    st.markdown("</div>", unsafe_allow_html=True)

portfolio_col, action_col = st.columns([1.2, 1.0])

with portfolio_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    input_mode = st.radio(
        "Portfolio input mode",
        options=["Transactions", "Manual holdings"],
        horizontal=True,
        index=0 if not st.session_state.rdtb_transactions.empty else 1,
    )
    symbol_query = st.text_input("Search symbol", placeholder="Type FPT, VCB, HPG, ...")
    query = symbol_query.strip().lower()
    if query:
        filtered_catalog = symbol_catalog.loc[
            symbol_catalog["symbol"].str.lower().str.contains(query)
            | symbol_catalog["label"].str.lower().str.contains(query)
        ].reset_index(drop=True)
    else:
        filtered_catalog = symbol_catalog

    selected_symbol = None
    if filtered_catalog.empty:
        st.warning("No symbol matches the current search query.")
    else:
        selected_symbol = st.selectbox(
            "Matching symbols",
            options=filtered_catalog["symbol"].tolist(),
            index=0,
            format_func=lambda value: filtered_catalog.loc[filtered_catalog["symbol"] == value, "label"].iloc[0],
            key=f"matching_symbol_{query or 'all'}",
        )

    symbol_key = selected_symbol or "none"
    calendar, latest_market_date = _load_trading_calendar_and_latest_date(str(config.project_dir))

    if input_mode == "Transactions":
        st.subheader("Search And Add Transactions")
        with st.form("add_transaction_form"):
            tx_action = st.selectbox("Action", ["BUY", "SELL"], key=f"tx_action_{symbol_key}")
            tx_quantity = st.number_input("Quantity", min_value=0, step=100, value=0, key=f"tx_quantity_{symbol_key}")
            tx_price = st.number_input("Price", min_value=0.0, step=0.1, value=0.0, key=f"tx_price_{symbol_key}")
            tx_date = st.date_input("Transaction date", value=latest_market_date.date(), key=f"tx_date_{symbol_key}")
            tx_submit = st.form_submit_button("Add transaction", width="stretch")

        if tx_submit:
            if not selected_symbol:
                st.warning("Choose a symbol first.")
            elif tx_quantity <= 0:
                st.warning("Quantity must be greater than zero.")
            elif tx_price <= 0:
                st.warning("Price must be greater than zero.")
            else:
                st.session_state.rdtb_transactions = _append_transaction(
                    st.session_state.rdtb_transactions,
                    date=pd.Timestamp(tx_date).strftime("%Y-%m-%d"),
                    symbol=selected_symbol,
                    action=tx_action,
                    quantity=int(tx_quantity),
                    price=float(tx_price),
                )
                st.success(f"Added `{tx_action} {selected_symbol}` transaction.")
    else:
        st.subheader("Search And Add Positions")
        existing_row = None
        if selected_symbol and not st.session_state.rdtb_holdings.empty:
            existing = st.session_state.rdtb_holdings.loc[st.session_state.rdtb_holdings["symbol"] == selected_symbol]
            if not existing.empty:
                existing_row = existing.iloc[0]

        default_quantity = int(existing_row["quantity"]) if existing_row is not None else 0
        default_sellable = int(existing_row["sellable_quantity"]) if existing_row is not None else default_quantity
        default_avg_cost = float(existing_row["avg_cost"]) if existing_row is not None else 0.0
        default_buy_date = str(existing_row["buy_date"]) if existing_row is not None else ""
        default_settled = default_sellable >= default_quantity if default_quantity > 0 else True

        with st.form("add_position_form"):
            add_quantity = st.number_input(
                "Quantity",
                min_value=0,
                step=100,
                value=default_quantity,
                key=f"quantity_{symbol_key}",
            )
            add_avg_cost = st.number_input(
                "Average cost",
                min_value=0.0,
                step=0.1,
                value=default_avg_cost,
                key=f"avg_cost_{symbol_key}",
            )
            settled_toggle = st.checkbox(
                "All shares already settled",
                value=default_settled,
                key=f"settled_{symbol_key}",
            )
            unsettled_sellable = st.number_input(
                "Sellable quantity",
                min_value=0,
                step=100,
                value=default_sellable if not default_settled else default_quantity,
                disabled=settled_toggle,
                key=f"sellable_{symbol_key}",
            )
            add_buy_date = st.text_input("Buy date (YYYY-MM-DD)", value=default_buy_date, key=f"buy_date_{symbol_key}")
            add_submit = st.form_submit_button("Add / Update position", width="stretch")

        if add_submit:
            if not selected_symbol:
                st.warning("Choose a symbol first.")
            elif add_quantity <= 0:
                st.warning("Quantity must be greater than zero.")
            else:
                final_sellable = int(add_quantity if settled_toggle else min(add_quantity, unsettled_sellable))
                st.session_state.rdtb_holdings = _upsert_holding(
                    st.session_state.rdtb_holdings,
                    symbol=selected_symbol,
                    quantity=int(add_quantity),
                    sellable_quantity=final_sellable,
                    avg_cost=float(add_avg_cost),
                    buy_date=add_buy_date,
                )
                st.success(f"Saved `{selected_symbol}` to the portfolio table.")
    st.markdown("</div>", unsafe_allow_html=True)

with action_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Generate Decisions")
    cash_label = "Starting cash before transactions" if input_mode == "Transactions" else "Available cash"
    cash = st.number_input(cash_label, min_value=0.0, value=1_000_000_000.0, step=50_000_000.0)
    refresh_decisions = st.checkbox("Refresh market data before creating decisions", value=True)
    if st.button("Get daily decisions", width="stretch"):
        with st.spinner("Generating next-session decisions..."):
            recommendation = generate_daily_decisions(
                cash=cash,
                holdings=st.session_state.rdtb_holdings if input_mode == "Manual holdings" else None,
                config=config,
                refresh_data=refresh_decisions,
                transactions=st.session_state.rdtb_transactions if input_mode == "Transactions" else None,
                starting_cash=cash if input_mode == "Transactions" else None,
            )
        st.session_state.rdtb_result = recommendation
        st.session_state.rdtb_result_mtime = (
            config.latest_decision_path.stat().st_mtime if config.latest_decision_path.exists() else None
        )
    st.markdown("</div>", unsafe_allow_html=True)

if input_mode == "Transactions":
    st.subheader("Transaction Ledger")
    tx_left, tx_right = st.columns([1.0, 0.45])
    with tx_left:
        tx_filter = st.text_input("Filter transactions", placeholder="Optional symbol filter")
    with tx_right:
        tx_options = (
            [""] + [_transaction_label(row) for _, row in st.session_state.rdtb_transactions.iterrows()]
            if not st.session_state.rdtb_transactions.empty
            else [""]
        )
        remove_transaction_label = st.selectbox("Remove transaction", options=tx_options, index=0)
        if st.button("Remove selected transaction", disabled=remove_transaction_label == "", width="stretch"):
            tx_id = int(remove_transaction_label.split("·")[0].replace("#", "").strip())
            st.session_state.rdtb_transactions = _remove_transaction(st.session_state.rdtb_transactions, tx_id)
            st.success(f"Removed transaction `{remove_transaction_label}`.")

    tx_frame = st.session_state.rdtb_transactions.copy()
    if tx_filter.strip():
        tx_frame = tx_frame.loc[tx_frame["symbol"].str.contains(tx_filter.strip().upper(), na=False)]
    edited_transactions = st.data_editor(
        tx_frame,
        hide_index=True,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "transaction_id": st.column_config.NumberColumn("ID", disabled=True),
            "date": st.column_config.TextColumn("Date (YYYY-MM-DD)"),
            "symbol": st.column_config.TextColumn("Symbol"),
            "action": st.column_config.SelectboxColumn("Action", options=["BUY", "SELL"]),
            "quantity": st.column_config.NumberColumn("Quantity", min_value=0, step=100),
            "price": st.column_config.NumberColumn("Price", min_value=0.0, step=0.1),
        },
    )
    if not tx_filter.strip():
        st.session_state.rdtb_transactions = _normalize_transactions(edited_transactions)
    else:
        preserved = st.session_state.rdtb_transactions.loc[
            ~st.session_state.rdtb_transactions["transaction_id"].isin(tx_frame["transaction_id"])
        ].copy()
        st.session_state.rdtb_transactions = _normalize_transactions(pd.concat([preserved, edited_transactions], ignore_index=True))

    preview, preview_date = _build_transaction_preview(st.session_state.rdtb_transactions, starting_cash=float(cash))
    st.subheader(f"Derived Portfolio As Of {preview_date.strftime('%Y-%m-%d')}")
    if preview is None:
        st.info("Add at least one transaction to build the portfolio automatically.")
    else:
        preview_cols = st.columns(4)
        preview_cols[0].metric("Available settled cash", f"{preview.available_cash:,.2f}")
        preview_cols[1].metric("Pending cash", f"{preview.pending_cash_total:,.2f}")
        preview_cols[2].metric("Pending buy quantity", f"{preview.pending_buy_quantity:,}")
        preview_cols[3].metric("Active holdings", f"{len(preview.holdings)}")
        if preview.notes:
            st.warning("\n".join(preview.notes))
        st.dataframe(preview.holdings, width="stretch")
else:
    st.subheader("Current Portfolio Table")
    table_left, table_right = st.columns([1.0, 0.45])
    with table_left:
        holdings_filter = st.text_input("Filter current positions", placeholder="Optional symbol filter")
    with table_right:
        current_symbols = st.session_state.rdtb_holdings["symbol"].tolist()
        remove_symbol = st.selectbox("Remove symbol", options=[""] + current_symbols, index=0)
        if st.button("Remove selected symbol", disabled=remove_symbol == "", width="stretch"):
            st.session_state.rdtb_holdings = _remove_holding(st.session_state.rdtb_holdings, remove_symbol)
            st.success(f"Removed `{remove_symbol}` from the portfolio table.")

    table_frame = st.session_state.rdtb_holdings.copy()
    if holdings_filter.strip():
        table_frame = table_frame.loc[table_frame["symbol"].str.contains(holdings_filter.strip().upper(), na=False)]
    edited_holdings = st.data_editor(
        table_frame,
        hide_index=True,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "symbol": st.column_config.TextColumn("Symbol"),
            "quantity": st.column_config.NumberColumn("Quantity", min_value=0, step=100),
            "sellable_quantity": st.column_config.NumberColumn("Sellable Quantity", min_value=0, step=100),
            "avg_cost": st.column_config.NumberColumn("Avg Cost", min_value=0.0, step=0.1),
            "buy_date": st.column_config.TextColumn("Buy Date (YYYY-MM-DD)"),
        },
    )
    if not holdings_filter.strip():
        st.session_state.rdtb_holdings = _normalize_holdings(edited_holdings)
    else:
        preserved = st.session_state.rdtb_holdings.loc[
            ~st.session_state.rdtb_holdings["symbol"].isin(table_frame["symbol"])
        ].copy()
        st.session_state.rdtb_holdings = _normalize_holdings(pd.concat([preserved, edited_holdings], ignore_index=True))

if "rdtb_result" in st.session_state:
    result = st.session_state.rdtb_result
    st.subheader(f"Decision date: {pd.Timestamp(result['date']).strftime('%Y-%m-%d')}")
    contract = result.get("contract_summary", {})
    note_col, contract_col = st.columns([1, 1])
    with note_col:
        st.markdown("### Notes")
        if result["notes"]:
            for note in result["notes"]:
                st.write(f"- {note}")
        else:
            st.write("No special portfolio notes.")
    with contract_col:
        st.markdown("### Contract Snapshot")
        st.write(f"Deployable artifact: {'Yes' if result['deployable'] else 'No'}")
        if contract:
            st.write(f"Final annualized return: {contract.get('annualized_return', 0.0):.2%}")
            st.write(f"Final max drawdown: {contract.get('max_drawdown', 0.0):.2%}")
        if result.get("transaction_summary"):
            tx_summary = result["transaction_summary"]
            st.write(f"Settled cash used for decisions: {tx_summary['available_cash']:,.2f}")
            st.write(f"Pending cash: {tx_summary['pending_cash_total']:,.2f}")
    st.markdown("### Daily actions")
    st.dataframe(pd.DataFrame(result["actions"]), width="stretch")
    st.markdown("### Position status")
    st.dataframe(pd.DataFrame(result["position_status"]), width="stretch")
    st.markdown("### Strongest ranked names")
    st.dataframe(pd.DataFrame(result["top_ranked"]), width="stretch")
    if result.get("transaction_summary"):
        with st.expander("Processed transactions", expanded=False):
            st.dataframe(pd.DataFrame(result["transaction_summary"]["processed_transactions"]), width="stretch")
