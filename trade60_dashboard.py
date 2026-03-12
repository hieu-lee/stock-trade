from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from trade60_app import get_default_config
from trade60_app.service import build_holdings_template, generate_trade_plan, train_strategy
from trade60_app.utils import read_json

st.set_page_config(page_title="Trade60 Giao Dịch Hàng Ngày", layout="wide")

config = get_default_config(Path(__file__).resolve().parent)
strategy_overview_path = config.app_dir / "strategy_overview_vi.md"


def translate_text(text: str) -> str:
    translated = str(text)
    replacements = [
        ("No symbol cleared the risk-aware buy or sell filters for the next session.", "Không có mã nào vượt bộ lọc mua/bán theo rủi ro cho phiên kế tiếp."),
        ("No buy/sell adjustments are needed for the next session. Current holdings remain in place.", "Không cần điều chỉnh mua/bán cho phiên kế tiếp. Danh mục hiện tại được giữ nguyên."),
        ("Defensive exposure mode is active, so the system will size fewer positions and keep extra cash.", "Chế độ phòng thủ đang bật, vì vậy hệ thống sẽ giảm số vị thế và giữ lại thêm tiền mặt."),
        ("The regime filter is defensive today, so the system will not open new positions for tomorrow.", "Bộ lọc thị trường hôm nay đang ở trạng thái phòng thủ, nên hệ thống sẽ không mở vị thế mới cho phiên tới."),
        ("Reached the 2-month holding limit.", "Đã chạm giới hạn nắm giữ 2 tháng."),
        ("Current loss breached the stop-loss threshold.", "Khoản lỗ hiện tại đã chạm ngưỡng cắt lỗ."),
        ("Current gain reached the take-profit target.", "Khoản lãi hiện tại đã chạm mục tiêu chốt lời."),
        ("The stock score dropped below the exit threshold.", "Điểm số của cổ phiếu đã rơi xuống dưới ngưỡng thoát lệnh."),
        ("The market regime is no longer supportive.", "Điều kiện thị trường không còn ủng hộ vị thế này."),
        ("The latest market snapshot did not include this symbol, so the system kept it unchanged.", "Snapshot thị trường mới nhất chưa có mã này, nên hệ thống tạm giữ nguyên vị thế."),
        ("The position is strong enough to keep unchanged for now.", "Vị thế hiện tại vẫn đủ mạnh để tạm thời giữ nguyên."),
        ("The position was partially trimmed to lock in gains.", "Vị thế đã được giảm bớt một phần để khóa lợi nhuận."),
        ("The stock score softened, so the system trimmed part of the position.", "Điểm số của cổ phiếu đã yếu đi, nên hệ thống giảm bớt một phần vị thế."),
        ("Defensive posture reduced the position size.", "Trạng thái phòng thủ khiến hệ thống giảm quy mô vị thế."),
        ("The holding rank slipped behind stronger opportunities, so the system exited the remainder.", "Mã đang nắm giữ đã tụt lại sau các cơ hội mạnh hơn, nên hệ thống thoát nốt phần còn lại."),
        ("The score remains strong enough to add to the position.", "Điểm số vẫn đủ mạnh để tăng thêm vị thế."),
        ("beating VNINDEX over the last month", "đang mạnh hơn VNINDEX trong 1 tháng gần đây"),
        ("trading above its 50-day trend", "đang giao dịch phía trên xu hướng MA50"),
        ("the 60-symbol market breadth is supportive", "độ rộng thị trường của rổ 60 mã đang ủng hộ"),
        ("volume is stronger than its recent average", "khối lượng đang cao hơn mức trung bình gần đây"),
        ("best composite score in the live universe", "điểm tổng hợp tốt nhất trong rổ theo dõi hiện tại"),
        (" buy date was blank, so the system assumed ", " chưa có ngày mua, nên hệ thống tạm giả định ngày mua là "),
        (" (~1 month ago).", " (xấp xỉ 1 tháng trước)."),
    ]
    for source, target in replacements:
        translated = translated.replace(source, target)
    return translated


def localize_actions(frame: pd.DataFrame) -> pd.DataFrame:
    translated = frame.copy()
    action_map = {
        "BUY": "MUA",
        "SELL": "BÁN",
        "DO_NOTHING": "ĐỨNG NGOÀI",
    }
    translated["action"] = translated["action"].replace(action_map)
    if "rationale" in translated.columns:
        translated["rationale"] = translated["rationale"].map(translate_text)
    translated = translated.rename(
        columns={
            "action": "Hành động",
            "symbol": "Mã cổ phiếu",
            "quantity": "Số lượng",
            "reference_price": "Giá tham chiếu",
            "alpha_probability": "Xác suất cổ phiếu",
            "regime_probability": "Xác suất thị trường",
            "rationale": "Lý do",
        }
    )
    return translated


def localize_position_status(frame: pd.DataFrame) -> pd.DataFrame:
    translated = frame.copy()
    status_map = {
        "KEEP": "GIỮ",
        "TRIM": "GIẢM TỶ TRỌNG",
        "EXIT": "THOÁT",
        "TOP_UP": "TĂNG TỶ TRỌNG",
    }
    translated["status"] = translated["status"].replace(status_map)
    if "rationale" in translated.columns:
        translated["rationale"] = translated["rationale"].map(translate_text)
    return translated.rename(
        columns={
            "symbol": "Mã cổ phiếu",
            "status": "Trạng thái",
            "current_quantity": "Số lượng hiện tại",
            "next_quantity": "Số lượng sau điều chỉnh",
            "delta_quantity": "Chênh lệch",
            "avg_cost": "Giá vốn bình quân",
            "reference_price": "Giá tham chiếu",
            "alpha_probability": "Xác suất cổ phiếu",
            "regime_probability": "Xác suất thị trường",
            "rationale": "Lý do",
        }
    )


def localize_ranked(frame: pd.DataFrame) -> pd.DataFrame:
    translated = frame.copy()
    return translated.rename(
        columns={
            "symbol": "Mã cổ phiếu",
            "close": "Giá đóng cửa",
            "alpha_probability": "Xác suất cổ phiếu",
            "regime_probability": "Xác suất thị trường",
            "composite_score": "Điểm tổng hợp",
            "relative_strength_20d": "Sức mạnh tương đối 20 phiên",
            "distance_ma50": "Độ lệch so với MA50",
            "volume_zscore_20d": "Độ bất thường khối lượng 20 phiên",
        }
    )


def yes_no(value: bool) -> str:
    return "Có" if value else "Không"


def build_progress_reporter():
    status = st.status("Đang chuẩn bị...", expanded=True)
    detail = st.empty()
    progress_bar = st.progress(0, text="Đang chuẩn bị...")
    state = {"progress": 0.0}

    def callback(message: str, progress: float | None = None) -> None:
        if progress is not None:
            state["progress"] = progress
        progress_bar.progress(max(int(state["progress"] * 100), 1), text=message)
        detail.caption(message)
        status.update(label=message, state="running", expanded=True)

    return status, progress_bar, callback


st.title("Trade60 Giao Dịch Hàng Ngày")
st.caption("Hệ thống chỉ mua, dùng tiền mặt, đề xuất hành động cho phiên tiếp theo trên rổ 60 mã đã huấn luyện.")

with st.expander("Đọc tổng quan chiến lược của hệ thống", expanded=False):
    if strategy_overview_path.exists():
        st.markdown(strategy_overview_path.read_text(encoding="utf-8"))
    else:
        st.info("Chưa tìm thấy tài liệu giới thiệu chiến lược.")

summary_path = config.artifacts_dir / "summary.json"

col_train, col_refresh = st.columns([1, 1])
with col_train:
    train_refresh = st.checkbox("Cập nhật dữ liệu thị trường khi huấn luyện", value=False)
    if st.button("Chạy huấn luyện", width="stretch"):
        status, progress_bar, progress_callback = build_progress_reporter()
        try:
            summary = train_strategy(
                config,
                refresh_data=train_refresh,
                progress_callback=progress_callback,
            )
        except Exception:
            status.update(label="Huấn luyện bị lỗi", state="error", expanded=True)
            raise
        progress_bar.progress(100, text="Huấn luyện hoàn tất")
        status.update(label="Huấn luyện hoàn tất", state="complete", expanded=False)
        st.success(
            "Huấn luyện xong. Lợi nhuận năm hóa của tập test cuối: "
            f"{summary['holdout_metrics']['annualized_return']:.2%}"
        )

if summary_path.exists():
    summary = read_json(summary_path)
    holdout = summary["holdout_metrics"]
    validation = summary["validation_metrics"]
    st.subheader("Tóm tắt backtest")
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Lợi nhuận năm hóa tập test cuối", f"{holdout['annualized_return']:.2%}")
    metric2.metric("Vượt VNINDEX", f"{holdout['excess_return_vs_benchmark']:.2%}")
    metric3.metric("Mức sụt giảm tối đa", f"{holdout['max_drawdown']:.2%}")
    metric4.metric(f"Vượt mục tiêu {config.target_annual_return:.0%}/năm", yes_no(holdout["beat_bank_target"]))
    with st.expander("Xem chi tiết chỉ số backtest"):
        st.json({"validation": validation, "holdout": holdout, "best_params": summary["best_params"]})
else:
    st.info("Hãy chạy huấn luyện ít nhất một lần để tạo model và kết quả backtest.")

st.subheader("Nhập danh mục hiện tại")
budget = st.number_input("Số tiền mặt hiện có", min_value=0.0, value=100000.0, step=1000.0)
portfolio_help = "Nếu để trống `buy_date`, hệ thống sẽ tạm giả định ngày mua là khoảng 1 tháng trước để kiểm tra quy tắc giữ tối đa 2 tháng."
st.caption(portfolio_help)
if "trade60_holdings" not in st.session_state:
    st.session_state.trade60_holdings = build_holdings_template(config)
holdings = st.data_editor(
    st.session_state.trade60_holdings,
    width="stretch",
    num_rows="fixed",
    hide_index=True,
    column_config={
        "symbol": st.column_config.TextColumn("Mã cổ phiếu"),
        "quantity": st.column_config.NumberColumn("Số lượng", min_value=0, step=1),
        "avg_cost": st.column_config.NumberColumn("Giá vốn bình quân", min_value=0.0, step=0.1, format="%.2f"),
        "buy_date": st.column_config.TextColumn("Ngày mua (YYYY-MM-DD)"),
    },
)

refresh_recommendation = st.checkbox("Cập nhật dữ liệu trước khi tạo khuyến nghị", value=True)
if st.button("Tạo kế hoạch cho phiên tới", width="stretch"):
    status, progress_bar, progress_callback = build_progress_reporter()
    try:
        recommendation = generate_trade_plan(
            config=config,
            budget=budget,
            holdings=holdings,
            refresh_data=refresh_recommendation,
            progress_callback=progress_callback,
        )
    except Exception:
        status.update(label="Tạo kế hoạch bị lỗi", state="error", expanded=True)
        raise
    progress_bar.progress(100, text="Đã tạo xong kế hoạch")
    status.update(label="Đã tạo xong kế hoạch", state="complete", expanded=False)
    st.session_state.trade60_recommendation = recommendation
    st.session_state.trade60_holdings = holdings

recommendation = st.session_state.get("trade60_recommendation")
if recommendation:
    st.subheader("Khuyến nghị cho phiên tiếp theo")
    st.caption(f"Ngày tín hiệu gần nhất: {pd.Timestamp(recommendation['latest_signal_date']).date()}")
    if recommendation["actions"].empty:
        st.info("Không có lệnh mua/bán mới cho phiên tới. Xem trạng thái danh mục ở bảng bên dưới.")
    else:
        st.dataframe(localize_actions(recommendation["actions"]), width="stretch", hide_index=True)
    st.write(f"Số tiền mặt dự kiến sau khi thực hiện hành động: `{recommendation['cash_after_actions']:.2f}`")
    if not recommendation["position_status"].empty:
        st.subheader("Trạng thái danh mục hiện tại")
        st.dataframe(localize_position_status(recommendation["position_status"]), width="stretch", hide_index=True)
    if recommendation["notes"]:
        st.warning("\n".join(translate_text(note) for note in recommendation["notes"]))
    st.subheader("Các mã được xếp hạng cao nhất")
    top_columns = [
        "symbol",
        "close",
        "alpha_probability",
        "regime_probability",
        "composite_score",
        "relative_strength_20d",
        "distance_ma50",
        "volume_zscore_20d",
    ]
    st.dataframe(localize_ranked(recommendation["scored_panel"][top_columns].head(15)), width="stretch", hide_index=True)
