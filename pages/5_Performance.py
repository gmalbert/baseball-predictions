"""Page: Performance — Pick History · Model Performance · Bankroll"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from page_utils import (
    _kelly_fraction,
    _american_to_implied_prob,
    render_sidebar,
    init_session_state,
    add_betting_oracle_footer,
)

render_sidebar(show_year_filter=False)
init_session_state()

tab_history, tab_perf, tab_bankroll = st.tabs(["Pick History", "Model Performance", "Bankroll"])

_bt = st.session_state["eval_backtests"]

# ── Pick History ──────────────────────────────────────────────────────────────
with tab_history:
    st.subheader("Pick History")
    st.markdown(
        "Backtest history for all modeled picks. "
        "Once the daily pipeline runs, live picks will appear here automatically."
    )

    if _bt is None:
        st.info("No pick history yet. Run the daily pipeline or backtest scripts to populate data.")
    else:
        _hist_rows: list[dict] = []
        for _model_name, _bt_result in _bt.items():
            for _b in _bt_result.bets:
                _hist_rows.append({
                    "model":          _model_name,
                    "date":           _b.date,
                    "game_id":        _b.game_id,
                    "pick_type":      _b.pick_type,
                    "confidence":     _b.confidence,
                    "predicted_prob": _b.predicted_prob,
                    "edge":           _b.edge,
                    "american_odds":  _b.american_odds,
                    "result":         _b.result,
                    "profit_units":   _b.profit_units,
                })
        _hist_df = pd.DataFrame(_hist_rows)
        if _hist_df.empty:
            st.info("No picks data available.")
        else:
            _hist_df["model"]      = _hist_df["model"].str.title()
            _hist_df["date"]       = pd.to_datetime(_hist_df["date"])
            _PICK_TYPE_LABELS      = {"totals": "Totals", "over_under": "Over/Under"}
            _hist_df["pick_type"]  = _hist_df["pick_type"].map(
                lambda x: _PICK_TYPE_LABELS.get(x, x.title())
            )
            _hist_df["confidence"] = _hist_df["confidence"].str.title()

            _fc1, _fc2, _fc3, _fc4 = st.columns(4)
            with _fc1:
                _models_avail = ["All"] + sorted(_hist_df["model"].unique().tolist())
                _sel_model = st.selectbox("Model", _models_avail, key="hist_model_filter")
            with _fc2:
                _sel_result = st.selectbox("Result", ["All", "win", "loss"], key="hist_result_filter")
            with _fc3:
                _conf_opts = ["All"] + sorted(_hist_df["confidence"].dropna().unique().tolist())
                _sel_conf = st.selectbox("Confidence", _conf_opts, key="hist_conf_filter")
            with _fc4:
                _pt_opts = ["All"] + sorted(_hist_df["pick_type"].unique().tolist())
                _sel_pt = st.selectbox("Pick Type", _pt_opts, key="hist_pt_filter")

            _filtered = _hist_df.copy()
            if _sel_model != "All":
                _filtered = _filtered[_filtered["model"] == _sel_model]
            if _sel_result != "All":
                _filtered = _filtered[_filtered["result"] == _sel_result]
            if _sel_conf != "All":
                _filtered = _filtered[_filtered["confidence"] == _sel_conf]
            if _sel_pt != "All":
                _filtered = _filtered[_filtered["pick_type"] == _sel_pt]

            _total_bets  = len(_filtered)
            _wins        = int((_filtered["result"] == "win").sum())
            _losses      = int((_filtered["result"] == "loss").sum())
            _win_rate    = _wins / _total_bets if _total_bets > 0 else 0.0
            _total_units = float(_filtered["profit_units"].sum())
            _avg_edge    = float(_filtered["edge"].mean()) if _total_bets > 0 else 0.0

            _m1, _m2, _m3, _m4, _m5 = st.columns(5)
            _m1.metric("Total Picks",  _total_bets)
            _m2.metric("Record",       f"{_wins}–{_losses}")
            _m3.metric("Win Rate",     f"{_win_rate:.1%}")
            _m4.metric("Total Units",  f"{_total_units:+.2f}")
            _m5.metric("Avg Edge",     f"{_avg_edge:.1%}")

            st.divider()

            _cum_df = _filtered.sort_values("date").copy()
            _cum_df["cumulative_units"] = _cum_df.groupby("model")["profit_units"].cumsum()
            if not _cum_df.empty:
                _pnl_fig = px.line(
                    _cum_df, x="date", y="cumulative_units", color="model",
                    title="Cumulative P&L (units)",
                    labels={"date": "Date", "cumulative_units": "Cumulative Units", "model": "Model"},
                )
                _pnl_fig.add_hline(y=0, line_dash="dot", line_color="gray")
                st.plotly_chart(_pnl_fig, width="stretch")

            st.markdown("#### Detailed Ledger")
            _display_df = _filtered[[
                "date", "model", "pick_type", "confidence",
                "predicted_prob", "edge", "american_odds", "result", "profit_units",
            ]].copy()
            _display_df["date"]           = _display_df["date"].dt.strftime("%Y-%m-%d")
            _display_df["predicted_prob"] = (_display_df["predicted_prob"] * 100).round(1).astype(str) + "%"
            _display_df["edge"]           = (_display_df["edge"] * 100).round(1).astype(str) + "%"
            _display_df["american_odds"]  = _display_df["american_odds"].apply(lambda x: f"{int(x):,}")
            _display_df = _display_df.rename(columns={
                "date": "Date", "model": "Model", "pick_type": "Pick Type",
                "confidence": "Confidence", "predicted_prob": "Pred. Prob",
                "edge": "Edge", "american_odds": "Odds",
                "result": "Result", "profit_units": "P&L (Units)",
            }).sort_values("Date", ascending=False).reset_index(drop=True)
            st.dataframe(_display_df, hide_index=True, width="stretch")

# ── Model Performance ─────────────────────────────────────────────────────────
with tab_perf:
    st.subheader("Model Performance")
    st.markdown("Backtest-derived profitability metrics.")

    if _bt is None:
        st.info("No model performance data yet. Run the backtest scripts to populate data.")
    else:
        st.markdown("### Leaderboard")
        _lb_rows = [_btr.summary() for _btr in _bt.values()]
        _lb_df   = pd.DataFrame(_lb_rows).sort_values("roi", ascending=False)
        _lb_df["model"] = _lb_df["model"].str.title()
        if "pick_type" in _lb_df.columns:
            _lb_df["pick_type"] = _lb_df["pick_type"].map(
                lambda x: {"totals": "Totals", "over_under": "Over/Under"}.get(x, x.title())
            )
        if "period" in _lb_df.columns:
            _lb_df["period"] = _lb_df["period"].astype(str).str.replace(r" \d{2}:\d{2}:\d{2}", "", regex=True).str.strip()
        st.dataframe(
            _lb_df.rename(columns={
                "model": "Model", "pick_type": "Pick Type", "total_bets": "Bets",
                "wins": "Wins", "losses": "Losses", "pushes": "Pushes",
                "win_rate": "Win Rate", "total_units": "Units",
                "max_drawdown": "Max Drawdown", "roi": "ROI", "period": "Period",
            }),
            hide_index=True, width="stretch",
        )

        _mp_rows: list[dict] = []
        for _mn, _btr in _bt.items():
            for _b in _btr.bets:
                _mp_rows.append({
                    "model":        _mn,
                    "date":         pd.to_datetime(_b.date),
                    "profit_units": _b.profit_units,
                    "result":       _b.result,
                    "confidence":   _b.confidence,
                })
        _mp_df = pd.DataFrame(_mp_rows)
        _mp_df["model"]      = _mp_df["model"].str.title()
        _mp_df["confidence"] = _mp_df["confidence"].str.title()

        if not _mp_df.empty:
            st.markdown("### Cumulative P&L by Model")
            _mp_df_sorted = _mp_df.sort_values(["model", "date"]).copy()
            _mp_df_sorted["cum_units"] = _mp_df_sorted.groupby("model")["profit_units"].cumsum()
            _mp_fig = px.line(
                _mp_df_sorted, x="date", y="cum_units", color="model",
                title="Cumulative Units by Model",
                labels={"date": "Date", "cum_units": "Cumulative Units", "model": "Model"},
            )
            _mp_fig.add_hline(y=0, line_dash="dot", line_color="gray")
            st.plotly_chart(_mp_fig, width="stretch")

            st.markdown("### Performance by Confidence Tier")
            _tier_grp = (
                _mp_df.groupby(["model", "confidence"])
                .agg(bets=("profit_units", "count"),
                     wins=("result", lambda x: (x == "win").sum()),
                     total_units=("profit_units", "sum"))
                .reset_index()
            )
            _tier_grp["win_rate"] = (_tier_grp["wins"] / _tier_grp["bets"]).round(3)
            _tier_grp["roi"]      = (_tier_grp["total_units"] / _tier_grp["bets"]).round(3)
            st.dataframe(
                _tier_grp.rename(columns={
                    "model": "Model", "confidence": "Tier", "bets": "Bets",
                    "wins": "Wins", "win_rate": "Win Rate",
                    "total_units": "Units", "roi": "ROI/Bet",
                }),
                hide_index=True, width="stretch",
            )
            _tier_bar = px.bar(
                _tier_grp, x="confidence", y="roi", color="model", barmode="group",
                title="ROI per Bet by Confidence Tier",
                labels={"confidence": "Confidence", "roi": "ROI per Bet", "model": "Model"},
            )
            st.plotly_chart(_tier_bar, width="stretch")

# ── Bankroll ──────────────────────────────────────────────────────────────────
with tab_bankroll:
    st.subheader("Bankroll Management")
    st.markdown(
        "Use the Kelly calculator to size each bet and run a historical simulation "
        "against the backtest results."
    )

    _col_kelly, _col_sim = st.columns([1, 1], gap="large")

    with _col_kelly:
        st.markdown("#### Kelly Calculator")
        _bankroll_size = st.number_input("Bankroll ($)", min_value=100, max_value=1_000_000, value=1_000, step=100)
        _unit_size     = st.number_input("Unit size ($)", min_value=1, max_value=10_000, value=50, step=10)
        _kelly_conf    = st.selectbox("Confidence tier", options=["HIGH", "MEDIUM", "LOW"])
        _tier_fractions = {"HIGH": 0.5, "MEDIUM": 0.25, "LOW": 0.125}
        _tier_frac      = _tier_fractions[_kelly_conf]

        _american_odds = st.number_input(
            "American odds (e.g. -110, +150)",
            min_value=-2000, max_value=2000, value=-110, step=5,
        )
        _implied = _american_to_implied_prob(_american_odds)
        st.caption(f"Implied probability: {_implied:.1%}")

        _kelly_prob  = st.slider("Estimated win probability", min_value=0.01, max_value=0.99,
                                 value=round(_implied + 0.04, 2), step=0.01)
        _full_kelly  = _kelly_fraction(prob=_kelly_prob, american_odds=_american_odds)
        _applied_kelly = _full_kelly * _tier_frac

        st.divider()
        _kc1, _kc2 = st.columns(2)
        _kc1.metric("Full Kelly %",  f"{_full_kelly*100:.2f}%")
        _kc2.metric("Tier fraction", f"{_tier_frac:.2%}")
        _kc3, _kc4 = st.columns(2)
        _kc3.metric("Applied Kelly %", f"{_applied_kelly*100:.2f}%")
        _kc4.metric("Bet size ($)",    f"${_bankroll_size * _applied_kelly:,.2f}")
        _units_to_bet = (_bankroll_size * _applied_kelly) / max(_unit_size, 1)
        st.metric("Units to bet", f"{_units_to_bet:.2f} units")

        with st.expander("Kelly Formula"):
            st.latex(
                r"f^* = \frac{bp - q}{b}"
                r"\quad\text{where }b=\text{decimal payout},\;p=\hat{p},\;q=1-p"
            )

    with _col_sim:
        st.markdown("#### Historical Bankroll Simulation")
        if _bt is None:
            st.info("No backtest data available for simulation.")
        else:
            _sim_model = st.selectbox(
                "Select model", options=list(_bt.keys()),
                format_func=lambda x: x.title(),
            )
            _sim_start = st.number_input(
                "Starting bankroll ($)", min_value=100, max_value=1_000_000,
                value=_bankroll_size, step=100, key="sim_start_br",
            )
            _sim_unit  = st.number_input(
                "Unit size ($)", min_value=1, max_value=10_000,
                value=_unit_size, step=10, key="sim_unit_sz",
            )

            _btr      = _bt[_sim_model]
            _sim_rows = [
                {"date": pd.to_datetime(_b.date), "profit_units": _b.profit_units}
                for _b in sorted(_btr.bets, key=lambda x: x.date)
            ]
            _sim_df = pd.DataFrame(_sim_rows)

            if _sim_df.empty:
                st.info("No bets found for this model.")
            else:
                _sim_df["pnl_$"]    = _sim_df["profit_units"] * _sim_unit
                _sim_df["bankroll"] = _sim_start + _sim_df["pnl_$"].cumsum()
                _sim_df["date_str"] = _sim_df["date"].dt.strftime("%Y-%m-%d")

                _sim_fig = go.Figure()
                _sim_fig.add_trace(go.Scatter(
                    x=_sim_df["date_str"], y=_sim_df["bankroll"],
                    mode="lines", name="Bankroll",
                    line=dict(color="#002D72", width=2),
                ))
                _sim_fig.add_hline(y=_sim_start, line_dash="dot", line_color="gray")
                _sim_fig.update_layout(
                    title=f"{_sim_model.title()} — Bankroll Growth",
                    xaxis_title="Date", yaxis_title="Bankroll ($)",
                    yaxis_tickprefix="$", height=380,
                )
                st.plotly_chart(_sim_fig, width="stretch")

                _final_br  = _sim_df["bankroll"].iloc[-1]
                _total_pnl = _final_br - _sim_start
                _peak_br   = _sim_df["bankroll"].max()
                _drawdown  = (_peak_br - _sim_df["bankroll"].min()) / max(_peak_br, 1)

                _sc1, _sc2, _sc3 = st.columns(3)
                _sc1.metric("Final bankroll", f"${_final_br:,.0f}", delta=f"${_total_pnl:+,.0f}")
                _sc2.metric("Peak bankroll",  f"${_peak_br:,.0f}")
                _sc3.metric("Max drawdown",   f"{_drawdown:.1%}")

add_betting_oracle_footer()
