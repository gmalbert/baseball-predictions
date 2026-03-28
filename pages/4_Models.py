"""Page: Models — Betting Features · ML Models · Model Evaluation · Savant Research"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from page_utils import (
    READABLE_COLS,
    _load_precomputed,
    render_sidebar,
    init_session_state,
    add_betting_oracle_footer,
)
from retrosheet import load_gameinfo
from src.evaluation.calibration import calibration_plot_data

min_year, max_year = render_sidebar()

_pre        = _load_precomputed()
features_df = _pre["model_features"][_pre["model_features"]["season"].between(min_year, max_year)].copy()
init_session_state(features_df=features_df)

tab_feat, tab_models, tab_eval, tab_savant = st.tabs([
    "Betting Features", "ML Models", "Model Evaluation", "Savant Research",
])

# ── Betting Features ──────────────────────────────────────────────────────────
with tab_feat:
    st.subheader("Engineered Betting Features")
    st.markdown("Feature matrix built from season-level stats — designed as inputs for ML models.")

    all_standings = _pre["standings"]
    feat_season   = st.selectbox(
        "Season",
        sorted(all_standings["season"].unique().tolist(), reverse=True),
        key="feat_season",
    )

    with st.spinner("Building feature matrix…"):
        gi = load_gameinfo(min_year=feat_season, max_year=feat_season)

    if gi.empty:
        st.info("No games in selected season.")
    else:
        ts_yr = all_standings[all_standings["season"] == feat_season].set_index("team")
        rows = []
        for _, g in gi.iterrows():
            vt, ht = g["visteam"], g["hometeam"]
            if vt not in ts_yr.index or ht not in ts_yr.index:
                continue
            rows.append({
                "date": g["date"],
                "visitor": vt, "home_team": ht,
                "home_WPct": ts_yr.loc[ht, "WPct"], "vis_WPct": ts_yr.loc[vt, "WPct"],
                "home_RS_G": ts_yr.loc[ht, "RS_per_G"], "home_RA_G": ts_yr.loc[ht, "RA_per_G"],
                "vis_RS_G":  ts_yr.loc[vt, "RS_per_G"], "vis_RA_G":  ts_yr.loc[vt, "RA_per_G"],
                "home_RD_G": ts_yr.loc[ht, "RD_per_G"], "vis_RD_G":  ts_yr.loc[vt, "RD_per_G"],
                "home_PythWPct": ts_yr.loc[ht, "PythWPct"], "vis_PythWPct": ts_yr.loc[vt, "PythWPct"],
                "WPct_diff":     ts_yr.loc[ht, "WPct"] - ts_yr.loc[vt, "WPct"],
                "PythWPct_diff": ts_yr.loc[ht, "PythWPct"] - ts_yr.loc[vt, "PythWPct"],
                "RS_advantage":  ts_yr.loc[ht, "RS_per_G"] - ts_yr.loc[vt, "RS_per_G"],
                "RA_advantage":  ts_yr.loc[vt, "RA_per_G"] - ts_yr.loc[ht, "RA_per_G"],
                "daynight": g.get("daynight", ""),
                "attendance": g.get("attendance", None),
                "temp": g.get("temp", None),
                "windspeed": g.get("windspeed", None),
                "home_win": int(g["wteam"] == ht),
                "total_runs": g["total_runs"],
            })
        feat_df = pd.DataFrame(rows)
        st.markdown(f"**{len(feat_df)} games** in {feat_season} with full feature coverage.")
        display_feat = feat_df.head(50).copy()
        display_feat["date"] = display_feat["date"].dt.date
        feat_rename = {
            **READABLE_COLS,
            "visitor": "Visitor", "home_team": "Home",
            "home_WPct": "Home Win %", "vis_WPct": "Visitor Win %",
            "home_RS_G": "Home RS/G", "home_RA_G": "Home RA/G",
            "vis_RS_G": "Visitor RS/G", "vis_RA_G": "Visitor RA/G",
            "home_PythWPct": "Home Pyth W%", "vis_PythWPct": "Visitor Pyth W%",
            "WPct_diff": "Win % Diff", "PythWPct_diff": "Pyth W% Diff",
            "RS_advantage": "RS Advantage", "RA_advantage": "RA Advantage",
            "daynight": "Day/Night", "attendance": "Attendance",
            "temp": "Temperature", "windspeed": "Wind Speed",
            "home_win": "Home Win?", "total_runs": "Total Runs",
        }
        st.dataframe(display_feat.rename(columns=feat_rename), width="stretch", hide_index=True)

        num_feats = [
            "home_WPct", "vis_WPct", "WPct_diff", "PythWPct_diff",
            "home_RS_G", "home_RA_G", "vis_RS_G", "vis_RA_G",
            "RS_advantage", "RA_advantage", "home_win", "total_runs",
        ]
        readable_feats = [
            "Home Win %", "Visitor Win %", "Win % Diff", "Pyth W% Diff",
            "Home RS/G", "Home RA/G", "Visitor RS/G", "Visitor RA/G",
            "RS Advantage", "RA Advantage", "Home Win?", "Total Runs",
        ]
        corr = feat_df[num_feats].corr()
        corr.index   = readable_feats
        corr.columns = readable_feats
        fig = px.imshow(
            corr, title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu", zmin=-1, zmax=1,
            text_auto=".2f", aspect="auto",
        )
        st.plotly_chart(fig, width="stretch")

        with st.expander("Feature data dictionary"):
            dict_df = pd.DataFrame({
                "column": num_feats,
                "description": [
                    "Home Win %, season standings", "Visitor Win %, season standings",
                    "Home WPct minus visitor WPct", "Home Pythagorean WPct minus visitor",
                    "Home runs scored per game", "Home runs allowed per game",
                    "Visitor runs scored per game", "Visitor runs allowed per game",
                    "Home RS/G minus visitor RS/G", "Visitor RA/G minus home RA/G",
                    "Indicator (1=home team won)", "Total runs scored in game",
                ],
            })
            st.dataframe(dict_df.rename(columns=READABLE_COLS), width="stretch", hide_index=True)

        st.markdown("#### Home Win % over time (all seasons in range)")
        gi_all = load_gameinfo(min_year, max_year)
        gi_all["home_win"] = (gi_all["wteam"] == gi_all["hometeam"]).astype(int)
        hfa = gi_all.groupby("season")["home_win"].mean().reset_index()
        hfa.columns = ["season", "home_win_pct"]
        fig2 = px.line(hfa, x="season", y="home_win_pct",
                       title="Home Field Advantage — Win % by season",
                       labels={"home_win_pct": "Home Win %", "season": "Season"})
        fig2.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="50%")
        st.plotly_chart(fig2, width="stretch")

        if gi_all["temp"].notna().sum() > 100:
            temp_df = gi_all.dropna(subset=["temp", "total_runs"])
            temp_df = temp_df[temp_df["temp"] > 0]
            fig3 = px.scatter(
                temp_df.sample(min(5000, len(temp_df)), random_state=42),
                x="temp", y="total_runs", trendline="lowess",
                title="Temperature vs Total Runs (sample)",
                labels={"temp": "Temp (°F)", "total_runs": "Total Runs"},
                opacity=0.4,
            )
            st.plotly_chart(fig3, width="stretch")

        with st.expander("Download feature CSV"):
            st.download_button(
                label="Download features.csv",
                data=feat_df.to_csv(index=False),
                file_name=f"features_{feat_season}.csv",
                mime="text/csv",
            )

# ── ML Models ─────────────────────────────────────────────────────────────────
with tab_models:
    st.subheader("ML Betting Models")
    st.markdown(
        """
        Three XGBoost classifiers trained on **2020+ Retrosheet** game data.
        All models use a **chronological train/test split** (no lookahead).

        | Model | Target | Features |
        |-------|--------|----------|
        | **Moneyline** | P(home team wins) | Team stats, SP stats, weather |
        | **Spread** | P(home covers −1.5) | Same as moneyline |
        | **Over/Under** | P(total > expected) | Same + expected-total offset |
        """
    )

    results = st.session_state["ml_results"]
    if results is None:
        st.info("Pre-trained results not found. Run `scripts/train_models.py` to generate them.")
    else:
        _model_labels = {
            "moneyline": "🏆 Moneyline (P home win)",
            "spread":    "📏 Spread (P home covers −1.5)",
            "totals":    "📈 Over/Under (P went over)",
        }

        st.markdown("### Model Performance (test set)")
        c1, c2, c3 = st.columns(3)
        for col, key in zip([c1, c2, c3], ["moneyline", "spread", "totals"]):
            m = results[key]["metrics"]
            col.markdown(f"**{_model_labels[key]}**")
            col.metric("ROC-AUC",  f"{m['roc_auc']:.4f}")
            col.metric("Accuracy", f"{m['accuracy']:.4f}")
            col.metric("Brier",    f"{m['brier_score']:.4f}")
            col.metric("Log Loss", f"{m['log_loss']:.4f}")

        with st.expander("ℹ️ Metric guide"):
            st.markdown(
                """
                - **ROC-AUC** – 0.5 = random; 1.0 = perfect. 0.60–0.65 is solid for MLB.
                - **Accuracy** – fraction of correct binary picks on the test set.
                - **Brier Score** – mean squared error of predicted probabilities. Lower is better.
                - **Log Loss** – cross-entropy. Lower is better.
                """
            )

        st.markdown("### Feature Importances")
        _FEAT_LABELS = {
            "WPct_diff": "Win % Diff", "PythWPct_diff": "Pythagorean W% Diff",
            "sp_ERA_gap": "SP ERA Gap", "home_WPct": "Home Win %", "away_WPct": "Away Win %",
            "home_PythWPct": "Home Pyth W%", "away_PythWPct": "Away Pyth W%",
            "home_RS_G": "Home RS / G", "home_RA_G": "Home RA / G",
            "away_RS_G": "Away RS / G", "away_RA_G": "Away RA / G",
            "home_RD_G": "Home RD / G", "away_RD_G": "Away RD / G",
            "ERA_diff": "ERA Diff", "WHIP_diff": "WHIP Diff",
            "temp": "Temperature (°F)", "windspeed": "Wind Speed",
            "is_day": "Day game?", "exp_total": "Expected Total Runs",
        }

        def _importance_chart(model_key: str, top_n: int = 20) -> None:
            imp = results[model_key]["importances"].head(top_n).copy()
            imp["label"] = imp["feature"].map(lambda x: _FEAT_LABELS.get(x, x))
            fig = px.bar(
                imp.sort_values("importance"),
                x="importance", y="label", orientation="h",
                title=f"Top {top_n} features — {_model_labels[model_key]}",
                labels={"importance": "Importance", "label": "Feature"},
                color="importance", color_continuous_scale="Blues",
            )
            fig.update_layout(coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig, width="stretch")

        imp_tab_ml, imp_tab_sp, imp_tab_ou = st.tabs(["Moneyline", "Spread", "Over/Under"])
        with imp_tab_ml:
            _importance_chart("moneyline")
        with imp_tab_sp:
            _importance_chart("spread")
        with imp_tab_ou:
            _importance_chart("totals")

        st.markdown("### Calibration: Predicted vs Actual Win Rate")

        def _calibration_chart(model_key: str, prob_col: str, actual_col: str, label: str) -> None:
            tdf = results[model_key]["test_df"][[prob_col, actual_col]].copy()
            tdf["bin"] = pd.cut(tdf[prob_col], bins=10)
            cal = tdf.groupby("bin", observed=False).agg(
                mean_pred=(prob_col, "mean"),
                actual_rate=(actual_col, "mean"),
                count=(prob_col, "count"),
            ).reset_index().dropna()
            fig = px.scatter(
                cal, x="mean_pred", y="actual_rate", size="count",
                title=f"{_model_labels[model_key]} — calibration",
                labels={"mean_pred": f"Mean predicted {label}", "actual_rate": "Actual rate"},
            )
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="gray"))
            st.plotly_chart(fig, width="stretch")

        cal_tab_ml, cal_tab_sp, cal_tab_ou = st.tabs(["Moneyline", "Spread", "Over/Under"])
        with cal_tab_ml:
            _calibration_chart("moneyline", "pred_prob", "home_win", "home win probability")
        with cal_tab_sp:
            _calibration_chart("spread", "pred_prob", "home_cover", "P(cover −1.5)")
        with cal_tab_ou:
            _calibration_chart("totals", "pred_prob_over", "went_over", "P(over)")

        st.markdown("### Backtest Sample — Recent Test-Set Games")
        bt_model = st.selectbox(
            "Model", ["moneyline", "spread", "totals"],
            format_func=lambda x: _model_labels[x],
            key="bt_model_sel",
        )
        n_show = st.slider("Games to display", 25, 200, 50, key="bt_n_show")
        bt_df  = results[bt_model]["test_df"].tail(n_show).copy()
        bt_df["date"] = pd.to_datetime(bt_df["date"]).dt.date
        if bt_model == "moneyline" and "pred_win" in bt_df.columns:
            bt_df["pred_win"] = bt_df["pred_win"].map({1: "Home", 0: "Away"})
        if bt_model == "spread" and "pred_cover" in bt_df.columns:
            bt_df["pred_cover"] = bt_df["pred_cover"].map({1: "Home −1.5", 0: "Away +1.5"})
        if "correct" in bt_df.columns:
            bt_df["correct"] = bt_df["correct"].astype(bool).map({True: "✔", False: ""})
        for col in ("home_win", "home_cover", "went_over"):
            if col in bt_df.columns:
                bt_df[col] = bt_df[col].astype(bool).map({True: "✔", False: ""})

        if bt_model == "moneyline":
            display_cols = {
                "date": "Date", "hometeam": "Home", "visteam": "Away",
                "hruns": "H Runs", "vruns": "V Runs",
                "home_win": "Actually Won?", "pred_prob": "Pred. Prob",
                "pred_win": "Model Pick", "correct": "Correct?",
            }
        elif bt_model == "spread":
            display_cols = {
                "date": "Date", "hometeam": "Home", "visteam": "Away",
                "home_margin": "Margin", "home_cover": "Actually Covered?",
                "pred_prob": "P(cover −1.5)", "pred_cover": "Model Pick", "correct": "Correct?",
            }
        else:
            display_cols = {
                "date": "Date", "hometeam": "Home", "visteam": "Away",
                "total_runs": "Total Runs", "exp_total": "Exp Total",
                "went_over": "Went Over?", "pred_prob_over": "P(over)",
                "pick_side": "Pick", "correct": "Correct?",
            }
        existing = [c for c in display_cols if c in bt_df.columns]
        st.dataframe(
            bt_df[existing].reset_index(drop=True).rename(columns=display_cols),
            width="stretch", hide_index=True,
        )

        with st.expander("Download backtest CSVs"):
            c_dl1, c_dl2, c_dl3 = st.columns(3)
            for col, key, label in zip([c_dl1, c_dl2, c_dl3], ["moneyline", "spread", "totals"],
                                       ["moneyline", "spread", "totals"]):
                col.download_button(
                    label=f"Download {label}.csv",
                    data=results[key]["test_df"].to_csv(index=False),
                    file_name=f"backtest_{label}.csv",
                    mime="text/csv",
                )

# ── Model Evaluation ──────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("Model Evaluation")
    st.markdown(
        "Walk-forward backtests, calibration analyses, and profitability reports. "
        "Results are pre-loaded from parquet files generated by the evaluation scripts."
    )

    if st.session_state["eval_backtests"]:
        leaderboard = [bt.summary() for bt in st.session_state["eval_backtests"].values()]
        lb_df = pd.DataFrame(leaderboard).sort_values("roi", ascending=False)
        if "period" in lb_df.columns:
            lb_df["period"] = lb_df["period"].astype(str).str.strip().str.slice(0, 10)

        st.markdown("### Backtest Leaderboard")
        st.dataframe(
            lb_df.rename(columns={
                "model": "Model", "pick_type": "Pick Type", "period": "Period",
                "total_bets": "Bets", "wins": "Wins", "losses": "Losses",
                "pushes": "Pushes", "win_rate": "Win Rate",
                "total_units": "Units", "max_drawdown": "Max Drawdown", "roi": "ROI",
            }),
            hide_index=True, width="stretch",
        )

        st.markdown("### Calibration Charts")
        for name, bt in st.session_state["eval_backtests"].items():
            arr_true = [1 if b.result == "win" else 0 for b in bt.bets]
            arr_prob = [b.predicted_prob for b in bt.bets]
            cal_data = calibration_plot_data(np.array(arr_true), np.array(arr_prob))
            fig = px.scatter(
                x=cal_data["mean_predicted"], y=cal_data["fraction_positive"],
                size=[1] * len(cal_data["mean_predicted"]),
                title=f"{name.capitalize()} Calibration",
                labels={"x": "Mean pred", "y": "Actual rate"},
            )
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="gray"))
            st.plotly_chart(fig, width="stretch")
    else:
        st.info(
            "No evaluation data yet. Run `scripts/run_evaluation.py` or `scripts/train_models.py`."
        )

# ── Savant Research ───────────────────────────────────────────────────────────
with tab_savant:
    mc_ranking     = _pre.get("mc_ranking")
    mc_trials      = _pre.get("mc_trials")
    savant_metrics = _pre.get("savant_metrics")
    savant_imps    = _pre.get("savant_imps")

    st.subheader("Savant Feature Research — Monte Carlo Selection")
    st.markdown(
        "Results of a Monte Carlo search over Baseball Savant advanced metrics. "
        "1,000 random trials were run, sampling random subsets of Statcast columns, "
        "training XGBoost with TimeSeriesSplit CV across three bet targets. "
        "Features are ranked by how often they appeared in the **top 10% of trials by ROC-AUC**."
    )

    if mc_ranking is None or mc_ranking.empty:
        st.info(
            "Monte Carlo results not found. "
            "Run `python scripts/monte_carlo_features.py --trials 1000` to generate them."
        )
    else:
        BASELINE_AUC = {"moneyline": 0.6253, "spread": 0.6304, "totals": 0.6157}
        n_valid    = len(mc_trials) if mc_trials is not None else "—"
        top_cutoff = mc_trials["mean_auc"].quantile(0.90) if mc_trials is not None else None

        col_m, col_s, col_t, col_v = st.columns(4)
        col_m.metric("Baseline Moneyline AUC", f"{BASELINE_AUC['moneyline']:.4f}")
        col_s.metric("Baseline Spread AUC",    f"{BASELINE_AUC['spread']:.4f}")
        col_t.metric("Baseline Totals AUC",    f"{BASELINE_AUC['totals']:.4f}")
        col_v.metric("Valid Trials", f"{n_valid:,}" if isinstance(n_valid, int) else n_valid)

        if savant_metrics is not None and not savant_metrics.empty:
            st.markdown("---")
            st.markdown("#### Savant-Enriched Model Performance")
            perf_cols = st.columns(3)
            for i, model_name in enumerate(["moneyline", "spread", "totals"]):
                row = savant_metrics[savant_metrics["model"] == model_name]
                if row.empty:
                    continue
                row  = row.iloc[0]
                auc  = float(row["roc_auc"])
                perf_cols[i].metric(
                    label=f"{model_name.capitalize()} AUC (Savant)",
                    value=f"{auc:.4f}",
                    delta=f"{auc - BASELINE_AUC[model_name]:+.4f} vs baseline",
                )

        if mc_trials is not None:
            st.markdown("---")
            auc_plot_cols = [c for c in mc_trials.columns if c.endswith("_auc") and c != "mean_auc"]
            auc_long = mc_trials[auc_plot_cols + ["mean_auc"]].melt(var_name="target", value_name="auc")
            auc_long["target"] = auc_long["target"].str.replace("_auc", "").str.capitalize()
            fig_dist = px.box(
                auc_long[auc_long["target"] != "Mean"],
                x="target", y="auc", color="target", points=False,
                title="Trial AUC by Bet Target (1,000 trials)",
                labels={"target": "Bet Target", "auc": "ROC-AUC"},
            )
            for target_name, baseline in [
                ("Moneyline", BASELINE_AUC["moneyline"]),
                ("Spread",    BASELINE_AUC["spread"]),
                ("Totals",    BASELINE_AUC["totals"]),
            ]:
                fig_dist.add_hline(y=baseline, line_dash="dot", line_color="gray",
                                   annotation_text=f"{target_name} baseline",
                                   annotation_position="right")
            st.plotly_chart(fig_dist, width="stretch")

        bat_ranks = mc_ranking[mc_ranking["type"] == "batter"].head(20).copy()
        pit_ranks = mc_ranking[mc_ranking["type"] == "pitcher"].head(20).copy()
        bat_ranks["appearance_pct"] = bat_ranks["appearance_rate"] * 100
        pit_ranks["appearance_pct"] = pit_ranks["appearance_rate"] * 100

        st.markdown("#### Top Batter Features")
        fig_bat = px.bar(
            bat_ranks.sort_values("appearance_pct"),
            x="appearance_pct", y="feature", orientation="h",
            title="Batter Feature Selection Frequency",
            labels={"appearance_pct": "Appearance Rate in Top Trials (%)", "feature": "Savant Column"},
            color="appearance_pct", color_continuous_scale="Blues",
        )
        fig_bat.update_layout(coloraxis_showscale=False, height=500)
        st.plotly_chart(fig_bat, width="stretch")

        st.markdown("#### Top Pitcher Features")
        fig_pit = px.bar(
            pit_ranks.sort_values("appearance_pct"),
            x="appearance_pct", y="feature", orientation="h",
            title="Pitcher Feature Selection Frequency",
            labels={"appearance_pct": "Appearance Rate in Top Trials (%)", "feature": "Savant Column"},
            color="appearance_pct", color_continuous_scale="Reds",
        )
        fig_pit.update_layout(coloraxis_showscale=False, height=500)
        st.plotly_chart(fig_pit, width="stretch")

        with st.expander("Full feature ranking table"):
            display_rank = mc_ranking.copy()
            display_rank["appearance_rate"] = (display_rank["appearance_rate"] * 100).round(1)
            st.dataframe(
                display_rank.rename(columns={
                    "feature": "Savant Column", "type": "Type",
                    "top_trial_appearances": "Appearances in Top Trials",
                    "appearance_rate": "Rate (%)",
                }),
                hide_index=True, width="stretch",
            )

add_betting_oracle_footer()
