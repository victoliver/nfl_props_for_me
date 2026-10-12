#!/usr/bin/env python3
"""
nfl_props_model.py — NFL player-prop model (anytime TD, 2+ TD, yards, receptions, attempts).

Self-contained: downloads nflverse data, builds features, fits every market, writes a board,
and emits warnings + commentary about what it found.

USAGE
  python nfl_props_model.py --week 6                 # price the upcoming week
  python nfl_props_model.py --week 6 --season 2026
  python nfl_props_model.py --validate               # holdout validation only
  python nfl_props_model.py --week 6 --refresh       # force re-download of data
  python nfl_props_model.py --week 6 --json          # machine-readable output for a skill

OUTPUT
  <outdir>/week<N>_skill.csv   skill-position board
  <outdir>/week<N>_qb.csv      quarterback board
  <outdir>/week<N>_report.json warnings + commentary (also printed)

MODEL SPEC (as validated; see MARKET_RELIABILITY below)
  Touchdowns     Poisson GLM, lambda capped at the training 99.5th percentile
  Count markets  Negative binomial with MEAN-DEPENDENT dispersion, capped
  Yards markets  Two-stage simulation: attempts (NB) x yards-per-attempt (gamma)
  Deliberately excluded, each tested and rejected:
    - route participation   (0.99 correlated with snap share; destabilised coefficients)
    - league-environment features (worsened every QB market)
    - attempt/YPA correlation     (rho=0.12 for RB, changed prices by 0.1pp)
    - empirical Bayes shrinkage   (changed Brier by 0.00001)
    - a "due for a TD" bonus      (drought coefficient is NEGATIVE, p=0.0007)

Requires: pandas numpy pyarrow statsmodels scikit-learn scipy
"""
from __future__ import annotations
import argparse, json, os, sys, subprocess, warnings
from datetime import datetime, timezone
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from scipy import stats as st

# ----------------------------------------------------------------------------- config
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
TRAIN_SEASONS = [2022, 2023, 2024, 2025, 2026]
MIN_PRIOR_GAMES = 3
CAP_PCTILE = 99.5
N_SIM = 4000
RNG = np.random.default_rng(2026)

TD_F = ["rz_share_e4","gl_share_e4","tgt_share_e4","touches_e4","wopr_e4","implied_total",
        "team_spread","snap_trend","rz_x_total","games_since_td","dnp","rec_epa_e4","snap_e3"]
SKILL_F = ["rz_share_e4","gl_share_e4","tgt_share_e4","carry_share_e4","touches_e4","wopr_e4",
           "implied_total","team_spread","snap_trend","snap_e3","rec_e4","rush_att_e4","targets_e4",
           "tm_db_e4","tm_rush_e4","d_db_e4","d_rush_e4","d_epa_e4","dnp"]
QB_F = ["pass_att_e4","completions_e4","pass_yds_e4","pass_td_e4","dropbacks_e4","pass_epa_e4",
        "implied_total","team_spread","total_line","opp_implied","tm_db_e4","d_db_e4","d_epa_e4",
        "snap_e3","dnp"]

# Measured on the 2026 holdout. Drives the warnings the script emits.
MARKET_RELIABILITY = {
    "anytime_td":      ("GOOD",     "Brier 0.162, AUC 0.668, overall error -0.7pp"),
    "rush_yds":        ("GOOD",     "0 of 3 lines significantly miscalibrated"),
    "receptions":      ("GOOD",     "pooled RB/WR/TE; 1 of 4 lines off at 4.5 (-4.2pp), RB lines 0 of 3"),
    "two_plus_td":     ("FAIR",     "under-predicts: 3.7% modeled vs 5.3% actual"),
    "rush_att":        ("FAIR",     "1 of 3 lines off; -6.4pp at 14.5"),
    "rec_yds":         ("GOOD",     "normalized (team yards x share): 7 of 27 lines off across 2024-26, mean 1.6pp"),
    "rb_rec_yds":      ("GOOD",     "normalized: 0 of 6 lines off on 2025-26, mean 0.7pp"),
    "completions":     ("WEAK",     "under-predicts by up to 9.2pp"),
    "pass_yds":        ("WEAK",     "under-predicts by 5.7 to 8.8pp"),
    "pass_att":        ("UNRELIABLE","under-predicts by up to 13pp; do not price off this"),
}
# Beta dispersion (kappa) for the normalized receiving-yards model, by position group.
# Measured 2024/2025/2026: WR/TE 8.50, 8.50, 8.60 (sd 0.05); RB 11.10, 11.40, 11.16 (sd 0.13).
# Kappa is LOAD-BEARING: it sets both the spread AND the median of the simulated
# distribution, so a drift shifts every modeled line in the same direction at once.
# An 18pp calibration swing was measured at the distribution centre across kappa 4 to 32.
KAPPA_BANDS = {"WRTE": (7.5, 9.5), "RB": (10.0, 12.5)}

# Signs we expect from theory. A flip means something upstream broke.
EXPECTED_SIGNS = {"rz_share_e4": +1, "gl_share_e4": +1, "touches_e4": +1, "implied_total": +1,
                  "snap_trend": +1, "snap_e3": +1, "games_since_td": -1, "dnp": -1}

def _a(x): return np.asarray(x, dtype=float)
def american(p: float) -> str:
    p = float(min(max(p, 1e-6), 0.999))
    return f"+{int(round(100*(1-p)/p))}" if p < 0.5 else f"-{int(round(100*p/(1-p)))}"

# ----------------------------------------------------------------------------- data
def fetch(datadir: str, refresh: bool = False) -> None:
    os.makedirs(datadir, exist_ok=True)
    jobs = [(f"pbp/play_by_play_{s}", f"pbp_{s}") for s in TRAIN_SEASONS]
    jobs += [(f"snap_counts/snap_counts_{s}", f"snap_counts_{s}") for s in TRAIN_SEASONS]
    jobs += [(f"injuries/injuries_{s}", f"injuries_{s}") for s in TRAIN_SEASONS]
    jobs += [("players/players", "players")]
    missing = []
    for remote, local in jobs:
        path = os.path.join(datadir, local + ".parquet")
        if os.path.exists(path) and not refresh:
            continue
        url = f"{NFLVERSE}/{remote}.parquet"
        r = subprocess.run(["curl", "-sS", "-L", "-f", "-o", path, url],
                           capture_output=True)
        if r.returncode != 0:
            missing.append(local)
            if os.path.exists(path): os.remove(path)
    g = os.path.join(datadir, "games.csv")
    if not os.path.exists(g) or refresh:
        subprocess.run(["curl","-sS","-L","-f","-o",g,
                        "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"],
                       capture_output=True)
    if missing:
        print(f"[data] not available (normal for a season not yet played): {', '.join(missing)}")

def build_frame(datadir: str) -> pd.DataFrame:
    cols = ["game_id","season","week","season_type","posteam","defteam","home_team","play_type",
            "rush_attempt","pass_attempt","complete_pass","yardline_100","rusher_player_id",
            "receiver_player_id","passer_player_id","rush_touchdown","pass_touchdown","interception",
            "air_yards","passing_yards","rushing_yards","receiving_yards","epa","sack",
            "spread_line","total_line","qb_dropback"]
    frames = []
    for s in TRAIN_SEASONS:
        p = os.path.join(datadir, f"pbp_{s}.parquet")
        if os.path.exists(p): frames.append(pd.read_parquet(p, columns=cols))
    if not frames: raise SystemExit("no play-by-play found; run with --refresh")
    pbp = pd.concat(frames, ignore_index=True)
    pbp = pbp[(pbp.season_type == "REG") & pbp.posteam.notna()].copy()
    pbp["rz"] = (pbp.yardline_100 <= 20).astype(int)
    pbp["gl"] = (pbp.yardline_100 <= 5).astype(int)
    K = ["game_id","season","week","posteam","defteam"]

    ru = (pbp[(pbp.rush_attempt == 1) & pbp.rusher_player_id.notna()]
          .groupby(K + ["rusher_player_id"])
          .agg(rush_att=("rush_attempt","sum"), rush_yds=("rushing_yards","sum"),
               rz_carries=("rz","sum"), gl_carries=("gl","sum"),
               rush_td=("rush_touchdown","sum"), rush_epa=("epa","sum"))
          .reset_index().rename(columns={"rusher_player_id":"player_id"}))
    re_ = (pbp[(pbp.pass_attempt == 1) & pbp.receiver_player_id.notna()]
           .groupby(K + ["receiver_player_id"])
           .agg(targets=("pass_attempt","sum"), rec=("complete_pass","sum"),
                rec_yds=("receiving_yards","sum"), air_yards=("air_yards","sum"),
                rz_targets=("rz","sum"), gl_targets=("gl","sum"),
                rec_td=("pass_touchdown","sum"), rec_epa=("epa","sum"))
           .reset_index().rename(columns={"receiver_player_id":"player_id"}))
    pa = (pbp[(pbp.pass_attempt == 1) & pbp.passer_player_id.notna()]
          .groupby(K + ["passer_player_id"])
          .agg(pass_att=("pass_attempt","sum"), completions=("complete_pass","sum"),
               pass_yds=("passing_yards","sum"), pass_td=("pass_touchdown","sum"),
               ints=("interception","sum"), pass_epa=("epa","sum"))
          .reset_index().rename(columns={"passer_player_id":"player_id"}))
    sk = (pbp[pbp.sack == 1].groupby(K + ["passer_player_id"]).size().rename("sacks")
          .reset_index().rename(columns={"passer_player_id":"player_id"}))
    pa = pa.merge(sk, on=K + ["player_id"], how="left"); pa["sacks"] = pa.sacks.fillna(0)

    pg = ru.merge(re_, on=K + ["player_id"], how="outer").merge(pa, on=K + ["player_id"], how="outer")
    num = [c for c in pg.columns if c not in K + ["player_id"]]
    pg[num] = pg[num].fillna(0)
    pg["tds"] = pg.rush_td + pg.rec_td
    pg["scored"] = (pg.tds > 0).astype(int)
    pg["touches"] = pg.rush_att + pg.rec
    pg["opportunities"] = pg.rush_att + pg.targets
    pg["rz_opps"] = pg.rz_carries + pg.rz_targets
    pg["gl_opps"] = pg.gl_carries + pg.gl_targets
    pg["dropbacks"] = pg.pass_att + pg.sacks

    tm = (pg.groupby(["game_id","posteam"])
          .agg(tm_rz=("rz_opps","sum"), tm_gl=("gl_opps","sum"),
               tm_tgt=("targets","sum"), tm_air=("air_yards","sum"), tm_car=("rush_att","sum"))
          .reset_index())
    pg = pg.merge(tm, on=["game_id","posteam"])
    for a, b, nm in [("rz_opps","tm_rz","rz_share"), ("gl_opps","tm_gl","gl_share"),
                     ("targets","tm_tgt","tgt_share"), ("air_yards","tm_air","ay_share"),
                     ("rush_att","tm_car","carry_share")]:
        pg[nm] = np.where(pg[b] > 0, pg[a] / pg[b], 0.0)
    pg["wopr"] = 1.5 * pg.tgt_share + 0.7 * pg.ay_share

    players = pd.read_parquet(os.path.join(datadir, "players.parquet"),
                              columns=["gsis_id","display_name","position","pfr_id"])
    players = players.rename(columns={"gsis_id":"player_id","display_name":"player_name"})
    pg = pg.merge(players[["player_id","player_name","position"]], on="player_id", how="left")
    pg = pg[pg.position.isin(["RB","WR","TE","QB"])].copy()

    snaps = []
    for s in TRAIN_SEASONS:
        p = os.path.join(datadir, f"snap_counts_{s}.parquet")
        if os.path.exists(p):
            snaps.append(pd.read_parquet(p, columns=["pfr_player_id","season","week","offense_pct"]))
    if snaps:
        sn = pd.concat(snaps).merge(players[["player_id","pfr_id"]].dropna(),
                                    left_on="pfr_player_id", right_on="pfr_id")
        sn = sn[["player_id","season","week","offense_pct"]].drop_duplicates(["player_id","season","week"])
        pg = pg.merge(sn, on=["player_id","season","week"], how="left")
    else:
        pg["offense_pct"] = np.nan

    ctx = (pbp.groupby("game_id")
           .agg(h=("home_team","first"), sp=("spread_line","first"), tt=("total_line","first"))
           .reset_index())
    pg = pg.merge(ctx, on="game_id", how="left")
    pg["is_home"] = (pg.posteam == pg.h).astype(int)
    pg["implied_total"] = np.where(pg.is_home == 1, pg.tt/2 + pg.sp/2, pg.tt/2 - pg.sp/2)
    pg["opp_implied"] = pg.tt - pg.implied_total
    pg["team_spread"] = np.where(pg.is_home == 1, pg.sp, -pg.sp)
    pg["total_line"] = pg.tt

    tv = (pbp.groupby(["game_id","season","week","posteam"])
          .agg(tm_db=("qb_dropback","sum"), tm_rush=("rush_attempt","sum")).reset_index()
          .sort_values(["posteam","season","week"]))
    for c in ["tm_db","tm_rush"]:
        tv[c + "_e4"] = (tv.groupby("posteam", group_keys=False)[c]
                         .apply(lambda s: s.shift(1).ewm(halflife=4, min_periods=1).mean()))
    pg = pg.merge(tv[["game_id","posteam","tm_db_e4","tm_rush_e4"]], on=["game_id","posteam"], how="left")
    dv = (pbp.groupby(["game_id","season","week","defteam"])
          .agg(d_db=("qb_dropback","sum"), d_rush=("rush_attempt","sum"), d_epa=("epa","mean"))
          .reset_index().sort_values(["defteam","season","week"]))
    for c in ["d_db","d_rush","d_epa"]:
        dv[c + "_e4"] = (dv.groupby("defteam", group_keys=False)[c]
                         .apply(lambda s: s.shift(1).ewm(halflife=4, min_periods=1).mean()))
    pg = pg.merge(dv[["game_id","defteam","d_db_e4","d_rush_e4","d_epa_e4"]],
                  on=["game_id","defteam"], how="left")

    injs = []
    for s in TRAIN_SEASONS:
        p = os.path.join(datadir, f"injuries_{s}.parquet")
        if os.path.exists(p):
            injs.append(pd.read_parquet(p, columns=["season","week","gsis_id","report_status","practice_status"]))
    if injs:
        inj = (pd.concat(injs).rename(columns={"gsis_id":"player_id"})
               .drop_duplicates(["season","week","player_id"]))
        pg = pg.merge(inj, on=["season","week","player_id"], how="left")
    else:
        pg["report_status"] = np.nan; pg["practice_status"] = np.nan
    pg["dnp"] = pg.practice_status.fillna("").str.contains("Did Not Participate", case=False).astype(int)
    pg["out_flag"] = (pg.report_status == "Out").astype(int)
    return pg

LAG_COLS = ["rz_share","gl_share","tgt_share","carry_share","wopr","touches","rush_att","targets",
            "rec","rush_yds","rec_yds","pass_att","pass_yds","pass_td","completions","dropbacks",
            "rush_epa","rec_epa","pass_epa","ay_share"]

def add_lags(frame: pd.DataFrame) -> pd.DataFrame:
    """EWMA features, strictly prior games only (shift(1) before ewm)."""
    f = frame.sort_values(["player_id","season","week","is_pred"]).copy()
    g = f.groupby("player_id", group_keys=False)
    for c in LAG_COLS:
        f[f"{c}_e4"] = g[c].apply(lambda s: s.shift(1).ewm(halflife=4, min_periods=1).mean())
    f["snap_e3"] = g["offense_pct"].apply(lambda s: s.shift(1).ewm(halflife=3, min_periods=1).mean())
    f["snap_l6"] = g["offense_pct"].apply(lambda s: s.shift(1).ewm(halflife=6, min_periods=1).mean())
    f["snap_trend"] = f.snap_e3 - f.snap_l6
    def drought(s):
        out, c = [], 0
        for v in s.values:
            out.append(c); c = 0 if (pd.notna(v) and v > 0) else c + 1
        return pd.Series(out, index=s.index)
    f["games_since_td"] = g["tds"].apply(drought)
    f["prior_games"] = g.cumcount()
    f["rz_x_total"] = f.rz_share_e4 * f.implied_total
    return f

# ----------------------------------------------------------------------------- models
def fit_count(tr, F, y):
    """Negative binomial with MEAN-DEPENDENT dispersion.
    Plain NB forces variance to rise with the mean; for volume stats it falls
    (established starters are more predictable), so dispersion is refit on residuals."""
    med = tr[F].median(); X = sm.add_constant(tr[F].fillna(med))
    aux = sm.GLM(tr[y].values, X, family=sm.families.Poisson()).fit()
    mu0 = np.clip(_a(aux.predict(X)), 1e-6, None)
    r = ((tr[y].values - mu0) ** 2 - mu0) / mu0
    a0 = float(np.clip(np.asarray(sm.OLS(r, mu0).fit().params).ravel()[0], 1e-4, 10))
    m = sm.GLM(tr[y].values, X, family=sm.families.NegativeBinomial(alpha=a0)).fit()
    mu_tr = np.clip(_a(m.predict(X)), 1e-6, None)
    cap = float(np.percentile(mu_tr, CAP_PCTILE))
    vm = sm.OLS((tr[y].values - mu_tr) ** 2, sm.add_constant(np.c_[mu_tr, mu_tr ** 2])).fit()
    def predict(te):
        raw = np.clip(_a(m.predict(sm.add_constant(te[F].fillna(med), has_constant="add"))), 1e-6, None)
        mu = np.minimum(raw, cap)
        var = np.clip(_a(vm.predict(sm.add_constant(np.c_[mu, mu ** 2]))), mu * 1.01, None)
        alpha = np.clip((var - mu) / np.maximum(mu ** 2, 1e-9), 1e-4, 10)
        n_ = 1 / alpha
        return mu, n_, n_ / (n_ + mu), (raw > cap)
    return predict, m

def fit_yards(tr, F, att, yds):
    """Two-stage: attempts ~ NB, yards-per-attempt ~ gamma, simulated independently.
    Independence tested: residual corr +0.12 (RB, significant) but changes prices by 0.1pp."""
    pa, _ = fit_count(tr, F, att)
    med = tr[F].median()
    sub = tr[tr[att] > 0].copy(); sub["ypa"] = sub[yds] / sub[att]
    sub = sub[(sub.ypa > -5) & (sub.ypa < 40)]
    my = sm.GLM(sub.ypa.values + 10, sm.add_constant(sub[F].fillna(med)),
                family=sm.families.Gamma(link=sm.families.links.Log())).fit()
    res = (sub.ypa.values + 10) / np.clip(_a(my.predict(sm.add_constant(sub[F].fillna(med)))), 1e-6, None)
    k = max(1.0 / np.var(res), 0.5)
    def predict(te, S=N_SIM):
        mu, n_, p_, capped = pa(te)
        mu_y = np.clip(_a(my.predict(sm.add_constant(te[F].fillna(med), has_constant="add"))) - 10, 0.3, None)
        draws = np.minimum(RNG.negative_binomial(np.clip(n_, 0.1, 1e6)[:, None],
                                                 np.clip(p_, 1e-6, 1 - 1e-9)[:, None], size=(len(te), S)), 60)
        shape = np.clip(draws * k, 1e-6, None)
        yd = np.where(draws == 0, 0.0, RNG.gamma(shape, np.broadcast_to((mu_y / k)[:, None], shape.shape)))
        return mu, yd, capped
    return predict

TEAM_F = ["t_yds_e4","t_att_e4","t_cmp_e4","t_rush_e4","t_db_e4","implied_total",
          "team_spread","total_line","opp_implied","d_allow_e4"]
SHARE_F = ["yds_share_e4","tgt_share_e4","wopr_e4","rec_e4","targets_e4","snap_e3",
           "snap_trend","ay_share_e4","dnp","implied_total","team_spread"]

def build_team_frame(raw, frame):
    """Team-game passing totals with lagged features, plus each player's lagged share.
    Team passing is modelled directly rather than via the QB: lower absolute error
    (MAE 54 vs 66) and it avoids having to identify the starter."""
    tg = (raw.groupby(["game_id","season","week","posteam","defteam"])
          .agg(t_yds=("rec_yds","sum"), t_att=("pass_att","sum"), t_cmp=("completions","sum"),
               t_rush=("rush_att","sum"), t_db=("dropbacks","sum")).reset_index())
    ctx = (raw.groupby(["game_id","posteam"])
           .agg(implied_total=("implied_total","first"), team_spread=("team_spread","first"),
                total_line=("total_line","first"), opp_implied=("opp_implied","first")).reset_index())
    tg = tg.merge(ctx, on=["game_id","posteam"], how="left").sort_values(["posteam","season","week"])
    g = tg.groupby("posteam", group_keys=False)
    for c in ["t_yds","t_att","t_cmp","t_rush","t_db"]:
        tg[c+"_e4"] = g[c].apply(lambda s: s.shift(1).ewm(halflife=4, min_periods=1).mean())
    dg = (tg.groupby(["defteam","season","week"]).agg(d_allow=("t_yds","first")).reset_index()
          .sort_values(["defteam","season","week"]))
    dg["d_allow_e4"] = (dg.groupby("defteam", group_keys=False)["d_allow"]
                        .apply(lambda s: s.shift(1).ewm(halflife=4, min_periods=1).mean()))
    tg = tg.merge(dg[["defteam","season","week","d_allow_e4"]], on=["defteam","season","week"], how="left")
    tg["gidx"] = g.cumcount()
    frame = frame.merge(tg[["game_id","posteam","t_yds"]], on=["game_id","posteam"], how="left")
    frame["yds_share"] = np.where(frame.t_yds > 0, frame.rec_yds / frame.t_yds, 0.0)
    frame = frame.sort_values(["player_id","season","week","is_pred"])
    frame["yds_share_e4"] = (frame.groupby("player_id", group_keys=False)["yds_share"]
                             .apply(lambda s: s.shift(1).ewm(halflife=4, min_periods=1).mean()))
    return tg, frame

def fit_rec_yards_normalized(tr, te, tg, group, rep=None):
    """Receiving yards = team passing yards x player share of it.
    Consistent by construction, and materially better calibrated than fitting each
    player independently: 7 of 27 lines off vs 16 of 27, mean error 1.6pp vs 3.6pp
    across 2024-26. The independent version was worst at the DISTRIBUTION CENTRE
    (+5.1pp at the median), which is exactly where betting decisions get made."""
    tv = tg[(tg.gidx >= 3) & tg.implied_total.notna() & tg[TEAM_F].notna().all(axis=1)]
    tv = tv[tv.season.isin(tr.season.unique())]
    if len(tv) < 200 or len(tr) < 300:
        return None, None
    tmed = tv[TEAM_F].median()
    mt = sm.GLM(tv.t_yds.values, sm.add_constant(tv[TEAM_F].fillna(tmed)),
                family=sm.families.Gaussian()).fit()
    rsd = float(np.std(tv.t_yds.values - _a(mt.predict(sm.add_constant(tv[TEAM_F].fillna(tmed))))))
    tt = te[["game_id","posteam"]].merge(tg[["game_id","posteam"]+TEAM_F], on=["game_id","posteam"], how="left")
    mu_t = np.clip(_a(mt.predict(sm.add_constant(tt[TEAM_F].fillna(tmed), has_constant="add"))), 40, None)
    s_ = tr.dropna(subset=["yds_share_e4"])
    s_ = s_[(s_.yds_share >= 0) & (s_.yds_share <= 1)]
    smed = s_[SHARE_F].median()
    ys = np.clip(s_.yds_share.values, 1e-4, 1-1e-4)
    ms = sm.GLM(ys, sm.add_constant(s_[SHARE_F].fillna(smed)), family=sm.families.Binomial()).fit()
    mu_s = np.clip(_a(ms.predict(sm.add_constant(te[SHARE_F].fillna(smed), has_constant="add"))), 1e-4, 0.95)
    fs = np.clip(_a(ms.predict(sm.add_constant(s_[SHARE_F].fillna(smed)))), 1e-4, 0.95)
    # method of moments. The MLE alternative fits the full beta density and calibrates
    # WORSE here (mean error 3.1pp vs 1.1pp) because we care about the tails of a
    # product distribution, not the density. Do not "improve" this to MLE without retesting.
    kappa = max(np.mean(fs*(1-fs)) / max(np.var(ys - fs), 1e-6) - 1, 1.0)
    if rep is not None:
        lo, hi = KAPPA_BANDS.get(group, (0, 1e9))
        rep.note("KAPPA", f"{group} receiving-yards dispersion kappa = {kappa:.2f} "
                          f"(expected {lo}-{hi}); team model residual sd {rsd:.0f} yds")
        if not (lo <= kappa <= hi):
            rep.warn("KAPPA_DRIFT",
                     f"{group} kappa is {kappa:.2f}, outside the expected {lo}-{hi}. This parameter "
                     f"sets where the modelled MEDIAN sits, so every receiving-yards line on the "
                     f"board has shifted in the same direction. Treat apparent edges as the model's, "
                     f"not the market's, until this is investigated.")
    team_draw = np.maximum(RNG.normal(mu_t[:, None], rsd, size=(len(te), N_SIM)), 0)
    a_ = np.clip(mu_s*kappa, 0.05, None)[:, None]
    b_ = np.clip((1-mu_s)*kappa, 0.05, None)[:, None]
    share_draw = RNG.beta(np.broadcast_to(a_, (len(te), N_SIM)), np.broadcast_to(b_, (len(te), N_SIM)))
    return team_draw * share_draw, kappa

def fit_td(tr, F):
    """Poisson (dispersion tested at 1.10 — NB adds nothing), lambda capped.
    The cap matters: only ~0.6% of training rows exceed lambda=1.0, so anything
    above that is extrapolation. Uncapped, the model priced a player at -540."""
    med = tr[F].median(); X = sm.add_constant(tr[F].fillna(med))
    m = sm.GLM(tr.tds.values, X, family=sm.families.Poisson()).fit()
    cap = float(np.percentile(_a(m.predict(X)), CAP_PCTILE))
    def predict(te):
        raw = np.clip(_a(m.predict(sm.add_constant(te[F].fillna(med), has_constant="add"))), 1e-6, 10)
        return np.minimum(raw, cap), (raw > cap)
    return predict, m, cap

# ----------------------------------------------------------------------------- warnings
class Report:
    def __init__(self):
        self.warnings, self.notes, self.commentary = [], [], []
    def warn(self, code, msg): self.warnings.append({"code": code, "message": msg})
    def note(self, code, msg): self.notes.append({"code": code, "message": msg})
    def say(self, msg): self.commentary.append(msg)
    def to_dict(self):
        return {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "warnings": self.warnings, "notes": self.notes, "commentary": self.commentary}
    def render(self):
        out = []
        if self.warnings:
            out.append("WARNINGS")
            for w in self.warnings: out.append(f"  [{w['code']}] {w['message']}")
        if self.notes:
            out.append("\nNOTES")
            for n in self.notes: out.append(f"  [{n['code']}] {n['message']}")
        if self.commentary:
            out.append("\nCOMMENTARY")
            for c in self.commentary: out.append(f"  - {c}")
        return "\n".join(out)

def check_data_freshness(frame, season, week, rep):
    done = frame[(frame.season == season) & (frame.week < week)]
    if done.empty:
        rep.warn("NO_SEASON_DATA", f"no completed {season} games in the data; "
                                   f"every feature is carried over from prior seasons")
        return
    last = int(done.week.max())
    if last < week - 1:
        rep.warn("STALE_DATA", f"latest completed week in the data is {last}, but pricing week {week}. "
                               f"Re-run with --refresh; nflverse may not have posted week {week-1} yet.")
    cov = done[done.season == season].offense_pct.notna().mean()
    if cov < 0.5:
        rep.warn("SNAP_COVERAGE", f"snap counts present for only {100*cov:.0f}% of {season} rows; "
                                  f"snap_trend and snap_e3 will be unreliable")

# Variables that also appear inside an interaction. Their standalone coefficient is the
# effect at zero for the other term, which is meaningless here, so the sign check has to
# use the MARGINAL effect evaluated at the sample mean instead.
INTERACTIONS = {"rz_share_e4": ("rz_x_total", "implied_total")}

def check_signs(model, features, rep, data=None):
    params = dict(zip(["const"] + list(features), np.asarray(model.params).ravel()))
    for f, want in EXPECTED_SIGNS.items():
        if f not in params: continue
        got = params[f]
        detail = ""
        if f in INTERACTIONS and data is not None:
            inter, modifier = INTERACTIONS[f]
            if inter in params and modifier in data:
                at = float(np.nanmean(data[modifier]))
                got = params[f] + params[inter] * at
                detail = (f" (marginal effect at mean {modifier}={at:.1f}; "
                          f"standalone coefficient {params[f]:+.4f} is not interpretable "
                          f"because '{f}' also enters via '{inter}')")
        if np.sign(got) != want and abs(got) > 1e-3:
            rep.warn("SIGN_FLIP", f"effect of '{f}' is {got:+.4f}, expected "
                                  f"{'positive' if want > 0 else 'negative'}{detail}. "
                                  f"This usually means a data problem upstream, not a real finding.")

def check_drift(frame, season, week, td_predict, rep):
    """Compare modeled vs actual on completed weeks of the live season."""
    done = frame[(frame.season == season) & (frame.week < week) &
                 (frame.prior_games >= MIN_PRIOR_GAMES) & frame.implied_total.notna() &
                 frame.position.isin(["RB","WR","TE"])]
    if len(done) < 150:
        rep.note("DRIFT_SKIPPED", f"only {len(done)} completed rows this season; "
                                  f"drift check needs ~150. Nothing to compare yet.")
        return
    lam, _ = td_predict(done)
    pred, act = (1 - np.exp(-lam)).mean(), done.scored.mean()
    se = np.sqrt(act * (1 - act) / len(done))
    diff = 100 * (pred - act)
    if abs(pred - act) > 1.96 * se:
        rep.warn("SEASON_DRIFT", f"anytime-TD model is off by {diff:+.1f}pp on {len(done)} completed "
                                 f"{season} games (modeled {100*pred:.1f}% vs actual {100*act:.1f}%), "
                                 f"outside sampling error. Treat TD prices as directional.")
    else:
        rep.say(f"Anytime-TD calibration on {len(done)} completed {season} games: "
                f"modeled {100*pred:.1f}% vs actual {100*act:.1f}% ({diff:+.1f}pp) — within sampling error.")

def market_warnings(rep):
    for mkt, (tier, why) in MARKET_RELIABILITY.items():
        if tier == "UNRELIABLE":
            rep.warn("MARKET_UNRELIABLE", f"{mkt}: {why}")
        elif tier == "WEAK":
            rep.warn("MARKET_WEAK", f"{mkt}: {why}")
        elif tier == "FAIR":
            rep.note("MARKET_FAIR", f"{mkt}: {why}")

def roster_warnings(board, rep, season, week):
    thin = board[board.prior_games < 6]
    if len(thin):
        rep.note("THIN_HISTORY", f"{len(thin)} players have fewer than 6 prior games; "
                                 f"their role estimates are mostly positional priors")
    if "capped" in board and board.capped.any():
        names = ", ".join(board[board.capped].player_name.head(5))
        rep.note("LAMBDA_CAP", f"{int(board.capped.sum())} player(s) hit the extrapolation cap "
                               f"({names}). Their true probability may be higher; the cap is deliberate.")
    dnp = board[board.dnp == 1]
    if len(dnp):
        rep.note("DNP", f"{len(dnp)} priced player(s) did not practice: "
                        f"{', '.join(dnp.player_name.head(6))}")

# ----------------------------------------------------------------------------- pricing
def build_slate(datadir, frame, season, week, rep):
    g = pd.read_csv(os.path.join(datadir, "games.csv"))
    slate = g[(g.season == season) & (g.week == week) & (g.game_type == "REG")]
    if slate.empty:
        raise SystemExit(f"no scheduled games found for {season} week {week}")
    no_line = slate.spread_line.isna() | slate.total_line.isna()
    if no_line.all():
        raise SystemExit(
            f"REFUSING TO PRICE: none of the {len(slate)} games in {season} week {week} have a posted "
            f"spread or total yet.\n"
            f"Implied team total is the single largest input to this model, so without lines every "
            f"price would be a guess dressed up as a number.\n"
            f"Lines usually post ~6 days out. Re-run with --refresh closer to kickoff.")
    if no_line.any():
        rep.warn("MISSING_LINES", f"{int(no_line.sum())} of {len(slate)} game(s) have no posted "
                                  f"spread/total: {', '.join(slate[no_line].game_id.head(4))}. "
                                  f"Players in those games are priced off a median fallback and "
                                  f"should not be trusted.")
    rows = []
    for _, r in slate.iterrows():
        for t, o, h in [(r.home_team, r.away_team, 1), (r.away_team, r.home_team, 0)]:
            rows.append(dict(game_id=r.game_id, season=season, week=week, posteam=t, defteam=o,
                             is_home=h, sp=r.spread_line, tt=r.total_line))
    sl = pd.DataFrame(rows)
    recent = frame[(frame.season == season) & (frame.week >= week - 3) & (frame.week < week)]
    if recent.empty:
        recent = frame[frame.season == frame.season.max()]
        rep.warn("ROSTER_FALLBACK", "no recent games this season; roster built from the latest "
                                    "available season, so it may include players who have moved teams")
    roster = (recent.groupby(["posteam","player_id","player_name","position"])
              .agg(opps=("opportunities","sum"), patt=("pass_att","sum")).reset_index())
    roster = roster[(roster.opps >= 3) | (roster.patt >= 20)]
    pred = sl.merge(roster, on="posteam")
    pred["is_pred"] = 1
    base = frame.copy(); base["is_pred"] = 0
    allf = pd.concat([base, pred], ignore_index=True)
    allf = add_lags(allf)
    m = allf.is_pred == 1
    allf.loc[m, "implied_total"] = np.where(allf.loc[m,"is_home"] == 1,
                                            allf.loc[m,"tt"]/2 + allf.loc[m,"sp"]/2,
                                            allf.loc[m,"tt"]/2 - allf.loc[m,"sp"]/2)
    allf.loc[m, "team_spread"] = np.where(allf.loc[m,"is_home"] == 1, allf.loc[m,"sp"], -allf.loc[m,"sp"])
    allf.loc[m, "opp_implied"] = allf.loc[m,"tt"] - allf.loc[m,"implied_total"]
    allf.loc[m, "total_line"] = allf.loc[m,"tt"]
    allf["rz_x_total"] = allf.rz_share_e4 * allf.implied_total
    for c in ["tm_db_e4","tm_rush_e4"]:
        last = frame[frame.season == season].sort_values("week").groupby("posteam")[c].last()
        allf.loc[m, c] = allf.loc[m, "posteam"].map(last)
    for c in ["d_db_e4","d_rush_e4","d_epa_e4"]:
        last = frame[frame.season == season].sort_values("week").groupby("defteam")[c].last()
        allf.loc[m, c] = allf.loc[m, "defteam"].map(last)
    allf["dnp"] = allf.dnp.fillna(0); allf["out_flag"] = allf.out_flag.fillna(0)
    return allf, len(slate)

def price_week(datadir, outdir, season, week, rep, as_json=False):
    frame = build_frame(datadir)
    check_data_freshness(frame, season, week, rep)
    allf, n_games = build_slate(datadir, frame, season, week, rep)
    team_tg, allf = build_team_frame(frame, allf)
    TRN = allf[(allf.is_pred == 0) & (allf.prior_games >= MIN_PRIOR_GAMES) & allf.implied_total.notna()]
    TST = allf[(allf.is_pred == 1) & (allf.prior_games >= MIN_PRIOR_GAMES) & (allf.out_flag == 0)].copy()
    rep.say(f"Trained on {len(TRN):,} player-games; pricing {len(TST)} players across {n_games} games.")

    s_tr = TRN[TRN.position.isin(["RB","WR","TE"])]
    s_te = TST[TST.position.isin(["RB","WR","TE"])].copy()
    td_predict, td_model, td_cap = fit_td(s_tr, TD_F)
    check_signs(td_model, TD_F, rep, data=s_tr)
    check_drift(allf[allf.is_pred == 0], season, week, td_predict, rep)
    lam, capped = td_predict(s_te)
    s_te["AnyTD"] = 1 - np.exp(-lam)
    s_te["TD2"] = 1 - np.exp(-lam) - lam * np.exp(-lam)
    s_te["capped"] = capped
    rep.say(f"Anytime-TD ceiling this week is {100*(1-np.exp(-td_cap)):.1f}% "
            f"({american(1-np.exp(-td_cap))}); nothing prices shorter than that by design.")

    # --- RB rushing (unchanged) ---
    rb_tr = TRN[(TRN.position == "RB") & (TRN.rush_att_e4 > 3)]
    rb_te = TST[(TST.position == "RB") & (TST.rush_att_e4 > 3)].copy()
    if len(rb_te) and len(rb_tr) > 300:
        mu, yd, _ = fit_yards(rb_tr, SKILL_F, "rush_att", "rush_yds")(rb_te)
        rb_te["RushYds_ln"] = np.median(yd, axis=1)
        rb_te["RushYds_o"] = [american((yd[i] > np.floor(np.median(yd[i])) - 0.5).mean()) for i in range(len(rb_te))]
        pc, _ = fit_count(rb_tr, SKILL_F, "rush_att"); _, n_, p_, _ = pc(rb_te)
        rb_te["RushAtt_ln"] = np.floor(st.nbinom.ppf(0.5, n_, p_)) + 0.5
        rb_te["RushAtt_o"] = [american(1 - st.nbinom.cdf(np.floor(l), n_[i], p_[i]))
                              for i, l in enumerate(rb_te.RushAtt_ln)]

    # --- RECEPTIONS: one POOLED model across RB/WR/TE ---
    # RBs take ~17% of team targets. Pooling beat separate fits on BOTH groups
    # (RB mean error 1.98 -> 1.78pp; WR/TE 1.66 -> 1.39pp) on the 2026 holdout.
    rec_tr = TRN[TRN.position.isin(["RB","WR","TE"]) & (TRN.targets_e4 >= 1.5)]
    rec_te = TST[TST.position.isin(["RB","WR","TE"]) & (TST.targets_e4 >= 1.5)].copy()
    if len(rec_te) and len(rec_tr) > 300:
        pc, _ = fit_count(rec_tr, SKILL_F, "rec"); _, n_, p_, _ = pc(rec_te)
        rec_te["Rec_ln"] = np.floor(st.nbinom.ppf(0.5, n_, p_)) + 0.5
        rec_te["Rec_o"] = [american(1 - st.nbinom.cdf(np.floor(l), n_[i], p_[i]))
                           for i, l in enumerate(rec_te.Rec_ln)]

    # --- RECEIVING YARDS: normalized (team passing yards x player share) ---
    # Replaces independent per-player fits. Validated 2024-26: 7/27 lines significantly
    # miscalibrated vs 16/27, mean error 1.61pp vs 3.62pp. Kept position-split because
    # the dispersion parameter differs materially (WR/TE ~8.5, RB ~11.2).
    wr_tr = TRN[TRN.position.isin(["WR","TE"]) & (TRN.targets_e4 > 2)]
    wr_te = TST[TST.position.isin(["WR","TE"]) & (TST.targets_e4 > 2)].copy()
    if len(wr_te) and len(wr_tr) > 300:
        yd, _k = fit_rec_yards_normalized(wr_tr, wr_te, team_tg, "WRTE", rep)
        if yd is None:
            yd, _, _ = fit_yards(wr_tr, SKILL_F, "rec", "rec_yds")(wr_te)
            rep.warn("NORM_FALLBACK", "not enough team history for the normalized receiving-yards "
                                      "model; fell back to the older independent fit, which is "
                                      "miscalibrated by about +5pp near the middle of the distribution")
        wr_te["RecYds_ln"] = np.median(yd, axis=1)
        wr_te["RecYds_o"] = [american((yd[i] > np.floor(np.median(yd[i])) - 0.5).mean()) for i in range(len(wr_te))]
    rby_tr = TRN[(TRN.position == "RB") & (TRN.targets_e4 >= 1.5)]
    rby_te = TST[(TST.position == "RB") & (TST.targets_e4 >= 1.5)].copy()
    if len(rby_te) and len(rby_tr) > 300:
        yd, _k = fit_rec_yards_normalized(rby_tr, rby_te, team_tg, "RB", rep)
        if yd is None:
            yd, _, _ = fit_yards(rby_tr, SKILL_F, "rec", "rec_yds")(rby_te)
        rby_te["RecYds_ln"] = np.median(yd, axis=1)
        rby_te["RecYds_o"] = [american((yd[i] > np.floor(np.median(yd[i])) - 0.5).mean()) for i in range(len(rby_te))]

    board = s_te
    rec_yds_all = pd.concat([d for d in (wr_te if len(wr_te) else None,
                                         rby_te if len(rby_te) else None) if d is not None]) \
                  if (len(wr_te) or len(rby_te)) else wr_te
    for src, cols in [(rb_te, ["RushAtt_ln","RushAtt_o","RushYds_ln","RushYds_o"]),
                      (rec_te, ["Rec_ln","Rec_o"]),
                      (rec_yds_all, ["RecYds_ln","RecYds_o"])]:
        have = [c for c in cols if c in src.columns]
        if len(src) and have:
            board = board.merge(src[["player_id","game_id"] + have], on=["player_id","game_id"], how="left")
    board["AnyTD_odds"] = board.AnyTD.apply(american)
    board["TD2_odds"] = board.TD2.apply(american)
    roster_warnings(board, rep, season, week)
    market_warnings(rep)

    keep = ["player_name","position","posteam","defteam","implied_total","AnyTD","AnyTD_odds",
            "TD2","TD2_odds","RushAtt_ln","RushAtt_o","RushYds_ln","RushYds_o",
            "Rec_ln","Rec_o","RecYds_ln","RecYds_o"]
    keep = [c for c in keep if c in board.columns]
    out = board[keep].copy().sort_values("AnyTD", ascending=False)
    for c in ["AnyTD","TD2"]: out[c] = (100 * out[c]).round(1)
    for c in ["RushYds_ln","RecYds_ln"]:
        if c in out: out[c] = out[c].round(1)
    os.makedirs(outdir, exist_ok=True)
    skill_path = os.path.join(outdir, f"week{week}_skill.csv")
    out.to_csv(skill_path, index=False)

    qb_tr = TRN[(TRN.position == "QB") & (TRN.pass_att_e4 > 15)]
    qb_te = TST[(TST.position == "QB") & (TST.pass_att_e4 > 15)].copy()
    qb_path = None
    if len(qb_te) and len(qb_tr) > 200:
        rec = (frame[(frame.season == season) & (frame.week >= week - 2)]
               .groupby(["posteam","player_id"]).pass_att.sum().rename("r_att").reset_index())
        qb_te = qb_te.merge(rec, on=["posteam","player_id"], how="left")
        qb_te["r_att"] = qb_te.r_att.fillna(0)
        qb_te = qb_te.sort_values("r_att", ascending=False).groupby("posteam", as_index=False).head(1)
        rep.warn("QB_STARTER_GUESS", "the QB sheet picks one starter per team by recent attempt volume, "
                                     "not by depth chart. Verify before using any QB row.")
        for nm, col in [("PassAtt","pass_att"), ("Cmp","completions")]:
            pc, _ = fit_count(qb_tr, QB_F, col); mu, n_, p_, _ = pc(qb_te)
            qb_te[nm + "_ln"] = np.floor(st.nbinom.ppf(0.5, n_, p_)) + 0.5
            qb_te[nm + "_o"] = [american(1 - st.nbinom.cdf(np.floor(l), n_[i], p_[i]))
                                for i, l in enumerate(qb_te[nm + "_ln"])]
        mu, yd, _ = fit_yards(qb_tr, QB_F, "pass_att", "pass_yds")(qb_te)
        qb_te["PassYds_ln"] = np.median(yd, axis=1).round(1)
        medq = qb_tr[QB_F].median()
        mtd = sm.GLM(qb_tr.pass_td.values, sm.add_constant(qb_tr[QB_F].fillna(medq)),
                     family=sm.families.Poisson()).fit()
        cap = float(np.percentile(_a(mtd.predict(sm.add_constant(qb_tr[QB_F].fillna(medq)))), CAP_PCTILE))
        l = np.minimum(np.clip(_a(mtd.predict(sm.add_constant(qb_te[QB_F].fillna(medq), has_constant="add"))), 1e-6, 8), cap)
        qb_te["xPassTD"] = l.round(2)
        qb_te["o1.5TD"] = [american(1 - st.poisson.cdf(1, x)) for x in l]
        qcols = ["player_name","posteam","defteam","implied_total","PassAtt_ln","PassAtt_o",
                 "Cmp_ln","Cmp_o","PassYds_ln","xPassTD","o1.5TD"]
        qcols = [c for c in qcols if c in qb_te.columns]
        qb_path = os.path.join(outdir, f"week{week}_qb.csv")
        qb_te[qcols].sort_values("PassYds_ln", ascending=False).to_csv(qb_path, index=False)

    top = out.head(3)
    if len(top):
        rep.say("Shortest anytime-TD prices: " +
                ", ".join(f"{r.player_name} ({r.posteam}) {r.AnyTD_odds}" for _, r in top.iterrows()))
    hot = out.nlargest(1, "implied_total")
    if len(hot):
        r = hot.iloc[0]
        rep.say(f"Highest implied team total on the slate: {r.posteam} at {r.implied_total:.2f} "
                f"vs {r.defteam} — the model's TD prices lean on this heavily.")
    rep.say("A modelled line is a MEDIAN, not an expected value. On right-skewed markets "
            "(receiving and rushing yards) the mean sits well above it, so do not read a low "
            "yardage line as a projection of what the player will average.")
    rep.say("Modeled prices carry NO vig. Your book's number will be shorter on both sides, "
            "so a small gap is not an edge — only a large one is.")
    rep.say("Parlay construction: same-game legs from OPPOSITE teams ran 1.136x independent "
            "(95% CI 1.061-1.213). Same-backfield RB/RB ran 0.823x. Stack across sides, not within.")
    return skill_path, qb_path, out

# ----------------------------------------------------------------------------- validate
def validate(datadir, rep):
    frame = build_frame(datadir)
    f = add_lags(frame.assign(is_pred=0))
    f = f[(f.prior_games >= MIN_PRIOR_GAMES) & f.implied_total.notna()]
    seasons = sorted(f.season.unique())
    test = seasons[-1]
    tr, te = f[f.season < test], f[f.season == test]
    s_tr = tr[tr.position.isin(["RB","WR","TE"])]; s_te = te[te.position.isin(["RB","WR","TE"])].copy()
    if len(s_te) < 100:
        rep.warn("VALIDATE_THIN", f"only {len(s_te)} rows in {test}; validation is not meaningful yet")
        return
    from sklearn.metrics import brier_score_loss, roc_auc_score
    pred, _, _ = fit_td(s_tr, TD_F)
    lam, _ = pred(s_te); p = 1 - np.exp(-lam)
    print(f"\nHOLDOUT {test}: n={len(s_te)}  Brier={brier_score_loss(s_te.scored, p):.5f}  "
          f"AUC={roc_auc_score(s_te.scored, p):.4f}  modeled={100*p.mean():.2f}%  actual={100*s_te.scored.mean():.2f}%")
    wk = []
    for w, s in s_te.groupby("week"):
        if len(s) < 50: continue
        lw, _ = pred(s)
        try: wk.append(roc_auc_score(s.scored, 1 - np.exp(-lw)))
        except Exception: pass
    if len(wk) > 1:
        print(f"  weekly AUC: mean={np.mean(wk):.4f} sd={np.std(wk):.4f} "
              f"(one week tells you almost nothing; ~17 weeks before results separate from noise)")

# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description="NFL player-prop model")
    ap.add_argument("--week", type=int, help="week to price")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--datadir", default="./nfl_data")
    ap.add_argument("--outdir", default="./nfl_out")
    ap.add_argument("--refresh", action="store_true", help="force re-download")
    ap.add_argument("--validate", action="store_true", help="run holdout validation and exit")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    a = ap.parse_args()
    rep = Report()
    fetch(a.datadir, a.refresh)
    if a.validate:
        validate(a.datadir, rep)
        print("\n" + rep.render()); return
    if not a.week:
        ap.error("--week is required unless --validate")
    skill_path, qb_path, out = price_week(a.datadir, a.outdir, a.season, a.week, rep, a.json)
    if a.json:
        print(json.dumps({**rep.to_dict(), "skill_csv": skill_path, "qb_csv": qb_path,
                          "n_players": int(len(out))}, indent=2))
    else:
        print(f"\nwrote {skill_path}" + (f"\nwrote {qb_path}" if qb_path else ""))
        print("\n" + rep.render())
    with open(os.path.join(a.outdir, f"week{a.week}_report.json"), "w") as fh:
        json.dump(rep.to_dict(), fh, indent=2)

if __name__ == "__main__":
    main()
