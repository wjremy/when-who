"""Chunked CGM reduction for the REPLACE-BG daily analysis panel."""
from pathlib import Path
import argparse
import os
import sqlite3
import tempfile
import numpy as np
import pandas as pd


def slot_day(day, seconds):
    """Round CGM times to five-minute slots, carrying slot 288 into tomorrow."""
    raw_slot = np.floor((seconds + 150) / 300).astype("int32")
    rollover = raw_slot == 288
    assigned_day = np.asarray(day, dtype="int32") + rollover.astype("int32")
    return assigned_day, np.mod(raw_slot, 288).astype("int16"), rollover


def _old_slot_day(day, seconds):
    """Reproduce the pre-fix day assignment for the optional comparison only."""
    raw_slot = np.floor((seconds + 150) / 300).astype("int16")
    return np.asarray(day), raw_slot % 288, raw_slot == 288


def changed_patient_nights(old_panel, new_panel):
    """Count final panel nights with any changed label, feature, or panel field."""
    keys = ["pid", "day"]
    old = old_panel.copy()
    new = new_panel.copy()
    old["_old_present"] = True
    new["_new_present"] = True
    joined = old.merge(new, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True)
    changed = joined["_merge"].ne("both")
    for col in (set(old_panel.columns) & set(new_panel.columns)) - set(keys):
        same = joined[f"{col}_old"].eq(joined[f"{col}_new"])
        same |= joined[f"{col}_old"].isna() & joined[f"{col}_new"].isna()
        changed |= ~same.fillna(False)
    return int(changed.sum())


def sustained(v, threshold):
    z = np.isfinite(v) & (v < threshold)
    return bool(np.any(z[:-2] & z[1:-1] & z[2:]))


def global_duplicate_medians(path, cols, chunksize=350_000, slot_assigner=slot_day):
    """Find cross-chunk slot keys, then compute medians over every raw reading.

    The SQLite index stays on disk; only one input chunk and the small set of
    cross-chunk keys are held in memory at a time.
    """
    fd, dbpath = tempfile.mkstemp(prefix="replace_bg_slots_", suffix=".sqlite")
    os.close(fd)
    try:
        con = sqlite3.connect(dbpath)
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.execute("CREATE TABLE seen (p INTEGER,d INTEGER,s INTEGER,last_chunk INTEGER,n_chunks INTEGER,"
                     "PRIMARY KEY(p,d,s)) WITHOUT ROWID")
        upsert = ("INSERT INTO seen VALUES(?,?,?,?,1) ON CONFLICT(p,d,s) DO UPDATE SET "
                  "n_chunks=n_chunks+(last_chunk<>excluded.last_chunk),last_chunk=excluded.last_chunk")
        chunk_count = 0
        for chunk_count, ch in enumerate(pd.read_csv(path, sep="|", usecols=cols,
                dtype={"PtID":"int32", "DeviceDtTmDaysFromEnroll":"int16", "DeviceTm":"string",
                       "RecordType":"string", "GlucoseValue":"float32"}, chunksize=chunksize), start=1):
            ch = ch[ch.RecordType.eq("CGM") & ch.GlucoseValue.notna()]
            seconds = pd.to_timedelta(ch.DeviceTm, errors="coerce").dt.total_seconds().to_numpy()
            day, slot, _ = slot_assigner(ch.DeviceDtTmDaysFromEnroll.to_numpy(), seconds)
            keys = pd.DataFrame({"p":ch.PtID.to_numpy(), "d":day, "s":slot}).drop_duplicates()
            con.executemany(upsert, ((int(p),int(d),int(s),chunk_count-1) for p,d,s in keys.itertuples(index=False,name=None)))
            con.commit()
        dup = con.execute("SELECT p,d,s FROM seen WHERE n_chunks>1").fetchall()
        con.close()
        key_frame = pd.DataFrame(dup, columns=["p","d","s"])
        if not dup:
            print(f"Cross-chunk duplicate slots: 0 across {chunk_count} chunks")
            return {}
        dup_keys = {(int(p),int(d),int(s)) for p,d,s in dup}
        values = {k:[] for k in dup_keys}
        for ch in pd.read_csv(path, sep="|", usecols=cols,
                dtype={"PtID":"int32", "DeviceDtTmDaysFromEnroll":"int16", "DeviceTm":"string",
                       "RecordType":"string", "GlucoseValue":"float32"}, chunksize=chunksize):
            ch = ch[ch.RecordType.eq("CGM") & ch.GlucoseValue.notna()]
            if ch.empty: continue
            seconds = pd.to_timedelta(ch.DeviceTm, errors="coerce").dt.total_seconds().to_numpy()
            day, slot, _ = slot_assigner(ch.DeviceDtTmDaysFromEnroll.to_numpy(), seconds)
            z = pd.DataFrame({"p":ch.PtID.to_numpy(), "d":day,
                              "s":slot,
                              "g":ch.GlucoseValue.to_numpy("float32")})
            z = z.merge(key_frame, on=["p","d","s"], how="inner")
            for p,d,s,g in z.itertuples(index=False,name=None): values[(int(p),int(d),int(s))].append(float(g))
        medians = {k:np.float32(np.median(np.asarray(v,dtype=np.float32))) for k,v in values.items()}
        print(f"Cross-chunk duplicate slots: {len(dup):,}; patient-days: {key_frame[['p','d']].drop_duplicates().shape[0]:,}; "
              f"patient-nights: {key_frame.loc[key_frame.s<72,['p','d']].drop_duplicates().shape[0]:,}; chunks: {chunk_count}")
        return medians
    finally:
        try: os.remove(dbpath)
        except FileNotFoundError: pass


def build_panel(slot_assigner=slot_day, write_panel=True):
    root = Path(__file__).resolve().parent / "replace_bg" / "Data Tables"
    cgm_path = root / "HDeviceCGM.txt"
    slots = {}
    cols = ["PtID", "DeviceDtTmDaysFromEnroll", "DeviceTm", "RecordType", "GlucoseValue"]
    duplicate_medians = global_duplicate_medians(cgm_path, cols, slot_assigner=slot_assigner)
    nread = 0
    n_reassigned = 0
    for ch in pd.read_csv(cgm_path, sep="|", usecols=cols,
                          dtype={"PtID": "int32", "DeviceDtTmDaysFromEnroll": "int16",
                                 "DeviceTm": "string", "RecordType": "string",
                                 "GlucoseValue": "float32"}, chunksize=350_000):
        nread += len(ch)
        ch = ch[ch.RecordType.eq("CGM") & ch.GlucoseValue.notna()]
        if ch.empty:
            continue
        seconds = pd.to_timedelta(ch.DeviceTm, errors="coerce").dt.total_seconds().to_numpy()
        day, slot, rollover = slot_assigner(ch.DeviceDtTmDaysFromEnroll.to_numpy(), seconds)
        n_reassigned += int(rollover.sum())
        x = pd.DataFrame({"p": ch.PtID.to_numpy(),
                          "d": day,
                          "s": slot,
                          "g": ch.GlucoseValue.to_numpy()})
        x = x.groupby(["p", "d", "s"], sort=False, as_index=False).g.median()
        for (pid, day), g in x.groupby(["p", "d"], sort=False):
            a = slots.get((int(pid), int(day)))
            if a is None:
                a = np.full(288, np.nan, np.float32)
                slots[(int(pid), int(day))] = a
            a[g.s.to_numpy("int16")] = g.g.to_numpy("float32")
    for (pid, day, slot), value in duplicate_medians.items():
        a = slots.get((pid, day))
        if a is not None: a[slot] = value
    print(f"Read {nread:,} CGM rows; reduced to {len(slots):,} participant-days.")

    rows = []
    for (pid, day), v in slots.items():
        valid = np.isfinite(v); z = v[valid]; n = int(valid.sum())
        night = v[:72]; nn = int(np.isfinite(night).sum())
        if n:
            q = v[276:288]; qi = np.flatnonzero(np.isfinite(q))
            late_slope = float(np.polyfit(qi * 5, q[qi], 1)[0]) if len(qi) >= 2 else np.nan
            mean = float(z.mean()); sd = float(z.std(ddof=1)) if len(z) > 1 else 0.
            summary = [mean, sd, sd / mean if mean else np.nan,
                       float(np.sum(z < 70) / n), float(np.sum(z < 54) / n),
                       float(np.sum(z > 180) / n), float(z.min()), float(z[-1]), late_slope]
        else:
            summary = [np.nan] * 9
        rows.append([pid, day, n, *summary, nn, sustained(night, 54), sustained(night, 70),
                     sustained(v, 54), bool(np.any(np.isfinite(v[72:]) & (v[72:] < 54)))])
    daily = pd.DataFrame(rows, columns=["pid", "day", "coverage", "mean", "sd", "cv",
        "time_below70", "time_below54", "time_above180", "minimum", "last_value",
        "late_slope", "night_coverage", "night54", "night70", "day54_15m", "day54_any"])
    daily = daily.sort_values(["pid", "day"]).reset_index(drop=True)

    # Sum pump-recorded bolus components and wizard carbohydrate entries by enrollment day.
    bparts = []
    for c in pd.read_csv(root / "HDeviceBolus.txt", sep="|",
                         usecols=["PtID", "DeviceDtTmDaysFromEnroll", "Normal", "Extended"],
                         chunksize=500_000):
        c["dose"] = pd.to_numeric(c.Normal, errors="coerce").fillna(0) + pd.to_numeric(c.Extended, errors="coerce").fillna(0)
        c["has"] = c[["Normal", "Extended"]].notna().any(axis=1).astype(int)
        bparts.append(c.groupby(["PtID", "DeviceDtTmDaysFromEnroll"], as_index=False).agg(
            insulin_bolus=("dose", "sum"), bolus_records=("has", "sum")))
    bolus = pd.concat(bparts).groupby(["PtID", "DeviceDtTmDaysFromEnroll"], as_index=False).sum()
    bolus = bolus.rename(columns={"PtID": "pid", "DeviceDtTmDaysFromEnroll": "day"})
    wparts = []
    for c in pd.read_csv(root / "HDeviceWizard.txt", sep="|",
                         usecols=["PtId", "DeviceDtTmDaysFromEnroll", "CarbInput"], chunksize=500_000):
        c["carbs"] = pd.to_numeric(c.CarbInput, errors="coerce")
        c["n_carb"] = c.carbs.notna().astype(int)
        wparts.append(c.groupby(["PtId", "DeviceDtTmDaysFromEnroll"], as_index=False).agg(
            carbs=("carbs", "sum"), carb_records=("n_carb", "sum")))
    carbs = pd.concat(wparts).groupby(["PtId", "DeviceDtTmDaysFromEnroll"], as_index=False).sum()
    carbs = carbs.rename(columns={"PtId": "pid", "DeviceDtTmDaysFromEnroll": "day"})
    # Estimate basal units from the documented actual duration (ms) and rate (U/hour).
    # Split a record at midnight so its dose is assigned to the calendar day delivered.
    basal = pd.read_csv(root / "HDeviceBasal.txt", sep="|",
                        usecols=["PtID", "DeviceDtTmDaysFromEnroll", "DeviceTm", "Duration", "Rate"])
    basal["rate"] = pd.to_numeric(basal.Rate, errors="coerce")
    basal["duration_s"] = pd.to_numeric(basal.Duration, errors="coerce") / 1000
    basal["start_s"] = pd.to_timedelta(basal.DeviceTm, errors="coerce").dt.total_seconds()
    basal = basal[basal.rate.notna() & basal.duration_s.gt(0) & basal.start_s.notna()].copy()
    left = np.maximum(0, 86400 - basal.start_s.to_numpy(float))
    first = np.minimum(basal.duration_s.to_numpy(float), left)
    second = basal.duration_s.to_numpy(float) - first
    bp = basal.PtID.to_numpy(); bd = basal.DeviceDtTmDaysFromEnroll.to_numpy()
    br = basal.rate.to_numpy(float)
    basal_parts = [pd.DataFrame({"pid": bp, "day": bd, "dose": br * first / 3600,
                                 "basal_records": 1})]
    cross = second > 0
    if cross.any():
        basal_parts.append(pd.DataFrame({"pid": bp[cross], "day": bd[cross] + 1,
            "dose": br[cross] * second[cross] / 3600, "basal_records": 1}))
    basal_daily = pd.concat(basal_parts, ignore_index=True).groupby(["pid", "day"], as_index=False).agg(
        basal_insulin_est=("dose", "sum"), basal_records=("basal_records", "sum"))
    daily = daily.merge(bolus, on=["pid", "day"], how="left").merge(carbs, on=["pid", "day"], how="left")
    daily = daily.merge(basal_daily, on=["pid", "day"], how="left")
    daily["insulin_bolus_available"] = daily.bolus_records.notna().astype(int)
    daily["basal_insulin_available"] = daily.basal_records.notna().astype(int)
    daily["carbs_available"] = daily.carb_records.notna().astype(int)

    # Target night starts at day d; predictor summaries come from calendar day d-1.
    features = ["mean", "sd", "cv", "time_below70", "time_below54", "time_above180",
                "minimum", "last_value", "late_slope", "insulin_bolus", "basal_insulin_est", "carbs"]
    panel_rows = []
    for pid, g in daily.groupby("pid", sort=False):
        g = g.set_index("day").sort_index()
        for day, row in g.iterrows():
            if day - 1 not in g.index:
                continue
            prev = g.loc[day - 1]
            # 70% nominal coverage: 51/72 samples in night, 202/288 in predictor day.
            if row.night_coverage < 51 or prev.coverage < 202:
                continue
            rec = {"pid": pid, "day": day, "y": int(row.night54),
                   "y_night70": int(row.night70),
                   "y_24h54": int(row.day54_15m) if row.coverage >= 202 else np.nan,
                   "event_any54_24h": int(row.day54_any),
                   "night_coverage": int(row.night_coverage), "predictor_coverage": int(prev.coverage),
                   "insulin_bolus_available": int(prev.insulin_bolus_available),
                   "basal_insulin_available": int(prev.basal_insulin_available),
                   "carbs_available": int(prev.carbs_available)}
            rec.update({f: prev[f] for f in features})
            panel_rows.append(rec)
    panel = pd.DataFrame(panel_rows).sort_values(["pid", "day"]).reset_index(drop=True)
    pieces = []
    for pid, g in panel.groupby("pid", sort=False):
        g = g.copy().reset_index(drop=True)
        g["night_index"] = np.arange(len(g))
        g["model_runin"] = g.night_index < 14
        y = g.y.to_numpy(float)
        for w in (7, 28):
            rate = np.full(len(g), np.nan)
            for j in range(1, len(g)):
                rate[j] = y[max(0, j - w):j].mean()
            g[f"history_rate_{w}"] = rate
        g["expanding_rate"] = np.r_[np.nan, np.cumsum(y[:-1]) / np.arange(1, len(g))] if len(g) > 1 else np.nan
        g["runin_rate"] = float(y[:14].mean()) if len(g) >= 14 else np.nan
        pieces.append(g)
    panel = pd.concat(pieces, ignore_index=True)
    post_counts = panel.loc[~panel.model_runin].groupby("pid").size()
    panel = panel[panel.pid.isin(post_counts[post_counts >= 60].index)].copy().reset_index(drop=True)
    if write_panel:
        panel.to_csv(Path(__file__).resolve().parent / "replace_bg_panel_v2.csv", index=False)
    scored = panel[~panel.model_runin]
    rates = scored.groupby("pid").y.mean()
    gaps = np.concatenate([np.diff(g.day.to_numpy()) for _, g in panel.groupby("pid")])
    prefix = "Saved panel" if write_panel else "Compared panel"
    print(f"{prefix}: {len(panel):,} rows, {panel.pid.nunique():,} patients, {len(scored):,} post-run-in nights")
    print(f"Primary event rate {scored.y.mean():.5f}; patient rates mean {rates.mean():.5f}, SD {rates.std(ddof=1):.5f}")
    print(f"Median interval between eligible nights {np.median(gaps):.1f} days; median night/day coverage {panel.night_coverage.median():.0f}/{panel.predictor_coverage.median():.0f}")
    print(f"Night <70 event rate {scored.y_night70.mean():.5f}; valid 24h <54 rate {scored.y_24h54.mean():.5f}; 24h labels missing {scored.y_24h54.isna().mean():.3f}")
    print(f"Predictor days with pump bolus / carb entries {scored.insulin_bolus_available.mean():.3f}/{scored.carbs_available.mean():.3f}")
    print(f"Predictor days with basal dose estimates {scored.basal_insulin_available.mean():.3f}")
    return panel, n_reassigned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-rollover", action="store_true",
                        help="compare old and corrected final panels; do not write the panel")
    args = parser.parse_args()
    if args.report_rollover:
        old_panel, _ = build_panel(slot_assigner=_old_slot_day, write_panel=False)
        new_panel, n_reassigned = build_panel(slot_assigner=slot_day, write_panel=False)
        n_changed = changed_patient_nights(old_panel, new_panel)
        print(f"CGM readings reassigned to the following day: {n_reassigned:,}")
        print(f"Patient-nights with any changed event label or panel feature: {n_changed:,}")
        return
    build_panel()


if __name__ == "__main__":
    main()
