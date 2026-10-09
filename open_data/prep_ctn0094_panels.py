"""Rebuild CTN-0094 weekly and daily panels from the open harmonized tables.

Eligibility rules copied from the event-log builder: weekly patients need at
least 12 observed screen weeks, including at least 2 of study weeks 0-3 and
at least 8 observed weeks after the four-week run-in. Daily patients need an
inclusive TLFB span of at least 56 days; the first 28 days form run-in and
the remaining transitions are scored. Weekly targets begin at study week 4.
The script reports patient and row counts at each eligibility step.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent / "ctn0094" / "csv"
OUT = Path(__file__).resolve().parent


def read(name):
    return pd.read_csv(BASE / f"{name}.csv")


def build():
    rand = read("randomization").sort_values(["who", "which", "when"]).drop_duplicates("who").rename(
        columns={"when": "randomization_day", "treatment": "arm"})
    people = rand.merge(read("everybody"), on="who", how="left")
    print(f"Randomized people: {rand.who.nunique():,}; with demographics: {people.who.nunique():,}")

    temp = read("uds_temp")[["who", "when"]].dropna().drop_duplicates().rename(columns={"when": "study_day"})
    uds = read("uds")
    uds["study_day"] = uds["when"]
    uds["is_opioid_positive"] = uds.what.str.casefold().eq("opioid").astype(int)
    positive = uds[uds.is_opioid_positive.eq(1)][["who", "study_day"]].drop_duplicates()
    samples = pd.concat([temp, uds[["who", "study_day"]].dropna().drop_duplicates()], ignore_index=True).drop_duplicates(["who", "study_day"])
    samples = samples.merge(people[["who", "randomization_day", "project", "arm"]], on="who", how="inner")
    samples["study_week"] = np.floor((samples.study_day-samples.randomization_day)/7).astype(int)
    samples = samples[samples.study_week >= 0]
    posset = set(zip(positive.who, positive.study_day))
    samples["week_event"] = [int((int(w), float(d)) in posset) for w, d in zip(samples.who, samples.study_day)]
    weekobs = samples.groupby(["who", "study_week"], as_index=False).agg(screen_days=("study_day", "nunique"), event=("week_event", "max"))
    print(f"Weekly sample evidence: {samples.who.nunique():,} patients, {len(weekobs):,} observed patient-weeks")

    tlfb = read("tlfb")
    tlfb["study_day"] = tlfb["when"]
    tlfb = tlfb[tlfb.what.str.casefold().isin(["heroin", "opioid"])].merge(people[["who", "randomization_day"]], on="who", how="inner")
    tlfb["study_week"] = np.floor((tlfb.study_day-tlfb.randomization_day)/7).astype(int)
    tlfb_week = tlfb[tlfb.study_week >= 0].groupby(["who", "study_week"]).study_day.nunique().to_dict()

    dose = read("treatment").merge(people[["who", "randomization_day"]], on="who", how="inner")
    dose["study_week"] = np.floor((dose.when-dose.randomization_day)/7).astype(int)
    dose = dose[dose.study_week >= 0].sort_values(["who", "study_week", "when"])
    dose_week = {k: g[["study_week", "amount"]].values.tolist() for k, g in dose.groupby("who")}
    weekby = {int(pid): {int(r.study_week): (int(r.event), int(r.screen_days)) for r in g.itertuples()} for pid, g in weekobs.groupby("who")}
    people_ids = []
    for p, g in weekby.items():
        run_weeks = [w for w in range(4) if w in g]
        post_weeks = [w for w in g if w >= 4]
        if len(g) >= 12 and len(run_weeks) >= 2 and len(post_weeks) >= 8:
            people_ids.append(p)
    print(f"Weekly eligibility (>=12 observed weeks, >=2 run-in, >=8 later): {len(people_ids):,} patients")
    rows = []
    for p in people_ids:
        meta = people.loc[people.who.eq(p)].iloc[0]
        weeks = weekby[p]
        last = max(weeks)
        person_doses = dose_week.get(p, [])
        for target_week in range(4, last+1):
            origin = target_week-1
            hist = [(w, *weeks[w]) for w in sorted(weeks) if w <= origin]
            observed = target_week in weeks
            event = int(weeks[target_week][0]) if observed else np.nan
            recent = [weeks[w][0] for w in range(max(0, origin-3), origin+1) if w in weeks]
            hist_events = sum(z[1] for z in hist)
            hist_screens = sum(z[2] for z in hist)
            run = [weeks[w][0] for w in range(4) if w in weeks]
            initial_events = [weeks[w][0] if w in weeks else None for w in range(4)]
            initial_screens = [weeks[w][1] if w in weeks else 0 for w in range(4)]
            run_rate = float(sum(run)/len(run)) if run else np.nan
            dose_now = np.nan
            for dw, amt in person_doses:
                if int(dw) <= origin:
                    dose_now = float(amt)
                else:
                    break
            dose_week_amt = sum(float(a) for dw, a in person_doses if int(dw) == origin)
            runin_self_report_mean = float(np.mean([tlfb_week.get((p, w), 0) for w in range(4)]))
            runin_dose_mean = float(np.mean([sum(float(a) for dw, a in person_doses if int(dw) == w) for w in range(4)]))
            rows.append(dict(who=p, project=int(meta.project), arm=str(meta.arm), study_week=target_week,
                origin_week=origin, event=event, observed=int(observed), event_missing_positive=(int(event) if observed else 1),
                runin_rate=run_rate, runin_n=len(run),
                history_initial_events=json.dumps(initial_events, separators=(',', ':')),
                history_initial_screens=json.dumps(initial_screens, separators=(',', ':')),
                previous_result=(weeks[origin][0] if origin in weeks else np.nan), previous_observed=int(origin in weeks),
                expanding_rate=(hist_events/hist_screens if hist_screens else np.nan), expanding_n=hist_screens,
                recent_rate=(sum(recent)/len(recent) if recent else np.nan), recent_n=len(recent),
                persistence=(weeks[origin][0] if origin in weeks else np.nan),
                self_report_opioid_days_last_week=int(tlfb_week.get((p, origin), 0)),
                runin_self_report_mean=runin_self_report_mean, runin_dose_mean=runin_dose_mean,
                dose_last_week=dose_week_amt,
                dose_current=dose_now, screen_count_target=(weeks[target_week][1] if observed else 0),
                n_observed_weeks=len(weeks), n_runin_observed_weeks=len(run), retrospective_rate=np.nan))
    weekly = pd.DataFrame(rows)
    for pid, ix in weekly.groupby("who").groups.items():
        observed = weekly.loc[ix].observed.eq(1)
        weekly.loc[ix, "retrospective_rate"] = weekly.loc[ix[observed.values], "event"].mean()
    print(f"Weekly panel rows: {len(weekly):,}; patients: {weekly.who.nunique():,}; observed target rows: {int(weekly.observed.sum()):,}")

    alltfb = read("tlfb").merge(people[["who", "randomization_day", "project", "arm"]], on="who", how="inner")
    alltfb["opioid_event"] = alltfb.what.str.casefold().isin({"heroin", "opioid", "methadone", "buprenorphine"}).astype(int)
    agg = alltfb.groupby(["who", "when"], as_index=False).agg(opioid_event=("opioid_event", "max"))
    span = alltfb.groupby("who").when.agg(first_day="min", last_day="max").reset_index()
    span["span_days"] = span.last_day-span.first_day+1
    eligible_daily = set(span.loc[span.span_days >= 56, "who"].astype(int))
    print(f"Daily span eligibility (>=56 days): {len(eligible_daily):,} patients")
    dailyrows = []
    usemap = {int(p): set(g.loc[g.opioid_event.eq(1), "when"].astype(int)) for p, g in agg.groupby("who")}
    for s in span[span.who.astype(int).isin(eligible_daily)].itertuples():
        p = int(s.who)
        meta = people.loc[people.who.eq(p)].iloc[0]
        first, last = int(s.first_day), int(s.last_day)
        seq = [int(d in usemap[p]) for d in range(first, last+1)]
        initial_events = json.dumps(seq[:29], separators=(',', ':'))
        for offset in range(28, len(seq)-1):
            prior = seq[:offset+1]
            d = first+offset
            dailyrows.append(dict(who=p, project=int(meta.project), arm=str(meta.arm), study_day=d+1, origin_day=d,
                event=seq[offset+1], observed=1, runin_rate=float(np.mean(seq[:28])), previous_result=seq[offset],
                history_initial_events=initial_events,
                expanding_rate=float(np.mean(prior)), recent_rate=float(np.mean(seq[max(0, offset-27):offset+1])),
                persistence=seq[offset],
                self_report_opioid_days_last_28=int(sum(seq[max(0, offset-27):offset+1])),
                inferred_first_day=first, inferred_last_day=last, covered_span_days=len(seq), runin_days=28,
                retrospective_rate=np.nan))
    daily = pd.DataFrame(dailyrows)
    for pid, ix in daily.groupby("who").groups.items():
        daily.loc[ix, "retrospective_rate"] = daily.loc[ix, "event"].mean()
    print(f"Daily panel rows: {len(daily):,}; patients: {daily.who.nunique():,}; scored transitions: {len(daily):,}")
    weekly.to_csv(OUT / "ctn0094_weekly_panel_rebuilt.csv", index=False)
    daily.to_csv(OUT / "ctn0094_daily_panel_rebuilt.csv", index=False)
    print("Wrote rebuilt panels without modifying the originals.")


if __name__ == "__main__":
    build()
