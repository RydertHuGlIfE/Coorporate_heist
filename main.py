"""The Corporate Heist - Optimized main.py
Run:  python main.py
Writes: submission.csv (500 ranked candidate_ids)
"""
import re, unicodedata, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
SEED, TOP_K, NOW = 20260919, 500, 2026
np.random.seed(SEED)

# ----------------------------------------------------------------------------
# 1. PARSING & CLEANING 
# ----------------------------------------------------------------------------

def norm(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(v)).lower().strip())

def first_num(s):
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else np.nan

def p_exp(v):
    s = norm(v).replace(",", "")
    if not s: return np.nan
    if "fresher" in s: return 0.0
    x = first_num(s)
    if np.isnan(x): return np.nan
    if "month" in s or re.search(r"\bmo\b", s): return x / 12.0
    if s.startswith("<"): return x / 2.0
    return x

def p_score(v, scale):
    s = norm(v)
    if not s or s in {"nan", "not taken", "absent", "none", "na", "n/a"}: return np.nan
    x = first_num(s)
    if np.isnan(x): return np.nan
    if "%" in s: return x / 100.0 * scale
    m = re.search(r"/\s*(\d+(?:\.\d+)?)", s)
    if m: return x / float(m.group(1)) * scale
    return x

RATING = {"unsatisfactory": 1, "needs improvement": 2, "meets expectations": 3,
          "exceeds expectations": 4, "outstanding": 5}
def p_rating(v):
    s = norm(v)
    if not s: return np.nan
    if s in RATING: return float(RATING[s])
    if "not rated" in s: return np.nan
    return p_score(s, 5.0)

def p_ctc(v): # -> lakh INR per annum
    s = norm(v).replace(",", "")
    if not s: return np.nan
    x = first_num(s)
    if np.isnan(x): return np.nan
    if "$" in s: return x * 83.0 / 1e5
    if "cr" in s: return x * 100.0
    if any(u in s for u in ("lpa", "lakh", "lac")) or re.search(r"\d\s*l$", s): return x
    if "₹" in s or "inr" in s or x >= 1000: return x / 1e5
    return x

def p_notice(v): # -> days
    s = norm(v)
    if not s: return np.nan
    if "immediate" in s or "available now" in s or "0 day" in s: return 0.0
    x = first_num(s)
    if np.isnan(x): return np.nan
    return x * 30.0 if "month" in s else x

def p_contrib(v):
    s = norm(v)
    if not s or "not tracked" in s: return np.nan
    return first_num(s.replace("~", ""))

def p_yes(v): return float(norm(v) in {"y", "yes", "true", "1", "1.0"})

def p_job_change(v):
    s = norm(v)
    if not s: return np.nan
    if "never" in s: return 0.0
    if ">" in s: return 5.0
    x = first_num(s)
    return x if not np.isnan(x) else np.nan

# Institute normalization
CITY = {"d": "delhi", "b": "bombay", "mumbai": "bombay", "chennai": "madras",
        "kgp": "kharagpur", "k": "kanpur", "r": "roorkee", "g": "guwahati", "h": "hyderabad", "bhu": "varanasi"}

def inst_base(s):
    s = str(s).lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[–—\-_/,.&']", " ", s)
    s = re.sub(r"\bthe\b", "", s)
    return re.sub(r"\s+", " ", s).strip()

def inst_canon(x):
    b = inst_base(x)
    if b in {"iitb", "iit bombay", "indian institute of technology bombay"}: return "iit bombay"
    if b in {"iitd", "iit delhi", "indian institute of technology delhi"}: return "iit delhi"
    if b in {"iitm", "iit madras", "indian institute of technology madras"}: return "iit madras"
    if b in {"iitk", "iit kanpur", "indian institute of technology kanpur"}: return "iit kanpur"
    if b in {"iitkgp", "iit kharagpur", "indian institute of technology kharagpur"}: return "iit kharagpur"
    m = re.match(r"^(?:indian institute of technology|iit|i i t)\s*(\w+)?", b)
    if m and m.group(1):
        c = CITY.get(m.group(1), m.group(1))
        rest = b[m.end():].strip()
        return "iit " + c + ((" " + rest) if rest else "")
    b = re.sub(r"^nit\b", "national institute of technology", b)
    b = re.sub(r"\btrichy\b", "tiruchirappalli", b)
    b = b.replace("national institute of technology surathkal", "national institute of technology karnataka surathkal")
    if b.startswith("bits") or b.startswith("birla institute of technology and science"): return "bits pilani"
    if b in {"dtu", "delhi technological university", "delhi college of engineering"}: return "dtu"
    if b in {"nsit delhi", "nsit", "netaji subhas university of technology"}: return "nsut"
    if "indian institute of science" in b or b == "iisc": return "indian institute of science"
    return b

# Skills map
SYN = {
 "sql": "sql tsql structuredquerylanguage", "python": "python python3", "docker": "docker dockerization containersdocker",
 "react": "react reactjs", "sklearn": "scikitlearn sklearn", "cicd": "cicd", "js": "javascript js es6",
 "ts": "typescript ts", "aws": "aws amazonwebservices awsec2s3", "k8s": "kubernetes k8s eksgke",
 "node": "nodejs node expressnode", "powerbi": "powerbi", "mlops": "mlops modeldeployment",
 "rest": "restapis rest restfulservices restapi apidesign", "microservices": "microservices msa microservicearchitecture",
 "spring": "springboot spring", "rn": "reactnative rn", "vue": "vuejs vue", "angular": "angular angularjs",
 "git": "git github versioncontrol", "selenium": "selenium seleniumwebdriver", "cypress": "cypress cypressio",
 "stats": "stats statistics statisticalmodelling hypothesistesting", "abtest": "abtesting splittesting experimentation",
 "excel": "excel advancedexcel msexcel", "kafka": "kafka apachekafka eventstreamingkafka",
 "spark": "spark apachespark pyspark", "airflow": "airflow apacheairflow workfloworchestrationairflow",
 "gcp": "gcp googlecloudplatform googlecloud", "azure": "azure msazure", "terraform": "terraform iacterraform hcl",
 "linux": "linux bashlinux unixlinux", "nlp": "nlp naturallanguageprocessing", "cv": "computervision cvimages",
 "tf": "tensorflow tfkeras", "torch": "pytorch torch", "go": "golang go", "cpp": "cpp cplusplus c++",
 "flutter": "flutter dartflutter", "android": "android androidsdk", "swift": "swift swiftui",
 "html": "html html5", "css": "css css3 scss", "java": "java java8 corejava", "mongo": "mongodb mongo",
 "postgres": "postgres postgresql psql",
}
SKMAP = {a: k for k, v in SYN.items() for a in v.split()}

def skill_set(s):
    out = set()
    for t in re.split(r"[,;|]", str(s) if not pd.isna(s) else ""):
        k = re.sub(r"[^a-z0-9+#]", "", t.lower())
        if k: out.add(SKMAP.get(k, k))
    return out

# Career path parser
LEVEL_KW = [("intern", 0), ("apprentice", 0), ("trainee", 0), ("junior", 1), ("associate", 1),
            ("senior", 3), ("sr", 3), ("lead", 4), ("principal", 5), ("staff", 5),
            ("manager", 5), ("architect", 5), ("head", 6), ("director", 7),
            ("vp", 8), ("vice president", 8), ("chief", 9), ("cto", 9)]

def level(title):
    t = norm(title)
    best = None
    for kw, l in LEVEL_KW:
        if re.search(r"\b" + re.escape(kw) + r"\b", t):
            best = l if best is None else max(best, l)
    if best is None:
        m = re.search(r"\b(i{1,3}|iv)\b\s*$", t)
        best = {"i": 1, "ii": 2, "iii": 3, "iv": 3}.get(m.group(1), 2) if m else 2
    return best

STEP = re.compile(r"(.+?)\s*[\(\[]\s*([\d.]+)\s*(y|yrs|yr|years|mo|months)?\s*[\)\]]")

def parse_path(s):
    s = str(s) if not pd.isna(s) else ""
    out = []
    for p in re.split(r"\s*(?:->|→|>|\|)\s*", s):
        m = STEP.search(p.strip())
        if not m: continue
        yrs = float(m.group(2))
        u = (m.group(3) or "y").lower()
        out.append((m.group(1).strip(), yrs / 12.0 if u.startswith("mo") else yrs))
    return out

def career_feats(df):
    rows = []
    for s in df["career_path"]:
        p = parse_path(s)
        if not p:
            rows.append((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        lv = [level(t) for t, _ in p]
        yrs = [y for _, y in p]
        n_promotions = sum(1 for i in range(1, len(lv)) if lv[i] > lv[i-1])
        rows.append((len(p), sum(yrs), max(lv), lv[-1], yrs[-1], n_promotions))
    return pd.DataFrame(rows, columns=["cp_n", "cp_years", "cp_top", "cp_cur", "cp_last", "cp_promos"], index=df.index)

# Recruiter note signals
POS_NOTE_PATTERNS = [
    r"primary on-call", r"open to weekend", r"ahead of sprint", r"24x7 on-call",
    r"production incident", r"led the migration", r"shipped a feature", r"mentored two junior",
    r"cross-team initiative", r"excellent system-design", r"core apis"
]
NEG_NOTE_PATTERNS = [
    r"struggled with the debugging", r"needed frequent guidance", r"lukewarm",
    r"unclear", r"missed two sprint commitments", r"exploring 2-3 other offers"
]

def parse_notes(notes):
    pos_cnt, neg_cnt = [], []
    for note in notes:
        s = norm(note)
        if not s:
            pos_cnt.append(0)
            neg_cnt.append(0)
            continue
        p = sum(1 for pat in POS_NOTE_PATTERNS if re.search(pat, s))
        n = sum(1 for pat in NEG_NOTE_PATTERNS if re.search(pat, s))
        pos_cnt.append(p)
        neg_cnt.append(n)
    return pd.DataFrame({"note_pos": pos_cnt, "note_neg": neg_cnt})

def clean(df):
    o = pd.DataFrame(index=df.index)
    o["age"] = pd.to_numeric(df["age"], errors="coerce")
    o["grad"] = pd.to_numeric(df["graduation_year"], errors="coerce")
    o["exp"] = df["total_experience"].map(p_exp)
    o["n_emp"] = pd.to_numeric(df["num_employers"], errors="coerce")
    o["tech"] = df["technical_assessment"].map(lambda v: p_score(v, 100))
    o["apt"] = df["aptitude_score"].map(lambda v: p_score(v, 10))
    o["rating"] = df["last_rating"].map(p_rating)
    o["kpi"] = df["kpi_met"].map(p_yes)
    o["awards"] = df["awards"].map(lambda v: float(norm(v) not in {"", "0", "-", "nan", "no", "none"}))
    o["trainings"] = pd.to_numeric(df["trainings_last_year"], errors="coerce")
    o["train_hours"] = pd.to_numeric(df["training_hours"], errors="coerce")
    o["notice"] = df["notice_period"].map(p_notice)
    o["contrib"] = df["public_code_contributions"].map(p_contrib) if "public_code_contributions" in df else np.nan
    o["overtime"] = df["overtime_history"].map(p_yes)
    o["n_certs"] = df["certifications"].map(lambda v: len([c for c in str(v).split(";") if c.strip()]) if not pd.isna(v) else 0)
    o["n_skills"] = df["skills"].map(lambda v: len(skill_set(v)))
    o["inst"] = df["institute"].map(inst_canon)
    o["role"] = df["applied_role"].map(norm)
    o["c_ctc"] = df["current_ctc"].map(p_ctc)
    o["e_ctc"] = df["expected_ctc"].map(p_ctc)
    o["ctc_hike"] = (o["e_ctc"] - o["c_ctc"]) / (o["c_ctc"].clip(lower=1.0))
    o["job_change"] = df["last_job_change"].map(p_job_change)
    
    # Career progression
    cp = career_feats(df)
    o = pd.concat([o, cp], axis=1)
    
    # Recruiter note features
    nt = parse_notes(df["recruiter_note"])
    nt.index = df.index
    o = pd.concat([o, nt], axis=1)
    
    o["since_grad"] = NOW - o["grad"]
    o["age_grad_gap"] = o["age"] - (o["grad"] - 2000 + 22)
    return o

# ----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------------
def sentences(v):
    return [re.sub(r"\s+", " ", x.strip().lower().rstrip(".")) for x in re.split(r"(?<=\.)\s+", str(v)) if x.strip()] if not pd.isna(v) else []

class Featurizer:
    def fit(self, raw, cl):
        rows = [(r, k) for r, s in zip(cl["role"], raw["skills"]) for k in skill_set(s)]
        d = pd.DataFrame(rows, columns=["role", "skill"])
        self.p_role = d.groupby(["skill", "role"]).size().div(d.groupby("skill").size(), level="skill").to_dict()
        self.skills = [s for s, c in d.skill.value_counts().items() if c >= 300][:70]
        allsent = pd.Series([x for v in raw["recruiter_note"].map(sentences) for x in v]).value_counts()
        self.sents = list(allsent[allsent >= 50].index)[:40]
        return self

    def transform(self, raw, cl):
        f = cl[["tech", "apt", "rating", "kpi", "awards", "exp", "n_emp", "trainings", "train_hours",
                "overtime", "n_certs", "n_skills", "cp_n", "cp_years", "cp_top", "cp_cur", "cp_last",
                "cp_promos", "c_ctc", "e_ctc", "ctc_hike", "job_change", "note_pos", "note_neg",
                "since_grad", "age_grad_gap"]].copy()
        
        f["cp_gap"] = cl["cp_years"] - cl["exp"]
        f["exp_per_emp"] = cl["exp"] / cl["n_emp"].clip(lower=1)
        f["tech_pct"] = cl["tech"] / 100.0
        f["apt_pct"] = cl["apt"] / 10.0
        f["composite_eval"] = f["tech_pct"] * 0.6 + f["apt_pct"] * 0.4
        
        fit = []
        for r, s in zip(cl["role"], raw["skills"]):
            ks = skill_set(s)
            fit.append(np.mean([self.p_role.get((k, r), 0.0) for k in ks]) if ks else np.nan)
        f["role_fit"] = fit
        f["tech_x_fit"] = f["tech"] * f["role_fit"]
        
        sk = raw["skills"].map(skill_set)
        for k in self.skills:
            f["sk_" + k] = sk.map(lambda x: float(k in x))
            
        ns = raw["recruiter_note"].map(sentences)
        for i, s in enumerate(self.sents):
            f[f"nt{i}"] = ns.map(lambda x: float(s in x))
            
        return f.replace([np.inf, -np.inf], np.nan)

# ----------------------------------------------------------------------------
# 3. DEDUPLICATION (Union-Find)
# ----------------------------------------------------------------------------
def dedup_groups(df):
    n = len(df)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    keys = {}
    phone = df["phone"].map(lambda v: re.sub(r"\D", "", str(v))[-10:] if not pd.isna(v) else "")
    email = df["email"].map(lambda v: norm(v).replace(" ", ""))
    name = df["full_name"].map(lambda v: re.sub(r"[^a-z]", "", norm(v)))
    ng = name + "|" + df["graduation_year"].astype(str)

    for col in (phone, email, ng):
        for i, k in enumerate(col):
            if len(k) < 7 or k.startswith("|"): continue
            if k in keys:
                union(i, keys[k])
            else:
                keys[k] = i
    return np.array([find(i) for i in range(n)])

# ----------------------------------------------------------------------------
# 4. TRAINING & SELECTION
# ----------------------------------------------------------------------------
def main():
    print("Loading datasets...")
    train, dev, winners, test = (pd.read_csv(f) for f in ("train.csv", "dev.csv", "dev_winners.csv", "test.csv"))
    ctr, cdv, cte = clean(train), clean(dev), clean(test)
    y = pd.to_numeric(train["post_hire_score"], errors="coerce")
    y = y.fillna(y.median())
    y_top = (y >= y.quantile(0.95)).astype(int)

    print("Building engineered features...")
    fz = Featurizer().fit(train, ctr)
    Xtr = fz.transform(train, ctr)
    Xdv = fz.transform(dev, cdv)
    Xte = fz.transform(test, cte)

    # 1. Regressor for post_hire_score
    params_reg = dict(
        n_estimators=850, learning_rate=0.025, num_leaves=25, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.75, reg_lambda=4.0,
        random_state=SEED, n_jobs=2, verbose=-1
    )
    
    # 2. Classifier directly targeting top 5%
    params_clf = dict(
        n_estimators=650, learning_rate=0.025, num_leaves=20, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.75, reg_lambda=5.0,
        scale_pos_weight=2.2, random_state=SEED, n_jobs=2, verbose=-1
    )

    # 5-fold CV evaluation on Archive
    oof_reg = np.zeros(len(Xtr))
    oof_clf = np.zeros(len(Xtr))
    for a, b in KFold(5, shuffle=True, random_state=SEED).split(Xtr):
        m_reg = lgb.LGBMRegressor(**params_reg).fit(Xtr.iloc[a], y.iloc[a])
        oof_reg[b] = m_reg.predict(Xtr.iloc[b])
        m_clf = lgb.LGBMClassifier(**params_clf).fit(Xtr.iloc[a], y_top.iloc[a])
        oof_clf[b] = m_clf.predict_proba(Xtr.iloc[b])[:, 1]

    r2 = 1.0 - ((y - oof_reg) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    combined_oof = pd.Series(oof_reg).rank(pct=True) * 0.55 + pd.Series(oof_clf).rank(pct=True) * 0.45
    overlap_top5 = len(set(np.argsort(-combined_oof)[:1000]) & set(np.argsort(-y.values)[:1000])) / 1000.0
    print(f"Archive 5-fold CV -> R2: {r2:.3f}, Top-5% Overlap: {overlap_top5:.3f}")

    # Ensembling across 4 distinct seeds
    n_seeds = 4
    reg_models = [lgb.LGBMRegressor(**{**params_reg, "random_state": SEED + i*17}).fit(Xtr, y) for i in range(n_seeds)]
    clf_models = [lgb.LGBMClassifier(**{**params_clf, "random_state": SEED + i*17}).fit(Xtr, y_top) for i in range(n_seeds)]

    def predict_rank_score(X):
        p_r = np.mean([m.predict(X) for m in reg_models], axis=0)
        p_c = np.mean([m.predict_proba(X)[:, 1] for m in clf_models], axis=0)
        r_r = pd.Series(p_r).rank(pct=True).to_numpy()
        r_c = pd.Series(p_c).rank(pct=True).to_numpy()
        return r_r * 0.55 + r_c * 0.45

    # Check against Dev Ledger
    win = dev["candidate_id"].isin(set(winners["candidate_id"])).astype(int).to_numpy()
    pdv = predict_rank_score(Xdv)
    top150 = np.argsort(-pdv)[:150]
    prec150 = win[top150].mean()
    rec150 = win[top150].sum() / win.sum()
    auc = roc_auc_score(win, pdv)
    print(f"Ledger Validation -> AUC: {auc:.3f}, Precision@150: {prec150:.3f}, Recall@150: {rec150:.3f}")

    # ---- Current cycle (Vault / test.csv) adjustments ----
    pte = predict_rank_score(Xte)
    z = (pte - pte.mean()) / (pte.std() + 1e-8)
    score = z.copy()

    # Clue 1: Public code contributions (sustained dozens or more fast-tracked)
    con = cte["contrib"].fillna(0).to_numpy()
    contrib_bonus = np.where(con >= 24, 0.95 + np.clip((con - 24) / 40.0, 0, 0.5), np.clip(con / 24.0, 0, 1.0) * 0.35)
    score += contrib_bonus

    # Clue 4: Candidates from never-hired colleges who topped technical assessment & fit role
    seen_colleges = set(ctr["inst"].dropna()) | set(cdv["inst"].dropna())
    is_new_college = ~cte["inst"].isin(seen_colleges)
    tech = cte["tech"].fillna(0).to_numpy()
    fit = Xte["role_fit"].fillna(0).to_numpy()
    
    tech_p70 = np.nanpercentile(tech, 70)
    fit_p50 = np.nanpercentile(fit, 50)
    new_col_strong = is_new_college.to_numpy() & (tech >= tech_p70) & (fit >= fit_p50)
    score += np.where(is_new_college.to_numpy(), 0.15, 0.0) + np.where(new_col_strong, 0.55, 0.0)

    # Clue 5: Old-boys' network colleges (shrunk empirical residual)
    resid = pd.Series(y.to_numpy() - oof_reg).groupby(ctr["inst"].to_numpy()).agg(["mean", "count"])
    resid["adj"] = resid["mean"] * resid["count"] / (resid["count"] + 30)
    old_boys = resid[resid["count"] >= 35].sort_values("adj", ascending=False).head(6)
    print("Old-boys network top colleges:", list(old_boys.index), old_boys["adj"].round(2).tolist())
    score += cte["inst"].map(old_boys["adj"] / float(y.std()) * 0.5).fillna(0.0).to_numpy()

    # Clue 3: Hard exclusion: notice period > 60 days
    notice_bad = (cte["notice"].fillna(30) > 60).to_numpy()

    # Clue 2 & 6: Suspicious career progression and fabricated timeline profiles
    # 1. Inflated titles with insufficient experience
    mins_exp = ctr.assign(e=ctr[["exp", "cp_years", "since_grad"]].min(axis=1)).groupby("cp_cur")["e"].quantile(0.01)
    lim = (mins_exp - 0.5).clip(lower=0.5).to_dict()
    e_te = cte[["exp", "cp_years", "since_grad"]].min(axis=1)
    fake_titles = (e_te < cte["cp_cur"].map(lim).fillna(0)).to_numpy() & (cte["cp_cur"] >= 3).to_numpy()
    
    # 2. Fabricated timeline contradictions (e.g. experience before age 16 or graduated before age 16)
    c_exp_age = (cte["age"] - cte["exp"] < 16).fillna(False).to_numpy()
    c_grad_age = (cte["age"] < cte["since_grad"] + 16).fillna(False).to_numpy()
    c_exp_grad = (cte["exp"] > cte["since_grad"] + 3.5).fillna(False).to_numpy()
    fake_timeline = c_exp_age | c_grad_age | c_exp_grad
    
    all_exclusions = notice_bad | fake_titles | fake_timeline
    print(f"Exclusions: notice>60d = {notice_bad.sum()}, fake_titles = {fake_titles.sum()}, fake_timeline = {fake_timeline.sum()}, total = {all_exclusions.sum()}")
    score = np.where(all_exclusions, -1e9, score)

    # Clue 6: Deduplication — keep best scoring profile per person
    grp = dedup_groups(test)
    order = np.argsort(-score, kind="stable")
    seen_groups = set()
    chosen_indices = []

    for idx in order:
        if score[idx] <= -1e8:
            break
        g = grp[idx]
        if g in seen_groups:
            continue
        seen_groups.add(g)
        chosen_indices.append(idx)
        if len(chosen_indices) == TOP_K:
            break

    print(f"Selected: {len(chosen_indices)} candidates (Target: {TOP_K})")
    assert len(chosen_indices) == TOP_K, f"Failed to pick exactly {TOP_K} valid candidates!"

    sub = pd.DataFrame({
        "rank": np.arange(1, TOP_K + 1),
        "candidate_id": test["candidate_id"].iloc[chosen_indices].to_numpy()
    })
    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv successfully!")
    print(f"  - Candidates with contrib >= 24: {(con[chosen_indices] >= 24).sum()}")
    print(f"  - Candidates from new colleges: {is_new_college.to_numpy()[chosen_indices].sum()}")

if __name__ == "__main__":
    main()