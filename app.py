"""ShieldDM - Online Harassment & Threat Classifier for DMs
Run:  streamlit run app.py
Needs: model.pkl (+ metrics.json) created by `python train.py`
"""
import datetime
import html
import json
import os
import re

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
from fpdf.enums import XPos, YPos

BASE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
CATS = ["Threat", "Blackmail", "Intimidation", "Sexual Harassment", "Stalking", "Abusive"]  # priority order
SEV = {"Threat": 4, "Blackmail": 4, "Intimidation": 3, "Sexual Harassment": 3, "Stalking": 2, "Abusive": 2, "Safe": 0}
LEVEL = {0: "Safe", 1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
ICON = {"Threat": "🔴", "Blackmail": "🟣", "Intimidation": "⚠️", "Sexual Harassment": "💗", "Stalking": "🟠",
        "Abusive": "🟡", "Safe": "🟢"}
CAT_COLORS = {"Threat": "#ff5c7a", "Blackmail": "#e58bd0", "Intimidation": "#ffc27a", "Sexual Harassment": "#ffb3d1",
              "Stalking": "#ff9a8a", "Abusive": "#f4cfe0", "Safe": "#4a3a45"}
LEVEL_COLORS = {"Critical": "#ff5c7a", "High": "#ff9a8a", "Medium": "#ffc27a", "Low": "#f4cfe0", "Safe": "#4a3a45"}

LAW = {
    "Threat": "BNS Sec 351 - Criminal intimidation",
    "Blackmail": "BNS Sec 308 - Extortion; BNS Sec 351 - Criminal intimidation",
    "Intimidation": "BNS Sec 351 - Criminal intimidation (veiled threats can qualify depending on context)",
    "Sexual Harassment": "BNS Sec 75 - Sexual harassment; IT Act Sec 67 - Publishing/transmitting obscene material",
    "Stalking": "BNS Sec 78 - Stalking (including monitoring online activity)",
    "Abusive": "BNS Sec 79 - Word/gesture intended to insult the modesty of a woman",
}

# ---------------------------------------------------------------------------
# CUE LEXICON.  Each cue = (category, weight 0-1, regex, human label).
# A cue is ONE linguistic signal. A single weak cue ("last chance") is not enough to flag a message,
# but several signals together are: the cue weights of a category are combined with a noisy-OR,
#     score = 1 - (1 - w1)(1 - w2)...(1 - wn)
# so "ultimatum + veiled warning + conditional warning + final warning" becomes a confident
# Intimidation detection even though no single violent word appears. The weights are what the
# sensitivity slider compares against, so the slider changes detection, counts and risk scores.
# ---------------------------------------------------------------------------
CUES = [
    # ---- Threat: explicit and strongly implied violence / consequences
    ("Threat", 0.95, r"\b(kill|murder|stab|shoot|burn|slap|beat)\b.{0,12}\b(you|u|ur|tujhe|tumhe|teri|tera)\b", "violent threat"),
    ("Threat", 0.95, r"\bi('| wi)ll (kill|hurt|destroy|ruin|end) (you|u|ur life|your life)", "threat to harm"),
    ("Threat", 0.95, r"maar (dunga|dalunga|daalunga)|jaan se|teri jaan|kaat dunga|tod (dunga|dalunga)", "violent threat (Hinglish)"),
    ("Threat", 0.80, r"dekh lunga|dekh lena|chh?odunga nahi|pachtayega|pachtana padega", "intimidation (Hinglish)"),
    ("Threat", 0.80, r"you('| wi)ll (regret|pay for|pay dearly|pay the price|be sorry)|tujhe barbaad", "threat of consequences"),
    ("Threat", 0.95, r"\bacid\b.{0,15}\b(face|you|tujh)", "acid threat"),
    ("Threat", 0.85, r"(make|let) you pay|make you (regret|suffer)|make your life (hell|miserable)", "threat to make you suffer"),
    ("Threat", 0.75, r"(something|bad things|things) (bad |terrible )?(will|could|can|might) happen to you|you (might|could|will) get hurt|watch your back|sleep with one eye open", "indirect threat of harm"),
    # ---- Blackmail: threats to expose / leak / ruin, demands under pressure
    ("Blackmail", 0.95, r"viral kar|leak kar|leak (your|ur|the)", "threat to leak/viral"),
    ("Blackmail", 0.90, r"i('| wi)ll (post|share|upload|leak|send|show).{0,25}(photos?|pics?|videos?|nudes?|chat|screenshots?)", "threat to share private content"),
    ("Blackmail", 0.85, r"(photos?|pics?|videos?|screenshots?|chats?).{0,25}(everyone|sabko|sab ko|your (parents|family|friends))", "threat to expose content"),
    ("Blackmail", 0.85, r"mere paas .{0,30}(photos?|pics?|videos?|screenshots?)", "claims to hold private content"),
    ("Blackmail", 0.80, r"tell (your|ur) (parents|family|husband|boyfriend|friends)|ghar wal(on|e) ko (bata|bol)", "threat to tell family"),
    ("Blackmail", 0.85, r"(badnam|expose) (kar|you|u\b)", "threat to defame/expose"),
    ("Blackmail", 0.90, r"\b(pay|send) (me )?(money|rs\.?|\d{3,}).{0,30}(or|else|otherwise|warna)", "demand for money with threat"),
    ("Blackmail", 0.70, r"\bi (have|still have|saved|kept) (all )?(your|ur|the) (photos?|pics?|videos?|screenshots?|chats?|messages?|voice notes?|secrets?)\b", "says it holds your private content"),
    ("Blackmail", 0.70, r"(everyone|everybody|the whole (college|world|class|office)|your (family|parents|friends|boss|husband|boyfriend)) (will|should|would|is going to) (know|find out|see|hear)", "threat to expose to others"),
    ("Blackmail", 0.85, r"you (don'?t|do not) want (me|this|people|everyone) to (share|tell|show|post|leak|find out|know)", "implied threat to share"),
    ("Blackmail", 0.80, r"i can (ruin|destroy|expose|finish|end) you|ruin your (life|career|reputation|image|name)", "threat to ruin you"),
    ("Blackmail", 0.50, r"don'?t tell (anyone|anybody|your (parents|family|friends))|if you tell (anyone|anybody)|keep (this )?(quiet|secret|silent)", "demand for silence"),
    ("Blackmail", 0.75, r"(want|like) (everyone|everybody|people|the world|your (family|parents|friends)) to (see|know|find out|hear)", "implied threat to expose"),
    ("Blackmail", 0.85, r"\b(send|give|show|meet|come|sleep|pay).{0,30}\b(or|otherwise|else|warna)\b.{0,40}\b(share|post|tell|leak|expose|everyone|photos?|pics?|videos?|screenshots?)", "demand under threat of exposure"),
    # ---- Intimidation: indirect / veiled threats, ultimatums, pressure (no violent words needed)
    ("Intimidation", 0.50, r"\b(last|final|one last) (chance|warning|opportunity)\b", "ultimatum"),
    ("Intimidation", 0.55, r"consider (this|it|yourself) (your |a |as )?(final |last )?(warning|warned)|(you have|you are|you've) been warned|consider yourself warned", "formal warning"),
    ("Intimidation", 0.60, r"\b(last|final) time (i('| a)m|i will|i'?ll|i am) (telling|asking|warning|saying)", "last-time ultimatum"),
    ("Intimidation", 0.70, r"(don'?t|do not) (mistake|confuse) my (silence|kindness|patience|calm|politeness|restraint|good nature)", "veiled warning"),
    ("Intimidation", 0.55, r"(don'?t|do not|stop) (push|test|provoke|try) (me|my patience|my limits)|you('| a)re (testing|pushing) my (patience|limits)|my patience (is|has) (running|run|wearing)", "pressure on patience"),
    ("Intimidation", 0.70, r"(no idea|don'?t know|have no clue|can'?t imagine).{0,30}(how (serious|bad|ugly|far)|what i('| a)m capable|what i can do|what i will do|what (i'?m|i am) going to do)", "hints at consequences"),
    ("Intimidation", 0.70, r"\b(if|unless) you .{0,70}\b(regret|sorry|suffer|consequences?|pay for|pay the price|trouble|worse|ugly|hell|see what happens|find out|end badly|get (serious|hurt|bad|messy))", "conditional warning"),
    ("Intimidation", 0.60, r"you('| wi)ll (see|find out|learn) (what|how)|wait and see", "veiled warning"),
    ("Intimidation", 0.55, r"i know (more|a lot|so much) (than you (think|know|realise|realize)|about you)|i know everything about you", "claims hidden knowledge"),
    ("Intimidation", 0.70, r"i know (what you did|what you've done|what you have done|your (dirty )?secrets?)", "claims to know your secret"),
    ("Intimidation", 0.45, r"i('| a)m (done|tired|sick) (of )?(being nice|waiting|letting|playing|with your|pretending)|i('| ha)ve had enough|enough is enough|i('| wi)ll not (tolerate|stay quiet|stay silent|let this)|(done|tired of) letting (things|this|it) slide", "escalating tone"),
    ("Intimidation", 0.50, r"you('| a)re making a (big |huge |terrible |serious |grave )?mistake", "warning of a 'mistake'"),
    ("Intimidation", 0.70, r"you (don'?t|do not) know who you('| a)re (dealing|messing) with|do you know who i am", "intimidation by status"),
    ("Intimidation", 0.70, r"\bmy (men|boys|guys|people|gang|friends|brothers) (will|are going to|can|shall)\b.{0,30}(deal|handle|teach|find|come|visit|sort)", "threat through others"),
    ("Intimidation", 0.60, r"\bor else\b|otherwise you('| wi)ll|warna (tu|tum|aap)", "'or else' ultimatum"),
    ("Intimidation", 0.60, r"(do|obey|listen).{0,15}what i (say|want|tell)|you will do (as|what) i|obey me", "coercive control"),
    ("Intimidation", 0.70, r"mujhe mat (aazma|chhed|cheda)|sabr ka imtehaan", "warning not to provoke (Hinglish)"),
    ("Intimidation", 0.75, r"(ab bhi|abhi bhi) samajh (ja|jao)|anjaam bura|bura hoga|akhri (baar|warning)|last warning hai", "warning (Hinglish)"),
    ("Intimidation", 0.65, r"you (really |truly )?(don'?t|do not) want to (find out|see|know|learn|test me|make me|push me)", "warns you not to find out"),
    ("Intimidation", 0.70, r"what i('| a)m capable of", "hints at what it can do"),
    ("Intimidation", 0.70, r"i can make (things|life|it|everything) (very |really |so )?(difficult|hard|bad|worse|miserable|painful|ugly)", "threat to make life difficult"),
    ("Intimidation", 0.65, r"(difficult|hard|bad|worse|trouble|pay|suffer|hurt|regret|unsafe).{0,30}(your (family|parents|mother|father|mom|dad|brother|sister|kids?)|people you love)|your (family|parents|mother|father|mom|dad|brother|sister|kids?).{0,40}(pay|suffer|hurt|regret|trouble|difficult|worse|unsafe)", "threat involving family or loved ones"),
    ("Intimidation", 0.55, r"(don'?t|do not) (underestimate|force|corner) (me|my hand)|i (won'?t|will not) (warn|ask|tell) you (again|twice|nicely)|i won'?t ask (nicely|again|twice)", "warns it will escalate"),
    ("Intimidation", 0.45, r"i('| ha)ve been (patient|quiet|silent|nice|waiting) (for )?(long enough|too long|a long time)|(my )?patience (is|has) (almost |about to )?(over|gone|ended|run out)", "patience running out"),
    ("Intimidation", 0.60, r"(learn|understand|realise|realize) (it )?(soon|the hard way)|some people (only )?learn", "'you will learn the hard way'"),
    ("Intimidation", 0.60, r"(try|test) me and (see|find out)|(see|find out) why people (fear|avoid) me|people who cross me|wouldn'?t (do|try) that if i were you", "warning not to cross me"),
    # ---- Sexual harassment
    ("Sexual Harassment", 0.90, r"\bsend (me )?(your |ur )?(nudes?|nude pics?|sexy|boobs|body)", "demand for nude/sexual content"),
    ("Sexual Harassment", 0.70, r"\bnudes?\b|\bdick pic\b", "sexual content request"),
    ("Sexual Harassment", 0.90, r"sleep with me|sex with me|have sex|hook ?up|raat (ko )?(milo|aa)", "sexual proposition"),
    ("Sexual Harassment", 0.70, r"\bhot (body|figure)\b|sexy figure|figure kaisa", "sexualised remark"),
    ("Sexual Harassment", 0.80, r"rate kya hai|kitna lega|kitne ka hai", "treats person as a commodity"),
    ("Sexual Harassment", 0.85, r"\b(show|dikha).{0,12}(body|boobs|figure)", "demand to show body"),
    # ---- Stalking: tracking, surveillance, pressure to respond
    ("Stalking", 0.60, r"where (?:are|r) (?:you|u)\b(?! (?:going|coming|headed|from|planning|living|studying|off))|kahan (?:hai|ho|hain)\b|kaha ho\b", "tracking whereabouts"),
    ("Stalking", 0.85, r"\bi('| a)m (outside|downstairs|near your|at your|watching)", "claims to be physically near"),
    ("Stalking", 0.90, r"i know where you (live|work|study)|tera ghar|tumhara ghar|ghar ke bahar", "knows home/work location"),
    ("Stalking", 0.80, r"i('| wi)ll find you|\bi saw you\b|saw you (today|at|near)|dekha tujhe", "surveillance / following"),
    ("Stalking", 0.85, r"i('| a)m (following|tracking|watching) you|following you", "following / tracking"),
    ("Stalking", 0.65, r"why (are|r) (you|u) (ignoring|not replying|online)|(seen|online) kyun|reply (kyun|nahi)", "pressure over ignored messages"),
    ("Stalking", 0.60, r"\breply (now|me|fast)\b|answer (me|the phone|my call)|pick up (the|my) (phone|call)", "demanding immediate response"),
    ("Stalking", 0.65, r"who (are|r) you (talking|chatting) (to|with)|kis se baat", "monitoring contacts"),
    ("Stalking", 0.65, r"(share|send) (me )?(your |ur )?location|teri location", "demands location"),
    ("Stalking", 0.75, r"keeping an eye on you|eyes on you|i can see you|i see everything you|i know (everything|all) you do|i know who you('| a)re (with|talking|meeting|chatting)|i know where you (were|went|are)\b", "claims to watch you"),
    ("Stalking", 0.75, r"you can'?t (hide|escape|run)|no matter where you (go|hide|run)|wherever you go|i('| wi)ll always find you|i will be (there|waiting)", "'you cannot escape'"),
    ("Stalking", 0.80, r"i know your (daily |usual )?(routine|schedule|timings?|movements|every move)", "knows your routine"),
    ("Stalking", 0.85, r"i('| a)ve been (tracking|checking|monitoring) your (location|phone|account|profile|activity)|i (saw|checked|read) your (location|phone|messages|chats)", "monitors your location or accounts"),
    ("Stalking", 0.70, r"i know (exactly )?where you('| ha)ve been|who you (were|are) (with|meeting)", "knows where you have been"),
    # ---- Abusive language
    ("Abusive", 0.65, r"\b(bitch|slut|whore|bastard|asshole|idiot|stupid|ugly|loser|shut up)\b", "abusive language"),
    ("Abusive", 0.80, r"\b(randi|harami|kamina|kamini|kutti|kutta|madarchod|behenchod|bhosdi\w*|chutiya|gandu|bc|mc)\b", "abusive language (Hinglish)"),
]
CUES_C = [(c, w, re.compile(p, re.I), lbl) for c, w, p, lbl in CUES]
REFUSAL = re.compile(
    r"stop (?:messaging|texting|calling|it|this)|leave me alone|don'?t (?:message|text|call|contact) me|"
    r"please stop|i(?:'| wi)ll block|blocked you|mat karo|pareshan mat|mujhe (?:message|call) mat|not interested", re.I)

DEMO_CHAT = """12/10/26, 9:01 pm - Rohan: hey where are you
12/10/26, 9:02 pm - Rohan: reply now
12/10/26, 9:02 pm - Rohan: why are you ignoring me
12/10/26, 9:05 pm - Rohan: i am outside your building
12/10/26, 9:06 pm - Riya: please stop messaging me
12/10/26, 9:10 pm - Rohan: reply or i will kill you
12/10/26, 11:40 pm - Rohan: send nudes or i will viral kar dunga your photos
12/10/26, 11:42 pm - Riya: leave me alone
12/10/26, 11:45 pm - Rohan: mere paas tere screenshots hai, sabko bhej dunga
12/10/26, 11:50 pm - Rohan: call my number 9876543210 or mail me at rohan@example.com
"""


# ----------------------------------------------------------------------------
# PARSING
# ----------------------------------------------------------------------------
HDR = re.compile(
    r"^\[?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}),?\s+(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?\s?(?:[AaPp]\.?[Mm]\.?)?)\]?\s*(?:[-–—]\s*)?(.*)$")
PLAIN = re.compile(r"^([^:\n]{1,30}):\s+(.+)$")


def _to_dt(series):
    try:
        return pd.to_datetime(series, dayfirst=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, dayfirst=True, errors="coerce")


def parse_chat(raw: str, default_sender: str = "") -> pd.DataFrame:
    """Parse a WhatsApp export (Android / iPhone), 'Name: msg' lines or plain lines."""
    for ch in ("\u202f", "\u00a0"):
        raw = raw.replace(ch, " ")
    for ch in ("\u200e", "\u200f", "\ufeff", "\r"):
        raw = raw.replace(ch, "")
    raw = raw.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')  # phone keyboards use curly quotes
    lines = raw.split("\n")
    rows, has_header = [], any(HDR.match(l.strip()) for l in lines)
    cur = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if has_header:
            m = HDR.match(s)
            if m:
                d, t, rest = m.groups()
                if ": " in rest and len(rest.split(": ", 1)[0]) <= 40:
                    sender, msg = rest.split(": ", 1)
                    cur = [(d + " " + t).strip(), sender.strip(), msg.strip()]
                    rows.append(cur)
                else:
                    cur = None  # system line such as "Messages are end-to-end encrypted"
            elif cur is not None:
                cur[2] += " " + s  # multi-line message continues
        else:
            m = PLAIN.match(s)
            rows.append(["-", m.group(1).strip(), m.group(2).strip()] if m else ["-", default_sender or "Unknown", s])
    df = pd.DataFrame(rows, columns=["time", "sender", "message"])
    if df.empty:
        return df
    df["message"] = df["message"].str.replace(r'^"(.*)"$', r"\1", regex=True).str.strip()  # drop quotes wrapped around a pasted line
    df["ts"] = _to_dt(df["time"].where(df["time"] != "-"))
    df.insert(0, "no", range(1, len(df) + 1))
    return df


# ----------------------------------------------------------------------------
# DETECTION ENGINE (rules + machine learning)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(os.path.join(BASE, "model.pkl"))
    except Exception:
        return None


@st.cache_data
def load_metrics():
    try:
        return json.load(open(os.path.join(BASE, "metrics.json")))
    except Exception:
        return None


def sens_to_thr(flag_pct):
    """Slider value = confidence (%) a message needs before it is flagged. Lower value = flags more."""
    return round(flag_pct / 100.0, 4)


def noisy_or(weights):
    p = 1.0
    for w in weights:
        p *= (1.0 - w)
    return 1.0 - p


def cue_scan(text):
    """Every linguistic cue found in the message, grouped by category: {cat: [(weight, label, matched text)]}."""
    out = {}
    for cat, w, rx, lbl in CUES_C:
        m = rx.search(text)
        if m:
            out.setdefault(cat, []).append((w, lbl, m.group(0).strip()))
    return out


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def ml_scores(texts):
    """Sentence-level ML. Each message is split into sentences; the model scores every sentence (and the whole
    message when it has several), and the highest Threat / Abusive probability wins. This stops a menacing
    sentence from being diluted by harmless ones around it."""
    res = [dict() for _ in texts]
    model = load_model()
    if model is None:
        return res
    vec, clf = model
    classes = list(clf.classes_)
    units, owner = [], []
    for i, t in enumerate(texts):
        if REFUSAL.search(t):  # a refusal ("stop messaging me") is never ML-flagged
            continue
        sents = [s.strip() for s in SENT_SPLIT.split(t) if len(s.split()) >= 3]
        cand = sents + ([t] if len(sents) != 1 and len(t.split()) >= 3 else [])
        for u in cand:
            units.append(u)
            owner.append(i)
    if not units:
        return res
    P = clf.predict_proba(vec.transform(units))
    for u, i, row in zip(units, owner, P):
        for c in ("Threat", "Abusive"):
            if c in classes:
                p = float(row[classes.index(c)])
                if p > res[i].get(c, (0.0, ""))[0]:
                    res[i][c] = (p, u)
    return res


def top_words(vec, clf, text, cls):
    v = vec.transform([text])
    i = list(clf.classes_).index(cls)
    names = vec.get_feature_names_out()
    c = {names[j]: v[0, j] * clf.coef_[i][j] for j in v.nonzero()[1]}
    return ", ".join(w for w in sorted(c, key=c.get, reverse=True)[:3] if c[w] > 0)


@st.cache_data(show_spinner=False)
def analyze(raw: str, sens: int, default_sender: str = ""):
    """Analyse the REAL text supplied by the user. sens = sensitivity slider (0-100). Returns (dataframe, patterns).

    Three stages: (1) every message is scanned for linguistic cues and scored per category, (2) a sentence-level ML
    model adds Threat / Abusive, (3) a CONTEXT pass: veiled threats are often sent as several short messages in a row
    ("Don't test me." / "You really don't want to find out."), so Intimidation cues from consecutive messages by the
    same sender are combined before deciding."""
    df = parse_chat(raw, default_sender)
    if df.empty:
        return df, []
    thr = sens_to_thr(sens)          # rule / cue confidence needed to flag
    ml_thr = min(0.95, thr + 0.15)   # the ML model is noisier, so it needs a little more confidence
    texts = df["message"].tolist()
    senders = df["sender"].tolist()
    ml = ml_scores(texts)
    model = load_model()
    all_cues, flagged_all = [], []
    for i, text in enumerate(texts):
        cues = cue_scan(text)
        all_cues.append(cues)
        flagged = {}
        # 1) linguistic cues -> noisy-OR score per category
        for cat, items in cues.items():
            sc = noisy_or([w for w, _, _ in items])
            if sc + 1e-9 >= thr:
                n = len(items)
                sev = SEV[cat]
                if cat == "Intimidation" and n >= 3 and sc >= 0.9:
                    sev = 4  # several independent warning signals in one message
                shown = "; ".join(f"{lbl} '{m[:60]}'" for _, lbl, m in sorted(items, key=lambda x: -x[0])[:5])
                flagged[cat] = {"score": sc, "sev": sev, "rule": True,
                                "why": f"{cat} ({n} signal{'s' if n > 1 else ''}, {sc:.0%}): {shown}"}
        # 2) machine learning (sentence level) for Threat / Abusive
        for cat in ("Threat", "Abusive"):
            if cat in ml[i]:
                p, sent = ml[i][cat]
                if p >= ml_thr and cat not in flagged:
                    sev = (4 if p >= 0.85 else 3) if cat == "Threat" else (2 if p >= 0.75 else 1)
                    tw = top_words(*model, sent, cat) if model else ""
                    flagged[cat] = {"score": p, "sev": sev, "rule": False,
                                    "why": f"ML model: {cat} ({p:.0%}) in \"{sent[:70]}\"" + (f"; key words: {tw}" if tw else "")}
        flagged_all.append(flagged)
    # 3) context pass: combine Intimidation cues of up to 4 consecutive messages from the same sender
    for i in range(len(texts)):
        if "Intimidation" not in all_cues[i]:
            continue
        j0 = i
        while j0 > 0 and i - j0 < 3 and senders[j0 - 1] == senders[i]:
            j0 -= 1
        win = [j for j in range(j0, i + 1) if "Intimidation" in all_cues[j]]
        if len(win) < 2:
            continue
        items = {}
        for j in win:
            for w, lbl, m in all_cues[j]["Intimidation"]:
                items.setdefault(lbl, (w, lbl, m))
        items = list(items.values())
        sc = noisy_or([w for w, _, _ in items])
        # a pile of purely administrative cues ("last chance", "final warning") must not be flagged on their own
        if sc + 1e-9 >= thr and (len(items) >= 3 or max(w for w, _, _ in items) >= 0.6):
            sev = 4 if len(items) >= 3 and sc >= 0.9 else SEV["Intimidation"]
            shown = "; ".join(f"{lbl} '{m[:60]}'" for _, lbl, m in sorted(items, key=lambda x: -x[0])[:5])
            why = f"Intimidation (context: {len(win)} consecutive messages from {senders[i]}, {sc:.0%}): {shown}"
            for j in win:
                if "Intimidation" not in flagged_all[j]:
                    flagged_all[j]["Intimidation"] = {"score": sc, "sev": sev, "rule": True, "why": why}
    labels, cats, sevs, reasons, confs = [], [], [], [], []
    for i, flagged in enumerate(flagged_all):
        if flagged:
            # most severe wins; on a tie a rule-based category beats an ML-only one, then the priority order decides
            main = max(flagged, key=lambda c: (flagged[c]["sev"], flagged[c]["rule"], -CATS.index(c)))
            order = [main] + [c for c in CATS if c in flagged and c != main]
            labels.append(main)
            cats.append(" + ".join(order))
            sevs.append(flagged[main]["sev"])
            reasons.append(" | ".join(flagged[c]["why"] for c in order))
            confs.append(round(max(f["score"] for f in flagged.values()), 2))
        else:
            labels.append("Safe")
            cats.append("Safe")
            sevs.append(0)
            reasons.append("")
            confs.append(round(max([p for p, _ in ml[i].values()], default=0.0), 2))
    df["label"], df["categories"], df["sev"], df["reason"], df["conf"] = labels, cats, sevs, reasons, confs
    df["level"] = df["sev"].map(LEVEL)
    df["points"] = df["sev"].map(lambda s: s * s * 3)
    return df, find_patterns(df)


def find_patterns(df):
    pats = []
    # 1. message flood
    run = (df.sender != df.sender.shift()).cumsum()
    for _, g in df.groupby(run):
        if len(g) >= 4:
            pats.append({"title": "Message flooding", "sender": g.sender.iloc[0], "bonus": 8,
                         "detail": f"{g.sender.iloc[0]} sent {len(g)} messages in a row without any reply."})
    # 2. contact after refusal
    ref = df[df.message.str.contains(REFUSAL)]
    if not ref.empty:
        r0 = ref.iloc[0]
        after = df[(df.no > r0.no) & (df.sender != r0.sender)]
        if len(after) >= 2:
            who = after.sender.value_counts().index[0]
            pats.append({"title": "Contact continued after refusal", "sender": who, "bonus": 12,
                         "detail": f"{len(after)} message(s) from {who} after {r0.sender} asked them to stop (message #{r0.no})."})
    # 3. repeated threats / blackmail
    for s, g in df[df.label.isin(["Threat", "Blackmail", "Intimidation"])].groupby("sender"):
        if len(g) >= 2:
            pats.append({"title": "Repeated threats", "sender": s, "bonus": 6,
                         "detail": f"{s} made {len(g)} threatening / blackmail / intimidating messages."})
    # 4. escalation
    for s, g in df[df.sev > 0].groupby("sender"):
        g = g.sort_values("no")
        if len(g) >= 4:
            h = len(g) // 2
            if g.sev.iloc[h:].mean() >= g.sev.iloc[:h].mean() + 0.5:
                pats.append({"title": "Escalating severity", "sender": s, "bonus": 8,
                             "detail": f"Messages from {s} become more severe over time "
                                       f"(average severity {g.sev.iloc[:h].mean():.1f} rising to {g.sev.iloc[h:].mean():.1f})."})
    # 5. late-night harassment
    if df.ts.notna().any():
        late = df[(df.sev > 0) & (df.ts.dt.hour < 5)]
        for s, g in late.groupby("sender"):
            if len(g) >= 2:
                pats.append({"title": "Late-night harassment", "sender": s, "bonus": 4,
                             "detail": f"{len(g)} flagged messages from {s} between 12 am and 5 am."})
    return pats


def risk_level(score):
    return "Low" if score < 20 else "Medium" if score < 45 else "High" if score < 70 else "Critical"


def sender_risk(df, pats):
    out = {}
    for s, g in df.groupby("sender"):
        pts = g.points.sum() + sum(p["bonus"] for p in pats if p["sender"] == s)
        out[s] = int(min(100, pts))
    return out


# ----------------------------------------------------------------------------
# PRIVACY HELPERS
# ----------------------------------------------------------------------------
PII = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
       (re.compile(r"https?://\S+|www\.\S+"), "[link]"),
       (re.compile(r"\+?\d[\d\s\-]{8,14}\d"), "[phone]")]


def mask_pii(t):
    for rx, rep in PII:
        t = rx.sub(rep, t)
    return t


def make_view(df, pats, mask, anon):
    """Return display copy: masked text / anonymised senders. Raw analysis stays untouched."""
    v = df.copy()
    pv = [dict(p) for p in pats]
    if mask:
        v["message"] = v["message"].map(mask_pii)
    if anon:
        names = list(dict.fromkeys(v.sender))
        mp = {n: f"Person {chr(65 + i)}" if n != "Unknown" else "Unknown" for i, n in enumerate(names)}
        v["sender"] = v["sender"].map(mp)
        for p in pv:
            for old, new in mp.items():
                p["detail"] = p["detail"].replace(old, new)
            p["sender"] = mp.get(p["sender"], p["sender"])
    return v, pv


# ----------------------------------------------------------------------------
# PDF REPORT
# ----------------------------------------------------------------------------
def pdf_safe(s):
    s = str(s)
    for k, v in {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", "₹": "Rs.", "•": "-"}.items():
        s = s.replace(k, v)
    s = s.encode("latin-1", "replace").decode("latin-1")
    return re.sub(r"(\S{45})(?=\S)", r"\1 ", s)  # break very long words so the PDF never errors


class ReportPDF(FPDF):
    def header(self):
        self.set_fill_color(255, 194, 217)
        self.rect(0, 0, 210, 14, "F")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(58, 11, 34)
        self.set_xy(10, 4)
        self.cell(0, 6, "ShieldDM  |  Online Harassment Evidence & Complaint Report")
        self.set_text_color(0, 0, 0)
        self.set_y(20)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}  |  Auto-generated draft. Verify with a legal professional before filing.", align="C")


def _p(pdf, text, h=5.5, style="", size=10, color=(0, 0, 0)):
    pdf.set_font("Helvetica", style, size)
    pdf.set_text_color(*color)
    pdf.multi_cell(pdf.epw, h, pdf_safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _h(pdf, text):
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(190, 60, 110)
    pdf.multi_cell(pdf.epw, 7, pdf_safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(255, 182, 213)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + pdf.epw, pdf.get_y())
    pdf.ln(2)
    pdf.set_text_color(0, 0, 0)


def _kv(pdf, k, v):
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(48, 6, pdf_safe(k))
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(pdf.epw - 48, 6, pdf_safe(v), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


JURIS = {
    "India": [
        "Do NOT delete the chat. Back up the original export (Chat > Export chat) and keep the phone safe.",
        "Take screenshots showing the sender's name/number, date and time of each flagged message.",
        "Report online at cybercrime.gov.in or call the National Cyber Crime helpline 1930 as soon as possible.",
        "Visit the nearest police station / cyber cell with this report, screenshots and the original chat export.",
        "Women can also call the Women Helpline 181; if you are in immediate danger, call 112.",
        "Report and block the account on the platform (Instagram / WhatsApp) AFTER saving all evidence.",
        "Tell a trusted friend or family member, and avoid replying to or meeting the accused.",
    ],
    "United States": [
        "Do NOT delete the chat. Back up the original export and keep the phone safe.",
        "Take screenshots showing the sender's name/number, date and time of each flagged message.",
        "Report online to the FBI Internet Crime Complaint Center (ic3.gov) and to your local police department.",
        "If you are in immediate danger, call 911.",
        "If intimate images are involved, the Cyber Civil Rights Initiative (cybercivilrights.org) offers support.",
        "Report and block the account on the platform AFTER saving all evidence.",
        "Tell a trusted friend or family member, and avoid replying to or meeting the accused.",
    ],
    "United Kingdom": [
        "Do NOT delete the chat. Back up the original export and keep the phone safe.",
        "Take screenshots showing the sender's name/number, date and time of each flagged message.",
        "Report to the police: call 101 (non-emergency) or use your local force's online form. In immediate danger call 999.",
        "Report Harmful Content (reportharmfulcontent.com) can help with removal; the Revenge Porn Helpline (revengepornhelpline.org.uk) helps if intimate images are involved.",
        "Report and block the account on the platform AFTER saving all evidence.",
        "Tell a trusted friend or family member, and avoid replying to or meeting the accused.",
    ],
    "Other / not sure": [
        "Do NOT delete the chat. Back up the original export and keep the phone safe.",
        "Take screenshots showing the sender's name/number, date and time of each flagged message.",
        "Report to your local police or cyber-crime unit with this report, the screenshots and the original chat export.",
        "If you are in immediate danger, call your local emergency number.",
        "Report and block the account on the platform AFTER saving all evidence.",
        "Tell a trusted friend or family member, and avoid replying to or meeting the accused.",
    ],
}
GENERIC_LAW = {
    "Threat": "criminal threats / criminal intimidation laws",
    "Blackmail": "extortion / blackmail laws",
    "Intimidation": "criminal intimidation / threatening communications laws",
    "Sexual Harassment": "sexual harassment and explicit-image laws",
    "Stalking": "stalking / cyberstalking / harassment laws",
    "Abusive": "abusive or harassing communications laws",
}
LAW_EXAMPLES = {
    "United States": "For example 18 U.S.C. 875(c) (threats in interstate communications), 18 U.S.C. 2261A (stalking) and state harassment laws.",
    "United Kingdom": "For example the Protection from Harassment Act 1997 and the Malicious Communications Act 1988.",
    "Other / not sure": "Check your country's criminal code for threats, harassment, stalking and extortion.",
}


def build_pdf(view, pats, evidence_from, accused, complainant, contact, platform, place, statement, source, risk, anon,
              jurisdiction="India"):
    """evidence_from = sender whose messages are evidence (or "All senders"); accused = name printed in the report."""
    all_s = evidence_from == "All senders"
    mine = view if all_s else view[view.sender == evidence_from]
    ev = mine[mine.sev > 0].sort_values("no")
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(True, margin=18)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(pdf.epw, 10, "Complaint Report: Online Harassment / Threats", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    case_id = "SDM-" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
    # --- incident details
    _h(pdf, "1. Incident details")
    first = ev.ts.min() if ev.ts.notna().any() else None
    last = ev.ts.max() if ev.ts.notna().any() else None
    period = f"{first:%d %b %Y %H:%M}  to  {last:%d %b %Y %H:%M}" if first is not None and pd.notna(first) else \
        (f"{ev.time.iloc[0]}  to  {ev.time.iloc[-1]}" if len(ev) and ev.time.iloc[0] != "-" else "Not available in chat text")
    _kv(pdf, "Case reference", case_id)
    _kv(pdf, "Report generated", datetime.datetime.now().strftime("%d %b %Y, %H:%M"))
    _kv(pdf, "Complainant", complainant or "Not provided")
    if contact:
        _kv(pdf, "Complainant contact", contact)
    _kv(pdf, "Accused (name / handle)", accused)
    _kv(pdf, "Platform", platform)
    _kv(pdf, "Jurisdiction", jurisdiction)
    _kv(pdf, "Place (city)", place or "Not provided")
    _kv(pdf, "Incident period", period)
    _kv(pdf, "Evidence source", source)
    _kv(pdf, "Messages analysed", f"{len(view)} total, {len(mine)} sent by accused, {len(ev)} flagged")
    _kv(pdf, "Overall risk", f"{risk}/100  ({risk_level(risk)})")
    if anon:
        _kv(pdf, "Privacy note", "Names were anonymised in this report. Turn anonymisation off before filing officially.")
    # --- complainant statement
    if statement.strip():
        _h(pdf, "2. Complainant's statement")
        _p(pdf, statement.strip())
    # --- evidence summary
    _h(pdf, "3. Evidence summary")
    counts = ev.label.value_counts()
    cat_txt = ", ".join(f"{n} {c.lower()}" for c, n in counts.items()) or "none"
    top = LEVEL[int(ev.sev.max())] if len(ev) else "none"
    pct = (len(ev) / len(mine) * 100) if len(mine) else 0
    _p(pdf, f"The supplied chat contains {len(mine)} message(s) from {accused}, of which {len(ev)} ({pct:.0f}%) were flagged "
            f"as harmful by an automated classifier. Flagged content: {cat_txt}. The highest severity found is {top}. "
            f"The overall risk score for this conversation is {risk}/100 ({risk_level(risk)}).")
    mp = [p for p in pats if all_s or p["sender"] == evidence_from]
    if mp:
        pdf.ln(1)
        _p(pdf, "Behaviour patterns detected:", style="B")
        for p in mp:
            _p(pdf, f"-  {p['title']}: {p['detail']}")
    # --- flagged messages
    _h(pdf, "4. Flagged messages (evidence log)")
    if ev.empty:
        _p(pdf, "No flagged messages from the selected sender.")
    for n, (_, r) in enumerate(ev.head(80).iterrows(), 1):
        if pdf.get_y() > 245:
            pdf.add_page()
        pdf.set_fill_color(255, 232, 242)
        pdf.set_font("Helvetica", "B", 9)
        pdf.multi_cell(pdf.epw, 6, pdf_safe(f"#{n}   " + ("" if r.time == "-" else f"{r.time}   |   ") + f"{r.categories}   |   Severity: {r.level}"),
                       fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        _p(pdf, f'Message: "{r.message}"', size=10)
        if r.reason:
            _p(pdf, f"Detected because: {r.reason}", style="I", size=9, color=(90, 90, 90))
        pdf.ln(2)
    if len(ev) > 80:
        _p(pdf, f"... and {len(ev) - 80} more flagged messages (see CSV export).", style="I")
    # --- legal
    _h(pdf, "5. Possible legal provisions (indicative)")
    all_cats = {c for s in ev.categories for c in s.split(" + ")}
    for c in [c for c in CATS if c in all_cats]:
        _p(pdf, f"-  {c}: " + (LAW[c] if jurisdiction == "India" else f"may fall under {GENERIC_LAW[c]}"))
    if jurisdiction != "India":
        _p(pdf, LAW_EXAMPLES.get(jurisdiction, LAW_EXAMPLES["Other / not sure"]), size=9)
    _p(pdf, "These sections are suggestions to help you start the conversation with police or a lawyer. "
            "The final decision on applicable sections rests with the authorities.", style="I", size=9, color=(90, 90, 90))
    # --- next steps
    _h(pdf, "6. Suggested next steps")
    for step in JURIS.get(jurisdiction, JURIS["Other / not sure"]):
        _p(pdf, f"-  {step}")
    # --- declaration
    pdf.ln(6)
    _p(pdf, "Declaration: I confirm that the above information and attached chat are true to the best of my knowledge.", size=9)
    pdf.ln(10)
    _p(pdf, f"Signature: ____________________        Name: {complainant or '____________________'}        Date: ______________", size=9)
    return bytes(pdf.output())


# ----------------------------------------------------------------------------
# ----  UI  ----
# ----------------------------------------------------------------------------
st.set_page_config(page_title="ShieldDM", page_icon="🛡️", layout="wide")

CSS = """
<style>
.stApp { background:
  radial-gradient(1000px 520px at 0% -10%, rgba(255,182,213,.20), transparent 60%),
  radial-gradient(800px 500px at 100% 0%, rgba(255,107,129,.14), transparent 55%),
  radial-gradient(700px 500px at 50% 115%, rgba(244,163,200,.12), transparent 60%), #0a0509; color:#fff; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#1c0a15,#12070d); border-right:1px solid rgba(255,182,213,.22); }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.10); }
h1,h2,h3,h4, label, p, span, div { color:#f7eef2; }
.hero { background: linear-gradient(120deg,#ffd6e6,#ffbcd5,#ffa6c4,#ff9aa8); background-size:300% 300%;
  animation: shift 10s ease infinite; border-radius:28px; padding:34px 40px; margin-bottom:22px;
  box-shadow:0 18px 50px rgba(255,182,213,.30); }
.hero h1 { margin:0 0 6px; font-weight:800; font-size:3rem; color:#3a0b22; letter-spacing:-.5px; }
.hero p { margin:0; color:#5c1a38; font-size:1.05rem; }
@keyframes shift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
.kpi { background:rgba(255,255,255,.05); border:1px solid rgba(255,182,213,.35); border-radius:20px; padding:18px 16px;
  text-align:center; transition:transform .25s ease, box-shadow .25s ease; }
.kpi:hover { transform:translateY(-4px); box-shadow:0 10px 30px rgba(255,182,213,.28); }
.kpi .v { font-size:2rem; font-weight:800; line-height:1.1; }
.kpi .t { font-size:.82rem; opacity:.75; margin-bottom:4px; }
.badge { display:inline-block; padding:3px 12px; border-radius:99px; font-size:.78rem; font-weight:600; color:#2a0a18; }
.crit { animation: pulse 1.6s infinite; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(255,107,129,.7)} 70%{box-shadow:0 0 0 12px rgba(255,107,129,0)} 100%{box-shadow:0 0 0 0 rgba(255,107,129,0)} }
.msg { background:rgba(255,255,255,.045); border-left:5px solid #ffb6d2; border-radius:14px; padding:12px 16px; margin-bottom:10px; }
.msg .m { font-size:1rem; margin:6px 0 4px; }
.msg .r { font-size:.8rem; opacity:.7; }
.pat { background:linear-gradient(90deg,rgba(255,107,129,.18),rgba(255,182,213,.08)); border:1px solid rgba(255,107,129,.45);
  border-radius:16px; padding:14px 18px; margin-bottom:10px; }
.step { background:rgba(255,255,255,.05); border-radius:18px; padding:18px 22px; border:1px dashed rgba(255,182,213,.5); }
.crow { display:flex; margin:7px 0; }
.crow.right { justify-content:flex-end; }
.bub { max-width:74%; padding:10px 15px; border-radius:20px; background:rgba(255,255,255,.07); }
.crow.left .bub { border-bottom-left-radius:5px; }
.crow.right .bub { background:rgba(255,182,213,.14); border-bottom-right-radius:5px; }
.bub.flag { border:1px solid var(--c); box-shadow:0 0 16px -5px var(--c); }
.bub .who { font-size:.72rem; opacity:.65; margin-bottom:2px; }
.bub .txt { font-size:1rem; }
.bub .why { font-size:.76rem; opacity:.7; margin-top:5px; }
.stButton, .stDownloadButton, [data-testid="stButton"], [data-testid="stDownloadButton"] { width:100%; }
[data-testid="stElementContainer"]:has(.stButton), [data-testid="stElementContainer"]:has(.stDownloadButton),
[data-testid="stElementContainer"]:has([data-testid="stButton"]), [data-testid="stElementContainer"]:has([data-testid="stDownloadButton"]) { width:100% !important; }
.stButton>button, .stDownloadButton>button { width:100%; background:linear-gradient(90deg,#ffc9dc,#ff9fbd); color:#3a0b22; border:0;
  border-radius:14px; font-weight:600; padding:.6rem 1.2rem; transition:transform .2s ease, box-shadow .2s ease; }
.stButton>button:hover, .stDownloadButton>button:hover { transform:translateY(-2px); box-shadow:0 8px 24px rgba(255,182,213,.45); color:#3a0b22; }
.stButton>button:disabled, .stDownloadButton>button:disabled { opacity:.45; box-shadow:none; transform:none; color:#3a0b22; }
.stTabs [data-baseweb="tab-list"] { gap:20px; border-bottom:1px solid rgba(255,255,255,.12); }
.stTabs [data-baseweb="tab"] { background:transparent; padding:10px 2px; }
.stTabs [aria-selected="true"] p { color:#ffb6d2; font-weight:700; }
.stTabs [data-baseweb="tab-highlight"] { background:#ffb6d2; }
[data-testid="stSlider"] [role="slider"] { background-color:#ffb6d2 !important; }
.hero h1, .hero h1 * { color:#3a0b22 !important; }
.stButton>button p, .stButton>button span, .stDownloadButton>button p, .stDownloadButton>button span { color:#3a0b22 !important; }
textarea, input { background:#1a0f16 !important; color:#fff !important; }
@media (prefers-reduced-motion: reduce) { .hero, .crit { animation:none; } .kpi, .stButton>button { transition:none; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def H(s):
    st.markdown("".join(l.strip() for l in s.splitlines()), unsafe_allow_html=True)


def style_fig(fig, h=330):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#f7eef2", family="Source Sans Pro, Arial, sans-serif"), height=h,
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


def badge(level):
    cls = "badge crit" if level == "Critical" else "badge"
    return f'<span class="{cls}" style="background:{LEVEL_COLORS[level]}">{level}</span>'


# ---- session state ---------------------------------------------------------
for k, v in {"raw": "", "source": "", "last_sig": None, "pdf": None, "paste_text": ""}.items():
    st.session_state.setdefault(k, v)


def clear_all():
    st.session_state.raw, st.session_state.source, st.session_state.pdf = "", "", None
    st.session_state["paste_text"] = ""


# ---- sidebar ---------------------------------------------------------------
with st.sidebar:
    mode = st.radio("Input method", ["📋 Paste chat text", "📁 Upload WhatsApp .txt"])
    if mode.startswith("📋"):
        st.text_area("Paste the chat here", key="paste_text", height=230,
                     placeholder="Paste your chat here...")
        if st.button("🔍 Analyze pasted text"):
            if st.session_state.paste_text.strip():
                st.session_state.raw, st.session_state.source, st.session_state.pdf = st.session_state.paste_text, "Pasted text", None
            else:
                st.warning("The box is empty. Paste some messages first.")
    else:
        f = st.file_uploader("Upload WhatsApp chat (.txt)", type=["txt"],
                             help="WhatsApp: open the chat > ⋮ > More > Export chat > Without media")
        if f is not None:
            sig = (f.name, f.size)
            if sig != st.session_state.last_sig:  # load once per new file
                st.session_state.raw = f.getvalue().decode("utf-8-sig", errors="ignore")
                st.session_state.source, st.session_state.pdf, st.session_state.last_sig = f"Uploaded file: {f.name}", None, sig
        else:
            st.session_state.last_sig = None
    st.divider()
    if st.button("🎬 Load demo data"):
        st.session_state.raw, st.session_state.source, st.session_state.pdf = DEMO_CHAT, "Demo data", None
    st.divider()
    st.markdown("### ⚙️ Detection")
    flag_pct = st.slider("Flag sensitivity (lower = flags more)", 20, 95, 60, 5,
                         help="Every message gets a confidence score. It is flagged only when that score reaches "
                              "this percentage, so changing the slider changes what is flagged, the flagged counts and the risk scores.")
    slider_info = st.empty()  # filled after the analysis, so the slider effect is visible
    st.markdown("### 🔒 Privacy")
    mask = st.toggle("Mask phone numbers & emails", value=True, help="Phone numbers, emails and links become [phone], [email], [link].")
    anon = st.toggle("Replace sender names with Person A/B", value=False,
                     help="Applies on screen, in the CSV and in the PDF.")
    st.caption("Your chat is processed in memory only. Nothing is written to disk or sent anywhere.")
    st.button("🗑️ Clear all data", on_click=clear_all)

if load_model() is None:
    st.warning("model.pkl not found: running with the cue lexicon only. Run `python train.py` to enable the ML model.")

# ---- hero ------------------------------------------------------------------
H("""<div class="hero"><h1>🛡️ ShieldDM</h1>
<p>AI-powered detection of threats, abuse, sexual harassment, blackmail and stalking in chats — with court-ready evidence reports.</p></div>""")

raw = st.session_state.raw
if not raw.strip():
    H("""<div class="step"><b>👈 Nothing analyzed yet.</b><br>Paste a chat, upload a WhatsApp export or load demo data
from the sidebar to see the dashboard, flagged messages, timeline, full chat and complaint report.</div>""")
    st.stop()

# ---- analysis (always on the user's current input) -------------------------
df, pats = analyze(raw, flag_pct, "")
if df.empty:
    st.error("No messages could be read. Paste at least one line of chat text.")
    st.stop()
view, pview = make_view(df, pats, mask, anon)
risks = sender_risk(view, pview)
accused_default = max(risks, key=risks.get)
overall = max(risks.values())
lvl = risk_level(overall)
flagged = view[view.sev > 0]
slider_info.caption(f"At {flag_pct}%: **{len(flagged)}** of {len(view)} messages flagged, risk **{overall}/100**")

tab_dash, tab_msgs, tab_time, tab_chat, tab_model, tab_rep = st.tabs(
    ["📊 Dashboard", "🚩 Flagged Messages", "🕒 Timeline", "💬 Full Chat", "🤖 Model Metrics", "⚖️ Legal Report"])

# ---------------- dashboard
with tab_dash:
    st.caption(f"Analyzing: **{st.session_state.source}**  |  {len(view)} messages from {view.sender.nunique()} sender(s)")
    k = st.columns(5)
    kp = [("Messages", len(view)), ("Flagged", len(flagged)),
          ("Flagged %", f"{len(flagged) / len(view) * 100:.0f}%"),
          ("Top severity", LEVEL[int(view.sev.max())]), ("Risk score", f"{overall}/100")]
    for col, (t, v) in zip(k, kp):
        col.markdown(f'<div class="kpi"><div class="t">{t}</div><div class="v">{v}</div></div>', unsafe_allow_html=True)
    st.write("")
    H(f'<div style="margin-bottom:6px">Overall risk level: {badge(lvl)}</div>')
    c1, c2, c3 = st.columns([1, 1, 1.2])
    g = go.Figure(go.Indicator(mode="gauge+number", value=overall, title={"text": "Risk score"},
                               gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#ffb6d2"},
                                      "bgcolor": "#1a0f16",
                                      "steps": [{"range": [0, 20], "color": "#2a1b24"}, {"range": [20, 45], "color": "#4a2a3a"},
                                                {"range": [45, 70], "color": "#6b2f4a"}, {"range": [70, 100], "color": "#8a2b46"}]}))
    c1.plotly_chart(style_fig(g), key="gauge", theme=None)
    cc = view.label.value_counts().reset_index()
    cc.columns = ["Category", "Messages"]
    c2.plotly_chart(style_fig(px.pie(cc, names="Category", values="Messages", hole=.55, color="Category",
                                     color_discrete_map=CAT_COLORS, title="Category split")), key="pie", theme=None)
    if len(flagged):
        sb = flagged.groupby(["sender", "label"]).size().reset_index(name="Messages")
        c3.plotly_chart(style_fig(px.bar(sb, x="sender", y="Messages", color="label", color_discrete_map=CAT_COLORS,
                                         title="Flagged messages by sender", labels={"label": "Category", "sender": "Sender"})),
                        key="bar", theme=None)
    else:
        c3.success("No harmful messages detected in this chat. 🎉")
    st.markdown("#### 🧩 Behavior patterns")
    if pview:
        for p in pview:
            H(f'<div class="pat"><b>⚠️ {html.escape(p["title"])}</b><br>{html.escape(p["detail"])}</div>')
    else:
        st.info("No harassment patterns (flooding, escalation, contact after refusal…) found.")
    st.markdown("#### 🔥 Most severe messages")
    for _, r in flagged.sort_values(["sev", "no"], ascending=[False, True]).head(5).iterrows():
        H(f'<div class="msg" style="border-left-color:{CAT_COLORS[r.label]}">{ICON[r.label]} <b>{html.escape(r.categories)}</b> '
          f'{badge(r.level)} <span class="r">  #{r.no}  {html.escape(r.sender)}  {html.escape(r.time)}</span>'
          f'<div class="m">{html.escape(r.message)}</div><div class="r">Why: {html.escape(r.reason or "-")}</div></div>')

# ---------------- flagged messages with filters
with tab_msgs:
    f1, f2, f3, f4 = st.columns([1.3, 1, 1, 1.2])
    cat_sel = f1.multiselect("Category", CATS + ["Safe"], default=CATS)
    snd_sel = f2.multiselect("Sender", sorted(view.sender.unique()), default=sorted(view.sender.unique()))
    min_sev = f3.select_slider("Minimum severity", ["Low", "Medium", "High", "Critical"], value="Low")
    kw = f4.text_input("Search text")
    ms = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}[min_sev]
    flt = view[view.label.isin(cat_sel) & view.sender.isin(snd_sel) & ((view.sev >= ms) | (view.label == "Safe"))]
    if kw:
        flt = flt[flt.message.str.contains(kw, case=False, regex=False)]
    out = flt[["no", "time", "sender", "message", "categories", "level", "points", "reason", "conf"]].rename(
        columns={"no": "#", "time": "Time", "sender": "Sender", "message": "Message", "categories": "Category",
                 "level": "Severity", "points": "Risk pts", "reason": "Reason", "conf": "Confidence"})
    st.caption(f"Showing {len(out)} of {len(view)} messages")
    st.dataframe(out, hide_index=True, column_config={
        "Risk pts": st.column_config.ProgressColumn("Risk pts", min_value=0, max_value=48, format="%d"),
        "Message": st.column_config.TextColumn(width="large"), "Reason": st.column_config.TextColumn(width="large")})
    st.download_button("⬇️ Export results as CSV", out.to_csv(index=False).encode("utf-8-sig"),
                       "shielddm_results.csv", "text/csv")

# ---------------- timeline
with tab_time:
    tl = view.copy()
    use_ts = tl.ts.notna().all()
    tl["x"] = tl.ts if use_ts else tl.no
    tl["short"] = tl.message.str.slice(0, 70)
    fig = px.scatter(tl, x="x", y="sev", color="label", color_discrete_map=CAT_COLORS, hover_data=["sender", "short"],
                     title="Incident timeline (severity of each message)",
                     labels={"x": "Time" if use_ts else "Message #", "sev": "Severity (0-4)", "label": "Category"})
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color="#fff")))
    fig.update_yaxes(tickvals=[0, 1, 2, 3, 4], ticktext=["Safe", "Low", "Medium", "High", "Critical"])
    st.plotly_chart(style_fig(fig, 380), key="tl1", theme=None)
    tl["cum"] = tl.points.cumsum().clip(upper=100)
    fig2 = px.area(tl, x="x", y="cum", title="Escalation curve (cumulative risk)",
                   labels={"x": "Time" if use_ts else "Message #", "cum": "Risk"})
    fig2.update_traces(line_color="#ffb6d2", fillcolor="rgba(255,182,213,.35)")
    st.plotly_chart(style_fig(fig2, 300), key="tl2", theme=None)
    if use_ts:
        hr = flagged.assign(hour=flagged.ts.dt.hour).groupby("hour").size().reset_index(name="Flagged")
        if len(hr):
            st.plotly_chart(style_fig(px.bar(hr, x="hour", y="Flagged", title="Flagged messages by hour of day",
                                             color_discrete_sequence=["#ff9a8a"]), 260), key="tl3", theme=None)

# ---------------- full chat
with tab_chat:
    only_flag = st.toggle("Show only flagged messages", value=False)
    sub_chat = view[view.sev > 0] if only_flag else view
    first_sender = view.sender.iloc[0]
    st.caption(f"{len(sub_chat)} message(s) shown. Highlighted bubbles were flagged; the reason is shown under each one.")
    parts = []
    for _, r in sub_chat.head(300).iterrows():
        side = "left" if r.sender == first_sender else "right"
        col, fl = CAT_COLORS[r.label], r.sev > 0
        tag = (f'<span class="badge" style="background:{col}">{ICON[r.label]} {html.escape(r.categories)} · {r.level}</span>'
               if fl else "")
        why = f'<div class="why">{html.escape(r.reason[:220])}</div>' if fl and r.reason else ""
        tm = "" if r.time == "-" else f" · {html.escape(r.time)}"
        parts.append(f'<div class="crow {side}"><div class="bub{" flag" if fl else ""}" style="--c:{col}">'
                     f'<div class="who">{html.escape(r.sender)}{tm} · #{r.no}</div><div class="txt">{html.escape(r.message)}</div>'
                     f'{tag}{why}</div></div>')
    H("".join(parts))
    if len(sub_chat) > 300:
        st.caption("Showing the first 300 messages. Use Flagged Messages for the complete list.")

# ---------------- model metrics
with tab_model:
    m = load_metrics()
    st.markdown("**Detection engine:** hybrid. A cue lexicon (English + Hinglish) scores threats, blackmail, intimidation "
                "(indirect threats), sexual harassment, stalking and abuse, and combines several weak signals into one "
                "confidence. A sentence-level ML model (TF-IDF + Logistic Regression) adds threat and abuse detection. "
                "The sensitivity slider decides how much confidence is needed to flag a message.")
    if m:
        # Every part is optional, and any problem is caught here, so an old or incomplete metrics.json can never
        # break this tab or the tabs after it (Legal Report).
        try:
            a_, b_, c_, d_ = st.columns(4)
            acc_v = f"{m['accuracy']:.1%}" if "accuracy" in m else "n/a"
            f1_v = f"{m['macro_f1']:.2f}" if "macro_f1" in m else "n/a"
            for col, (t, v) in zip((a_, b_, c_, d_), [("Accuracy", acc_v), ("Macro F1", f1_v),
                                                       ("Train rows", m.get("train_rows", "n/a")), ("Test rows", m.get("test_rows", "n/a"))]):
                col.markdown(f'<div class="kpi"><div class="t">{t}</div><div class="v">{v}</div></div>', unsafe_allow_html=True)
            st.write("")
            r1, r2 = st.columns(2)
            if m.get("report"):
                r1.markdown("**Per-class report**")
                r1.dataframe(pd.DataFrame(m["report"]).T.round(2))
            if m.get("confusion_matrix") and m.get("labels"):
                cm = pd.DataFrame(m["confusion_matrix"], index=m["labels"], columns=m["labels"])
                r2.plotly_chart(style_fig(px.imshow(cm, text_auto=True, color_continuous_scale=["#1a0f16", "#ffb6d2", "#ff5c7a"],
                                                    title="Confusion matrix", labels=dict(x="Predicted", y="Actual")), 330),
                                key="cm", theme=None)
            if m.get("comparison"):
                st.markdown("**Model comparison**")
                st.dataframe(pd.DataFrame(m["comparison"]), hide_index=True)
            v = m.get("veiled_validation")
            if v:
                st.markdown("**Indirect-threat check** (hand-written sentences that were never used for training)")
                va, vb = st.columns(2)
                va.metric("Veiled threats caught by the ML model alone", f"{v['ml_recall']:.0%}", help=f"{v['veiled_sentences']} sentences")
                vb.metric("Harmless look-alikes wrongly flagged", f"{v['ml_false_alarm']:.0%}", help=f"{v['safe_lookalikes']} sentences")
                st.caption("The cue lexicon and the context pass catch more of these on top of the ML model.")
            st.caption("These numbers come from held-out Jigsaw comments only (hand-written chat samples are used for training, never for testing). "
                       "The ML model handles Safe / Abusive / Threat; blackmail, intimidation, sexual harassment and stalking come from the cue lexicon.")
        except Exception:
            st.warning("metrics.json is incomplete or from an older version. Run `python train.py` once to regenerate it. "
                       "The rest of the app is not affected.")
    else:
        st.info("metrics.json not found. Run `python train.py` to generate evaluation metrics.")

# ---------------- legal report
with tab_rep:
    st.markdown("### ⚖️ Legal Complaint Report")
    st.caption("Builds a PDF from the chat you actually loaded: incident details, flagged messages, timestamps, evidence summary and next steps.")
    senders = sorted(view.sender.unique())
    MANUAL = "✍️ Enter manually"
    ev_opts = senders + (["All senders"] if len(senders) > 1 else [])
    opts = ev_opts + [MANUAL]
    c1, c2 = st.columns(2)
    choice = c1.selectbox("Accused person", opts, index=opts.index(accused_default) if accused_default in opts else 0,
                          help="Pick a sender from the chat, or choose 'Enter manually' to type any name or username.")
    comp = c2.text_input("Complainant (your name)")
    if choice == MANUAL:
        typed = c1.text_input("Accused name or username", placeholder="e.g. Rohan Mehta or @rohan_insta")
        ev_from = c1.selectbox("Whose messages are the evidence?", ev_opts,
                               index=ev_opts.index(accused_default) if accused_default in ev_opts else 0)
        acc_name = typed.strip()
    else:
        ev_from, acc_name = choice, ("All senders in this chat" if choice == "All senders" else choice)
    contact = c2.text_input("Your contact details (optional)")
    juris = st.selectbox("Jurisdiction for next-step guidance", list(JURIS))
    stmt = st.text_area("Your statement / additional context (optional)", height=100)
    sub = view if ev_from == "All senders" else view[view.sender == ev_from]
    n_ev = int((sub.sev > 0).sum())
    need_name = choice == MANUAL and not acc_name
    if need_name:
        st.warning("Type the accused person's name or username to continue.")
    elif n_ev == 0:
        st.warning("No flagged messages for this selection, so there is no evidence to put in a report.")
    else:
        st.write(f"This report will include **{n_ev}** flagged message(s) against **{acc_name}**.")
    if st.button("⚖️ Generate Legal Complaint Report", disabled=(n_ev == 0 or need_name)):
        try:
            plat = "WhatsApp" if (view.time != "-").any() else "Not specified"
            st.session_state.pdf = build_pdf(view, pview, ev_from, acc_name, comp.strip(), contact.strip(), plat, "",
                                             stmt, st.session_state.source,
                                             overall if ev_from == "All senders" else risks.get(ev_from, overall), anon, juris)
            st.session_state.pdf_name = f"complaint_report_{re.sub(r'[^A-Za-z0-9]+', '_', acc_name).strip('_') or 'accused'}.pdf"
        except Exception as e:  # never crash the app
            st.session_state.pdf = None
            st.error(f"Could not create the PDF: {e}")
    if st.session_state.pdf:
        st.success("Report ready. Download it below.")
    st.download_button("⬇️ Download PDF report", st.session_state.pdf or b"",
                       st.session_state.get("pdf_name", "complaint_report.pdf"), "application/pdf",
                       disabled=not st.session_state.pdf)
    if st.session_state.pdf and mask:
        st.caption("Phone numbers, emails and links are masked in this report (Privacy settings in the sidebar).")