"""
ShieldDM - chat harassment analyser
===================================
Run:  streamlit run app.py

Everything shown is computed from the chat YOU paste / upload.
Demo data (sample_chat.txt) is only loaded when you press the demo button.
"""
import datetime as dt
import hashlib
import html
import io
import json
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model.joblib")
METRICS_PATH = os.path.join(BASE, "metrics.json")
SAMPLE_PATH = os.path.join(BASE, "sample_chat.txt")

SAFE_LABELS = {"safe", "normal", "neutral", "clean", "ok", "none", "non-toxic",
               "nontoxic", "not harmful", "non harmful", "benign", "0", "no",
               "false", "harmless"}

CATEGORY_COLORS = {
    "Threat": "#ff2d55",
    "Abuse": "#ff8a65",
    "Sexual Harassment": "#ff2ea6",
    "Blackmail": "#b0003a",
    "Stalking": "#ffb3d1",
    "Other Harmful": "#b39ba5",
}
SEVERITY_ORDER = ["Critical", "High", "Medium", "Low"]
SEVERITY_COLORS = {"Critical": "#ff1744", "High": "#ff5c8a", "Medium": "#ff9e80", "Low": "#ffc2d6"}

# --------------------------------------------------------------------------
# 1. Rule engine  (category, weight 0-1, reason, regex)
# --------------------------------------------------------------------------
_RULES_RAW = [
    # ---- Threat
    ("Threat", 0.92, "Explicit threat of violence",
     r"\b(i('| a)?m|i will|i'll|gonna|going to|will)\s+(kill|murder|hurt|beat|break|destroy|burn|stab|shoot|finish)\b"),
    ("Threat", 0.92, "Threatens to kill / physically harm the recipient",
     r"\b(kill|murder|stab|shoot|beat up|break)\s+(you|u|ur|your|her|him|them)\b"),
    ("Threat", 0.85, "Intimidating warning of harm",
     r"\b(watch your back|you'?re dead|you are dead|dead meat|you won'?t live|end you|make you pay|teach you a lesson|ruin your life)\b"),
    ("Threat", 0.70, "Warns of consequences (intimidation)",
     r"\byou('ll| will)\s+(regret|pay for|be sorry|suffer)\b"),
    ("Threat", 0.88, "Threat in Hindi/Hinglish",
     r"\b(maar dunga|jaan se (maar|khatam)|kaat dunga|nahi (chhodunga|chodunga)|barbaad kar dunga|dekh lunga tujhe)\b"),
    # ---- Abuse
    ("Abuse", 0.75, "Strong abusive language / slur",
     r"\b(bitch|bastard|slut|whore|asshole|motherfucker|scum|retard|fuck (you|off)|chutiya|madarchod|bhenchod|behenchod|harami|randi|gandu|kutti|kamina|haramzada)\b"),
    ("Abuse", 0.50, "Insulting / demeaning language",
     r"\b(idiot|stupid|dumb|moron|loser|ugly|worthless|useless|trash|pathetic|disgusting|shut up|bakwas|pagal|gadha)\b"),
    ("Abuse", 0.55, "Demeaning statement about the person's worth",
     r"\b(nobody|no one)\s+(likes|loves|wants|cares about)\s+(you|u)\b"),
    ("Abuse", 0.90, "Encourages self-harm / suicide",
     r"\b(go (die|to hell)|kill yourself|kys|you should (die|disappear)|hope you die)\b"),
    # ---- Sexual harassment
    ("Sexual Harassment", 0.92, "Requests nude / explicit images",
     r"\b(send|show|share)\s+(me\s+)?(your\s+)?(nudes?|nude (pics?|photos?)|naked|boobs|dick pic|private (pics?|photos?)|body pics?)\b"),
    ("Sexual Harassment", 0.85, "Unwanted sexual proposition",
     r"\b(sleep with me|have sex|sex with me|hook ?up with me|bed with me|spend the night with me|sexual favou?rs?)\b"),
    ("Sexual Harassment", 0.92, "Quid-pro-quo sexual demand",
     r"\b(you owe me|if you want (the job|marks|promotion|passing)).{0,40}\b(sex|night|favou?r|sleep)\b"),
    ("Sexual Harassment", 0.55, "Unsolicited sexual / objectifying remark",
     r"\b(sexy|hot body|boobs|tits|horny|nude|naked|touch you|kiss you|your body|figure)\b"),
    ("Sexual Harassment", 0.60, "Inappropriate invitation / personal probing",
     r"\b(what are you wearing|come to my (room|place|hotel|flat)|send (me )?(a )?(hot|sexy) (pic|photo|selfie))\b"),
    # ---- Blackmail
    ("Blackmail", 0.92, "Threatens to leak / expose content",
     r"\b(i('| wi)ll|gonna|going to|will)\s+(leak|post|share|expose|upload|send|tell|publish|viral)\b.{0,60}\b(photos?|pics?|videos?|screenshots?|chats?|nudes?|secret|everyone|family|parents|friends|boss|college|internet|online|social media|instagram|facebook)\b"),
    ("Blackmail", 0.88, "Conditional threat ('or else')",
     r"\b(or else|otherwise)\b.{0,50}\b(leak|expose|post|share|tell|viral|send)\b"),
    ("Blackmail", 0.90, "Claims to possess private material",
     r"\bi have (your|ur)\s+(private\s+)?(photos?|pics?|videos?|nudes?|screenshots?|chats?|recordings?)\b"),
    ("Blackmail", 0.88, "Demands money under threat",
     r"\b(pay|send|transfer)\s+(me\s+)?(₹|rs\.?|inr|\$|money|\d{3,}).{0,40}\b(or|else|otherwise|delete)\b"),
    ("Blackmail", 0.80, "Coercion ('do what I say or...')",
     r"\bdo (what|as) i (say|tell).{0,40}\b(or|else|otherwise)\b"),
    ("Blackmail", 0.90, "Blackmail threat in Hindi/Hinglish",
     r"\b(viral kar|leak kar|sabko bata dunga|sabko dikha dunga|photo viral)\b"),
    # ---- Stalking
    ("Stalking", 0.90, "Claims to know the person's location",
     r"\bi know where you (live|work|study|stay)\b"),
    ("Stalking", 0.90, "States they are following / watching the person",
     r"\bi('| a)?m (following|watching|tracking) you\b"),
    ("Stalking", 0.80, "States they are physically near the person",
     r"\bi('| a)?m (outside|waiting outside|standing outside|near)\b.{0,25}\b(house|home|hostel|office|college|school|work|door|flat|room|you)\b"),
    ("Stalking", 0.85, "Says the person cannot escape",
     r"\byou can'?t (hide|escape|run)\b|\bwherever you go\b|\bevery(where)? you go\b"),
    ("Stalking", 0.85, "Intends to track / find the person",
     r"\b(i('| wi)ll|gonna)\s+(find|follow|track|come for)\s+you\b"),
    ("Stalking", 0.60, "Reports surveilling the person's movements",
     r"\bi saw you (at|in|near|today|yesterday|with|going|coming)\b"),
    ("Stalking", 0.50, "Demands location / whereabouts",
     r"\b(where are you|where r u|who are you with|who were you with|send (me )?(your )?(live )?location)\b"),
    ("Stalking", 0.50, "Pressures for reply after being ignored / blocked",
     r"\b(why (are you|r u|did you|didn'?t you) (ignoring|ignore|block|reply|answer|respond)|unblock me|reply (now|me)|pick up (the )?(phone|call))\b"),
]
RULES = [(c, w, reason, re.compile(rx, re.IGNORECASE)) for c, w, reason, rx in _RULES_RAW]

# --------------------------------------------------------------------------
# 2. Chat parsing
# --------------------------------------------------------------------------
_LINE_RE = re.compile(
    r"^\[?(?P<date>\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?\s?(?:[APap]\.?\s?[Mm]\.?)?)\]?\s*[-\u2013\u2014]?\s*(?P<rest>.*)$"
)
_SENDER_RE = re.compile(r"^(?P<sender>[^:]{1,60}?):\s(?P<msg>.*)$", re.DOTALL)
_PLAIN_RE = re.compile(r"^(?P<sender>[^:\n]{1,40}?):\s+(?P<msg>.+)$")
_SKIP_MESSAGES = {"<media omitted>", "image omitted", "video omitted", "sticker omitted",
                  "audio omitted", "gif omitted", "document omitted"}


def decode_bytes(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _clean_chars(s: str) -> str:
    for ch in ("\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\ufeff"):
        s = s.replace(ch, "")
    return s.replace("\u202f", " ").replace("\u00a0", " ")


def parse_chat(raw: str) -> pd.DataFrame:
    """Parse a WhatsApp export (or plain 'Name: message' text) into a DataFrame."""
    rows, cur = [], None
    for line in _clean_chars(raw).splitlines():
        m = _LINE_RE.match(line)
        if m:
            sm = _SENDER_RE.match(m.group("rest"))
            if sm:
                cur = {"date": m.group("date"), "time": re.sub(r"[.\s]", "", m.group("time")).upper(),
                       "sender": sm.group("sender").strip(), "message": sm.group("msg")}
                rows.append(cur)
            else:
                cur = None  # system line such as "Messages are end-to-end encrypted"
        elif cur is not None and line.strip():
            cur["message"] += "\n" + line
        elif cur is not None and not line.strip():
            continue

    if not rows:  # fallback: plain pasted text, one message per line
        for line in raw.splitlines():
            line = _clean_chars(line).strip()
            if not line:
                continue
            pm = _PLAIN_RE.match(line)
            rows.append({"date": None, "time": None,
                         "sender": pm.group("sender").strip() if pm else "Unknown",
                         "message": pm.group("msg") if pm else line})

    df = pd.DataFrame(rows, columns=["date", "time", "sender", "message"])
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "sender", "message"])

    df["message"] = df["message"].astype(str).str.strip()
    df = df[~df["message"].str.lower().isin(_SKIP_MESSAGES)]
    df = df[df["message"] != ""].reset_index(drop=True)

    stamp = (df["date"].fillna("") + " " + df["time"].fillna("")).str.strip()
    try:
        ts = pd.to_datetime(stamp.replace("", np.nan), dayfirst=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        ts = pd.to_datetime(stamp.replace("", np.nan), dayfirst=True, errors="coerce")
    df["timestamp"] = ts
    return df[["timestamp", "sender", "message"]]


# --------------------------------------------------------------------------
# 3. Analysis
# --------------------------------------------------------------------------
def label_to_category(label: str) -> str:
    l = str(label).lower()
    if "sex" in l:
        return "Sexual Harassment"
    if "black" in l or "extort" in l:
        return "Blackmail"
    if "stalk" in l:
        return "Stalking"
    if "threat" in l or "violen" in l:
        return "Threat"
    if "abus" in l or "insult" in l or "toxic" in l or "hate" in l or "harass" in l or "bully" in l:
        return "Abuse"
    return "Other Harmful"


def match_rules(text: str):
    return [(c, w, reason) for c, w, reason, rx in RULES if rx.search(text)]


def detect_bursts(df: pd.DataFrame, min_run=6, max_minutes=30) -> dict:
    """Runs of >= min_run consecutive messages from one sender (stalking pattern)."""
    out = {}
    if df["sender"].nunique() < 2:
        return out
    senders, ts = df["sender"].tolist(), df["timestamp"].tolist()
    i, n = 0, len(df)
    while i < n:
        j = i
        while j + 1 < n and senders[j + 1] == senders[i]:
            j += 1
        run = j - i + 1
        if run >= min_run:
            ok = True
            if pd.notna(ts[i]) and pd.notna(ts[j]):
                ok = (ts[j] - ts[i]).total_seconds() / 60 <= max_minutes
            if ok:
                for k in range(i, j + 1):
                    out[k] = run
        i = j + 1
    return out


def severity_from_score(score: float, flagged: bool) -> str:
    if not flagged:
        return "None"
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def analyze(df: pd.DataFrame, model, threshold: float) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    n = len(df)
    texts = df["message"].tolist()

    # ML probabilities (optional - the rule engine works without a model)
    ml_prob = np.zeros(n)
    ml_cat = [""] * n
    ml_label = [""] * n
    if model is not None and n:
        try:
            proba = model.predict_proba(texts)
            classes = list(model.classes_)
            harmful_idx = [i for i, c in enumerate(classes) if str(c).strip().lower() not in SAFE_LABELS]
            ml_prob = proba[:, harmful_idx].sum(axis=1) if harmful_idx else np.zeros(n)
            for r in range(n):
                harm_only = [(proba[r, i], classes[i]) for i in harmful_idx]
                if harm_only:
                    best = max(harm_only)[1]
                    ml_label[r], ml_cat[r] = str(best), label_to_category(best)
        except Exception:
            ml_prob = np.zeros(n)

    bursts = detect_bursts(df)
    out = []
    for i, text in enumerate(texts):
        matches = match_rules(text)
        if i in bursts:
            matches.append(("Stalking", 0.35, f"Part of {bursts[i]} consecutive messages sent without a reply"))

        keep = 1.0
        for _, w, _ in matches:
            keep *= (1 - w)
        rule_score = 100 * (1 - keep)
        letters = [c for c in text if c.isalpha()]
        if matches and len(letters) > 8 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
            rule_score = min(100, rule_score + 5)
            matches.append((matches[0][0], 0, "Message written in aggressive ALL CAPS"))

        p = float(ml_prob[i])
        if matches:
            score = max(rule_score, 0.75 * rule_score + 25 * p)
        else:
            score = p * 70 if p >= 0.5 else p * 20
        ts = df.at[i, "timestamp"]
        if pd.notna(ts) and ts.hour < 5 and score >= threshold:
            score = min(100, score + 5)
            matches.append((matches[0][0] if matches else "Other Harmful", 0, "Sent late at night (00:00-05:00)"))
        score = float(min(100, score))

        flagged = score >= threshold
        best_per_cat = {}
        for c, w, _ in matches:
            best_per_cat[c] = max(best_per_cat.get(c, 0), w)
        reasons = []
        for _, _, r in matches:
            if r not in reasons:
                reasons.append(r)

        if best_per_cat:
            real = {c: w for c, w in best_per_cat.items() if w > 0}
            primary = max(real, key=real.get) if real else list(best_per_cat)[0]
        elif flagged:
            primary = ml_cat[i] or "Other Harmful"
        else:
            primary = "Safe"
        if flagged and not matches and p >= 0.5:
            reasons.append(f"ML model classed it as '{ml_label[i]}' ({p*100:.0f}% harmful probability)")
        elif flagged and p >= 0.5:
            reasons.append(f"ML model also rates it {p*100:.0f}% harmful")

        out.append({
            "category": primary if flagged else "Safe",
            "all_categories": ", ".join(sorted(best_per_cat)) if flagged else "",
            "rule_score": round(rule_score, 1),
            "ml_prob": round(p, 3),
            "score": round(score, 1),
            "flagged": bool(flagged),
            "severity": severity_from_score(score, flagged),
            "reasons": "; ".join(reasons) if flagged else "",
        })
    res = pd.concat([df, pd.DataFrame(out)], axis=1)
    res.insert(0, "id", np.arange(1, n + 1))
    res["hour"] = res["timestamp"].dt.hour
    return res


def overall_risk(res: pd.DataFrame) -> float:
    fl = res[res["flagged"]]
    if fl.empty:
        return 0.0
    return float(min(100, 0.6 * fl["score"].max() + 0.25 * fl["score"].mean() + min(15, len(fl) * 1.5)))


def risk_label(score: float) -> str:
    if score >= 80:
        return "CRITICAL"
    if score >= 60:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    if score > 0:
        return "LOW"
    return "NO RISK FOUND"


# --------------------------------------------------------------------------
# 4. Privacy helpers
# --------------------------------------------------------------------------
_PHONE_RE = re.compile(r"(?<!\w)(\+?\d[\d\s\-()]{8,}\d)(?!\w)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def apply_privacy(df: pd.DataFrame, mask_contacts: bool, pseudonymize: bool) -> pd.DataFrame:
    out = df.copy()
    if mask_contacts:
        out["message"] = out["message"].map(lambda s: _EMAIL_RE.sub("[EMAIL]", _PHONE_RE.sub("[PHONE]", s)))
        out["sender"] = out["sender"].map(lambda s: _PHONE_RE.sub("[PHONE]", s))
    if pseudonymize:
        names = sorted(out["sender"].unique())
        mapping = {nm: f"Person {chr(65 + i) if i < 26 else i + 1}" for i, nm in enumerate(names)}
        out["sender"] = out["sender"].map(mapping)
    return out


# --------------------------------------------------------------------------
# 5. Legal PDF report
# --------------------------------------------------------------------------
def pdf_safe(s) -> str:
    """ReportLab's built-in fonts only support Latin-1, so strip everything else."""
    s = "" if s is None else str(s)
    s = s.encode("latin-1", "replace").decode("latin-1")
    return html.escape(s).replace("\n", "<br/>")


LEGAL_HINTS_IN = {
    "Threat": "BNS s.351 (criminal intimidation)",
    "Abuse": "BNS s.79 / s.351 (insult / intimidation), depending on content",
    "Sexual Harassment": "BNS s.75 (sexual harassment); IT Act s.67 / 67A if explicit material was sent",
    "Blackmail": "BNS s.308 (extortion); IT Act s.66E / 67 if intimate images are involved",
    "Stalking": "BNS s.78 (stalking, including online monitoring / repeated contact)",
}


def build_pdf(res: pd.DataFrame, flagged: pd.DataFrame, accused: str, complainant: str,
              case_title: str, contact: str, notes: str, jurisdiction: str,
              source_label: str, raw_hash: str, overall: float, privacy_note: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    PINK, DARK, SOFT = colors.HexColor("#d6006f"), colors.HexColor("#1a0a12"), colors.HexColor("#fde7f0")
    ss = getSampleStyleSheet()
    h_title = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20,
                             textColor=PINK, spaceAfter=4, alignment=0)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5,
                        textColor=PINK, spaceBefore=12, spaceAfter=5)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.5)
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor("#555555"))
    cell = ParagraphStyle("c", parent=body, fontSize=7.6, leading=9.6)
    cellb = ParagraphStyle("cb", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)

    def P(t, st=body):
        return Paragraph(pdf_safe(t), st)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(15 * mm, 9 * mm, "ShieldDM - Confidential - automated analysis, requires human verification")
        canvas.drawRightString(195 * mm, 9 * mm, f"Page {doc.page}")
        canvas.setStrokeColor(PINK)
        canvas.setLineWidth(1.2)
        canvas.line(15 * mm, 13 * mm, 195 * mm, 13 * mm)
        canvas.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=20 * mm,
                            title="Legal Complaint Report - ShieldDM", author="ShieldDM")
    story = []

    # ---- header
    story += [P("LEGAL COMPLAINT REPORT", h_title),
              P(case_title or "Digital harassment complaint", ParagraphStyle("sub", parent=body, fontSize=11, textColor=DARK)),
              Spacer(1, 4)]
    stamp = dt.datetime.now().strftime("%d %b %Y, %H:%M")

    # ---- incident details
    flagged_acc = flagged
    ts_valid = flagged_acc["timestamp"].dropna()
    period = (f"{ts_valid.min():%d %b %Y %H:%M} to {ts_valid.max():%d %b %Y %H:%M}"
              if not ts_valid.empty else "Not available (chat contains no timestamps)")
    cat_counts = flagged_acc["category"].value_counts()
    top_sev = next((s for s in SEVERITY_ORDER if (flagged_acc["severity"] == s).any()), "None")

    details = [
        ["Report generated", stamp],
        ["Complainant", complainant or "Not provided"],
        ["Complainant contact", contact or "Not provided"],
        ["Accused / sender", accused],
        ["Evidence source", source_label],
        ["Incident period", period],
        ["Messages analysed", str(len(res))],
        ["Flagged messages (accused)", str(len(flagged_acc))],
        ["Highest severity", top_sev],
        ["Overall risk score", f"{overall:.0f} / 100 ({risk_label(overall).title()})"],
        ["Evidence SHA-256", raw_hash],
        ["Privacy settings", privacy_note],
    ]
    t = Table([[P(a, cellb if False else ParagraphStyle('k', parent=cell, fontName='Helvetica-Bold')), P(b, cell)] for a, b in details],
              colWidths=[45 * mm, 135 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), SOFT),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5b7cb")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story += [P("1. Incident Details", h2), t]

    # ---- evidence summary
    story.append(P("2. Evidence Summary", h2))
    if flagged_acc.empty:
        story.append(P(f"No messages from {accused} were flagged at the selected sensitivity."))
    else:
        parts = [f"Automated analysis of the supplied chat flagged {len(flagged_acc)} message(s) from {accused}, "
                 f"out of {len(res)} messages in total."]
        parts.append("Category breakdown: " + ", ".join(f"{k} ({v})" for k, v in cat_counts.items()) + ".")
        sev_counts = flagged_acc["severity"].value_counts()
        parts.append("Severity breakdown: " + ", ".join(f"{s} ({sev_counts[s]})" for s in SEVERITY_ORDER if s in sev_counts) + ".")
        worst = flagged_acc.sort_values("score", ascending=False).iloc[0]
        wt = f"{worst['timestamp']:%d %b %Y %H:%M}" if pd.notna(worst["timestamp"]) else "unknown time"
        parts.append(f"The highest-scoring message (risk {worst['score']:.0f}/100, {worst['category']}) was sent at {wt}.")
        story.append(P(" ".join(parts)))
    if notes.strip():
        story += [Spacer(1, 4), P("Complainant's statement / additional notes:", ParagraphStyle('nb', parent=body, fontName='Helvetica-Bold')), P(notes)]

    # ---- flagged messages
    story.append(P("3. Flagged Messages", h2))
    if flagged_acc.empty:
        story.append(P("None."))
    else:
        data = [[Paragraph(h, cellb) for h in ("#", "Date / time", "Sender", "Category / severity", "Message", "Why flagged")]]
        for k, (_, r) in enumerate(flagged_acc.sort_values(["timestamp", "id"]).iterrows(), 1):
            when = f"{r['timestamp']:%d-%m-%Y %H:%M}" if pd.notna(r["timestamp"]) else "n/a"
            msg = r["message"] if len(r["message"]) <= 500 else r["message"][:500] + " [...]"
            data.append([P(str(k), cell), P(when, cell), P(r["sender"], cell),
                         P(f"{r['category']} / {r['severity']} ({r['score']:.0f})", cell),
                         P(msg, cell), P(r["reasons"], cell)])
        mt = Table(data, colWidths=[8 * mm, 23 * mm, 22 * mm, 27 * mm, 56 * mm, 44 * mm], repeatRows=1)
        mt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PINK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fff4f8")]),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#e5b7cb")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(mt)

    # ---- legal pointers + next steps
    story.append(P("4. Suggested Next Steps", h2))
    steps = [
        "Preserve the evidence: do not delete the chat. Keep the original exported .txt file (its SHA-256 fingerprint is listed above) and take screenshots showing the sender's number/name and timestamps.",
        "Back up the chat on a second device or cloud storage. Do not edit the exported file; any change alters its fingerprint.",
        "Avoid replying, arguing or deleting messages. You may block the sender after the evidence is saved.",
        "Report the account to the messaging platform (in WhatsApp: open chat > Report).",
        "File a written complaint with the local police station or cyber-crime cell and ask for an acknowledgement / FIR number. Attach this report and the original chat export.",
        "Consult a lawyer about a protection / restraining order and the sections that apply to your situation.",
        "Tell a trusted person and consider counselling or a support organisation. You do not have to handle this alone.",
    ]
    if jurisdiction == "India":
        steps.insert(5, "India: you can also file online at the National Cyber Crime Reporting Portal (cybercrime.gov.in); "
                        "emergency number 112, Women Helpline 181, Cyber Crime Helpline 1930.")
    for i, s in enumerate(steps, 1):
        story.append(P(f"{i}. {s}"))

    if jurisdiction == "India" and not flagged_acc.empty:
        story.append(P("Provisions that may be relevant (indicative only - confirm with a lawyer)", ParagraphStyle('lb', parent=body, fontName='Helvetica-Bold', spaceBefore=6)))
        for c in cat_counts.index:
            if c in LEGAL_HINTS_IN:
                story.append(P(f"- {c}: {LEGAL_HINTS_IN[c]}"))

    story.append(Spacer(1, 10))
    story.append(KeepTogether([
        P("Disclaimer", ParagraphStyle('d', parent=body, fontName='Helvetica-Bold', fontSize=8.5)),
        P("This report was generated automatically by ShieldDM using keyword rules and a machine-learning model. "
          "It can contain false positives and false negatives and is not legal advice. The complete original chat must be "
          "reviewed by the complainant, police or a qualified lawyer before any legal action.", small),
        Spacer(1, 14),
        P("Complainant signature: ______________________      Date: ________________", body),
    ]))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


# --------------------------------------------------------------------------
# 6. Styling
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Poppins', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #3a0a24 0%, transparent 60%),
                     radial-gradient(900px 500px at 100% 0%, #3d0a1f 0%, transparent 55%), #0b0508; color:#ffe9f1; }
#MainMenu, footer { visibility:hidden; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#16060d 0%,#220a15 100%); border-right:1px solid #ff2ea633; }
section[data-testid="stSidebar"] * { color:#ffe9f1; }
h1,h2,h3,h4 { color:#fff; letter-spacing:.2px; }
@keyframes fadeUp { from {opacity:0; transform:translateY(18px);} to {opacity:1; transform:translateY(0);} }
@keyframes pulseGlow { 0%,100% { box-shadow:0 0 0 0 #ff2ea655;} 50% { box-shadow:0 0 22px 4px #ff2ea655;} }
@keyframes gradientMove { 0%{background-position:0% 50%;} 50%{background-position:100% 50%;} 100%{background-position:0% 50%;} }
@keyframes floaty { 0%,100%{transform:translateY(0);} 50%{transform:translateY(-6px);} }
.hero { background: linear-gradient(120deg,#ff2d55,#ff2ea6,#c2185b,#ff8a65); background-size:300% 300%;
        animation: gradientMove 9s ease infinite, fadeUp .7s ease both; border-radius:22px; padding:28px 34px; margin-bottom:22px;
        box-shadow:0 14px 40px #ff2ea633; }
.hero h1 { margin:0; font-size:2.2rem; font-weight:800; color:#fff; }
.hero p { margin:6px 0 0; color:#fff; opacity:.92; font-size:1.02rem; }
.hero .shield { display:inline-block; animation: floaty 3s ease-in-out infinite; }
.sd-card { background: linear-gradient(160deg,#1c0b13,#14070d); border:1px solid #ff2ea633; border-radius:18px;
           padding:18px 20px; animation: fadeUp .6s ease both; transition: transform .25s, border-color .25s, box-shadow .25s; }
.sd-card:hover { transform: translateY(-4px); border-color:#ff2ea6; box-shadow:0 10px 30px #ff2ea622; }
.metric .lbl { font-size:.78rem; text-transform:uppercase; letter-spacing:1.2px; color:#ff9fc2; }
.metric .val { font-size:2.1rem; font-weight:800; color:#fff; line-height:1.15; }
.metric .sub { font-size:.8rem; color:#d9a6ba; }
.metric .ico { float:right; font-size:1.7rem; opacity:.9; }
.flag-card { background:#170a10; border:1px solid #ff2ea633; border-left:6px solid #ff2ea6; border-radius:14px;
             padding:14px 16px; margin-bottom:12px; animation: fadeUp .5s ease both; }
.flag-card.sev-Critical { border-left-color:#ff1744; animation: fadeUp .5s ease both, pulseGlow 2.6s infinite; }
.flag-card.sev-High { border-left-color:#ff5c8a; }
.flag-card.sev-Medium { border-left-color:#ff9e80; }
.flag-card.sev-Low { border-left-color:#ffc2d6; }
.flag-top { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.badge { padding:3px 11px; border-radius:999px; font-size:.72rem; font-weight:600; color:#fff; }
.badge.sev-Critical { background:#ff1744; } .badge.sev-High { background:#ff5c8a; }
.badge.sev-Medium { background:#e8704f; } .badge.sev-Low { background:#b8567a; }
.badge.cat { background:#2a0f1c; border:1px solid #ff2ea688; color:#ffb3d1; }
.flag-score { margin-left:auto; font-weight:700; color:#ff8fb8; }
.flag-meta { font-size:.78rem; color:#c796aa; margin:6px 0; }
.flag-msg { font-size:.97rem; color:#fff; background:#0e0509; padding:10px 12px; border-radius:10px; margin:6px 0; }
.flag-reason { font-size:.82rem; color:#ffc2d6; }
.riskbar { height:10px; background:#2a0f1c; border-radius:99px; overflow:hidden; }
.riskbar > div { height:100%; background:linear-gradient(90deg,#ff8a65,#ff2d55,#ff2ea6); border-radius:99px; }
.stButton > button, .stDownloadButton > button { background:linear-gradient(120deg,#ff2d55,#ff2ea6); color:#fff; border:0;
        border-radius:12px; font-weight:600; padding:.55rem 1.1rem; transition: transform .2s, box-shadow .2s; }
.stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-2px) scale(1.01); box-shadow:0 8px 22px #ff2ea64d; color:#fff; border:0; }
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { background:#1c0b13; border-radius:12px 12px 0 0; padding:10px 18px; color:#ffb3d1; }
.stTabs [aria-selected="true"] { background:linear-gradient(120deg,#ff2d5533,#ff2ea633); color:#fff !important; border-bottom:3px solid #ff2ea6; }
div[data-baseweb="input"] input, textarea, div[data-baseweb="select"] > div { background:#14070d !important; color:#fff !important; }
.step { display:flex; gap:12px; align-items:flex-start; }
.step .n { background:linear-gradient(120deg,#ff2d55,#ff2ea6); border-radius:50%; width:34px; height:34px; min-width:34px;
           display:flex; align-items:center; justify-content:center; font-weight:700; }
.banner { background:#2a0f1c; border:1px dashed #ff2ea6; border-radius:12px; padding:10px 14px; margin-bottom:12px; color:#ffc2d6; }
</style>
"""


def metric_card(icon, label, value, sub=""):
    return (f'<div class="sd-card metric"><span class="ico">{icon}</span><div class="lbl">{label}</div>'
            f'<div class="val">{value}</div><div class="sub">{sub}</div></div>')


def style_fig(fig, height=330):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#ffe9f1", family="Poppins"), margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#3a1626", zerolinecolor="#3a1626")
    fig.update_yaxes(gridcolor="#3a1626", zerolinecolor="#3a1626")
    return fig


def stretch(fn, *args, **kwargs):
    """Works on both old and new Streamlit versions (width='stretch' vs use_container_width)."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def ts_str(ts):
    return f"{ts:%d %b %Y, %H:%M}" if pd.notna(ts) else "no timestamp"


def render_flag_cards(df, limit=25):
    parts = []
    for _, r in df.head(limit).iterrows():
        parts.append(
            f'<div class="flag-card sev-{r["severity"]}"><div class="flag-top">'
            f'<span class="badge cat">{html.escape(r["category"])}</span>'
            f'<span class="badge sev-{r["severity"]}">{r["severity"]}</span>'
            f'<span class="flag-score">Risk {r["score"]:.0f}/100</span></div>'
            f'<div class="flag-meta">#{r["id"]} &bull; {html.escape(str(r["sender"]))} &bull; {ts_str(r["timestamp"])}</div>'
            f'<div class="flag-msg">{html.escape(r["message"]).replace(chr(10), "<br>")}</div>'
            f'<div class="flag-reason">&#9888; {html.escape(r["reasons"])}</div></div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 7. Model loading
# --------------------------------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def load_metrics():
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------
# 8. App
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="ShieldDM", page_icon="🛡️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    ss = st.session_state
    for k, v in (("raw", ""), ("source", ""), ("upload_sig", None), ("is_demo", False)):
        ss.setdefault(k, v)

    model = st.cache_resource(load_model)()
    metrics = load_metrics()

    # ------------------------------------------------------------ sidebar
    with st.sidebar:
        st.markdown("## 🛡️ ShieldDM")
        st.caption("Detect harassment. Build evidence. Stay protected.")
        mode = st.radio("Input method", ["📋 Paste chat text", "📁 Upload WhatsApp .txt"])
        if mode.startswith("📋"):
            txt = st.text_area("Paste the chat here", height=230, key="paste_box",
                               placeholder="12/03/2024, 22:14 - Rahul: where are you?\n12/03/2024, 22:15 - Rahul: reply now")
            if stretch(st.button, "🔍 Analyze pasted text", type="primary"):
                if txt.strip():
                    ss.raw, ss.source, ss.is_demo = txt, "Pasted text", False
                else:
                    st.warning("Paste some chat text first.")
        else:
            up = st.file_uploader("WhatsApp export (.txt)", type=["txt"])
            if up is not None:
                sig = (up.name, up.size)
                if ss.upload_sig != sig:
                    ss.raw, ss.source, ss.is_demo = decode_bytes(up.getvalue()), f"Uploaded file: {up.name}", False
                    ss.upload_sig = sig

        st.divider()
        if stretch(st.button, "🎬 Load demo data"):
            if os.path.exists(SAMPLE_PATH):
                with open(SAMPLE_PATH, "rb") as fh:
                    ss.raw, ss.source, ss.is_demo = decode_bytes(fh.read()), "Demo data (sample_chat.txt)", True
            else:
                st.error("sample_chat.txt not found next to app.py.")

        st.divider()
        st.markdown("**⚙️ Detection**")
        threshold = st.slider("Flag sensitivity (lower = flags more)", 10, 80, 35,
                              help="Messages scoring at or above this value are flagged.")
        st.markdown("**🔒 Privacy**")
        mask_contacts = st.toggle("Mask phone numbers & emails", value=True)
        pseudonymize = st.toggle("Replace sender names with Person A/B", value=False,
                                 help="Also applies to exports and the PDF. Turn off to put real names in the legal report.")
        st.caption("Your chat is processed in memory only. Nothing is written to disk or sent anywhere.")
        if stretch(st.button, "🗑️ Clear all data"):
            for k in ("raw", "source", "upload_sig", "pdf", "pdf_sig"):
                ss.pop(k, None)
            st.rerun()

    # ------------------------------------------------------------ hero
    st.markdown('<div class="hero"><h1><span class="shield">🛡️</span> ShieldDM</h1>'
                '<p>AI-powered detection of threats, abuse, sexual harassment, blackmail and stalking in chats '
                '&mdash; with court-ready evidence reports.</p></div>', unsafe_allow_html=True)

    if not ss.get("raw", "").strip():
        c = st.columns(3)
        info = [("1", "Add your chat", "Paste text or upload a WhatsApp export (.txt) from the sidebar."),
                ("2", "Instant analysis", "Every message is scored for threats, abuse, harassment, blackmail and stalking."),
                ("3", "Generate evidence", "Download a legal complaint PDF built from your real chat.")]
        for col, (n, t, d) in zip(c, info):
            col.markdown(f'<div class="sd-card step"><div class="n">{n}</div><div><b>{t}</b><br>'
                         f'<span style="color:#d9a6ba;font-size:.88rem">{d}</span></div></div>', unsafe_allow_html=True)
        st.info("Nothing is analysed until you add a chat. Use **Load demo data** in the sidebar if you just want to explore.")
        if model is None:
            st.warning("No trained model found. Run `python train.py` once to enable ML scoring. "
                       "The rule-based detector still works without it.")
        return

    # ------------------------------------------------------------ analysis
    raw = ss.raw
    parsed = parse_chat(raw)
    if parsed.empty:
        st.error("I couldn't find any messages in that text. Check the format and try again.")
        return
    res_full = analyze(parsed, model, threshold)
    res = apply_privacy(res_full, mask_contacts, pseudonymize)
    raw_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    flagged_all = res[res["flagged"]].copy()
    overall = overall_risk(res)
    has_time = res["timestamp"].notna().any()

    if ss.is_demo:
        st.markdown('<div class="banner">🎬 You are viewing <b>demo data</b>. Paste or upload your own chat to analyse real messages.</div>',
                    unsafe_allow_html=True)
    if model is None:
        st.warning("ML model not found - running with the rule-based detector only. Run `python train.py` to enable ML scoring.")

    tabs = st.tabs(["📊 Dashboard", "🚩 Flagged Messages", "🕒 Timeline", "💬 Full Chat", "🤖 Model Metrics", "⚖️ Legal Report"])

    # ------------------------------------------------------------ dashboard
    with tabs[0]:
        top_cat = flagged_all["category"].value_counts().idxmax() if not flagged_all.empty else "None"
        crit = int(flagged_all["severity"].isin(["Critical", "High"]).sum())
        cols = st.columns(4)
        cols[0].markdown(metric_card("💬", "Messages analysed", len(res), f"{res['sender'].nunique()} participant(s)"), unsafe_allow_html=True)
        cols[1].markdown(metric_card("🚩", "Flagged", len(flagged_all),
                                     f"{(len(flagged_all) / len(res) * 100):.0f}% of the chat"), unsafe_allow_html=True)
        cols[2].markdown(metric_card("🔥", "Critical / High", crit, "needs attention"), unsafe_allow_html=True)
        cols[3].markdown(metric_card("🎯", "Top category", top_cat, f"source: {ss.source}"), unsafe_allow_html=True)
        st.write("")

        g1, g2 = st.columns([1, 1])
        with g1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=overall, number={"suffix": " / 100", "font": {"size": 34}},
                title={"text": f"Overall risk: {risk_label(overall)}"},
                gauge={"axis": {"range": [0, 100], "tickcolor": "#ffb3d1"}, "bar": {"color": "#ff2ea6"},
                       "bgcolor": "#1c0b13", "borderwidth": 0,
                       "steps": [{"range": [0, 40], "color": "#2a0f1c"}, {"range": [40, 60], "color": "#4a1230"},
                                 {"range": [60, 80], "color": "#7a1240"}, {"range": [80, 100], "color": "#b0003a"}]}))
            stretch(st.plotly_chart, style_fig(fig, 320))
        with g2:
            if flagged_all.empty:
                st.success("No harmful messages were flagged at the current sensitivity.")
            else:
                cc = flagged_all["category"].value_counts().reset_index()
                cc.columns = ["Category", "Count"]
                fig = px.pie(cc, names="Category", values="Count", hole=0.58, color="Category",
                             color_discrete_map=CATEGORY_COLORS, title="Flagged messages by category")
                fig.update_traces(textinfo="value+percent", marker=dict(line=dict(color="#0b0508", width=2)))
                stretch(st.plotly_chart, style_fig(fig, 320))

        if not flagged_all.empty:
            g3, g4 = st.columns(2)
            with g3:
                sv = flagged_all["severity"].value_counts().reindex(SEVERITY_ORDER).fillna(0).reset_index()
                sv.columns = ["Severity", "Count"]
                fig = px.bar(sv, x="Severity", y="Count", color="Severity", color_discrete_map=SEVERITY_COLORS,
                             title="Severity levels", text="Count")
                fig.update_layout(showlegend=False)
                stretch(st.plotly_chart, style_fig(fig))
            with g4:
                tot = res.groupby("sender").size().rename("Total")
                fl = flagged_all.groupby("sender").size().rename("Flagged")
                sd = pd.concat([tot, fl], axis=1).fillna(0).reset_index().rename(columns={"index": "sender"})
                sd = sd.melt(id_vars="sender", var_name="Type", value_name="Messages")
                fig = px.bar(sd, x="sender", y="Messages", color="Type", barmode="group", title="Messages per sender",
                             color_discrete_map={"Total": "#5b2a40", "Flagged": "#ff2ea6"})
                fig.update_xaxes(title="")
                stretch(st.plotly_chart, style_fig(fig))

            ra = flagged_all.groupby("sender")["score"].agg(["count", "mean", "max"]).reset_index()
            ra.columns = ["Sender", "Flagged", "Avg risk", "Max risk"]
            st.markdown("#### 👤 Sender risk profile")
            for _, r in ra.sort_values("Max risk", ascending=False).iterrows():
                st.markdown(
                    f'<div class="sd-card" style="margin-bottom:10px"><b>{html.escape(str(r["Sender"]))}</b> '
                    f'<span style="color:#d9a6ba">&mdash; {int(r["Flagged"])} flagged &bull; avg {r["Avg risk"]:.0f} &bull; max {r["Max risk"]:.0f}</span>'
                    f'<div class="riskbar" style="margin-top:8px"><div style="width:{r["Max risk"]:.0f}%"></div></div></div>',
                    unsafe_allow_html=True)

            if has_time:
                hh = flagged_all.dropna(subset=["hour"]).groupby("hour").size().reindex(range(24), fill_value=0).reset_index()
                hh.columns = ["Hour", "Flagged"]
                fig = px.bar(hh, x="Hour", y="Flagged", title="When flagged messages are sent (hour of day)",
                             color="Flagged", color_continuous_scale=["#4a1230", "#ff2ea6", "#ff2d55"])
                fig.update_coloraxes(showscale=False)
                stretch(st.plotly_chart, style_fig(fig, 280))

    # ------------------------------------------------------------ flagged
    with tabs[1]:
        if flagged_all.empty:
            st.success("Nothing flagged. Try lowering the sensitivity slider in the sidebar.")
        else:
            f1, f2, f3 = st.columns(3)
            cats = f1.multiselect("Category", sorted(flagged_all["category"].unique()), default=sorted(flagged_all["category"].unique()))
            sevs = f2.multiselect("Severity", [s for s in SEVERITY_ORDER if s in set(flagged_all["severity"])],
                                  default=[s for s in SEVERITY_ORDER if s in set(flagged_all["severity"])])
            senders = f3.multiselect("Sender", sorted(flagged_all["sender"].unique()), default=sorted(flagged_all["sender"].unique()))
            f4, f5 = st.columns([1, 2])
            min_score = f4.slider("Minimum risk score", 0, 100, int(threshold))
            query = f5.text_input("🔎 Search in messages / reasons")
            view = flagged_all[flagged_all["category"].isin(cats) & flagged_all["severity"].isin(sevs)
                               & flagged_all["sender"].isin(senders) & (flagged_all["score"] >= min_score)]
            if has_time:
                vt = view["timestamp"].dropna()
                if not vt.empty and vt.min().date() != vt.max().date():
                    d1, d2 = st.slider("Date range", vt.min().date(), vt.max().date(), (vt.min().date(), vt.max().date()))
                    view = view[view["timestamp"].isna() | ((view["timestamp"].dt.date >= d1) & (view["timestamp"].dt.date <= d2))]
            if query.strip():
                q = query.strip().lower()
                view = view[view["message"].str.lower().str.contains(q, regex=False) | view["reasons"].str.lower().str.contains(q, regex=False)]
            view = view.sort_values("score", ascending=False)

            st.markdown(f"**{len(view)}** of {len(flagged_all)} flagged messages shown")
            export = view[["id", "timestamp", "sender", "message", "category", "all_categories", "severity", "score", "reasons"]]
            stretch(st.download_button, "⬇️ Export flagged messages (CSV)", export.to_csv(index=False).encode("utf-8-sig"),
                    file_name="shielddm_flagged_messages.csv", mime="text/csv")
            render_flag_cards(view, limit=25)
            if len(view) > 25:
                st.caption(f"Showing the 25 highest-risk cards. All {len(view)} rows are in the table below and the CSV.")
            with st.expander("📑 Table view"):
                stretch(st.dataframe, export, hide_index=True)

    # ------------------------------------------------------------ timeline
    with tabs[2]:
        if flagged_all.empty:
            st.info("No flagged messages to place on a timeline.")
        else:
            plot = flagged_all.copy()
            xcol = "timestamp" if has_time and plot["timestamp"].notna().any() else "id"
            if xcol == "timestamp":
                plot = plot.dropna(subset=["timestamp"])
            plot["Message"] = plot["message"].str.slice(0, 120)
            fig = px.scatter(plot, x=xcol, y="score", color="category", size="score", size_max=22,
                             color_discrete_map=CATEGORY_COLORS, hover_data={"sender": True, "severity": True, "Message": True, "score": ":.0f"},
                             title="Incident timeline (each dot is a flagged message)")
            fig.update_yaxes(range=[0, 105], title="Risk score")
            fig.update_xaxes(title="" if xcol == "timestamp" else "Message number")
            stretch(st.plotly_chart, style_fig(fig, 420))
            if xcol == "id":
                st.caption("This chat has no timestamps, so the timeline is ordered by message number.")
            st.markdown("#### Chronological log")
            log = flagged_all.sort_values(["timestamp", "id"])
            for _, r in log.iterrows():
                color = SEVERITY_COLORS.get(r["severity"], "#ff2ea6")
                st.markdown(
                    f'<div class="sd-card" style="margin-bottom:8px;border-left:5px solid {color}"><b>{ts_str(r["timestamp"])}</b> '
                    f'&bull; {html.escape(str(r["sender"]))} &bull; <span style="color:{color}">{r["category"]} / {r["severity"]}</span><br>'
                    f'<span style="color:#ffd9e6">{html.escape(r["message"][:220])}</span></div>', unsafe_allow_html=True)

    # ------------------------------------------------------------ full chat
    with tabs[3]:
        only_flagged = st.toggle("Show only flagged", value=False)
        show = res[res["flagged"]] if only_flagged else res
        cols_show = ["id", "timestamp", "sender", "message", "category", "severity", "score"]
        stretch(st.dataframe, show[cols_show], hide_index=True)
        stretch(st.download_button, "⬇️ Export full analysis (CSV)",
                res[["id", "timestamp", "sender", "message", "category", "all_categories", "severity", "score", "rule_score", "ml_prob", "reasons"]]
                .to_csv(index=False).encode("utf-8-sig"), file_name="shielddm_full_analysis.csv", mime="text/csv")

    # ------------------------------------------------------------ metrics
    with tabs[4]:
        if not metrics:
            st.warning("No metrics found. Run `python train.py` to train the model and generate metrics.json.")
        else:
            m = st.columns(4)
            m[0].markdown(metric_card("✅", "Accuracy", f"{metrics['accuracy']*100:.1f}%", "hold-out test set"), unsafe_allow_html=True)
            m[1].markdown(metric_card("🎯", "Precision", f"{metrics['precision']*100:.1f}%", "weighted"), unsafe_allow_html=True)
            m[2].markdown(metric_card("📡", "Recall", f"{metrics['recall']*100:.1f}%", "weighted"), unsafe_allow_html=True)
            m[3].markdown(metric_card("⚖️", "F1 score", f"{metrics['f1']*100:.1f}%", "weighted"), unsafe_allow_html=True)
            st.caption(f"Trained {metrics.get('trained_at', '')} on {metrics['n_rows']} rows "
                       f"({metrics['n_train']} train / {metrics['n_test']} test).")
            if metrics["n_test"] < 30:
                st.warning("The test set is very small, so these numbers are only indicative. Add more rows to train.csv for reliable metrics.")
            b = metrics.get("binary", {})
            if b:
                st.markdown(f"**Harmful vs safe:** accuracy {b['accuracy']*100:.1f}% &bull; precision {b['precision']*100:.1f}% "
                            f"&bull; recall {b['recall']*100:.1f}% &bull; F1 {b['f1']*100:.1f}%")
            if metrics.get("cv_f1_mean") is not None:
                st.markdown(f"**5-fold cross-validation F1:** {metrics['cv_f1_mean']*100:.1f}% +/- {metrics['cv_f1_std']*100:.1f}%")
            lc, rc = st.columns(2)
            with lc:
                cm = np.array(metrics["confusion_matrix"])
                fig = px.imshow(cm, x=metrics["labels"], y=metrics["labels"], text_auto=True, aspect="auto",
                                color_continuous_scale=["#1c0b13", "#c2185b", "#ff2d55"], title="Confusion matrix")
                fig.update_xaxes(title="Predicted")
                fig.update_yaxes(title="Actual")
                fig.update_coloraxes(showscale=False)
                stretch(st.plotly_chart, style_fig(fig, 380))
            with rc:
                cc = pd.DataFrame({"Label": list(metrics["class_counts"]), "Rows": list(metrics["class_counts"].values())})
                fig = px.bar(cc, x="Label", y="Rows", title="Training data distribution", color_discrete_sequence=["#ff2ea6"])
                stretch(st.plotly_chart, style_fig(fig, 380))
            rep = pd.DataFrame(metrics["report"]).T.drop(index=["accuracy"], errors="ignore")
            rep = rep[["precision", "recall", "f1-score", "support"]].round(3)
            stretch(st.dataframe, rep)
        st.caption("Detection combines the ML model with a transparent keyword/pattern rule engine. "
                   "Rule-based reasons are shown for every flagged message.")

    # ------------------------------------------------------------ legal report
    with tabs[5]:
        st.markdown("### ⚖️ Legal Complaint Report")
        st.caption("Builds a PDF from the chat you actually loaded: incident details, flagged messages, timestamps, evidence summary and next steps.")
        if pseudonymize:
            st.warning("Sender names are currently replaced by Person A/B. Turn that off in the sidebar to put real names in the report.")
        senders_all = sorted(res["sender"].unique())
        flagged_counts = flagged_all["sender"].value_counts()
        default_acc = flagged_counts.idxmax() if not flagged_counts.empty else senders_all[0]
        r1, r2 = st.columns(2)
        accused = r1.selectbox("Accused person", senders_all + ["All senders"],
                               index=senders_all.index(default_acc))
        others = [s for s in senders_all if s != accused]
        complainant = r2.text_input("Complainant (your name)", value=others[0] if len(others) == 1 else "")
        r3, r4 = st.columns(2)
        case_title = r3.text_input("Case title", value="Online harassment through messaging app")
        contact = r4.text_input("Your contact details (optional)")
        jurisdiction = st.selectbox("Jurisdiction for next-step guidance", ["India", "Other / generic"])
        notes = st.text_area("Your statement / additional context (optional)", height=110,
                             placeholder="Describe when the harassment started, how you know the person, and how it affected you.")

        acc_flagged = flagged_all if accused == "All senders" else flagged_all[flagged_all["sender"] == accused]
        st.markdown(f"This report will include **{len(acc_flagged)}** flagged message(s).")
        privacy_note = ("Phone numbers/emails masked" if mask_contacts else "No masking") + (
            "; sender names pseudonymised" if pseudonymize else "")
        sig = (raw_hash, accused, complainant, case_title, contact, jurisdiction, notes, threshold, mask_contacts, pseudonymize)

        if stretch(st.button, "⚖️ Generate Legal Complaint Report", type="primary"):
            try:
                with st.spinner("Building your PDF..."):
                    ss.pdf = build_pdf(res, acc_flagged, accused, complainant, case_title, contact, notes,
                                       jurisdiction, ss.source, raw_hash, overall, privacy_note)
                    ss.pdf_sig = sig
                st.success("Report ready. Download it below.")
            except ModuleNotFoundError:
                st.error("The PDF library is missing. In your terminal run:  pip install reportlab")
            except Exception as e:  # keep the app alive and show what went wrong
                st.error(f"Could not build the PDF: {e}")
        if ss.get("pdf") and ss.get("pdf_sig") == sig:
            stretch(st.download_button, "⬇️ Download PDF report", ss.pdf,
                    file_name=f"ShieldDM_Complaint_Report_{dt.datetime.now():%Y%m%d_%H%M}.pdf", mime="application/pdf")
        elif ss.get("pdf"):
            st.info("Your inputs changed. Press Generate again to refresh the PDF.")


if __name__ == "__main__":
    main()