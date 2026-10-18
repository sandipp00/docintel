"""Synthetic document generator for DocIntel.

ALL DATA IS SYNTHETIC (Faker + templates). No real personal documents are used.

Outputs
-------
* ``training/data/train.jsonl``       text + label + ground-truth fields (>=200 per class)
* ``training/data/rendered/``         a small rendered subset (PDF / PNG) to exercise OCR
* ``evaluation/test_docs/``           held-out docs (PDF / PNG / JPG / DOCX), a different seed,
                                      ``labels.json`` (type + field ground truth) and ``qa.json``
* ``sample_docs/``                    a few demo files for the website

Held-out realism: the last layout variant of each structured type (invoice/resume/contract) is
*never* used for training, so ~25% of test docs use a layout the models have not seen.

Usage::

    python -m training.generate_data            # defaults
    python -m training.generate_data --train-per-class 300 --seed 7
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import textwrap
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageFilter, ImageFont

logger = logging.getLogger("generate_data")

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["invoice", "resume", "contract", "report", "other"]
N_LAYOUTS = 4  # layout 3 is held out from training for structured types
HELD_OUT_LAYOUT = 3


@dataclass
class SynthDoc:
    """A generated document: text (pages split by form-feed), labels and QA pairs."""

    doc_id: str
    doc_type: str
    text: str
    fields: dict = field(default_factory=dict)
    qa: list[dict] = field(default_factory=list)
    layout: int = 0


# --------------------------------------------------------------------------- text noise helpers
def ascii_clean(text: str) -> str:
    """Fold accents/dashes to ASCII, keeping the currency symbols £ and €."""
    text = text.replace("–", "-").replace("—", "-").replace("’", "'")
    out = []
    for ch in text:
        if ch in "£€\n\f":
            out.append(ch)
        else:
            out.append(unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode())
    return "".join(out)


def typo(rng: random.Random, s: str, p: float = 0.04) -> str:
    """Introduce OCR/typing style typos into alphabetic words (never digits)."""
    words = s.split(" ")
    for i, w in enumerate(words):
        if len(w) > 4 and w.isalpha() and rng.random() < p:
            j = rng.randrange(1, len(w) - 2)
            if rng.random() < 0.5:
                w = w[:j] + w[j + 1] + w[j] + w[j + 2:]  # swap
            else:
                w = w[:j] + w[j + 1:]  # drop
            words[i] = w
    return " ".join(words)


def whitespace_noise(rng: random.Random, text: str) -> str:
    """Randomly double spaces, add trailing spaces and extra blank lines."""
    lines = []
    for line in text.split("\n"):
        if rng.random() < 0.12 and " " in line.strip():
            parts = line.split(" ")
            k = rng.randrange(len(parts))
            parts[k] = parts[k] + " " * rng.randint(1, 2)
            line = " ".join(parts)
        if rng.random() < 0.08:
            line += "  "
        lines.append(line)
        if rng.random() < 0.03:
            lines.append("")
    return "\n".join(lines)


def fmt_date(rng: random.Random, d: date) -> str:
    """Unambiguous date formats only (so ground truth ISO date is always recoverable)."""
    fmts = ["%Y-%m-%d", "%B %d, %Y", "%d %b %Y", "%d %B %Y", "%b %d, %Y"]
    if d.day > 12:
        fmts.append("%d/%m/%Y")
        fmts.append("%d.%m.%Y")
    return d.strftime(rng.choice(fmts))


def rand_date(rng: random.Random, start: int = 2021, end: int = 2025) -> date:
    base = date(start, 1, 1)
    return base + timedelta(days=rng.randrange((date(end, 12, 28) - base).days))


def money_formatter(rng: random.Random, cur: str):
    """Return a function formatting a float in one of several currency styles."""
    sym = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "Rs."}[cur]
    style = rng.choice(["sym", "sym", "code", "eu", "plain"])

    def f(x: float) -> str:
        if style == "sym":
            return f"{sym}{x:,.2f}"
        if style == "code":
            return f"{cur} {x:,.2f}"
        if style == "eu":
            s = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            return f"{s} {sym}"
        return f"{x:.2f}"

    return f


def person(fake: Faker) -> tuple[str, str]:
    return fake.first_name(), fake.last_name()


def company(rng: random.Random, fake: Faker) -> str:
    base = rng.choice([fake.last_name(), fake.last_name(), fake.word().title()])
    suffix = rng.choice(["LLC", "Inc.", "Ltd", "GmbH", "Co.", "Solutions", "Group", "Partners",
                         "Industries", "Systems", "Labs", "Consulting", "Supplies"])
    mid = rng.choice(["", "", "Global ", "Digital ", "Prime ", "Blue ", "Apex ", "United "])
    return f"{mid}{base} {suffix}".strip()


def address(fake: Faker) -> list[str]:
    return [fake.street_address(), f"{fake.city()}, {fake.postcode()}"]


# --------------------------------------------------------------------------- invoices
ITEM_BANK = [
    "Web design services", "Consulting hours", "Cloud hosting (monthly)", "Office chairs",
    "Printer toner cartridges", "Software license", "Catering service", "Logistics and freight",
    "Marketing campaign", "Data analysis report", "Laptop docking station", "Security audit",
    "Training workshop", "Maintenance contract", "Packaging materials", "Photography session",
    "Legal review", "Translation services", "Server rack mounting", "Stationery bundle",
    "Mobile app development", "SEO optimisation", "Equipment rental", "Courier delivery",
]


def gen_invoice(rng: random.Random, fake: Faker, layout: int, doc_id: str) -> SynthDoc:
    vendor = company(rng, fake)
    year = rng.randint(2021, 2025)
    inv_no = rng.choice([
        f"INV-{rng.randint(1000, 99999)}", f"{year}-{rng.randint(1, 999):04d}",
        f"{rng.randint(100000, 999999)}", f"{rng.choice(['AB', 'XQ', 'BL'])}{rng.randint(10000, 99999)}",
        f"INV/{year}/{rng.randint(1, 9999):04d}",
    ])
    d = rand_date(rng)
    due = d + timedelta(days=rng.choice([7, 14, 30, 45, 60]))
    cur = rng.choices(["USD", "EUR", "GBP", "INR"], weights=[5, 2, 2, 1])[0]
    m = money_formatter(rng, cur)
    client = company(rng, fake)

    items = []
    for desc in rng.sample(ITEM_BANK, rng.randint(1, 6)):
        qty = rng.randint(1, 20)
        unit = round(rng.uniform(5, 900), 2)
        items.append({"description": desc, "quantity": qty, "unit_price": unit,
                      "amount": round(qty * unit, 2)})
    subtotal = round(sum(i["amount"] for i in items), 2)
    rate = rng.choice([0, 5, 7.5, 8.25, 10, 19, 20])
    tax = round(subtotal * rate / 100, 2)
    total = round(subtotal + tax, 2)

    inv_lbl = rng.choice(["Invoice No:", "Invoice #", "Invoice Number:", "Inv. No.", "Bill No:"])
    date_lbl = rng.choice(["Date:", "Invoice Date:", "Issued:", "Date of Issue:"])
    total_lbl = rng.choice(["Total", "Amount Due", "Grand Total", "Total Due", "Balance Due"])
    tax_name = rng.choice(["Tax", "VAT", "GST", "Sales Tax"])
    tax_lbl = f"{tax_name} ({rate:g}%)"
    date_str = fmt_date(rng, d)
    thanks = typo(rng, rng.choice(["Thank you for your business.", "Payment is due within the terms stated.",
                                   "Please reference the invoice number with your payment.",
                                   "Late payments may incur a 1.5% monthly fee."]), 0.05)
    terms = f"Payment terms: Net {(due - d).days}. Due date: {fmt_date(rng, due)}"

    def table() -> list[str]:
        style = rng.choice(["table", "compact"]) if layout != 3 else "hours"
        rows: list[str] = []
        if style == "table":
            rows.append(f"{'Description':<34}{'Qty':>5}{'Unit Price':>14}{'Amount':>14}")
            rows.append("-" * 67)
            for i in items:
                rows.append(f"{i['description']:<34}{i['quantity']:>5}{m(i['unit_price']):>14}{m(i['amount']):>14}")
        elif style == "compact":
            for i in items:
                rows.append(f"- {i['description']}: {i['quantity']} x {m(i['unit_price'])} = {m(i['amount'])}")
        else:
            for i in items:
                rows.append(f"{i['description']} ({i['quantity']} hrs @ {m(i['unit_price'])}/hr) .... {m(i['amount'])}")
        return rows

    sub_lbl = rng.choice(["Subtotal", "Sub-total", "Net amount"])
    totals = [f"{sub_lbl}: {m(subtotal)}", f"{tax_lbl}: {m(tax)}", f"{total_lbl}: {m(total)}"]
    va, ca = address(fake), address(fake)

    if layout == 0:
        lines = [vendor, *va, "", "INVOICE", f"{inv_lbl} {inv_no}", f"{date_lbl} {date_str}", "",
                 "Bill To:", client, *ca, "", *table(), "", *totals, "", terms, thanks]
    elif layout == 1:
        lines = ["TAX INVOICE", f"From: {vendor}", *va, f"{inv_lbl} {inv_no}   |   {date_lbl} {date_str}",
                 f"Customer: {client}", "", *table(), "",
                 *[f"{t.split(':')[0]:>50}   {t.split(': ', 1)[1]}" for t in totals], "", thanks, terms]
    elif layout == 2:
        lines = [f"{total_lbl}: {m(total)}", "", "Invoice details", f"{inv_lbl} {inv_no}",
                 f"{date_lbl} {date_str}", f"Billed to {client}", "", *table(), "", totals[0], totals[1],
                 "", terms, "", f"Please remit payment to: {vendor}", *va, thanks]
    else:  # held-out layout: key/value form
        lines = ["Document type : Invoice", f"Supplier : {vendor}", f"Reference : {inv_no}",
                 f"Issue date : {date_str}", f"Customer : {client}", "", "Services rendered", *table(), "",
                 *[t.replace(": ", " : ", 1) for t in totals], "", thanks]

    fields = {"vendor": vendor, "invoice_number": inv_no, "date": d.isoformat(), "line_items": items,
              "subtotal": subtotal, "tax": tax, "total": total, "currency": cur}
    qa = [
        {"question": "What is the invoice number?", "answer": inv_no, "answerable": True},
        {"question": "Who issued this invoice (the vendor)?", "answer": vendor, "answerable": True},
        {"question": "What is the total amount on this invoice?", "answer": m(total), "answerable": True},
        {"question": "What is the vendor's bank account number?", "answer": "", "answerable": False},
    ]
    return SynthDoc(doc_id, "invoice", "\n".join(lines), fields, qa, layout)


# --------------------------------------------------------------------------- resumes
SKILLS = ["Python", "SQL", "Java", "JavaScript", "TypeScript", "React", "Docker", "Kubernetes", "AWS",
          "Machine Learning", "Data Analysis", "Project Management", "Excel", "Tableau", "Figma",
          "Communication", "Leadership", "Salesforce", "Accounting", "Public Speaking", "C++", "Go",
          "Agile", "Git", "Marketing Analytics", "Customer Support", "Negotiation", "Photoshop"]
TITLES = ["Software Engineer", "Data Analyst", "Product Manager", "Marketing Specialist", "Account Executive",
          "DevOps Engineer", "UX Designer", "Business Analyst", "Operations Manager", "Research Assistant",
          "Financial Analyst", "Customer Success Manager", "QA Engineer", "HR Coordinator"]
DEGREES = ["B.Sc. Computer Science", "B.A. Economics", "M.Sc. Data Science", "MBA", "B.Eng. Electrical Engineering",
           "B.A. Graphic Design", "M.A. Marketing", "B.Sc. Statistics", "Ph.D. Physics", "B.Com. Accounting"]
BULLETS = [
    "Led a team of {n} to deliver projects ahead of schedule",
    "Improved process efficiency by {n}% through automation",
    "Managed a portfolio of {n} key client accounts",
    "Built dashboards used by {n}+ stakeholders",
    "Reduced operating costs by {n}% year over year",
    "Mentored {n} junior colleagues",
]


def gen_resume(rng: random.Random, fake: Faker, layout: int, doc_id: str) -> SynthDoc:
    first, last = person(fake)
    name = f"{first} {last}"
    email = f"{first}.{last}{rng.choice(['', '', '1', '_dev', '88'])}@{rng.choice(['gmail.com', 'outlook.com', 'proton.me', 'mail.com', 'yahoo.com'])}".lower()
    phone = fake.phone_number()
    skills = rng.sample(SKILLS, rng.randint(5, 12))

    exp, end_year = [], 2025
    for _ in range(rng.randint(1, 4)):
        length = rng.randint(1, 4)
        start = end_year - length
        exp.append({"title": rng.choice(TITLES), "company": company(rng, fake), "start": str(start),
                    "end": "Present" if not exp else str(end_year)})
        end_year = start - rng.randint(0, 1)
    edu = [{"degree": rng.choice(DEGREES),
            "institution": rng.choice([f"University of {fake.city()}", f"{fake.last_name()} Institute of Technology",
                                       f"{fake.city()} State College"]),
            "year": str(rng.randint(2005, 2023))} for _ in range(rng.randint(1, 2))]

    def bullets() -> list[str]:
        return ["  * " + rng.choice(BULLETS).format(n=rng.randint(3, 60)) for _ in range(rng.randint(1, 3))]

    summary = typo(rng, f"{rng.choice(['Results-driven', 'Detail-oriented', 'Self-motivated'])} professional with "
                        f"{rng.randint(2, 15)} years of experience in {rng.choice(['technology', 'finance', 'operations', 'marketing'])}.")
    L: list[str] = []
    if layout == 0:
        L += [name.upper(), f"{email} | {phone} | {fake.city()}", "", "SUMMARY", summary, "", "SKILLS", ", ".join(skills), "",
              "EXPERIENCE"]
        for e in exp:
            L += [f"{e['title']} | {e['company']} | {e['start']} - {e['end']}", *bullets()]
        L += ["", "EDUCATION"] + [f"{e['degree']}, {e['institution']}, {e['year']}" for e in edu]
    elif layout == 1:
        L += ["Curriculum Vitae", f"Name: {name}", f"Email: {email}", f"Phone: {phone}", "", "Technical Skills:",
              *[f"* {s}" for s in skills], "", "Work History"]
        for e in exp:
            L += [f"{e['company']} - {e['title']} ({e['start']}-{e['end']})", *bullets()]
        L += ["", "Education"] + [f"{e['institution']}: {e['degree']} ({e['year']})" for e in edu]
    elif layout == 2:
        L += ["PROFESSIONAL EXPERIENCE"]
        for e in exp:
            L += [f"{e['title']}, {e['company']}", f"{e['start']} to {e['end']}", *bullets()]
        L += ["", "EDUCATION"] + [f"{e['year']} - {e['degree']} - {e['institution']}" for e in edu]
        L += ["", "CORE COMPETENCIES", "; ".join(skills), "", "CONTACT", name, email, phone]
    else:  # held-out layout
        L += [f"Resume of {name}", f"E-mail: {email}", f"Tel: {phone}", "", "Profile", summary, "",
              f"Key skills: {' / '.join(skills)}", "", "Career"]
        for e in exp:
            L += [f"{e['start']} - {e['end']}  {e['title']} at {e['company']}", *bullets()]
        L += ["", "Studies"] + [f"{e['degree']} - {e['institution']} - {e['year']}" for e in edu]

    fields = {"name": name, "email": email, "skills": skills, "experience": exp, "education": edu}
    qa = [
        {"question": "What is the candidate's email address?", "answer": email, "answerable": True},
        {"question": "Which company did the candidate most recently work for?", "answer": exp[0]["company"],
         "answerable": True},
        {"question": "What is the candidate's salary expectation?", "answer": "", "answerable": False},
    ]
    return SynthDoc(doc_id, "resume", "\n".join(L), fields, qa, layout)


# --------------------------------------------------------------------------- contracts
CLAUSE_BANK = {
    "Confidentiality": ["Each party shall keep the other party's confidential information strictly confidential and shall not disclose it to any third party without prior written consent.",
                        "The receiving party agrees to protect all proprietary information using at least the same degree of care it uses for its own confidential materials."],
    "Termination": ["Either party may terminate this Agreement upon thirty (30) days written notice if the other party materially breaches its obligations and fails to cure.",
                    "This Agreement may be terminated by either party for convenience with sixty (60) days prior written notice."],
    "Payment Terms": ["All fees are payable within thirty (30) days of receipt of invoice. Overdue amounts accrue interest at 1.5% per month.",
                      "The Client shall pay the agreed fees in monthly installments, due on the first business day of each month."],
    "Indemnification": ["Each party shall indemnify and hold harmless the other from claims arising out of its negligence or willful misconduct.",
                        "The Provider shall defend and indemnify the Client against third-party claims arising from the Provider's breach."],
    "Limitation of Liability": ["In no event shall either party be liable for indirect, incidental or consequential damages arising out of this Agreement.",
                                "Total liability under this Agreement shall not exceed the fees paid in the twelve months preceding the claim."],
    "Intellectual Property": ["All work product created under this Agreement shall be owned by the Client upon full payment.",
                              "Each party retains ownership of its pre-existing intellectual property."],
    "Force Majeure": ["Neither party shall be liable for delay caused by events beyond its reasonable control, including natural disasters and strikes."],
    "Dispute Resolution": ["Any dispute shall first be submitted to good-faith negotiation, and thereafter to binding arbitration."],
    "Non-Compete": ["During the term and for twelve months thereafter, the Provider shall not engage in directly competing business within the territory."],
}
AGREEMENT_TYPES = ["Service Agreement", "Consulting Agreement", "License Agreement", "Master Services Agreement",
                   "Supply Agreement", "Software Development Agreement"]
ROLE_PAIRS = [("Client", "Provider"), ("Company", "Contractor"), ("Licensor", "Licensee"), ("Buyer", "Supplier")]
STATES = ["Delaware", "California", "New York", "Texas", "England and Wales", "Ontario", "Washington"]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def gen_contract(rng: random.Random, fake: Faker, layout: int, doc_id: str) -> SynthDoc:
    a, b = company(rng, fake), company(rng, fake)
    while b == a:
        b = company(rng, fake)
    ra, rb = rng.choice(ROLE_PAIRS)
    kind = rng.choice(AGREEMENT_TYPES)
    eff = rand_date(rng)
    eff_str = fmt_date(rng, eff)
    term = rng.choice([f"{rng.choice([6, 12, 18, 24, 36])} months", f"{rng.randint(1, 5)} years"])
    clause_names = rng.sample(list(CLAUSE_BANK), rng.randint(4, 7))
    law = rng.choice(STATES)

    def head(n: int, title: str) -> str:
        return {0: f"{n}. {title.upper()}", 1: f"ARTICLE {ROMAN[n - 1]} - {title.upper()}",
                2: f"Section {n}. {title}", 3: f"Clause {n}: {title}"}[layout]

    L = [kind.upper(), ""]
    if layout in (0, 1):
        L += [f'This {kind} (the "Agreement") is entered into as of {eff_str} (the "Effective Date") by and between '
              f'{a} (the "{ra}") and {b} (the "{rb}").']
    elif layout == 2:
        L += [f"Effective Date: {eff_str}", f"PARTY A: {a} ({ra})", f"PARTY B: {b} ({rb})"]
    else:
        L += [f"This agreement takes effect on {eff_str}.", "Parties:", f'1) {a} (the "{ra}")', f'2) {b} (the "{rb}")']
    L += ["", head(1, "Term"),
          f"This Agreement shall commence on the Effective Date and continue for a term of {term}, unless terminated earlier in accordance with its provisions.", ""]
    for i, name in enumerate(clause_names, start=2):
        L += [head(i, name), typo(rng, rng.choice(CLAUSE_BANK[name]), 0.03), ""]
        if name == "Dispute Resolution" or rng.random() < 0.3:
            L += [f"This Agreement shall be governed by the laws of {law}.", ""]
    L += [head(len(clause_names) + 2, "Governing Law"), f"This Agreement shall be governed by the laws of {law}.", "",
          "IN WITNESS WHEREOF, the parties have executed this Agreement.", "",
          f"{a}: ____________________    {b}: ____________________"]
    # paginate every ~32 lines
    pages, cur = [], []
    for ln in L:
        cur.append(ln)
        if len(cur) >= 32 and ln == "":
            pages.append("\n".join(cur))
            cur = []
    pages.append("\n".join(cur))
    key_clauses = clause_names + ["Governing Law"]
    fields = {"parties": [a, b], "effective_date": eff.isoformat(), "term": term, "key_clauses": key_clauses}
    qa = [
        {"question": "What is the effective date of the agreement?", "answer": eff_str, "answerable": True},
        {"question": "What is the term of the agreement?", "answer": term, "answerable": True},
        {"question": "What is the penalty for late delivery in dollars?", "answer": "", "answerable": False},
    ]
    return SynthDoc(doc_id, "contract", "\f".join(pages), fields, qa, layout)


# --------------------------------------------------------------------------- reports
REGIONS = ["North America", "EMEA", "APAC", "Latin America"]


def gen_report(rng: random.Random, fake: Faker, layout: int, doc_id: str) -> SynthDoc:
    flavor = rng.choice(["business", "incident", "research"])
    org = company(rng, fake)
    period = f"Q{rng.randint(1, 4)} {rng.randint(2022, 2025)}"
    growth = f"{rng.uniform(2, 35):.1f}%"
    churn = f"{rng.uniform(0.5, 9):.1f}%"
    headcount = str(rng.randint(40, 4000))
    L: list[str] = []
    if flavor == "business":
        L += [f"{org} Quarterly Business Review - {period}", "", "Executive Summary",
              typo(rng, f"Revenue grew {growth} year over year in {period}, driven by strong demand and improved retention."),
              f"Customer churn was {churn} for the period, and total headcount reached {headcount} employees.", "",
              "Regional Performance", f"{'Region':<18}{'Revenue':>12}{'Growth':>10}"]
        for r in REGIONS:
            L.append(f"{r:<18}{'$' + format(rng.randint(2, 90), ',') + 'M':>12}{rng.uniform(-3, 25):>9.1f}%")
        L += ["", "Outlook", typo(rng, "Management expects continued momentum next quarter, subject to macroeconomic conditions and supply constraints."),
              "Risks include currency volatility, hiring delays and contract renewals."]
        facts = [("What was the revenue growth year over year?", growth),
                 ("What was the customer churn rate?", churn)]
    elif flavor == "incident":
        dur = rng.randint(12, 400)
        L += [f"Incident Report: Service Degradation ({period})", "", "Summary",
              typo(rng, f"On {fmt_date(rng, rand_date(rng))} the platform experienced elevated latency lasting {dur} minutes."),
              "", "Root Cause Analysis", typo(rng, "A misconfigured cache layer caused cascading timeouts across dependent services."),
              "", "Findings", f"Error rate peaked at {churn} of requests. {headcount} customers were affected.",
              "", "Recommendations", "Add canary deployments, tighten alerting thresholds and run quarterly failure drills."]
        facts = [("How long did the service degradation last?", f"{dur} minutes"),
                 ("What was the peak error rate?", churn)]
    else:
        n = rng.randint(80, 3000)
        L += [f"Research Report: Consumer Adoption Survey {rng.randint(2022, 2025)}", "", "Abstract",
              typo(rng, f"We surveyed {n} participants about adoption of digital tools and measured satisfaction."),
              "", "Methodology", "Participants were recruited online and responded to a structured questionnaire.",
              "", "Results", f"Overall adoption was {growth}, with dropout at {churn}.",
              "", "Discussion", typo(rng, "Findings suggest that usability and trust are the strongest predictors of adoption."),
              "", "Conclusion", "Further longitudinal work is needed to confirm these preliminary results."]
        facts = [("How many participants were surveyed?", str(n)), ("What was the overall adoption rate?", growth)]
    qa = [{"question": q, "answer": a, "answerable": True} for q, a in facts]
    qa.append({"question": "Who is the chief executive officer of the company?", "answer": "", "answerable": False})
    return SynthDoc(doc_id, "report", "\n".join(L), {"title": L[0]}, qa, layout)


# --------------------------------------------------------------------------- other (incl. hard negatives)
def gen_other(rng: random.Random, fake: Faker, layout: int, doc_id: str) -> SynthDoc:
    kind = rng.choice(["cover_letter", "email", "minutes", "recipe", "newsletter", "job_posting", "memo_invoice",
                       "flyer", "memo_contract"])
    first, last = person(fake)
    qa: list[dict] = []
    if kind == "cover_letter":
        L = [fake.city(), fmt_date(rng, rand_date(rng)), "", "Dear Hiring Manager,", "",
             typo(rng, f"I am writing to apply for the {rng.choice(TITLES)} position at {company(rng, fake)}. I have {rng.randint(2, 12)} years of experience and strong skills in {rng.choice(SKILLS)}."),
             typo(rng, "I would welcome the opportunity to discuss how my background fits your team."), "", "Sincerely,", f"{first} {last}"]
    elif kind == "email":
        L = [f"From: {first}.{last}@example.com".lower(), f"To: team@{fake.domain_name()}", f"Subject: {rng.choice(['Lunch plans', 'Schedule change', 'Quick question', 'Weekend trip'])}",
             "", "Hi all,", typo(rng, f"Just a reminder that we are meeting on {fmt_date(rng, rand_date(rng))} at {rng.randint(1, 5)}pm in the main room. Please bring your notes."),
             "Thanks,", first]
    elif kind == "minutes":
        chair = f"{first} {last}"
        L = [f"Meeting Minutes - {fmt_date(rng, rand_date(rng))}", f"Chair: {chair}", f"Attendees: {fake.name()}, {fake.name()}, {fake.name()}", "",
             "1. Review of previous actions", typo(rng, "All outstanding actions were completed except the website refresh."),
             "2. Budget", typo(rng, f"The group discussed spending of ${rng.randint(1, 90)}k for next quarter."), "3. Any other business", "None."]
        qa = [{"question": "Who chaired the meeting?", "answer": chair, "answerable": True}]
    elif kind == "recipe":
        dish = rng.choice(["Lentil Soup", "Banana Bread", "Veggie Stir Fry", "Tomato Pasta", "Pancakes"])
        L = [dish, f"Serves {rng.randint(2, 8)}  |  Prep time {rng.randint(10, 45)} minutes", "", "Ingredients",
             *[f"- {rng.randint(1, 4)} {rng.choice(['cups', 'tbsp', 'tsp', 'pieces'])} {fake.word()}" for _ in range(rng.randint(4, 8))],
             "", "Method", typo(rng, "Combine the ingredients, stir well and cook on medium heat until golden. Season to taste and serve warm.")]
    elif kind == "newsletter":
        L = [f"{fake.city()} Community Newsletter", f"Issue {rng.randint(1, 120)}", "", "In this issue",
             typo(rng, "Spring festival preview, volunteer spotlight, and a new walking group."),
             "", typo(rng, f"Join us on {fmt_date(rng, rand_date(rng))} at the community hall. Everyone is welcome and refreshments will be provided.")]
    elif kind == "job_posting":
        L = [f"We are hiring: {rng.choice(TITLES)}", f"Company: {company(rng, fake)}", "", "Requirements",
             *[f"- {rng.choice(SKILLS)}" for _ in range(4)], "", "Apply by sending your resume to careers@example.com"]
    elif kind == "memo_invoice":  # hard negative: invoice vocabulary, not an invoice
        L = ["INTERNAL MEMO", "To: Finance team", f"From: {first} {last}", "",
             typo(rng, "Reminder: please submit all vendor invoices for approval by Friday. Do not pay any invoice without a purchase order number."),
             "Totals will be reviewed in the monthly budget meeting."]
    elif kind == "memo_contract":  # hard negative: contract vocabulary, not a contract
        L = ["POLICY NOTICE", "From: Legal Department", "",
             typo(rng, "Staff should route every agreement through legal review before signature. Confidentiality and termination clauses must be checked."),
             "Questions may be sent to the legal team."]
    else:
        L = [rng.choice(["GARAGE SALE", "BAKE SALE", "YARD SALE", "BOOK FAIR"]), f"Saturday {fmt_date(rng, rand_date(rng))}",
             f"{fake.street_address()}, {fake.city()}", typo(rng, "Great prices, friendly neighbours and free coffee all morning!")]
    return SynthDoc(doc_id, "other", "\n".join(L), {"kind": kind}, qa, layout)


GENERATORS = {"invoice": gen_invoice, "resume": gen_resume, "contract": gen_contract,
              "report": gen_report, "other": gen_other}


def make_doc(rng: random.Random, fake: Faker, doc_type: str, doc_id: str, allow_held_out: bool) -> SynthDoc:
    layouts = list(range(N_LAYOUTS)) if allow_held_out else list(range(N_LAYOUTS - 1))
    layout = rng.choice(layouts)
    doc = GENERATORS[doc_type](rng, fake, layout, doc_id)
    doc.text = whitespace_noise(rng, ascii_clean(doc.text))
    for q in doc.qa:
        q["answer"] = ascii_clean(q["answer"])
    return doc


# --------------------------------------------------------------------------- rendering
_FONT_CANDIDATES = ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def render_pdf(doc: SynthDoc, path: Path, rng: random.Random) -> None:
    """Render text to a digital PDF (real text layer) using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    font, size = rng.choice([("Helvetica", 10), ("Courier", 9), ("Times-Roman", 11)])
    width_chars = 92 if font == "Courier" else 100 if font == "Helvetica" else 105
    lh = size + 4
    w, h = A4
    c = canvas.Canvas(str(path), pagesize=A4)
    for page_text in doc.text.split("\f"):
        y = h - 60
        c.setFont(font, size)
        for raw in page_text.split("\n"):
            for line in (textwrap.wrap(raw, width_chars, replace_whitespace=False, drop_whitespace=False) or [""]):
                if y < 50:
                    c.showPage()
                    c.setFont(font, size)
                    y = h - 60
                c.drawString(50, y, line)
                y -= lh
        c.showPage()
    c.save()


def render_image(doc: SynthDoc, path: Path, rng: random.Random, fmt: str = "png") -> None:
    """Render page 1 as a 'scanned' image: slight rotation, noise, blur, JPEG artefacts."""
    size = rng.randint(22, 28)
    font = _load_font(size)
    margin, width = 90, 1240
    lines: list[str] = []
    maxw = width - 2 * margin
    for raw in doc.text.split("\f")[0].split("\n"):
        cur = ""
        for word in raw.split(" "):
            trial = f"{cur} {word}" if cur else word
            if font.getlength(trial) > maxw and cur:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
    lh = int(size * 1.45)
    height = max(900, 2 * margin + lh * len(lines))
    img = Image.new("L", (width, height), 255)
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        d.text((margin, margin + i * lh), ln, fill=rng.randint(0, 40), font=font)
    img = img.rotate(rng.uniform(-2.5, 2.5), expand=True, fillcolor=255, resample=Image.BICUBIC)
    if rng.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.7)))
    arr = np.asarray(img, dtype=np.float32)
    arr += np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(4, 12), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    if fmt == "jpg":
        img.save(path, "JPEG", quality=rng.randint(60, 85))
    else:
        img.save(path, "PNG")


def render_docx(doc: SynthDoc, path: Path) -> None:
    """Render text to a DOCX with explicit page breaks between pages."""
    from docx import Document

    d = Document()
    pages = doc.text.split("\f")
    for pi, page in enumerate(pages):
        for ln in page.split("\n"):
            d.add_paragraph(ln)
        if pi < len(pages) - 1:
            d.add_page_break()
    d.save(str(path))


def render(doc: SynthDoc, path_no_ext: Path, fmt: str, rng: random.Random) -> Path:
    """Render a doc in the requested format (images fall back to PDF for multi-page docs)."""
    if fmt in ("png", "jpg") and "\f" in doc.text:
        fmt = "pdf"
    path = path_no_ext.with_suffix(f".{fmt}")
    if fmt == "pdf":
        render_pdf(doc, path, rng)
    elif fmt == "docx":
        render_docx(doc, path)
    else:
        render_image(doc, path, rng, fmt)
    return path


TEST_FORMATS = {
    "invoice": ["pdf", "pdf", "png", "jpg"], "resume": ["pdf", "pdf", "docx"],
    "contract": ["pdf", "pdf", "docx", "png"], "report": ["pdf", "docx", "pdf"],
    "other": ["pdf", "png", "docx", "jpg"],
}


# --------------------------------------------------------------------------- orchestration
def norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def generate(out_dir: Path, test_dir: Path, sample_dir: Path, train_per_class: int, test_per_class: int,
             seed: int) -> dict:
    """Generate train/test/sample sets. Returns a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rendered").mkdir(exist_ok=True)

    # ---- train
    fake = Faker(["en_US", "en_GB", "en_CA"])
    fake.seed_instance(seed)
    rng = random.Random(seed)
    train: list[SynthDoc] = []
    for cls in CLASSES:
        for i in range(train_per_class):
            train.append(make_doc(rng, fake, cls, f"train_{cls}_{i:04d}", allow_held_out=False))
    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as fh:
        for d in train:
            fh.write(json.dumps({"id": d.doc_id, "label": d.doc_type, "text": d.text, "fields": d.fields,
                                 "layout": d.layout}) + "\n")
    seen = {norm_key(d.text) for d in train}
    # rendered subset: 8 per class, alternating PDF / PNG (exercises ingestion + OCR on training-style docs)
    for cls in CLASSES:
        for k, d in enumerate([x for x in train if x.doc_type == cls][:8]):
            render(d, out_dir / "rendered" / d.doc_id, "pdf" if k % 2 == 0 else "png", rng)

    # ---- held-out test (different seed, de-duplicated against train)
    fake_t = Faker(["en_US", "en_GB", "en_CA"])
    fake_t.seed_instance(seed + 10_000)
    rng_t = random.Random(seed + 10_000)
    labels, qa_all = {}, []
    for cls in CLASSES:
        for i in range(test_per_class):
            while True:
                d = make_doc(rng_t, fake_t, cls, f"test_{cls}_{i:02d}", allow_held_out=True)
                if norm_key(d.text) not in seen:
                    break
            seen.add(norm_key(d.text))
            fmt = TEST_FORMATS[cls][i % len(TEST_FORMATS[cls])]
            path = render(d, test_dir / f"{cls}_{i:02d}", fmt, rng_t)
            labels[path.name] = {"doc_type": cls, "fields": d.fields, "layout": d.layout,
                                 "format": path.suffix[1:]}
            for q in d.qa:
                qa_all.append({**q, "file": path.name, "doc_type": cls})
    # global off-topic unanswerable questions, asked against random test docs
    for q in ["What is the capital of France?", "Who won the 2018 football World Cup?",
              "What is the airspeed velocity of an unladen swallow?"]:
        for name in rng_t.sample(sorted(labels), 3):
            qa_all.append({"question": q, "answer": "", "answerable": False, "file": name,
                           "doc_type": labels[name]["doc_type"]})
    (test_dir / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    (test_dir / "qa.json").write_text(json.dumps(qa_all, indent=1), encoding="utf-8")

    # ---- website samples (own seed so they are never in train/test)
    fake_s = Faker("en_US")
    fake_s.seed_instance(seed + 99)
    rng_s = random.Random(seed + 99)
    for cls, fmt, name in [("invoice", "pdf", "sample_invoice"), ("resume", "pdf", "sample_resume"),
                           ("contract", "pdf", "sample_contract"), ("invoice", "png", "sample_scanned_invoice")]:
        d = make_doc(rng_s, fake_s, cls, name, allow_held_out=False)
        render(d, sample_dir / name, fmt, rng_s)

    return {"train": len(train), "per_class_train": train_per_class, "test": len(labels),
            "per_class_test": test_per_class, "qa_pairs": len(qa_all)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-per-class", type=int, default=220)
    ap.add_argument("--test-per-class", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "training" / "data")
    ap.add_argument("--test-out", type=Path, default=ROOT / "evaluation" / "test_docs")
    ap.add_argument("--sample-out", type=Path, default=ROOT / "sample_docs")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = generate(a.out, a.test_out, a.sample_out, a.train_per_class, a.test_per_class, a.seed)
    logger.info("Generated: %s", summary)


if __name__ == "__main__":
    main()
