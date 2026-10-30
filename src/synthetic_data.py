from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .policy_engine import evaluate_policy
from .routing import get_department_for_issue, get_default_severity, load_taxonomy
from .preprocessing import normalize_text

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "synthetic_feedback.csv"

REGIONS = ["Hyderabad", "Visakhapatnam", "Chennai", "Pune", "Mumbai", "Bengaluru"]
CHANNELS = ["Online Review", "In-app Rating", "Store Staff Note", "Call Centre"]
LANGUAGES = ["English", "Hinglish", "Telugu Transliteration", "Tamil Transliteration", "Marathi Transliteration"]
STORE_SIZES = ["Small", "Medium", "Large"]

CATEGORY_PRIORITY = [
    "Refrigeration Failure",
    "Hygiene",
    "Allergen Label Error",
    "Stock Shortage",
    "Delivery Delay",
    "Incorrect Price Label",
    "App Payment Failure",
    "Staff Misconduct",
    "Checkout Delay",
    "Store Cleanliness",
    "Food Contamination",
    "Product Damage",
    "Shelf Tag Missing",
    "Equipment Failure",
    "Freshness Concern",
    "Customer Service Delay",
    "Lighting Failure",
    "Security Concern",
    "Promotion Mismatch",
    "Billing Issue",
    "App Login Failure",
    "Queue Management",
    "Inventory Mismatch",
    "Packaging Damage",
    "Fresh Produce Quality",
    "Return Issue",
    "Parking Issue",
    "Unsafe Product",
    "Harassment",
    "Customer Praise",
]

TEMPLATES: dict[str, dict[str, list[str]]] = {
    "Refrigeration Failure": {
        "English": [
            "The freezer is not working and milk packets are spoiling.",
            "Fridge temperature is too high; the frozen section is warm.",
            "Cold storage failed again and products are getting spoiled.",
        ],
        "Hinglish": [
            "Store lo freezer work avvatledu, milk packets spoil avtunnayi!",
            "Fridge temperature chala ekkuva undi, ice cream melt avtundi.",
            "Freezer band ayindi, items spoil avuthunnayi.",
        ],
        "Telugu Transliteration": [
            "Freezer pani cheyatledu, milk packets spoil avtunnayi.",
            "Cold room lo temperature ekkuva undi, items spoil avtunnayi.",
        ],
        "Tamil Transliteration": [
            "Freezer seri illa, milk packets spoil aaguthu.",
            "Cold storage work aagala, goods spoil aaguthu.",
        ],
        "Marathi Transliteration": [
            "Freezer chalat nahi aahe, dudh packets spoil hot aahet.",
            "Cold storage madhye problem aahe ani mal spoil hotoy.",
        ],
    },
    "Hygiene": {
        "English": [
            "The washroom and aisle are dirty and smell bad.",
            "Hygiene is poor; spills were not cleaned on time.",
            "Cockroaches near the bakery area and unclean shelves.",
        ],
        "Hinglish": [
            "Store bahut dirty hai, smell aa rahi hai aur cleaning nahi hui.",
            "Washroom ganda hai aur floor pe spill pada hai.",
            "Safai theek nahi hai, cockroach dikha.",
        ],
        "Telugu Transliteration": [
            "Store lo cleanliness ledu, smell vastondi.",
            "Washroom chala dirty ga undi, spill clean cheyatledu.",
        ],
        "Tamil Transliteration": [
            "Store romba dirty irukku, smell varudhu.",
            "Washroom clean aagala, floor spill irukku.",
        ],
        "Marathi Transliteration": [
            "Store khup dirty aahe, smell yetey.",
            "Washroom ani floor var safai nahi zali.",
        ],
    },
    "Allergen Label Error": {
        "English": [
            "Incorrect allergen information printed on the food packet.",
            "The label says no nuts, but the ingredient list includes nuts.",
            "Allergen warning missing from the packaged item.",
        ],
        "Hinglish": [
            "Wrong allergen label on the packet, very risky.",
            "Food packet pe allergen info galat hai.",
            "Pack label me nuts mention nahi hai but andar hai.",
        ],
        "Telugu Transliteration": [
            "Food packet me allergen label tappu ga undi.",
            "Nuts unnayi kani label lo mention cheyaledu.",
        ],
        "Tamil Transliteration": [
            "Food packet la allergen label thappu ah irukku.",
            "Nuts irukku aana label sollala.",
        ],
        "Marathi Transliteration": [
            "Food packet var allergen label chukicha aahe.",
            "Nuts aahet pan label var dile nahi.",
        ],
    },
    "Stock Shortage": {
        "English": ["The shelf is empty and the product is out of stock.", "Stock shortage in the dairy aisle."],
        "Hinglish": ["Shelf khali hai, stock nahi hai.", "Dairy section me stock shortage hai."],
        "Telugu Transliteration": ["Shelf empty undi, stock ledu.", "Dairy lo stock shortage undi."],
        "Tamil Transliteration": ["Shelf la stock illa.", "Dairy section la stock shortage irukku."],
        "Marathi Transliteration": ["Shelf rikami aahe, stock nahi aahe.", "Dairy madhye stock shortage aahe."],
    },
    "Delivery Delay": {
        "English": ["Delivery arrived late and the fresh stock missed peak hours.", "Truck delay caused late replenishment."],
        "Hinglish": ["Delivery late aayi, replenish time miss ho gaya.", "Truck late tha aur stock time pe nahi aaya."],
        "Telugu Transliteration": ["Delivery late ga vachindi.", "Truck delay valla stock late ga refill ayindi."],
        "Tamil Transliteration": ["Delivery late ah vandhuchu.", "Truck delay nala stock late refill aachu."],
        "Marathi Transliteration": ["Delivery late ali.", "Truck delay mule stock late ala."],
    },
    "Incorrect Price Label": {
        "English": ["The price label is incorrect and the billing amount is wrong.", "MRP on shelf does not match billing."],
        "Hinglish": ["Price tag galat hai, billing bhi mismatch hai.", "Shelf label aur bill match nahi kar raha."],
        "Telugu Transliteration": ["Price label tappu undi.", "Billing amount shelf price tho match avvatledu."],
        "Tamil Transliteration": ["Price label thappu ah irukku.", "Bill amount shelf price kooda match aagala."],
        "Marathi Transliteration": ["Price tag chukicha aahe.", "Bill ani shelf price match hot nahi."],
    },
    "App Payment Failure": {
        "English": ["Payment keeps failing in the app checkout flow.", "UPI payment failed multiple times."],
        "Hinglish": ["App payment fail ho raha hai.", "Checkout pe transaction decline ho gaya."],
        "Telugu Transliteration": ["App payment fail avtundi.", "UPI payment work avvatledu."],
        "Tamil Transliteration": ["App payment fail aaguthu.", "Checkout la payment work aagala."],
        "Marathi Transliteration": ["App payment fail hot aahe.", "UPI payment work hot nahi."],
    },
    "Staff Misconduct": {
        "English": ["A staff member behaved inappropriately with a customer.", "The employee was rude and threatening."],
        "Hinglish": ["Staff member ne inappropriate behavior kiya.", "Employee rude tha aur customer ko threaten kiya."],
        "Telugu Transliteration": ["Staff member inappropriate ga behave chesaru.", "Customer tho rude ga matladaru."],
        "Tamil Transliteration": ["Staff member inappropriate ah nadandhukitanga.", "Customer kitta rude ah pesinaanga."],
        "Marathi Transliteration": ["Staff member ne inappropriate vatanuk keli.", "Customer shi rude vaatla."],
    },
    "Checkout Delay": {
        "English": ["Checkout lines are too long during evening peak hours.", "Waiting time at billing counter is high."],
        "Hinglish": ["Checkout queue bahut lambi hai.", "Billing counter pe wait zyada hai."],
        "Telugu Transliteration": ["Billing queue chala pedda ga undi.", "Checkout lo delay avtundi."],
        "Tamil Transliteration": ["Checkout queue romba perusa irukku.", "Billing counter la wait adhigama irukku."],
        "Marathi Transliteration": ["Checkout line khup mothi aahe.", "Billing counter var wait jasta aahe."],
    },
    "Store Cleanliness": {
        "English": ["The store floor is dirty and trash is visible near the entrance.", "Cleaning was missed in the aisles."],
        "Hinglish": ["Store floor dirty hai aur trash pada hai.", "Aisle ki cleaning nahi hui."],
        "Telugu Transliteration": ["Store floor dirty ga undi.", "Entrance daggara trash undi."],
        "Tamil Transliteration": ["Store floor dirty ah irukku.", "Entrance pakkathula trash irukku."],
        "Marathi Transliteration": ["Store floor dirty aahe.", "Entrance jawal kachra aahe."],
    },
    "Customer Praise": {
        "English": ["Great shopping experience and helpful staff.", "Thanks, the store was clean and well organized."],
        "Hinglish": ["Great experience, staff helpful tha.", "Store clean tha aur shopping smooth thi."],
        "Telugu Transliteration": ["Shopping experience chala bagundi.", "Staff helpful ga unnaru."],
        "Tamil Transliteration": ["Shopping experience romba nalla irundhuchu.", "Staff helpful ah irundhaanga."],
        "Marathi Transliteration": ["Shopping experience khup changla hota.", "Staff helpful hote."],
    },
}

INCIDENT_PROFILES = [
    {
        "incident_tag": "refrigeration_spike_st042",
        "store_id": "ST-042",
        "region": "Visakhapatnam",
        "issue_category": "Refrigeration Failure",
        "start_offset_weeks": 6,
        "weekly_counts": [1, 2, 3, 5, 7, 9],
        "known_escalation_offset_weeks": 9,
    },
    {
        "incident_tag": "hygiene_spike_st007",
        "store_id": "ST-007",
        "region": "Hyderabad",
        "issue_category": "Hygiene",
        "start_offset_weeks": 8,
        "weekly_counts": [1, 1, 2, 3, 4, 4, 5, 6],
        "known_escalation_offset_weeks": 11,
    },
    {
        "incident_tag": "allergen_spike_st019",
        "store_id": "ST-019",
        "region": "Pune",
        "issue_category": "Allergen Label Error",
        "start_offset_weeks": 4,
        "weekly_counts": [1, 1, 2, 2, 3, 4],
        "known_escalation_offset_weeks": 7,
    },
    {
        "incident_tag": "low_volume_store_st031",
        "store_id": "ST-031",
        "region": "Bengaluru",
        "issue_category": "Stock Shortage",
        "start_offset_weeks": 5,
        "weekly_counts": [1, 2, 2, 3, 3],
        "known_escalation_offset_weeks": 8,
    },
]


@dataclass(slots=True)
class SyntheticDatasetBundle:
    dataframe: pd.DataFrame
    incident_profiles: list[dict]
    path: Path



def _build_store_master() -> pd.DataFrame:
    rows = []
    store_idx = 1
    region_cycle = REGIONS * 9
    for region in region_cycle[:50]:
        size = np.random.choice(STORE_SIZES, p=[0.35, 0.4, 0.25])
        rows.append(
            {
                "store_id": f"ST-{store_idx:03d}",
                "region": region,
                "store_size": size,
                "store_name": f"Store {store_idx:03d}",
            }
        )
        store_idx += 1
    return pd.DataFrame(rows)


def _choose_language(rng: np.random.Generator) -> str:
    return rng.choice(LANGUAGES, p=[0.46, 0.2, 0.14, 0.1, 0.1])


def _template_for(category: str, language: str, rng: np.random.Generator) -> str:
    category_templates = TEMPLATES.get(category)
    if category_templates is None:
        category_templates = TEMPLATES["Customer Praise"]
    options = category_templates.get(language) or category_templates.get("English") or [f"{category} reported."]
    text = str(rng.choice(options))
    noise_suffixes = ["", " 😟", " !!", " pls check", " urgent", " asap"]
    return text + str(rng.choice(noise_suffixes))


def _base_category_weights() -> dict[str, float]:
    weights = {
        "Customer Praise": 0.12,
        "Customer Service Delay": 0.08,
        "Store Cleanliness": 0.08,
        "Incorrect Price Label": 0.07,
        "Stock Shortage": 0.07,
        "Checkout Delay": 0.06,
        "Delivery Delay": 0.05,
        "App Payment Failure": 0.05,
        "Equipment Failure": 0.05,
        "Shelf Tag Missing": 0.05,
        "Hygiene": 0.05,
        "Fresh Produce Quality": 0.04,
        "Packaging Damage": 0.04,
        "Billing Issue": 0.04,
        "Inventory Mismatch": 0.04,
        "Promotion Mismatch": 0.03,
        "App Login Failure": 0.03,
        "Store Cleanliness": 0.03,
        "Product Damage": 0.03,
        "Freshness Concern": 0.03,
        "Refrigeration Failure": 0.02,
        "Allergen Label Error": 0.02,
        "Food Contamination": 0.02,
        "Unsafe Product": 0.02,
        "Security Concern": 0.02,
        "Staff Misconduct": 0.02,
        "Harassment": 0.01,
        "Queue Management": 0.01,
        "Lighting Failure": 0.01,
        "Parking Issue": 0.01,
        "Return Issue": 0.01,
    }
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


BASE_WEIGHTS = _base_category_weights()


def _sample_category(rng: np.random.Generator, weights: dict[str, float] | None = None) -> str:
    categories = list((weights or BASE_WEIGHTS).keys())
    probs = list((weights or BASE_WEIGHTS).values())
    return str(rng.choice(categories, p=probs))


def _build_weekly_schedule(store_master: pd.DataFrame, weeks: int, rng: np.random.Generator) -> list[dict]:
    base_end = pd.Timestamp.today().normalize().to_period("W-MON").start_time
    week_starts = [base_end - pd.Timedelta(weeks=offset) for offset in range(weeks - 1, -1, -1)]
    schedule: list[dict] = []
    for _, store in store_master.iterrows():
        size = store["store_size"]
        if size == "Small":
            low, high = 8, 13
        elif size == "Large":
            low, high = 24, 36
        else:
            low, high = 14, 24
        for week_start in week_starts:
            base_volume = int(rng.integers(low, high + 1))
            schedule.append(
                {
                    "store_id": store["store_id"],
                    "region": store["region"],
                    "store_size": size,
                    "week_start": week_start,
                    "base_volume": base_volume,
                }
            )
    return schedule


def _incident_targets(week_start: pd.Timestamp) -> dict[tuple[str, str], int]:
    targets: dict[tuple[str, str], int] = {}
    for profile in INCIDENT_PROFILES:
        start_week = week_start + pd.Timedelta(weeks=profile["start_offset_weeks"])
        if week_start >= start_week:
            offset = int((week_start - start_week) / pd.Timedelta(weeks=1))
            counts = profile["weekly_counts"]
            if 0 <= offset < len(counts):
                targets[(profile["store_id"], profile["issue_category"])] = counts[offset]
    return targets


def _incident_profile_lookup() -> dict[tuple[str, str], dict]:
    return {(item["store_id"], item["issue_category"]): item for item in INCIDENT_PROFILES}


def generate_synthetic_dataset(seed: int = 2026, weeks: int = 12) -> SyntheticDatasetBundle:
    rng = np.random.default_rng(seed)
    store_master = _build_store_master()
    schedule = _build_weekly_schedule(store_master, weeks=weeks, rng=rng)
    profile_lookup = _incident_profile_lookup()
    rows: list[dict] = []
    feedback_id = 1

    for item in schedule:
        store_id = item["store_id"]
        region = item["region"]
        week_start = pd.Timestamp(item["week_start"])
        base_volume = int(item["base_volume"])
        store_rows: list[dict] = []

        # Determine if this store has a seeded incident active this week.
        active_incident = None
        for profile in INCIDENT_PROFILES:
            if profile["store_id"] == store_id:
                incident_start = pd.Timestamp.today().normalize().to_period("W-MON").start_time - pd.Timedelta(weeks=profile["start_offset_weeks"])
                if week_start >= incident_start:
                    offset = int((week_start - incident_start) / pd.Timedelta(weeks=1))
                    if 0 <= offset < len(profile["weekly_counts"]):
                        active_incident = {
                            "profile": profile,
                            "offset": offset,
                            "count": profile["weekly_counts"][offset],
                        }
                        break

        incident_count = active_incident["count"] if active_incident else 0
        total_feedback = base_volume + incident_count
        if total_feedback < 1:
            total_feedback = 1

        # General background distribution.
        categories = list(BASE_WEIGHTS.keys())
        weights = list(BASE_WEIGHTS.values())
        week_categories = rng.choice(categories, size=total_feedback, p=weights)

        # If an incident is active, overwrite some records with the incident category.
        if active_incident:
            incident_category = active_incident["profile"]["issue_category"]
            incident_injections = min(incident_count, total_feedback)
            if incident_injections > 0:
                inject_idx = rng.choice(np.arange(total_feedback), size=incident_injections, replace=False)
                for idx in inject_idx:
                    week_categories[idx] = incident_category

        for idx in range(total_feedback):
            category = str(week_categories[idx])
            language = _choose_language(rng)
            channel = str(rng.choice(CHANNELS))
            day_offset = int(rng.integers(0, 7))
            hour = int(rng.integers(7, 22))
            minute = int(rng.integers(0, 60))
            timestamp = week_start + pd.Timedelta(days=day_offset, hours=hour, minutes=minute)

            feedback_text = _template_for(category, language, rng)
            if active_incident and category == active_incident["profile"]["issue_category"]:
                feedback_text = _template_for(category, language, rng)
                if active_incident["offset"] >= 2 and idx % 2 == 0:
                    feedback_text += " repeated complaint from another channel"

            policy = evaluate_policy(feedback_text, category)
            department = get_department_for_issue(category, feedback_text)
            severity = policy.severity if policy.severity != "MEDIUM" else get_default_severity(category)
            actionable = bool(policy.actionable)

            incident_tag = ""
            known_incident_flag = False
            if active_incident and category == active_incident["profile"]["issue_category"]:
                incident_tag = active_incident["profile"]["incident_tag"]
                known_incident_flag = True

            store_rows.append(
                {
                    "feedback_id": f"FB-{feedback_id:06d}",
                    "store_id": store_id,
                    "region": region,
                    "store_size": item["store_size"],
                    "timestamp": timestamp,
                    "channel": channel,
                    "feedback_text": feedback_text,
                    "issue_category": category,
                    "severity": severity,
                    "actionable": actionable,
                    "department": department,
                    "dataset_mode": "Synthetic Demonstration Data",
                    "incident_tag": incident_tag,
                    "known_incident_flag": known_incident_flag,
                    "language_hint": language,
                    "feedback_text_normalized": normalize_text(feedback_text),
                }
            )
            feedback_id += 1

        rows.extend(store_rows)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["timestamp", "store_id", "feedback_id"]).reset_index(drop=True)
    df["week_start"] = df["timestamp"].dt.to_period("W-MON").dt.start_time
    df["store_week_id"] = df["store_id"] + "-" + df["week_start"].dt.strftime("%Y-%m-%d")
    df["week_label"] = df["week_start"].dt.strftime("%Y-%m-%d")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return SyntheticDatasetBundle(dataframe=df, incident_profiles=INCIDENT_PROFILES, path=CSV_PATH)


@np.vectorize
def _safe_str(value: object) -> str:
    return str(value)


@lru_cache(maxsize=1)
def load_or_create_synthetic_dataset() -> SyntheticDatasetBundle:
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "week_start" not in df.columns and "timestamp" in df.columns:
            df["week_start"] = df["timestamp"].dt.to_period("W-MON").dt.start_time
        return SyntheticDatasetBundle(dataframe=df, incident_profiles=INCIDENT_PROFILES, path=CSV_PATH)
    return generate_synthetic_dataset()


def incident_profiles_dataframe() -> pd.DataFrame:
    today = pd.Timestamp.today().normalize().to_period("W-MON").start_time
    rows = []
    for profile in INCIDENT_PROFILES:
        start_week = today - pd.Timedelta(weeks=profile["start_offset_weeks"])
        escalation_week = today - pd.Timedelta(weeks=profile["known_escalation_offset_weeks"])
        rows.append(
            {
                **profile,
                "incident_start_date": start_week,
                "known_escalation_date": escalation_week,
            }
        )
    return pd.DataFrame(rows)
