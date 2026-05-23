"""
Disease Knowledge Base for Skin Disease Detection.
Single source of truth for all disease info used by app.py, api.py, and tests.
"""

# ─────────────────────────────────────────────────────────────
# Disease Database — all 5 classes
# ─────────────────────────────────────────────────────────────
DISEASE_DATABASE = {
    "acne": {
        "display_name": "Acne",
        "description": (
            "A common skin condition that occurs when hair "
            "follicles become plugged with oil and dead skin cells."
        ),
        "icon": "🧴",
        "specialist": "General Dermatologist",
        "base_severity": 2,
        "symptoms": [
            "Red or inflamed bumps on the skin",
            "Whiteheads and blackheads",
            "Pus-filled pimples or cysts",
            "Oily skin with enlarged pores",
            "Scarring or dark spots from healed lesions",
        ],
        "recommendations": [
            "Use a gentle, non-comedogenic cleanser twice daily",
            "Apply Benzoyl Peroxide (2.5–5%) or Salicylic Acid topically",
            "Avoid touching or picking at affected areas",
            "Consider Adapalene gel (OTC retinoid) for persistent acne",
            "Consult a dermatologist if acne is severe or cystic",
        ],
    },
    "eczema": {
        "display_name": "Eczema (Atopic Dermatitis)",
        "description": (
            "A condition that makes your skin red and itchy. "
            "Common in children but can occur at any age."
        ),
        "icon": "💧",
        "specialist": "Dermatologist / Allergist",
        "base_severity": 4,
        "symptoms": [
            "Dry, itchy, and inflamed skin patches",
            "Red or brownish-grey discoloration",
            "Small raised bumps that may leak fluid",
            "Thickened, cracked, or scaly skin",
            "Sensitive or swollen skin from scratching",
        ],
        "recommendations": [
            "Keep skin well-moisturized (CeraVe, Cetaphil, or Vanicream)",
            "Apply Hydrocortisone cream (1%) for mild flare-ups",
            "Avoid harsh soaps, hot showers, and known allergens",
            "Wear soft, breathable clothing — avoid wool",
            "See a dermatologist for prescription options if OTC fails",
        ],
    },
    "fungal": {
        "display_name": "Fungal Infection (Ringworm / Tinea)",
        "description": "A common fungal skin infection caused by dermatophytes.",
        "icon": "🍄",
        "specialist": "General Practitioner / Dermatologist",
        "base_severity": 3,
        "symptoms": [
            "Ring-shaped, red, scaly rash with raised edges",
            "Intense itching or burning in the affected area",
            "Clear or scaly center within the ring",
            "Multiple rings that may overlap",
            "Blistering or oozing in severe cases",
        ],
        "recommendations": [
            "Apply OTC antifungal cream (Clotrimazole, Terbinafine, or Miconazole)",
            "Keep the area clean and dry at all times",
            "Avoid sharing towels, clothing, or personal items",
            "Continue treatment for 1–2 weeks after symptoms resolve",
            "See a doctor if the infection spreads or does not improve",
        ],
    },
    "normal": {
        "display_name": "Healthy / Normal Skin",
        "description": "No skin disease detected. Skin appears healthy.",
        "icon": "✅",
        "specialist": "None required",
        "base_severity": 0,
        "symptoms": [
            "No visible signs of disease or abnormality",
            "Even skin tone and smooth texture",
            "No inflammation, redness, or irritation",
        ],
        "recommendations": [
            "Continue your regular skincare routine",
            "Apply SPF 30+ sunscreen daily",
            "Stay hydrated — drink 8 glasses of water per day",
            "Eat a balanced diet rich in vitamins C and E",
            "See a dermatologist annually for a routine skin check",
        ],
    },
    "psoriasis": {
        "display_name": "Psoriasis",
        "description": (
            "A skin disease that causes red, itchy scaly patches "
            "on the knees, elbows, trunk and scalp."
        ),
        "icon": "🩹",
        "specialist": "Medical Dermatologist / Rheumatologist",
        "base_severity": 7,
        "symptoms": [
            "Thick, red patches covered with silvery-white scales",
            "Dry, cracked skin that may bleed",
            "Itching, burning, or soreness around patches",
            "Stiff or swollen joints (psoriatic arthritis risk)",
            "Pitting or ridging on fingernails or toenails",
        ],
        "recommendations": [
            "Use Salicylic Acid or Coal Tar shampoo / ointment",
            "Apply topical corticosteroids to reduce inflammation",
            "Keep skin moisturized daily to reduce scaling",
            "Controlled sun exposure may help — avoid sunburn",
            "Consult a dermatologist — prescription biologics may be needed",
        ],
    },
}

# Fallback for unrecognised disease names
_DEFAULT_INFO = {
    "display_name": "Unknown Condition",
    "description": "Could not identify the condition. Please consult a dermatologist.",
    "icon": "❓",
    "specialist": "Dermatologist",
    "base_severity": 5,
    "symptoms": [
        "Symptoms could not be determined from the image alone",
    ],
    "recommendations": [
        "Please consult a dermatologist for an accurate diagnosis",
        "Do not self-medicate based solely on this prediction",
    ],
}

MEDICAL_DISCLAIMER = (
    "⚠️ MEDICAL DISCLAIMER: This system is for educational and "
    "informational purposes only. It is NOT a substitute for "
    "professional medical advice, diagnosis, or treatment. Always "
    "consult a qualified dermatologist before taking any medication "
    "or starting any treatment."
)


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────


def get_disease_info(disease_name: str) -> dict:
    """
    Returns the info dict for a given disease name.
    Tries exact match first, then partial match.
    Falls back to _DEFAULT_INFO if not found.
    """
    key = disease_name.strip().lower()

    if key in DISEASE_DATABASE:
        return DISEASE_DATABASE[key]

    for db_key in DISEASE_DATABASE:
        if db_key in key or key in db_key:
            return DISEASE_DATABASE[db_key]

    return _DEFAULT_INFO


def get_severity(confidence: float, disease_name: str = "") -> tuple:
    """
    Returns (severity_label, colour_hex) based on confidence and disease base severity.

    Args:
        confidence:   float 0–100
        disease_name: optional — used to factor in disease base severity

    Returns:
        (severity_label, colour_hex)
    """
    if disease_name:
        info = get_disease_info(disease_name)
        base = info.get("base_severity", 0)
        if base == 0:
            return "Not Applicable", "#10B981"
        risk = (confidence / 100.0) * base
        if risk >= 5.0 or base >= 7:
            return "High Priority — Seek Medical Advice", "#EF4444"
        elif risk >= 2.5:
            return "Moderate — Monitor Closely", "#F59E0B"
        else:
            return "Low / Mild — Routine Care", "#3B82F6"
    else:
        # Fallback: confidence-only logic (used by api.py)
        if confidence >= 85.0:
            return "High Confidence", "#EF4444"
        elif confidence >= 60.0:
            return "Moderate Confidence", "#F59E0B"
        else:
            return "Low Confidence", "#3B82F6"


def get_disclaimer() -> str:
    """Returns the standard medical disclaimer."""
    return MEDICAL_DISCLAIMER
