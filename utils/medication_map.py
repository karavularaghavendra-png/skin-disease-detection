"""Over-the-counter medication guidance per disease class."""

MEDICATION_MAP: dict[str, list[dict]] = {
    "acne": [
        {
            "name": "Benzoyl Peroxide 2.5% gel",
            "use": "Kills acne-causing bacteria, reduces inflammation",
        },
        {
            "name": "Salicylic Acid cleanser",
            "use": "Unclogs pores, removes dead skin cells",
        },
        {
            "name": "Adapalene 0.1% gel (Differin)",
            "use": "Retinoid that prevents new comedones",
        },
    ],
    "eczema": [
        {
            "name": "1% Hydrocortisone cream",
            "use": "Reduces itching and mild inflammation",
        },
        {
            "name": "Cerave Moisturising Cream",
            "use": "Restores skin barrier with ceramides",
        },
        {"name": "Antihistamine (Loratadine 10mg)", "use": "Reduces allergic itching"},
    ],
    "fungal": [
        {
            "name": "Clotrimazole 1% cream",
            "use": "Antifungal — apply twice daily for 4 weeks",
        },
        {
            "name": "Miconazole nitrate cream",
            "use": "Broad-spectrum antifungal treatment",
        },
        {
            "name": "Terbinafine 1% cream (Lamisil AT)",
            "use": "Highly effective for ringworm",
        },
    ],
    "psoriasis": [
        {"name": "Coal Tar 2% cream", "use": "Slows skin cell growth, reduces scaling"},
        {"name": "Salicylic Acid ointment", "use": "Softens and removes plaques"},
        {
            "name": "1% Hydrocortisone cream",
            "use": "Reduces redness and itch for mild cases",
        },
    ],
    "normal": [
        {
            "name": "SPF 30+ sunscreen (daily)",
            "use": "Prevents UV damage and premature ageing",
        },
        {
            "name": "Moisturiser with hyaluronic acid",
            "use": "Maintains skin hydration and elasticity",
        },
    ],
}
