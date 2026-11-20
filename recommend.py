"""
recommend.py
------------
Simple rule-based recommendation engine that maps a predicted skin type
to a short skincare summary and two representative product categories
(Representative Product A / B, matching the project's block diagram).

Product links point to brand product pages. Advice is educational and does
not claim that diet, hydration, or weather can diagnose or treat skin issues.
"""

from copy import copy

PROFILE_OPTIONS = {
    "image": "Use photo estimate",
    "oily": "Oily (self-assessed)",
    "dry": "Dry (self-assessed)",
    "normal": "Normal (self-assessed)",
    "combination": "Combination (self-assessed)",
    "sensitive": "Sensitive (self-assessed)",
}

RECOMMENDATIONS = {
    "oily": {
        "summary": (
            "Your skin shows characteristics typical of an OILY skin type "
            "(visible shine, larger pores, tendency toward breakouts)."
        ),
        "tips": [
            "Cleanse twice daily with a gentle, oil-free foaming cleanser.",
            "Use lightweight, non-comedogenic (won't clog pores) moisturizers.",
            "Look for ingredients like salicylic acid or niacinamide.",
            "Avoid heavy creams and alcohol-based products that can irritate skin.",
        ],
        "product_a": "Foaming Facial Cleanser",
        "product_a_url": "https://www.cerave.com/skincare/cleansers/foaming-facial-cleanser",
        "product_b": "PM Facial Moisturizing Lotion",
        "product_b_url": "https://www.cerave.com/skincare/moisturizers/pm-facial-moisturizing-lotion",
        "hydration_message": "Sip water regularly through the day; hydration needs vary by person and activity.",
        "diet_tips": [
            "Choose a varied mix of vegetables, fruit, whole grains, protein, and unsaturated fats.",
            "No single food or supplement is proven to change your skin type.",
        ],
        "morning_tasks": [
            "Cleanse gently; skip a second cleanse if your skin feels stripped.",
            "Apply a lightweight moisturizer that feels comfortable.",
            "Finish with broad-spectrum SPF 30 or higher.",
        ],
        "evening_tasks": [
            "Remove sunscreen and cleanse gently.",
            "Apply a light, non-comedogenic moisturizer.",
            "Avoid scrubbing or adding several new active products at once.",
        ],
    },
    "dry": {
        "summary": (
            "Your skin shows characteristics typical of a DRY skin type "
            "(tightness, flaking, reduced natural shine)."
        ),
        "tips": [
            "Use a hydrating, cream-based cleanser instead of foaming ones.",
            "Apply a rich moisturizer with ceramides or hyaluronic acid.",
            "Avoid long hot showers and harsh, alcohol-heavy toners.",
            "Consider a humidifier in dry weather to reduce moisture loss.",
        ],
        "product_a": "Hydrating Facial Cleanser",
        "product_a_url": "https://www.cerave.com/skincare/cleansers/hydrating-facial-cleanser",
        "product_b": "Moisturizing Cream",
        "product_b_url": "https://www.cerave.com/skincare/moisturizers/moisturizing-cream",
        "hydration_message": "Keep water nearby and drink to thirst; topical moisturizer helps support the skin barrier.",
        "diet_tips": [
            "Aim for a varied, balanced eating pattern rather than restrictive skin-type diets.",
            "Include vegetables, fruit, whole grains, protein, and sources of unsaturated fats.",
        ],
        "morning_tasks": [
            "Use a gentle, non-stripping cleanser or rinse with lukewarm water.",
            "Apply moisturizer while skin is slightly damp.",
            "Use broad-spectrum SPF 30 or higher.",
        ],
        "evening_tasks": [
            "Remove sunscreen and wash with a gentle cleanser.",
            "Apply a comfortable cream moisturizer.",
            "Keep showers lukewarm and avoid rough exfoliation.",
        ],
    },
    "normal": {
        "summary": (
            "Your skin shows characteristics typical of a NORMAL skin type "
            "(balanced oil/moisture levels, few visible concerns)."
        ),
        "tips": [
            "Maintain balance with a gentle daily cleanser.",
            "Use a lightweight moisturizer to keep hydration levels stable.",
            "Apply sunscreen (SPF 30+) daily to protect your skin barrier.",
            "Avoid over-exfoliating, which can disrupt naturally balanced skin.",
        ],
        "product_a": "Hydrating Facial Cleanser",
        "product_a_url": "https://www.cerave.com/skincare/cleansers/hydrating-facial-cleanser",
        "product_b": "AM Facial Moisturizing Lotion with SPF 30",
        "product_b_url": "https://www.cerave.com/skincare/moisturizers/am-facial-moisturizing-lotion-with-sunscreen",
        "hydration_message": "Drink water regularly according to thirst, activity, and your own health needs.",
        "diet_tips": [
            "A varied mix of vegetables, fruit, whole grains, protein, and unsaturated fats supports general wellbeing.",
            "There is no one-size-fits-all diet for normal skin.",
        ],
        "morning_tasks": [
            "Wash with a gentle cleanser.",
            "Apply a lightweight moisturizer if needed.",
            "Finish with broad-spectrum SPF 30 or higher.",
        ],
        "evening_tasks": [
            "Remove sunscreen and cleanse gently.",
            "Apply a simple moisturizer.",
            "Keep the routine consistent rather than adding unnecessary steps.",
        ],
    },
    "combination": {
        "summary": (
            "Combination skin often feels oilier through the forehead, nose, and chin, "
            "while the cheeks feel balanced or dry. This profile is self-selected, not photo-predicted."
        ),
        "tips": [
            "Use a gentle cleanser over the whole face; avoid trying to dry out the T-zone.",
            "Use a light moisturizer where skin feels oily and add more cream only to dry areas.",
            "Use broad-spectrum SPF 30 or higher every morning.",
            "Change one product at a time and watch how different areas respond.",
        ],
        "product_a": "Foaming Facial Cleanser",
        "product_a_url": "https://www.cerave.com/skincare/cleansers/foaming-facial-cleanser",
        "product_b": "PM Facial Moisturizing Lotion",
        "product_b_url": "https://www.cerave.com/skincare/moisturizers/pm-facial-moisturizing-lotion",
        "hydration_message": "Drink according to thirst and your activity; use moisturizer on areas that feel dry.",
        "diet_tips": [
            "Choose varied meals with vegetables, fruit, whole grains, protein, and unsaturated fats.",
            "Food choices do not determine whether skin is combination type.",
        ],
        "morning_tasks": [
            "Cleanse gently, especially around the T-zone.",
            "Use light moisturizer on oily areas and more on dry patches if needed.",
            "Apply broad-spectrum SPF 30 or higher.",
        ],
        "evening_tasks": [
            "Remove sunscreen and cleanse the whole face gently.",
            "Apply a light moisturizer, adding extra only to dry-feeling areas.",
            "Skip harsh scrubs and avoid over-cleansing shiny areas.",
        ],
    },
    "sensitive": {
        "summary": (
            "Sensitive describes a self-selected skin profile that may react easily; "
            "it is not a category predicted by the photo model."
        ),
        "tips": [
            "Keep the routine simple and choose fragrance-free products when possible.",
            "Patch-test a new product on a small area before wider use.",
            "Pause products that sting or cause persistent irritation.",
            "Seek professional advice for ongoing or severe reactions.",
        ],
        "product_a": "Hydrating Facial Cleanser",
        "product_a_url": "https://www.cerave.com/skincare/cleansers/hydrating-facial-cleanser",
        "product_b": "Moisturizing Cream",
        "product_b_url": "https://www.cerave.com/skincare/moisturizers/moisturizing-cream",
        "hydration_message": "Drink to thirst and follow any fluid guidance from your healthcare professional.",
        "diet_tips": [
            "A balanced, varied diet supports general health; no restrictive diet is prescribed for this profile.",
            "If you suspect a food reaction, discuss it with a qualified healthcare professional.",
        ],
        "morning_tasks": [
            "Rinse or use a gentle fragrance-free cleanser if tolerated.",
            "Apply a simple moisturizer that you have already patch-tested.",
            "Use broad-spectrum SPF 30 or higher that your skin tolerates.",
        ],
        "evening_tasks": [
            "Remove sunscreen gently with a familiar cleanser.",
            "Apply your tested moisturizer without adding new actives.",
            "Note any irritation and stop products that trigger a reaction.",
        ],
    },
}

DISCLAIMER = (
    "This tool is a prototype for general guidance only and does not "
    "replace professional dermatological advice. Please consult a "
    "dermatologist for persistent or serious skin concerns."
)

WEATHER_CONDITIONS = {
    "sunny-hot": "Sunny and hot",
    "cold-windy": "Cold and windy",
    "dry-air": "Dry air / indoor heating",
    "humid": "Humid",
    "mild": "Mild weather",
}

WEATHER_ADVICE = {
    "sunny-hot": "Use broad-spectrum SPF 30+ and reapply as directed; choose a lighter moisturizer if heat feels uncomfortable.",
    "cold-windy": "Wind and cold can feel drying. Use a comfortable moisturizer and protect exposed skin from the elements.",
    "dry-air": "Low humidity and indoor heating can feel drying. Apply moisturizer after washing and consider a room humidifier.",
    "humid": "In humid weather, a light, non-comedogenic moisturizer may feel more comfortable; cleanse gently after sweating.",
    "mild": "Keep your routine steady and use broad-spectrum SPF 30+ during daylight exposure.",
}

WEATHER_MOISTURIZERS = {
    "oily": {
        "sunny-hot": ("AM Facial Moisturizing Lotion with SPF 30", "https://www.cerave.com/skincare/moisturizers/am-facial-moisturizing-lotion-with-sunscreen"),
        "cold-windy": ("Moisturizing Cream", "https://www.cerave.com/skincare/moisturizers/moisturizing-cream"),
        "dry-air": ("Moisturizing Cream", "https://www.cerave.com/skincare/moisturizers/moisturizing-cream"),
    },
    "dry": {
        "sunny-hot": ("AM Facial Moisturizing Lotion with SPF 30", "https://www.cerave.com/skincare/moisturizers/am-facial-moisturizing-lotion-with-sunscreen"),
        "humid": ("PM Facial Moisturizing Lotion", "https://www.cerave.com/skincare/moisturizers/pm-facial-moisturizing-lotion"),
    },
    "normal": {
        "cold-windy": ("Moisturizing Cream", "https://www.cerave.com/skincare/moisturizers/moisturizing-cream"),
        "dry-air": ("Moisturizing Cream", "https://www.cerave.com/skincare/moisturizers/moisturizing-cream"),
        "humid": ("PM Facial Moisturizing Lotion", "https://www.cerave.com/skincare/moisturizers/pm-facial-moisturizing-lotion"),
    },
    "combination": {
        "sunny-hot": ("AM Facial Moisturizing Lotion with SPF 30", "https://www.cerave.com/skincare/moisturizers/am-facial-moisturizing-lotion-with-sunscreen"),
        "cold-windy": ("Moisturizing Cream", "https://www.cerave.com/skincare/moisturizers/moisturizing-cream"),
        "dry-air": ("Moisturizing Cream", "https://www.cerave.com/skincare/moisturizers/moisturizing-cream"),
    },
    "sensitive": {
        "sunny-hot": ("AM Facial Moisturizing Lotion with SPF 30", "https://www.cerave.com/skincare/moisturizers/am-facial-moisturizing-lotion-with-sunscreen"),
        "humid": ("PM Facial Moisturizing Lotion", "https://www.cerave.com/skincare/moisturizers/pm-facial-moisturizing-lotion"),
    },
}


def get_recommendation(skin_type: str, weather: str = "mild") -> dict:
    """
    Returns the recommendation dict for a given skin type
    ('oily', 'dry', 'normal', 'combination', or 'sensitive'), adjusted for
    the selected weather condition. Raises KeyError for unsupported inputs.
    """
    key = skin_type.strip().lower()
    if key not in RECOMMENDATIONS:
        raise KeyError(f"Unknown skin type: {skin_type}")
    weather_key = weather.strip().lower()
    weather_key = next(
        (value for value, label in WEATHER_CONDITIONS.items() if label.lower() == weather_key),
        weather_key,
    )
    if weather_key not in WEATHER_CONDITIONS:
        raise KeyError(f"Unknown weather condition: {weather}")

    recommendation = copy(RECOMMENDATIONS[key])
    moisturizer = WEATHER_MOISTURIZERS.get(key, {}).get(weather_key)
    if moisturizer:
        recommendation["product_b"], recommendation["product_b_url"] = moisturizer
    return recommendation


def resolve_profile(profile_choice: str, predicted_skin_type: str) -> tuple[str, bool]:
    """Return the selected profile and whether it was manually chosen."""
    key = profile_choice.strip().lower()
    if key == "image":
        return predicted_skin_type.strip().lower(), False
    if key not in RECOMMENDATIONS:
        raise KeyError(f"Unknown skin profile: {profile_choice}")
    return key, True


def get_weather_advice(condition: str, skin_type: str) -> str:
    """Return general weather-aware advice for a supported skin type."""
    key = condition.strip().lower()
    key = next(
        (value for value, label in WEATHER_CONDITIONS.items() if label.lower() == key),
        key,
    )
    advice = WEATHER_ADVICE.get(key, WEATHER_ADVICE["mild"])
    skin = skin_type.strip().lower()
    if key in {"cold-windy", "dry-air"} and skin == "dry":
        return f"{advice} Apply a fragrance-free moisturizer while skin is slightly damp if that suits you."
    if key == "humid" and skin == "oily":
        return f"{advice} Avoid adding extra exfoliating steps just because of shine."
    return advice
