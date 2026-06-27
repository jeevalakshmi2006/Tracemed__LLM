import urllib.request
import urllib.error
import json
import os
# ─────────────────────────────────────────
#  GROQ API  (free — uses Llama 3)
# ─────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama3-8b-8192"           # free model on Groq

# ─────────────────────────────────────────
#  FEATURE RANGES  (for denormalizing)
# ─────────────────────────────────────────
FEATURE_RANGES = {
    'age':      {'min': 29,  'max': 77,   'label': 'Age',                  'unit': 'years'},
    'sex':      {'min': 0,   'max': 1,    'label': 'Sex',                  'unit': '(1=Male, 0=Female)'},
    'cp':       {'min': 1,   'max': 4,    'label': 'Chest Pain Type',      'unit': '(1=typical, 4=asymptomatic)'},
    'trestbps': {'min': 94,  'max': 200,  'label': 'Blood Pressure',       'unit': 'mmHg'},
    'chol':     {'min': 126, 'max': 564,  'label': 'Cholesterol',          'unit': 'mg/dL'},
    'fbs':      {'min': 0,   'max': 1,    'label': 'Fasting Blood Sugar',  'unit': '(1=>120mg/dL)'},
    'restecg':  {'min': 0,   'max': 2,    'label': 'Resting ECG',          'unit': ''},
    'thalach':  {'min': 71,  'max': 202,  'label': 'Max Heart Rate',       'unit': 'bpm'},
    'exang':    {'min': 0,   'max': 1,    'label': 'Exercise Angina',      'unit': '(1=Yes, 0=No)'},
    'oldpeak':  {'min': 0,   'max': 6.2,  'label': 'ST Depression',        'unit': ''},
    'slope':    {'min': 1,   'max': 3,    'label': 'Slope',                'unit': ''},
    'ca':       {'min': 0,   'max': 3,    'label': 'Major Vessels',        'unit': 'count'},
    'thal':     {'min': 3,   'max': 7,    'label': 'Thal',                 'unit': '(3=normal, 6=fixed, 7=reversible)'},
}

FEATURE_KEYS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]


# ─────────────────────────────────────────
#  DENORMALIZE  (0–1 back to real value)
# ─────────────────────────────────────────
def denormalize_value(norm_val, key):
    r   = FEATURE_RANGES[key]
    raw = norm_val * (r['max'] - r['min']) + r['min']
    return round(raw, 1)


# ─────────────────────────────────────────
#  BUILD PROMPT
# ─────────────────────────────────────────
def build_prompt(risk_score, contributions, patient_row):
    lines = []
    for c in contributions:
        key       = c['feature_key']
        raw_val   = denormalize_value(c['patient_value'], key)
        direction = "increases" if c['contribution'] > 0 else "decreases"
        unit      = FEATURE_RANGES[key]['unit']
        lines.append(
            f"- {c['feature_name']}: {raw_val} {unit} "
            f"-> {direction} risk by {abs(c['contribution'])}%"
        )

    factors_text = "\n".join(lines)

    prompt = f"""You are a clinical AI assistant helping doctors understand heart disease risk predictions.

A patient has been assessed with a heart disease risk score of {risk_score}%.

Top contributing factors from the AI model:
{factors_text}

Write a clear, professional clinical explanation (3-4 sentences) for the doctor that:
1. States the overall risk level
2. Explains which factors are most concerning and why
3. Mentions any protective factors if present
4. Suggests what the doctor should focus on

Use medical terminology but keep it readable. Write in paragraph form only."""

    return prompt


# ─────────────────────────────────────────
#  CALL GROQ API  (free)
# ─────────────────────────────────────────
def call_groq(prompt):
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a clinical AI assistant. Always respond in clear paragraph form."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    try:
        data    = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            GROQ_API_URL, data=data, headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        print(f"Groq API Error {e.code}: {error_body}")
        return None
    except Exception as e:
        print(f"Connection Error: {str(e)}")
        return None


# ─────────────────────────────────────────
#  FALLBACK EXPLANATION  (no API needed)
# ─────────────────────────────────────────
def get_fallback_explanation(risk_score, contributions, mode="doctor"):
    level = "high" if risk_score >= 70 else "moderate" if risk_score >= 40 else "low"

    positives = [c for c in contributions if c['contribution'] > 0]
    negatives = [c for c in contributions if c['contribution'] < 0]

    top_pos = positives[0] if positives else None
    top_neg = negatives[0] if negatives else None

    if mode == "doctor":
        text = f"This patient presents a {level} cardiovascular risk profile with a predicted score of {risk_score}%. "
        if top_pos:
            text += (f"The primary risk driver is {top_pos['feature_name']}, "
                     f"contributing +{abs(top_pos['contribution'])}% to the overall risk elevation, "
                     f"suggesting significant cardiovascular burden. ")
        if top_neg:
            text += (f"A mitigating factor is {top_neg['feature_name']}, "
                     f"which reduces risk by {abs(top_neg['contribution'])}%. ")
        text += "Immediate clinical evaluation and cardiology referral is strongly recommended."
    else:
        text = f"Your heart health check shows a {level} risk level ({risk_score}%). "
        if top_pos:
            text += (f"The main factor affecting your score is {top_pos['feature_name'].lower()}, "
                     f"which is pushing your risk higher. ")
        if top_neg:
            text += (f"The good news is that your {top_neg['feature_name'].lower()} "
                     f"is helping keep your risk lower. ")
        text += "Please speak with your doctor soon to discuss ways to improve your heart health."

    return text


# ─────────────────────────────────────────
#  MAIN FUNCTIONS  (called from dashboard)
# ─────────────────────────────────────────
def get_clinical_explanation(risk_score, contributions, patient_row):
    if GROQ_API_KEY == "your-groq-api-key-here":
        print("  [No API key - using fallback explanation]")
        return get_fallback_explanation(risk_score, contributions, mode="doctor")

    prompt = build_prompt(risk_score, contributions, patient_row)
    result = call_groq(prompt)

    if result is None:
        print("  [API failed - using fallback explanation]")
        return get_fallback_explanation(risk_score, contributions, mode="doctor")

    return result


def get_patient_explanation(risk_score, contributions, patient_row):
    if GROQ_API_KEY == "your-groq-api-key-here":
        print("  [No API key - using fallback explanation]")
        return get_fallback_explanation(risk_score, contributions, mode="patient")

    patient_prompt = build_prompt(risk_score, contributions, patient_row)
    patient_prompt += """

Now rewrite this in very simple language for a patient with no medical background.
Avoid all medical jargon. Be warm, empathetic and encouraging. 3-4 sentences only."""

    result = call_groq(patient_prompt)

    if result is None:
        print("  [API failed - using fallback explanation]")
        return get_fallback_explanation(risk_score, contributions, mode="patient")

    return result