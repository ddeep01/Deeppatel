import uuid


# =========================
# TOOL 1: SymptomChecker
# =========================
def symptom_checker(symptoms):
    """
    Input: list of symptoms
    Output: possible conditions + risk level
    """

    symptom_map = {
        "fever": ["Infection", "Flu", "COVID-19"],
        "cough": ["Common Cold", "Bronchitis", "Pneumonia"],
        "chest pain": ["Heart Disease", "Pneumonia"],
        "fatigue": ["Diabetes", "Anemia"],
        "thirst": ["Diabetes"],
        "headache": ["Migraine", "Stress"]
    }

    conditions = set()

    for s in symptoms:
        s = s.lower()
        if s in symptom_map:
            conditions.update(symptom_map[s])

    # Risk logic
    if "chest pain" in symptoms:
        risk = "high"
    elif len(symptoms) >= 3:
        risk = "medium"
    else:
        risk = "low"

    return {
        "tool": "SymptomChecker",
        "possible_conditions": list(conditions),
        "risk_level": risk
    }


# =========================
# TOOL 2: GetGuideline
# =========================
def get_guideline(condition):
    """
    Input: condition
    Output: medical guideline text
    """

    guidelines = {
        "diabetes": "Maintain blood sugar levels through diet, exercise, and medication. Regular monitoring is required.",
        "pneumonia": "Use antibiotics if bacterial. Rest, hydration, and medical supervision are recommended.",
        "covid-19": "Isolate, monitor oxygen levels, and seek medical help if symptoms worsen.",
        "heart disease": "Adopt healthy lifestyle, avoid smoking, and follow prescribed medications.",
    }

    condition = condition.lower()

    return {
        "tool": "GetGuideline",
        "guideline_text": guidelines.get(
            condition,
            "No guideline available. Please consult a doctor."
        )
    }


# =========================
# TOOL 3: CreateMedicalTicket
# =========================
def create_medical_ticket(summary, severity):
    """
    Input: summary + severity
    Output: ticket_id
    """

    ticket_id = str(uuid.uuid4())[:8]

    # Simulate ticket storage (can extend to DB)
    print(f"📌 Ticket Created | ID: {ticket_id} | Severity: {severity}")

    return {
        "tool": "CreateMedicalTicket",
        "ticket_id": ticket_id
    }