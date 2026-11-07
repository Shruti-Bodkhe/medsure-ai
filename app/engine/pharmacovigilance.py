import re

# Simple deterministic rule base for common clinical drug conflicts
ALLERGY_CONFLICTS = {
    "ibuprofen": ["ibuprofen", "advil", "brufen", "nsaid", "aspirin", "diclofenac"],
    "aspirin": ["aspirin", "disprin", "nsaid", "ibuprofen"],
    "penicillin": ["penicillin", "amoxicillin", "ampicillin", "augmentin"],
    "sulfa": ["sulfamethoxazole", "sulfa", "sulfadiazine"]
}

CONDITION_CAUTIONS = {
    "asthma": ["aspirin", "ibuprofen", "nsaid", "beta-blocker"],
    "gastritis": ["ibuprofen", "aspirin", "diclofenac", "naproxen", "steroid"],
    "hypertension": ["pseudoephedrine", "decongestant", "high sodium"]
}

def check_pharmacovigilance_safety(query: str, patient_allergies: str, patient_conditions: str) -> list:
    """Scans the user query or extracted text for explicit allergy or condition contraindications."""
    warnings = []
    text_lower = query.lower()
    
    allergies = [a.strip().lower() for a in patient_allergies.split(",") if a.strip() and "none" not in a.lower()]
    conditions = [c.strip().lower() for c in patient_conditions.split(",") if c.strip() and "none" not in c.lower()]
    
    for allergy in allergies:
        triggers = ALLERGY_CONFLICTS.get(allergy, [allergy])
        for t in triggers:
            if t in text_lower:
                warnings.append(f"🚨 **CRITICAL ALLERGY ALERT:** You have a registered allergy to **{allergy.capitalize()}**. This query/substance involves or mentions '{t}', which is contraindicated.")
                
    for cond in conditions:
        risky_drugs = CONDITION_CAUTIONS.get(cond, [])
        for drug in risky_drugs:
            if drug in text_lower:
                warnings.append(f"⚠️ **CONDITION PRECAUTION:** Given your history of **{cond.capitalize()}**, substances like '{drug}' require strict physician supervision.")
                
    return warnings