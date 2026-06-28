HR_KEYWORDS = {
    "leave",
    "salary",
    "benefits",
    "employee",
    "holiday",
    "attendance",
    "travel",
    "expense",
    "probation",
    "performance",
    "promotion",
    "conduct",
    "security",
    "posh",
    "joining",
    "resignation",
    "termination",
    "insurance",
    "bonus",
    "policy",
    "hr",
    "wfh",
    "work from home",
    "maternity",
    "paternity"
}

def is_hr_question(question: str):
    question = question.lower()

    return any(word in question for word in HR_KEYWORDS)