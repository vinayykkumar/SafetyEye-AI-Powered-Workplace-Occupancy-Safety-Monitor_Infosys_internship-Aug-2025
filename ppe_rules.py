# ppe_rules.py

# Mapping of detected classes to violation messages
violation_classes = {
    "NO-Gloves": "Gloves missing",
    "NO-Goggles": "Goggles missing",
    "NO-Hardhat": "Helmet missing",
    "NO-Mask": "Mask missing",
    "NO-Safety Vest": "Safety vest missing"
}

# Optional: list of required PPE classes (positive detection)
required_ppe = ["Gloves", "Goggles", "Hardhat", "Mask", "Safety Vest"]

def check_violations(detected_classes):
    """
    Input: detected_classes - list of class names detected by YOLOv8
    Output: list of violation messages
    """
    violations = []
    for cls in detected_classes:
        if cls in violation_classes:
            violations.append(violation_classes[cls])
    return violations
