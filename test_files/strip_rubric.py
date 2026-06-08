import json
import copy

INPUT_FILE = "AssignmentRahul.json"
OUTPUT_FILE = "AssignmentRahul_noRubric.json"

FIELDS_TO_CLEAR = {"rubric", "correctAnswer"}


def clear_fields(obj):
    if isinstance(obj, dict):
        for key in obj:
            if key in FIELDS_TO_CLEAR:
                obj[key] = ""
            else:
                clear_fields(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            clear_fields(item)


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

data_copy = copy.deepcopy(data)
clear_fields(data_copy)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data_copy, f, indent=4, ensure_ascii=False)

print(f"Saved cleaned assignment to {OUTPUT_FILE}")
