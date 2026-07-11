"""
Adds human grades (3 graders, per question) into a copy of
grading_results.csv / grading_results.json.

Human grades source: test_files/biomaterials_human_grades.csv
(whitespace-delimited, columns: question student pdf_id grader_1 grader_2 grader_3)

New data added per pdf_id:
  grader_1_total_score / grader_2_total_score / grader_3_total_score
      – sum of that grader's scores across all 5 questions
  total_grade
      – mean of the three graders' totals (out of total_points = 10)
  human_grades_by_question
      – JSON with each grader's score per question, e.g.
        {"1": {"grader_1": 2, "grader_2": 2, "grader_3": 2}, ...}

Usage:
  python3 add_human_grades.py
Output:
  grading_results/grading_results_with_human_score.csv
  grading_results/grading_results_with_human_score.json
"""

import csv
import json
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

HUMAN_GRADES_CSV = os.path.join(ROOT_DIR, "test_files", "biomaterials_human_grades.csv")
INPUT_CSV = os.path.join(BASE_DIR, "grading_results", "grading_results.csv")
INPUT_JSON = os.path.join(BASE_DIR, "grading_results", "grading_results.json")

OUTPUT_CSV = os.path.join(BASE_DIR, "grading_results", "grading_results_with_human_score.csv")
OUTPUT_JSON = os.path.join(BASE_DIR, "grading_results", "grading_results_with_human_score.json")

GRADERS = ["grader_1", "grader_2", "grader_3"]


def load_human_grades():
    """
    Returns a dict keyed by pdf_id (str):
      {
        "student_name": str,
        "by_question": { "1": {"grader_1": .., "grader_2": .., "grader_3": ..}, ... },
        "grader_1_total_score": float,
        "grader_2_total_score": float,
        "grader_3_total_score": float,
        "total_grade": float,
      }
    """
    df = pd.read_csv(HUMAN_GRADES_CSV, sep=r"\s+")

    human_grades = {}
    for pdf_id, group in df.groupby("pdf_id"):
        pdf_id = str(pdf_id)
        by_question = {}
        totals = {g: 0.0 for g in GRADERS}

        for _, row in group.iterrows():
            q_id = str(row["question"])
            by_question[q_id] = {g: float(row[g]) for g in GRADERS}
            for g in GRADERS:
                totals[g] += float(row[g])

        human_grades[pdf_id] = {
            "student_name": group["student"].iloc[0],
            "by_question": by_question,
            "grader_1_total_score": totals["grader_1"],
            "grader_2_total_score": totals["grader_2"],
            "grader_3_total_score": totals["grader_3"],
            "total_grade": round(sum(totals.values()) / len(GRADERS), 2),
        }

    return human_grades


def update_csv(human_grades):
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    new_fieldnames = fieldnames + [
        "student_name",
        "grader_1_total_score",
        "grader_2_total_score",
        "grader_3_total_score",
        "human_grades_by_question",
        "total_grade",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()

        for row in rows:
            pdf_id = str(row["pdf_id"])
            hg = human_grades.get(pdf_id)

            if hg:
                row["student_name"] = hg["student_name"]
                row["grader_1_total_score"] = hg["grader_1_total_score"]
                row["grader_2_total_score"] = hg["grader_2_total_score"]
                row["grader_3_total_score"] = hg["grader_3_total_score"]
                row["human_grades_by_question"] = json.dumps(hg["by_question"])
                row["total_grade"] = hg["total_grade"]
            else:
                row["student_name"] = ""
                row["grader_1_total_score"] = ""
                row["grader_2_total_score"] = ""
                row["grader_3_total_score"] = ""
                row["human_grades_by_question"] = json.dumps({})
                row["total_grade"] = ""

            writer.writerow(row)

    print(f"CSV written: {OUTPUT_CSV}")
    print(f"  Rows processed: {len(rows)}")


def update_json(human_grades):
    with open(INPUT_JSON, encoding="utf-8") as f:
        data = json.load(f)

    new_data = {}
    for pdf_id, runs in data.items():
        hg = human_grades.get(str(pdf_id), {})
        new_data[pdf_id] = {
            "student_name": hg.get("student_name", ""),
            "total_grade": hg.get("total_grade"),
            "grader_1_total_score": hg.get("grader_1_total_score"),
            "grader_2_total_score": hg.get("grader_2_total_score"),
            "grader_3_total_score": hg.get("grader_3_total_score"),
            "human_grades_by_question": hg.get("by_question", {}),
            "runs": runs,
        }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4)

    print(f"JSON written: {OUTPUT_JSON}")
    grades_written = {pdf_id: new_data[pdf_id]["total_grade"] for pdf_id in new_data}
    print(f"  Grades written: {grades_written}")


def main():
    human_grades = load_human_grades()
    update_csv(human_grades)
    update_json(human_grades)


if __name__ == "__main__":
    main()
