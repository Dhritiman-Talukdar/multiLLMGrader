"""
Re-run grading for Gemini models only, to replace the corrupted 0-score results
caused by a parsing bug in grading_service._parse_bulk_grading_response (it
mis-routed grade_pdf_direct's JSON "grades" response through a bracket-tag
regex parser meant only for grade_submission's Gemini prompt format).

Merges the corrected Gemini entries into grading_results.json/csv in place.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

import dotenv

dotenv.load_dotenv(os.path.join(ROOT_DIR, ".env"))

import json
import time
from typing import Dict, List
from config import logging

from grading_service import LLMGrader
from grade_biomaterials import load_assignment

logger = logging.getLogger(__name__)

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]


def rerun_gemini():
    max_runs = 3

    pdf_submissions = {
        "1": "biomaterials_submissions/1.pdf",
        "2": "biomaterials_submissions/2.pdf",
        "3": "biomaterials_submissions/3.pdf",
        "4": "biomaterials_submissions/4.pdf",
        "5": "biomaterials_submissions/5.pdf",
    }

    assignment_dict = load_assignment()

    new_results: Dict[str, List[Dict]] = {pdf_id: [] for pdf_id in pdf_submissions.keys()}

    for model in GEMINI_MODELS:
        try:
            grader = LLMGrader(model=model)
            for pdf_id, pdf_s3_key in pdf_submissions.items():
                for run in range(1, max_runs + 1):
                    start_time = time.time()
                    (
                        total_score,
                        total_points,
                        feedback_by_question,
                        overall_feedback,
                        llm_call_time_taken,
                    ) = grader.grade_pdf_direct(
                        assignment=assignment_dict,
                        pdf_s3_key=pdf_s3_key,
                        options={},
                    )
                    end_time = time.time()

                    new_results[pdf_id].append({
                        "model": model,
                        "run": run,
                        "total_score": total_score,
                        "total_points": total_points,
                        "feedback_by_question": feedback_by_question,
                        "overall_feedback": overall_feedback,
                        "llm_call_time_taken": llm_call_time_taken,
                        "total_time_taken": end_time - start_time,
                    })
        except Exception as e:
            logger.info(f"Error grading with model {model}: {str(e)}")
            for pdf_id in pdf_submissions.keys():
                new_results[pdf_id].append({
                    "model": model,
                    "error": str(e),
                })

    out_dir = os.path.join(BASE_DIR, "grading_results")
    results_json_path = os.path.join(out_dir, "grading_results.json")

    with open(results_json_path, "r") as f:
        existing_results: Dict[str, List[Dict]] = json.load(f)

    for pdf_id, entries in existing_results.items():
        non_gemini = [e for e in entries if e.get("model") not in GEMINI_MODELS]
        existing_results[pdf_id] = non_gemini + new_results.get(pdf_id, [])

    with open(results_json_path, "w") as f:
        json.dump(existing_results, f, indent=4)

    import pandas as pd
    all_results = []
    for pdf_id, pdf_results in existing_results.items():
        for res in pdf_results:
            all_results.append({"pdf_id": pdf_id, **res})
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(out_dir, "grading_results.csv"), index=False)

    print("Gemini re-run complete and merged into grading_results.json/csv")


if __name__ == "__main__":
    rerun_gemini()
