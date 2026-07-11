"""
Grade the 5 Biomaterials PDF submissions with different models of various providers.
Providers: OpenAI, Anthropic, Google Gemini
Models tested:
- OpenAI: gpt-5, gpt-4o
- Anthropic: claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5
- Google Gemini: gemini-2.5-flash, gemini-2.5-pro
Output: Save results for all models in a structured format (JSON and CSV) for easy comparison.
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

logger = logging.getLogger(__name__)


def load_assignment() -> Dict:
    """Biomaterials.json is a bare list of question objects; grading_service
    expects an assignment dict with a "questions" key."""
    with open(os.path.join(ROOT_DIR, "test_files", "Biomaterials.json"), "r") as f:
        questions = json.load(f)
    return {"questions": questions}


def grade_biomaterials():
    max_runs = 3  # Run each model 3 times for consistency

    pdf_submissions = {
        "1": "biomaterials_submissions/1.pdf",
        "2": "biomaterials_submissions/2.pdf",
        "3": "biomaterials_submissions/3.pdf",
        "4": "biomaterials_submissions/4.pdf",
        "5": "biomaterials_submissions/5.pdf",
    }

    assignment_dict = load_assignment()

    models_to_test = [
        "gpt-5",
        "gpt-4o",
        "global.anthropic.claude-opus-4-6-v1",
        "global.anthropic.claude-sonnet-4-6",
        "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    results: Dict[str, List[Dict]] = {pdf_id: [] for pdf_id in pdf_submissions.keys()}

    for model in models_to_test:
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

                    total_time_taken = end_time - start_time

                    results[pdf_id].append({
                        "model": model,
                        "run": run,
                        "total_score": total_score,
                        "total_points": total_points,
                        "feedback_by_question": feedback_by_question,
                        "overall_feedback": overall_feedback,
                        "llm_call_time_taken": llm_call_time_taken,
                        "total_time_taken": total_time_taken,
                    })
        except Exception as e:
            logger.info(f"Error grading with model {model}: {str(e)}")
            for pdf_id in pdf_submissions.keys():
                results[pdf_id].append({
                    "model": model,
                    "error": str(e),
                })

    out_dir = os.path.join(BASE_DIR, "grading_results")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "grading_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    import pandas as pd
    all_results = []
    for pdf_id, pdf_results in results.items():
        for res in pdf_results:
            all_results.append({
                "pdf_id": pdf_id,
                **res,
            })
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(out_dir, "grading_results.csv"), index=False)


if __name__ == "__main__":
    grade_biomaterials()
