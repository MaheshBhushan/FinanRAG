import unittest
from pathlib import Path

from src.evaluation.diagnostics import abstention_message, diagnose, evaluate_abstention, is_abstention

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "abstention_cases.json"


class AbstentionTests(unittest.TestCase):
    def test_fixed_cases_abstain_on_missing_and_conflicting_sources(self):
        report = evaluate_abstention(FIXTURES)
        by_id = {row["id"]: row for row in report["cases"]}
        self.assertEqual(report["summary"]["accuracy"], 1.0)
        self.assertEqual(report["summary"]["invented_answers"], 0)
        self.assertEqual(report["summary"]["evidence_link_failures"], 0)
        self.assertEqual(by_id["missing-year"]["abstain_reasons"], ["top_score_below_threshold"])
        self.assertEqual(by_id["missing-topic"]["abstain_reasons"], ["no_term_overlap"])
        self.assertEqual(by_id["empty-index"]["abstain_reasons"], ["no_chunks_retrieved"])
        self.assertEqual(by_id["conflicting-revenue"]["abstain_reasons"], ["conflicting_sources"])

    def test_diagnostics_link_evidence_and_explain_conflict(self):
        retrieved = [
            {"id": "a", "source": "annual.pdf", "score": 0.7, "text": "Revenue was $4.2 billion in 2023."},
            {"id": "b", "source": "press.html", "score": 0.6, "text": "2023 revenue was $3.9 billion."},
        ]
        diag = diagnose("What was revenue in 2023?", retrieved)
        self.assertEqual(diag["supporting_sources"], ["annual.pdf", "press.html"])
        self.assertEqual(diag["score_gap"], 0.1)
        self.assertEqual(diag["evidence"][0]["figures"], ["$4.2billion"])
        message = abstention_message(diag)
        self.assertTrue(is_abstention(message))
        self.assertIn("annual.pdf reports $4.2billion", message)

    def test_generator_abstains_without_calling_llm(self):
        from src.generation.generator import generate_answer
        from src.retrieval.retriever import RetrievedChunk

        answer = generate_answer("What is Fabrikam's current ratio?", [RetrievedChunk("x", "Unrelated tax text.", "tax.html", 0.1)])
        self.assertTrue(is_abstention(answer.text))
        self.assertEqual(answer.sources, [])
        self.assertTrue(answer.diagnostics["abstain"])


if __name__ == "__main__":
    unittest.main()
