import unittest

from src.synthesis.citations import (audit_answer_sentences, check_citations, claim_evidence_map, extract_markers,
                                     strip_unresolved_markers)

PASSAGES = [
    {"evidence_id": "qdrant_1", "doc_title": "Aulani"},
    {"evidence_id": "qdrant_2", "doc_title": "Aulani"},
    {"evidence_id": "qdrant_3", "doc_title": "Disney's Saratoga Springs Resort &amp; Spa"},
]


class TestCitations(unittest.TestCase):
    def test_extract_markers_dedup_and_order(self):
        self.assertEqual(extract_markers("A [X] b [Y] c [X]."), ["X", "Y"])
        self.assertEqual(extract_markers("no markers"), [])

    def test_resolves_by_title_case_and_entity_insensitive(self):
        out = check_citations("Fact one [aulani]. Fact two [Disney's Saratoga Springs Resort & Spa].", PASSAGES)
        self.assertEqual(out["unresolved"], [])
        self.assertEqual(out["resolved"]["aulani"], ["qdrant_1", "qdrant_2"])
        self.assertEqual(out["resolved"]["Disney's Saratoga Springs Resort & Spa"], ["qdrant_3"])

    def test_flags_fabricated_and_placeholder_citations(self):
        out = check_citations("Claim [Made Up Article] and [Title] and [Aulani].", PASSAGES)
        self.assertEqual(out["unresolved"], ["Made Up Article", "Title"])
        self.assertIn("Aulani", out["resolved"])
        self.assertEqual(out["n_markers"], 3)

    def test_no_evidence_means_everything_unresolved(self):
        out = check_citations("Claim [Aulani].", [])
        self.assertEqual(out["unresolved"], ["Aulani"])


class TestStripAndMap(unittest.TestCase):
    def test_strip_removes_only_unresolved_markers(self):
        text = "Fact [Aulani]. Other (Source: [Real Title], James) and [VERIFIED]: ok."
        cleaned, removed = strip_unresolved_markers(text, PASSAGES)
        self.assertEqual(removed, ["Real Title", "VERIFIED"])
        self.assertIn("[Aulani]", cleaned)
        self.assertNotIn("[Real Title]", cleaned)
        self.assertNotIn("[VERIFIED]", cleaned)

    def test_strip_is_noop_when_all_resolve(self):
        text = "Fact [Aulani]."
        cleaned, removed = strip_unresolved_markers(text, PASSAGES)
        self.assertEqual((cleaned, removed), (text, []))

    def test_claim_evidence_map_picks_best_passage_by_content_overlap(self):
        passages = [
            {"evidence_id": "e1", "doc_title": "Twins", "text": "James and Oliver Phelps were born in Sutton Coldfield on 25 February 1986."},
            {"evidence_id": "e2", "doc_title": "Other", "text": "Mark Williams plays Arthur Weasley in the films."},
        ]
        claims = [{"claim_id": "c1", "state": "VERIFIED", "text": "They were born in Sutton Coldfield on 25 February 1986."},
                  {"claim_id": "c2", "state": "UNRESOLVED", "text": "Completely unrelated quantum chromodynamics statement."}]
        m = claim_evidence_map(claims, passages)
        self.assertEqual(m[0]["best_evidence_id"], "e1")
        self.assertGreaterEqual(m[0]["content_word_overlap"], 0.8)
        self.assertEqual(m[1]["best_evidence_id"], None)  # no overlap at all -> no mapping claimed
        self.assertEqual(m[1]["content_word_overlap"], 0.0)


class TestAnswerAudit(unittest.TestCase):
    VERIFIED = ["James and Oliver Phelps played Fred and George Weasley.", "Rupert Grint played Ron Weasley."]

    def test_backed_sentences_pass_and_citations_are_ignored(self):
        ans = "James and Oliver Phelps played Fred and George Weasley. [James and Oliver Phelps]\nRupert Grint played Ron Weasley."
        rows = audit_answer_sentences(ans, self.VERIFIED)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["backed"] for r in rows))

    def test_sentence_outside_verified_claims_is_flagged_not_removed(self):
        ans = ("Rupert Grint played Ron Weasley. "
               "James and Oliver Phelps were born in the Sutton Coldfield area of Birmingham on 25 February 1986.")
        rows = audit_answer_sentences(ans, self.VERIFIED)
        flagged = [r for r in rows if not r["backed"]]
        self.assertEqual(len(flagged), 1)
        self.assertIn("Sutton Coldfield", flagged[0]["sentence"])

    def test_short_fragments_skipped_and_empty_inputs_safe(self):
        self.assertEqual(audit_answer_sentences("Yes. No.", self.VERIFIED), [])
        self.assertEqual(audit_answer_sentences("", self.VERIFIED), [])
        rows = audit_answer_sentences("Rupert Grint played Ron Weasley.", [])
        self.assertFalse(rows[0]["backed"])  # nothing verified -> nothing can back a sentence


if __name__ == "__main__":
    unittest.main()
