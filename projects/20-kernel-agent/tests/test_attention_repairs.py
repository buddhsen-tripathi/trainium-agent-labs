"""Regression checks for the failure modes observed in the Level 8 audit."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import agent
from lint import lint_kernel

FIXTURE = (ROOT / "examples/attention_kernel.py").read_text()


class AttentionRepairs(unittest.TestCase):
    def test_working_fixture_has_no_static_issues(self):
        self.assertEqual(lint_kernel(FIXTURE, level=8), [])

    def test_discarded_shape_dimension_is_not_treated_as_a_known_size(self):
        source = """def kernel(lhsT, rhs):
    K, M = lhsT.shape
    _, N = rhs.shape
    rhs_sb = nl.ndarray(shape=(K, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rhs_sb, src=rhs)
    return rhs_sb
"""
        self.assertEqual(lint_kernel(source, level=3), [])

    def test_observed_invalid_apis_are_reported_together(self):
        broken = FIXTURE.replace("row_sum = nl.sum", "row_sum = nisa.sum")
        broken = broken.replace("1.0 / (dim ** 0.5)", "nl.float32(1.0 / (dim ** 0.5))")
        feedback = "\n".join(lint_kernel(broken, level=8))
        self.assertIn("nisa.sum is not an API", feedback)
        self.assertIn("nl.float32 is a dtype", feedback)

    def test_psum_cannot_be_stored_directly_to_hbm(self):
        broken = FIXTURE.replace("nisa.dma_copy(dst=out, src=out_sb)",
                                 "nisa.dma_copy(dst=out, src=out_ps)")
        self.assertTrue(any("dma_copy moves" in issue for issue in lint_kernel(broken, level=8)))

    def test_invented_matmul_transpose_keyword_is_caught(self):
        broken = FIXTURE.replace("stationary=qT_sb, moving=kT_sb)",
                                 "stationary=qT_sb, moving=kT_sb, transpose_moving=True)")
        self.assertTrue(any("no transpose_moving keyword" in issue for issue in lint_kernel(broken, level=8)))

    def test_keyword_shape_preserves_output_contract(self):
        broken = FIXTURE.replace("out = nl.ndarray((seq, dim)", "out = nl.ndarray(shape=(dim, seq)")
        feedback = "\n".join(lint_kernel(broken, level=8))
        self.assertIn("attention must return Q's shape", feedback)
        self.assertNotIn("Allocate out with shape", feedback)

    def test_all_prompt_modes_include_attention_contract(self):
        for terse in (0, 1, 2):
            prompt = agent.first_prompt(8, terse)
            self.assertIn("WITHOUT a pre-transpose", prompt)
            self.assertIn("stationary=P_T, moving=V_sbuf", prompt)
            self.assertNotIn("left operand is already transposed", prompt)
        self.assertIn(agent.ATTENTION_CONTRACT, agent.repair_prompt(8, "broken", "fix layout"))

    def test_equal_reward_prefers_fewer_issues_before_novelty(self):
        old = (0.3, "old", "A static check found 1 problem(s)", {}, 0)
        novel = (0.3, "new", "A static check found 4 problem(s)", {}, 1)
        self.assertGreater(agent.attention_candidate_key(old, "old", old[2]),
                           agent.attention_candidate_key(novel, "old", old[2]))
        running = (0.5, "new", "numerical mismatch", {}, 2)
        self.assertGreater(agent.attention_candidate_key(running, "old", old[2]),
                           agent.attention_candidate_key(old, "old", old[2]))


if __name__ == "__main__":
    unittest.main()
