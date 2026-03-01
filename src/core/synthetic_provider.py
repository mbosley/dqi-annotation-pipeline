from .contracts import AnnotationSample


class SyntheticKeywordProvider:
    def annotate(self, sample: AnnotationSample) -> dict:
        text = sample.text.lower()
        reason_score = 2 if "because" in text or "evidence" in text else 1
        respect_score = 2 if "thank" in text or "agree" in text else 1
        return {
            "reason_giving": reason_score,
            "respect": respect_score,
            "contains_policy_claim": "policy" in text or "proposal" in text,
        }
