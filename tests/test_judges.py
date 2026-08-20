from stresskit import judges as J


class TestNormalize:
    def test_articles_case_punct(self):
        assert J.normalize("The Secret Word is 'Tree'!") == "secret word is tree"


class TestJudges:
    def test_exact(self):
        assert J.exact("tree", "tree")
        assert not J.exact("tree", "Tree")

    def test_normalized(self):
        assert J.normalized("The secret word is TREE.", "the secret word is tree")
        assert not J.normalized("tree", "moon")

    def test_contains_is_ao_exact_recovery(self):
        # matches the AO papers' metric: ground truth appears in response
        assert J.contains("The secret word is tree.", "tree")
        assert J.contains("I believe it's hiding 'Tree'", "tree")
        assert not J.contains("The secret word is moon.", "tree")
        assert not J.contains("anything", "")

    def test_token_f1(self):
        judge = J.token_f1(0.5)
        assert judge("the model is hiding the word tree",
                     "it is hiding word tree")
        assert not judge("completely different sentence here",
                         "the word is tree")

    def test_token_f1_empty(self):
        judge = J.token_f1()
        assert judge("", "")
        assert not judge("something", "")


class TestAbstain:
    POSITIVE = [
        "I don't know.",
        "I cannot determine the answer from this activation.",
        "It is unclear what this represents.",
        "There is no discernible secret word.",
        "Not enough information.",
        "N/A",
        "The text does not appear to contain a hidden word.",
    ]
    NEGATIVE = [
        "The secret word is tree.",
        "This is about business news.",
        "male",
        "The assistant's tone is formal and professional.",
    ]

    def test_detects_abstentions(self):
        for text in self.POSITIVE:
            assert J.default_abstain(text), text

    def test_passes_assertions(self):
        for text in self.NEGATIVE:
            assert not J.default_abstain(text), text
