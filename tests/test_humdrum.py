import unittest

from humdrum import HumdrumParser


class TestHumdrumParser(unittest.TestCase):

    def ok(self, text: str):
        parser = HumdrumParser.from_text(text)
        parser.parse()

    def fail(self, text: str):
        parser = HumdrumParser.from_text(text)
        with self.assertRaises(ValueError):
            parser.parse()

    def test_kerns(self):
        print("HERE")
        self.fail("")
        self.ok(
            "**kern\t**kern\n"
            "*clefF4\t*clefG2"
        )

    def test_spine_indicators(self):
        self.ok(
            "**kern\t**kern\n"
            "*-\t*"
        )


if __name__ == '__main__':
    unittest.main()
