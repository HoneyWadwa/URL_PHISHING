import unittest

from src.feature_extractor import FEATURE_COLUMNS, extract_features, features_as_dataframe


class FeatureExtractorTests(unittest.TestCase):
    def test_returns_exact_model_columns(self):
        features = extract_features("https://example.com/login?user=student")
        self.assertEqual(list(features), FEATURE_COLUMNS)
        self.assertEqual(list(features_as_dataframe("https://example.com").columns), FEATURE_COLUMNS)

    def test_detects_https_and_ip_address(self):
        https_features = extract_features("https://www.example.com")
        ip_features = extract_features("http://192.168.1.1:8080/login")
        self.assertEqual(https_features["https_flag"], 1)
        self.assertEqual(ip_features["has_ip_address"], 1)
        self.assertEqual(ip_features["tld_length"], len("1:8080"))
        self.assertEqual(https_features["token_count"], 5)

    def test_empty_url_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_features("  ")


if __name__ == "__main__":
    unittest.main()
