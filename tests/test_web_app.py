import unittest

from app.app import app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_page_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Is this URL safe?", response.data)

    def test_url_submission_returns_prediction(self):
        response = self.client.post("/", data={"url": "https://www.google.com"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Legitimate", response.data)
        self.assertIn(b"Top model signals", response.data)


if __name__ == "__main__":
    unittest.main()
