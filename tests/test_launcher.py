import unittest
from pathlib import Path


class WindowsLauncherTest(unittest.TestCase):
    def test_run_bat_is_ascii_crlf_and_has_required_fallbacks(self):
        path = Path(__file__).resolve().parents[1] / "run.bat"
        data = path.read_bytes()
        self.assertTrue(all(byte < 128 for byte in data))
        self.assertNotIn(b"\xef\xbb\xbf", data)
        self.assertNotIn(b"\n", data.replace(b"\r\n", b""))
        text = data.decode("ascii")
        self.assertIn('cd /d "%~dp0"', text)
        self.assertIn('set "PYTHONPATH=%CD%\\src"', text)
        self.assertLess(text.index('.venv\\Scripts\\python.exe'), text.index('py -3'))
        self.assertLess(text.index('py -3'), text.index('where python'))
        self.assertIn("pause", text)


if __name__ == "__main__":
    unittest.main()
