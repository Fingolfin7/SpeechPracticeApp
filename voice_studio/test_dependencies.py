from unittest import TestCase
from unittest.mock import patch

from check_studio_dependencies import dependencies_ok


class DependencyCheckTests(TestCase):
    def check_versions(self, django_version, openai_version):
        versions = {"django": django_version, "openai": openai_version}
        with patch("check_studio_dependencies.metadata.version", side_effect=lambda name: versions[name.lower()]), \
             patch("check_studio_dependencies.importlib.import_module"):
            return dependencies_ok()

    def test_supported_versions(self):
        self.assertTrue(self.check_versions("5.2.18", "3.42.1"))
        self.assertTrue(self.check_versions("5.0", "1.66"))

    def test_old_versions_and_unsupported_major_are_rejected(self):
        self.assertFalse(self.check_versions("4.2", "3.42.1"))
        self.assertFalse(self.check_versions("6.0", "3.42.1"))
        self.assertFalse(self.check_versions("5.2", "1.65.0"))
        self.assertFalse(self.check_versions("5.2", "1.66.0rc1"))

    def test_missing_or_broken_dependency_is_rejected(self):
        with patch("check_studio_dependencies.metadata.version", side_effect=ImportError):
            self.assertFalse(dependencies_ok())
        with patch("check_studio_dependencies.importlib.import_module", side_effect=ImportError):
            self.assertFalse(dependencies_ok())
