"""Check the local environment against the launcher's requirements."""
import importlib
from importlib import metadata
from pathlib import Path


def dependencies_ok():
    try:
        # pip ships packaging in a fresh venv, without needing another dependency.
        from pip._vendor.packaging.requirements import Requirement
        from pip._vendor.packaging.version import Version

        requirements = Path(__file__).with_name("requirements-studio.txt").read_text(encoding="utf-8").splitlines()
        module_names = {"django": "django", "openai": "openai"}
        for line in requirements:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            requirement = Requirement(line)
            module = module_names.get(requirement.name.lower())
            if not module or Version(metadata.version(requirement.name)) not in requirement.specifier:
                return False
            importlib.import_module(module)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(0 if dependencies_ok() else 1)
