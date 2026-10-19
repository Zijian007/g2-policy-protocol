"""Build a source handoff ZIP including docs/examples without weights or recordings."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "dist" / "g2-policy-protocol-handoff.zip"
    destination.parent.mkdir(exist_ok=True)
    files = [root / name for name in ("README.md", "pyproject.toml")]
    for directory in ("src", "docs", "examples", "tools", "tests"):
        files.extend(path for path in (root / directory).rglob("*") if path.is_file()
                     and "__pycache__" not in path.parts and not path.name.endswith(".pyc")
                     and not any(part.endswith(".egg-info") for part in path.parts))
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, Path("g2-policy-protocol") / path.relative_to(root))
    print(destination)


if __name__ == "__main__":
    main()
