"""Build a pull-request comment from CI outputs (used by .github/workflows/verify.yml, pilot.yml).

    python tools/ci_comment.py verify PYVER SHA [--full]
    python tools/ci_comment.py pilot PYVER SHA OUTDIR

verify: result of the table 3 comparison; with --full (or when table3.csv differs from expected/)
the full text of results/run_all.txt, pytest.txt, env_versions.md, table3.md and table3.csv, so
that expected/ can be refreshed from a real CI run, not from a local environment.
pilot: the pilot report and journal from OUTDIR (experiments/pilot.py).
"""
import difflib
import pathlib
import sys

LIMIT = 60000          # GitHub comment body limit is 65536 characters


def read(path):
    p = pathlib.Path(path)
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


def block(path, lang=""):
    text = read(path)
    text = "(нет файла)" if text is None else text.rstrip()
    return f"<details><summary>{path}</summary>\n\n````{lang}\n{text}\n````\n</details>\n"


def verify(pyver, sha, full):
    exp = (read("expected/table3.csv") or "").splitlines()
    res = read("results/table3.csv")
    run_all = (read("results/run_all.txt") or "").strip().splitlines()
    tail = "; ".join(run_all[-3:]) if run_all else "run_all.txt отсутствует"
    if res is None:
        status, diff = "results/table3.csv не создан", []
    else:
        diff = list(difflib.unified_diff(exp, res.splitlines(), "expected/table3.csv",
                                         "results/table3.csv", lineterm=""))
        status = "совпадает с expected/" if not diff else "ОТЛИЧАЕТСЯ от expected/"
    out = [f"### verify, Python {pyver}, {sha[:7]}\n", f"table3.csv: {status}. run_all: {tail}\n"]
    if diff:
        out.append("````diff\n" + "\n".join(diff) + "\n````\n")
    if full or diff or res is None:
        for f in ("results/run_all.txt", "results/pytest.txt", "results/env_versions.md",
                  "results/table3.md", "results/table3.csv"):
            out.append(block(f))
    return out


def pilot(pyver, sha, outdir):
    d = pathlib.Path(outdir)
    out = [f"### pilot, Python {pyver}, {sha[:7]}\n"]
    report = read(d / "report.md")
    out.append(report if report is not None else "report.md не создан (см. журнал ниже)\n")
    for name in ("tests.txt", "journal.md", "bench.txt", "pilot_log.txt"):
        out.append(block(d / name))
    return out


def main():
    kind, pyver, sha = sys.argv[1:4]
    if kind == "verify":
        parts = verify(pyver, sha, "--full" in sys.argv)
    else:
        parts = pilot(pyver, sha, sys.argv[4])
    body = "\n".join(parts)
    if len(body) > LIMIT:
        body = body[:LIMIT] + "\n\n… (обрезано; полный текст в артефакте запуска)\n"
    sys.stdout.write(body)


if __name__ == "__main__":
    main()
