#!/usr/bin/env python3
import argparse
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile

SBATCH_FILENAME = "gce_demc.sbatch"  # <-- put your sbatch filename here

def replace_file_path(sbatch_text: str, file_path: str) -> str:
    """
    Replace a line that defines FILE_PATH=... with a safely quoted value.
    If not found, insert it just above the srun line.
    """
    # Safely single-quote for bash (handle apostrophes)
    # bash trick: close quote, insert escaped quote, reopen
    def bash_single_quote(s: str) -> str:
        return "'" + s.replace("'", "'\"'\"'") + "'"

    quoted = bash_single_quote(file_path)

    # Try to replace existing assignment (any of: FILE_PATH=..., FILE_PATH='...', FILE_PATH="")
    pattern = re.compile(r"^(FILE_PATH\s*=).*$", flags=re.MULTILINE)
    if pattern.search(sbatch_text):
        return pattern.sub(rf"\1{quoted}", sbatch_text)

    # Otherwise, inject above the srun line
    srun_pat = re.compile(r"^\s*srun\b.*$", flags=re.MULTILINE)
    m = srun_pat.search(sbatch_text)
    inject_line = f"FILE_PATH={quoted}\n\n"
    if m:
        start = m.start()
        return sbatch_text[:start] + inject_line + sbatch_text[start:]
    else:
        # Fallback: append at end
        return sbatch_text + ("\n" if not sbatch_text.endswith("\n") else "") + inject_line

def submit(temp_path: pathlib.Path) -> int:
    proc = subprocess.run(
        ["sbatch", str(temp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or proc.stdout)
        raise SystemExit(proc.returncode)

    # Typical sbatch response: "Submitted batch job 123456"
    m = re.search(r"Submitted batch job\s+(\d+)", proc.stdout)
    if not m:
        print(proc.stdout.strip())
        raise SystemExit("Could not parse job ID from sbatch output.")
    jobid = int(m.group(1))
    print(f"Submitted batch job {jobid}")
    return jobid

def main():
    p = argparse.ArgumentParser(description="Inject FILE_PATH into sbatch and submit.")
    p.add_argument("file_path", help="Path to pass into the Slurm job (MDF input, config, etc.)")
    p.add_argument(
        "-s", "--sbatch",
        default=SBATCH_FILENAME,
        help=f"Path to the sbatch script (default: {SBATCH_FILENAME})",
    )
    args = p.parse_args()

    sbatch_path = pathlib.Path(args.sbatch).expanduser().resolve()
    if not sbatch_path.is_file():
        raise SystemExit(f"sbatch script not found: {sbatch_path}")

    # Normalize the file path to an absolute path so the job sees the same path
    file_path = str(pathlib.Path(args.file_path).expanduser().resolve())

    sbatch_text = sbatch_path.read_text()

    new_text = replace_file_path(sbatch_text, file_path)

    # Write to a secure temp file in the same directory (so relative paths still work)
    with tempfile.NamedTemporaryFile("w", prefix=sbatch_path.stem + "_", suffix=".sbatch", dir=str(sbatch_path.parent), delete=False) as tf:
        tf.write(new_text)
        tmp_path = pathlib.Path(tf.name)

    try:
        submit(tmp_path)
    finally:
        # Keep the temp file around for reproducibility? If you prefer auto-clean, uncomment:
        # tmp_path.unlink(missing_ok=True)
        pass

if __name__ == "__main__":
    main()
