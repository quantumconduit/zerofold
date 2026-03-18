"""python -m zerofold — runs the benchmark"""
import subprocess, sys

def main():
    import os, pathlib
    bench = pathlib.Path(__file__).parent.parent / "benchmark.py"
    os.execv(sys.executable, [sys.executable, "-X", "utf8", str(bench)] + sys.argv[1:])

if __name__ == "__main__":
    main()
