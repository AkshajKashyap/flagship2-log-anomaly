from pathlib import Path

LOG_PATH = Path("data/raw/hdfs/HDFS.log")

def main():
    n = 0
    with LOG_PATH.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            print(line.rstrip("\n"))
            n += 1
            if n >= 5:
                break

if __name__ == "__main__":
    main()
