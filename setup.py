import sys
import subprocess
import warnings
warnings.simplefilter("ignore", UserWarning)

def main(file_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", file_name])
    return


if __name__ == "__main__":
    main("requirements.txt" if len(sys.argv) < 2 else sys.argv[1])