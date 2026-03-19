import subprocess

def retrain():

    print("Running retraining pipeline...")

    subprocess.run(["python", "training/train.py"])

    print("Model retrained successfully")


if __name__ == "__main__":
    retrain()