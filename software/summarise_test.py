import pandas as pd

df = pd.read_csv("docs/live_test_log.csv")
total = len(df)
correct = (df["correct"] == 1).sum()
repeat = (df["predicted"] == "REPEAT").sum()
nohands = (df["predicted"] == "NO_HANDS").sum()
wrong = total - correct - repeat - nohands

print(f"Total attempts: {total}")
print(f"Correct: {correct}  ({correct/total:.0%})")
print(f"Wrong word: {wrong}")
print(f"Please repeat: {repeat}")
print(f"No hands seen: {nohands}")
print()
print(df.groupby("intended")["correct"].agg(["sum", "count"]).rename(
    columns={"sum": "correct", "count": "attempts"}))
print()
print(pd.crosstab(df["intended"], df["predicted"]))