import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\watermelon3.0.csv")
F = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感"]
tr, te = train_test_split(df, test_size=.2, random_state=1, stratify=df["好瓜"])
x = df[df["编号"] == 1].iloc[0]

p = {}
for c in ["是", "否"]:
    s = tr[tr["好瓜"] == c]
    p[c] = len(s) / len(tr)
    for f in F:
        p[c] *= ((s[f] == x[f]).sum() + 1) / (len(s) + tr[f].nunique())

z = p["是"] + p["否"]
print("train/test:", len(tr), len(te), sorted(te["编号"]))
print(f"P(好|x) = {p['是']/z:.4f}")
print(f"P(坏|x) = {p['否']/z:.4f}")
print("预测：好瓜" if p["是"] > p["否"] else "预测：坏瓜")
