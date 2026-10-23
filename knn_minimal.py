import numpy as np  # 导入数值计算库，用于距离和排序
import pandas as pd  # 导入数据分析库，用于读取、分层抽样和投票
df = pd.read_csv("iris_dataset.csv")  # 读取同目录下的鸢尾花数据
train = df.groupby("target_name").sample(frac=0.8, random_state=42)  # 每类抽 80% 作训练集
test = df.drop(train.index)  # 剩下的 20% 作测试集
X, y = train.iloc[:, :4].to_numpy(float), train["target_name"].to_numpy()  # 训练特征和标签
Xt, yt = test.iloc[:, :4].to_numpy(float), test["target_name"].to_numpy()  # 测试特征和标签
mu, sd = X.mean(0), X.std(0)  # 只用训练集算均值和标准差
X, Xt = (X - mu) / sd, (Xt - mu) / sd  # 对训练/测试集做 Z-score 标准化
def knn_predict(X, y, Xt, k):  # 定义欧氏距离 KNN 预测函数
    dist = np.linalg.norm(Xt[:, None, :] - X[None, :, :], axis=2)  # 计算测试样本到训练样本的欧氏距离
    idx = np.argsort(dist, axis=1)[:, :k]  # 取每行距离最近的 k 个邻居下标
    votes = y[idx]  # 取出 k 个邻居的类别
    return np.array([pd.Series(v).mode()[0] for v in votes])  # 多数投票，返回预测类别
folds = np.array_split(np.random.default_rng(42).permutation(len(X)), 5)  # 训练集随机均分成 5 折
trs = [np.setdiff1d(np.arange(len(X)), va) for va in folds]  # 预先算好每折对应的训练集下标
best_k, best_cv = 1, 0  # 初始化最优 k 和最高 CV 准确率
for k in range(1, 16, 2):  # 只尝试 1 到 15 的奇数 k
    cv = np.mean([(knn_predict(X[tr], y[tr], X[va], k) == y[va]).mean() for tr, va in zip(trs, folds)])  # 计算当前 k 的 5 折平均准确率
    if cv > best_cv:  # 如果当前 k 更好
        best_k, best_cv = k, cv  # 更新最优 k 和最高 CV 准确率
pred = knn_predict(X, y, Xt, best_k)  # 用最优 k 在测试集上预测
acc = (pred == yt).mean()  # 计算测试集准确率
print(f"best k={best_k}, cv={best_cv:.2%}, test={acc:.2%}")  # 输出最优 k、CV 和测试准确率
print(pd.crosstab(pd.Series(yt, name="actual"), pd.Series(pred, name="predicted")))  # 输出混淆矩阵
