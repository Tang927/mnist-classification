import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def load_mnist_data():
    """加载并预处理 MNIST 数据集"""
    print("正在加载 MNIST 数据集...")
    # 从 openml 加载 MNIST (70000张 28x28 灰度图)
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # 归一化像素值到 [0, 1]
    X = X / 255.0
    
    # 划分训练集和测试集 (使用 60000 训练, 10000 测试)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42
    )
    
    print(f"数据加载完成. 训练集: {X_train.shape}, 测试集: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_lda(X_train, y_train, n_components=None):
    """
    训练 LDA 模型
    n_components: 降维后的维度数 (最大为 类别数-1 = 9)
    """
    print(f"正在训练 LDA 模型 (降维至 {n_components} 维)...")
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X_train, y_train)
    print("LDA 模型训练完成.")
    return lda

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n模型测试集准确率: {acc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    return acc

def visualize_lda_projection(lda, X_test, y_test):
    """可视化 LDA 降维后的前两个维度 (仅当 n_components >=2 时)"""
    if lda.n_components < 2:
        print("降维维度小于 2，跳过可视化.")
        return
    
    X_lda = lda.transform(X_test)
    
    plt.figure(figsize=(10, 8))
    for digit in range(10):
        mask = y_test == str(digit)
        plt.scatter(X_lda[mask, 0], X_lda[mask, 1], label=str(digit), alpha=0.6, s=10)
    
    plt.xlabel('LDA Component 1')
    plt.ylabel('LDA Component 2')
    plt.title('MNIST Test Set Projection by LDA')
    plt.legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. 加载数据
    X_train, X_test, y_train, y_test = load_mnist_data()
    
    # 2. 训练 LDA (将 784 维数据降至 9 维 (10类-1))
    lda_model = train_lda(X_train, y_train, n_components=9)
    
    # 3. 评估
    evaluate_model(lda_model, X_test, y_test)
    
    # 4. (可选) 可视化前两维
    # 重新训练一个 2 维的模型用于可视化
    lda_2d = train_lda(X_train, y_train, n_components=2)
    visualize_lda_projection(lda_2d, X_test, y_test)
