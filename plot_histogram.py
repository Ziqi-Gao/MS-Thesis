#!/usr/bin/env python3
# 文件名：test_normality.py

import numpy as np
import matplotlib
matplotlib.use('Agg')   # 如果在无 GUI 环境下运行
import matplotlib.pyplot as plt
from scipy import stats

def main():
    # 1. 读取数据
    w_all = np.load('/home/del6500/CD/Gemma2b_Cities_Concept.npz')['w_all']
    # 形状 (1000, 18, 2048)
    data = w_all[:, 8, 1024]   # layer0，第0维，1000个样本

    # 2. 直方图 + 正态曲线
    plt.figure(figsize=(6,4))
    count, bins, _ = plt.hist(data, bins=100, density=True, alpha=0.6, label='Empirical')
    # 拟合正态分布
    mu, sigma = data.mean(), data.std(ddof=1)
    pdf = stats.norm.pdf(bins, loc=mu, scale=sigma)
    plt.plot(bins, pdf, 'r--', label=f'N({mu:.2f},{sigma:.2f}²)')
    plt.legend()
    plt.title('Histogram with Fitted Normal PDF')
    plt.tight_layout()
    plt.savefig('/home/del6500/CD/hist_normfit.png')
    print("Saved histogram + normal fit to hist_normfit.png")

    # 3. 正态性检验
    print("\n=== Normality Tests ===")
    # 3.1 Shapiro–Wilk
    stat_sw, p_sw = stats.shapiro(data)
    print(f"Shapiro–Wilk: W = {stat_sw:.4f}, p = {p_sw:.4e}")

    # 3.2 D’Agostino’s K^2
    stat_dk2, p_dk2 = stats.normaltest(data)
    print(f"D’Agostino’s K²: K² = {stat_dk2:.4f}, p = {p_dk2:.4e}")

    # 3.3 Anderson–Darling
    ad = stats.anderson(data, dist='norm')
    print(f"Anderson–Darling: A² = {ad.statistic:.4f}")
    for sl, cv in zip(ad.significance_level, ad.critical_values):
        print(f"  {sl}% level: critical = {cv:.4f}")

    # 4. Q–Q 图
    plt.figure(figsize=(6,6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title("Q–Q Plot vs Normal")
    plt.tight_layout()
    plt.savefig('/home/del6500/CD/qqplot.png')
    print("Saved Q–Q plot to qqplot.png")

if __name__ == "__main__":
    main()
