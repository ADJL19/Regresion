import matplotlib.pyplot as plt
import seaborn as sns

def boxplot(data, x, y):
    sns.set_theme(style="ticks")

    _, ax = plt.subplots(figsize=(7, 6))

    sns.boxplot(x=x, y=y, data=data)

    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)
    plt.show()

# print(sns.load_dataset("planets"))