import matplotlib.pyplot as plt
import seaborn as sns

def boxplot(data, x, y):
    # sns.set_theme(style="ticks")

    # Initialize the figure with a logarithmic x axis
    _, ax = plt.subplots(figsize=(7, 6))

    # Plot the orbital period with horizontal boxes
    sns.boxplot(x=x, y=y, data=data)

    # Tweak the visual presentation
    # ax.xaxis.grid(True)
    # ax.set(ylabel="")
    # sns.despine(trim=True, left=True)
    plt.show()

def main():
    data= sns.load_dataset("planets")
    print(data)
    boxplot(data = data)

if __name__ == '__main__':
    main()