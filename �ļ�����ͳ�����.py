import os
import matplotlib.pyplot as plt

def read_txt_files(folder_path):
    class_counts = {}

    # 遍历文件夹中的所有txt文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                # 读取每行数据并统计类别数量
                for line in file:
                    category = line.split()[0]
                    class_counts[category] = class_counts.get(category, 0) + 1

    return class_counts

def plot_bar_chart(class_counts):
    categories = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts, color='skyblue')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Class Counts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    folder_path = 'D:/参考文献/newtest/新建数据集/val/labels'  # 替换为实际文件夹路径
    class_counts = read_txt_files(folder_path)
    print(class_counts)
    # plot_bar_chart(class_counts)
