import matplotlib.pyplot as plt
import os

current_directory = os.getcwd()
output_directory = current_directory  # Output directory same as current directory

legend_labels = ['Noisy Accuracies', 'True Accuracies', 'Bayes Optimal Classifier']
line_styles = ['-', ':', '--']
line_colors = ['blue', 'yellow', 'red']

for filename in os.listdir(current_directory):
    if filename.endswith('.png'):
        filepath = os.path.join(current_directory, filename)
        image = plt.imread(filepath)

        plt.figure()
        plt.imshow(image)

        legend_elements = [plt.Line2D([], [], linestyle=style, color=color, label=label)
                           for style, color, label in zip(line_styles, line_colors, legend_labels)]
        plt.legend(handles=legend_elements)

        output_filepath = os.path.join(output_directory, filename)
        plt.savefig(output_filepath)

        plt.close()

