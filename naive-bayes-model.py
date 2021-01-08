from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import random


# calculates P(attribute_a = value_a)
def calc_probablity(dataset, attribute_a, value_a):
    total = 0
    count = 0
    for i in dataset[:, attribute_a]:
        total += 1
        if(value_a == i):
            count += 1
    return count/total


# calculates P(attribute_a = value_a| attribute_b = value_b)
def calc_cond_probablity(dataset, attribute_a, attribute_b, value_a, value_b):
    total = 0
    count = 0
    for i in range(len(dataset)):
        if(dataset[i, attribute_b] == value_b):
            total += 1
            if(dataset[i, attribute_a] == value_a):
                count += 1
    return count/total


def plot_confusion_matrix(cm, classes, count, cmap=plt.cm.Blues):
    title = 'Confusion matrix'

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Predicted label',
           xlabel='True label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('confusion_matrix' + str(count) + '.png', format='png')


dataset = read_csv("House-votes-data.TXT", header=None)
dataset = dataset.sample(frac=1).reset_index(drop=True)
dataset = dataset.replace("y", 1)
dataset = dataset.replace("n", 0)
dataset = dataset.replace("democrat", 1)
dataset = dataset.replace("republican", 0)
values = dataset.values

type_attr_value = 2
type_class_value = 2
class_column_no = 16

prob_data = []

for attribute in range(class_column_no):
    for attr_value in range(type_attr_value):
        for class_value in range(type_class_value):
            temp = calc_cond_probablity(values, attribute, 16, attr_value, class_value)
            prob_data.append(temp)

for i in range(0, len(values)):
    for j in range(0, len(values[i])):
        if(values[i, j] == "?"):
            if(prob_data[4*j + int(str(0)+str(values[i, 16]), 2)] > prob_data[4*j + int(str(1)+str(values[i, 16]), 2)]):
                values[i, j] = 0
            else:
                values[i, j] = 1

no_of_folds = 5
bin_size = len(values) // no_of_folds
precision = 0
recall = 0
specificity = 0
accuracy = 0

for i in range(no_of_folds):
    prob_data = []
    test_set = values[i*bin_size:(i+1)*bin_size]
    train_set = values[0:i*bin_size]
    train_set = np.append(train_set, values[(i+1)*bin_size:no_of_folds*bin_size], axis=0)

    for attribute in range(0, len(values[0])):
        for attr_value in range(type_attr_value):
            for class_value in range(type_class_value):
                prob_data.append(calc_cond_probablity(train_set, attribute, 16, attr_value, class_value))

    tp_count = 0
    fp_count = 0
    fn_count = 0
    tn_count = 0

    prob_demo = calc_probablity(train_set, class_column_no, 1)
    prob_repub = calc_probablity(train_set, class_column_no, 0)

    for j in test_set:
        prob_class = [prob_repub, prob_demo]
        for class_value in range(type_class_value):
            for attribute in range(0, len(values[0]) - 1, 2):
                prob_class[class_value] *= prob_data[4 * attribute + int(str(j[attribute]) + str(class_value), 2)]

        if(prob_class[0] > prob_class[1]):
            res = 0
            if(res == j[class_column_no]):
                tn_count += 1
            else:
                fn_count += 1
        else:
            res = 1
            if(res == j[class_column_no]):
                tp_count += 1
            else:
                fp_count += 1
#########################################################
    # conf_matrix = np.array([[tp_count, fp_count], [fn_count, tn_count]]);
    # plot_confusion_matrix(conf_matrix, ['Democrat', 'Republican'], i)
#########################################################
    precision += tp_count/(tp_count + fp_count)
    recall += tp_count/(tp_count + fn_count)
    specificity += tn_count/(tn_count + fp_count)
    accuracy += (tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count)
    # print(precision, recall, specificity, accuracy)
print("precision:", precision / no_of_folds)
print("recall:", recall/no_of_folds)
print("specificity:", specificity/no_of_folds)
print("accuracy:", accuracy/no_of_folds)
print("f1:", (2 * precision * recall)/((precision + recall) * no_of_folds))


# from pandas import read_csv
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.set(font_scale=1.2)
# fig, ax = plt.subplots(figsize=(12,7)) 
# sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False, cmap = 'viridis', ax=ax)

# sns.set(font_scale=1.1)
# sns.set_style('whitegrid')
# sns.countplot(x='handicapped-infants',hue='ClassName',data=dataset,palette='rainbow')

# fig, ax = plt.subplots(figsize=(7,4))
# sns.set(font_scale=1.1)
# sns.set_style('whitegrid')
# sns.countplot(x='water-project-cost-sharing',hue='ClassName',data=dataset,palette='rainbow', ax = ax)

# sns.set(font_scale=1.1)
# sns.set_style('whitegrid')
# sns.countplot(x='adoption-of-the-budget-resolution',hue='ClassName',data=dataset,palette='rainbow')

# sns.set(font_scale=1.1)
# sns.set_style('whitegrid')
# sns.countplot(x='adoption-of-the-budget-resolution',hue='ClassName',data=dataset,palette='rainbow')

# sns.set(font_scale=1.1)
# sns.set_style('whitegrid')
# sns.countplot(x='el-salvador-aid',hue='ClassName',data=dataset,palette='rainbow')

# fig, ax = plt.subplots(figsize=(7,5))
# sns.set(font_scale=1.1)
# sns.set_style('whitegrid')
# sns.countplot(x='religious-groups-in-schools',hue='ClassName',data=dataset,palette='rainbow', ax = ax)

# sns.set(font_scale=1.1)
# sns.set_style('whitegrid')
# sns.countplot(x='anti-satellite-test-ban',hue='ClassName',data=dataset,palette='rainbow')

# sns.set(font_scale=1.1)
# sns.set_style('whitegrid')
# sns.countplot(x='aid-to-nicaraguan-contras',hue='ClassName',data=dataset,palette='rainbow')

# sns.set(font_scale=1.1)
# sns.set_style('whitegrid')
# sns.countplot(x='mx-missile',hue='ClassName',data=dataset,palette='rainbow')

# fig, ax = plt.subplots(figsize=(7,5))
# sns.set(font_scale=1.1)
# sns.set_style('whitegrid')
# sns.countplot(x='immigration',hue='ClassName',data=dataset,palette='rainbow', ax=ax)

# fig, ax = plt.subplots(figsize=(7,8))
# sns.set_style('whitegrid')
# sns.set(font_scale=1.1)
# sns.countplot(x='synfuels-corporation-cutback',hue='ClassName',data=dataset,palette='rainbow', ax = ax)

# sns.set(font_scale=1.1)
# sns.set_style('whitegrid')
# sns.countplot(x='education-spending',hue='ClassName',data=dataset,palette='rainbow')

# sns.set(font_scale=1.1)
# sns.set_style('whitegrid')
# sns.countplot(x='superfund-right-to-sue',hue='ClassName',data=dataset,palette='rainbow')

# sns.set(font_scale=1.1)
# sns.set_style('whitegrid')
# sns.countplot(x='crime',hue='ClassName',data=dataset,palette='rainbow')

# sns.set(font_scale=1.1)
# sns.set_style('whitegrid')
# sns.countplot(x='duty-free-exports',hue='ClassName',data=dataset,palette='rainbow')

# sns.set(font_scale=1.1)
# sns.set_style('whitegrid')
# sns.countplot(x='export-administration-act-south-africa',hue='ClassName',data=dataset,palette='rainbow')

