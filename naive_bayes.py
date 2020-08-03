import cv2
import os
import numpy as np
from math import sqrt
from math import exp
from math import pi
from random import seed
from random import randrange
from img_to_array import kelas_0, kelas_1, kelas_2, kelas_3

class naive_bayes(object):
    def __init__(self,kelas_0=kelas_0,kelas_1=kelas_1, kelas_2=kelas_2, kelas_3=kelas_3):
        dataset=self.join_dataset(kelas_0=kelas_0,kelas_1=kelas_1, kelas_2=kelas_2, kelas_3=kelas_3)
        # convert class column to integers
        self.dict_kelas=self.str_column_to_int(dataset, len(dataset[0]) - 1)
        self.dataset = np.array(dataset, dtype=float).tolist()
        print(dataset)

    def join_dataset(self,kelas_0=kelas_0,kelas_1=kelas_1, kelas_2=kelas_2, kelas_3=kelas_3):
        temp=[]
        def parse(data):
            for i in range(len(data)):
                temp.append(data[i])

        parse(kelas_0)
        parse(kelas_1)
        parse(kelas_2)
        parse(kelas_3)
        return np.array(temp)

    # Convert string column to integer
    def str_column_to_int(self,dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()

        for i, value in enumerate(unique):
            lookup[value] = i


        for row in dataset:
            row[column] = lookup[row[column]]

        return lookup

    # Split the dataset by class values, returns a dictionary
    def separate_by_class(self,dataset):
        separated = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
            if (class_value not in separated):
                separated[class_value] = list()
            separated[class_value].append(vector)
        return separated

    # Calculate the mean of a list of numbers
    def mean(self,numbers):
        return sum(numbers) / float(len(numbers))

    # Calculate the standard deviation of a list of numbers
    def stdev(self,numbers):
        avg = self.mean(numbers)
        variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
        return sqrt(variance)

    # Calculate the mean, stdev and count for each column in a dataset
    def summarize_dataset(self,dataset):
        summaries = [(self.mean(column), self.stdev(column), len(column)) for column in zip(*dataset)]
        del (summaries[-1])
        return summaries

    # Split dataset by class then calculate statistics for each row
    def summarize_by_class(self,dataset):
        separated = self.separate_by_class(dataset)
        summaries = dict()
        for class_value, rows in separated.items():
            summaries[class_value] = self.summarize_dataset(rows)
        return summaries

    # Calculate the Gaussian probability distribution function for x
    def calculate_probability(self,x, mean, stdev):
        exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent

    # Calculate the probabilities of predicting each class for a given row
    def calculate_class_probabilities(self,summaries, row):
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, _ = class_summaries[i]
                probabilities[class_value] *= self.calculate_probability(row[i], mean, stdev)
        return probabilities

    # Predict the class for a given row
    def predict(self,summaries, row):
        probabilities = self.calculate_class_probabilities(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label
