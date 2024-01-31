# Your code goes here 
import re 
def binary_representation(text, target_words):
    # Convert to lowercase and remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.title())
    words = text.split()

    # Create a binary representation dictionary
    binary_dict = {word: 1 if word in words else 0 for word in target_words}
    
    return binary_dict

# Define the target words
target_words = ['Awful', 'Bad' ,'Boring', 'Dull', 'Effective', 'Enjoyable' ,'Great',' Hilarious']

# Read a sample text file
file_path = './review_polarity/txt_sentoken/pos/cv996_11592.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Get the binary representation
binary_representation_dict = binary_representation(text, target_words)

# Print the result
print(f"Binary Representation: {binary_representation_dict}")



import os
import numpy as np
from prettytable import PrettyTable
class SentimentAnalysis:
    def __init__(self, path="./review_polarity/txt_sentoken"):
        self.path = path
        self.data = []
        self.polarity = []
        self.pols = ['neg', 'pos']
        self.negvectors = []
        self.posvectors = []
        self.sumnegvectors = [0, 0, 0, 0, 0, 0, 0, 0]
        self.sumposvectors = [0, 0, 0, 0, 0, 0, 0, 0]
        self.words = ['awful', 'bad', 'boring', 'dull', 'effective', 'enjoyable', 'great', 'hilarious']
        self.pos_prior = 0
        self.neg_prior = 0
        self.pos_likelihood = np.zeros(len(self.words))
        self.neg_likelihood = np.zeros(len(self.words))
        self.predictions = []

    def read_data(self):
        for index, pol in enumerate(self.pols):
            files = os.listdir(os.path.join(self.path, pol))
            for file in files:
                masterlist = self.process_file(os.path.join(self.path, pol, file))
                self.data.append(masterlist)
                self.polarity.append(index)

    def process_file(self, file_path):
        masterlist = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                word_list = line.split(" ")
                masterlist += word_list
        return masterlist

    def create_vectors(self):
        for i, text in enumerate(self.data):
            vector = self.create_vector(text)
            if self.polarity[i] == 0:  # negative review
                self.negvectors.append(vector)
                self.update_sum_vectors(vector, self.sumnegvectors)
            else:
                self.posvectors.append(vector)  # positive review
                self.update_sum_vectors(vector, self.sumposvectors)

    def create_vector(self, text):
        vector = [1 if word in text else 0 for word in self.words]
        return vector

    def update_sum_vectors(self, vector, sum_vectors):
        for j, item in enumerate(vector):
            if vector[j] == 1:
                sum_vectors[j] += 1

    def calculate_likelihood(self, vectors_sum):
        return [count / (len(self.pols) * 1000) for count in vectors_sum]

    def train(self):
        total_reviews = len(self.data)
        self.pos_prior = self.pols.count('pos') / total_reviews
        self.neg_prior = self.pols.count('neg') / total_reviews

        self.pos_likelihood = self.calculate_likelihood(self.sumposvectors)
        self.neg_likelihood = self.calculate_likelihood(self.sumnegvectors)

    def classify(self, review):
        review_vector = self.create_vector(review)
        pos_prob = np.log(self.pos_prior) + np.sum(np.log(self.pos_likelihood) * review_vector)
        neg_prob = np.log(self.neg_prior) + np.sum(np.log(self.neg_likelihood) * review_vector)

        if pos_prob > neg_prob:
            return "Positive"
        else:
            return "Negative"
        
    def calculate_accuracy(self, dataset, ground_truth_labels):
        correct_predictions = 0
        total_reviews = len(dataset)

        for i in range(total_reviews):
            review = dataset[i]
            predicted_label = self.classify(review)
            ground_truth_label = ground_truth_labels[i]

            if (predicted_label == "Positive" and ground_truth_label == 1) or (predicted_label == "Negative" and ground_truth_label == 0):
                correct_predictions += 1

        accuracy = correct_predictions / total_reviews
        return accuracy

    def calculate_confusion_matrix(self, dataset, ground_truth_labels):
        confusion_matrix = np.zeros((len(self.pols), len(self.pols)))

        for i in range(len(dataset)):
            review = dataset[i]
            predicted_label = self.classify(review)
            ground_truth_label = ground_truth_labels[i]

            confusion_matrix[ground_truth_label][int(predicted_label == "Positive")] += 1

        return confusion_matrix

        
    def print_results(self):
        print("Results:")
        for word, pos_count, neg_count in zip(self.words, self.sumposvectors, self.sumnegvectors):
            pos_percentage = pos_count / (len(self.pols) * 1000)
            neg_percentage = neg_count / (len(self.pols) * 1000)

            print(f"Word: {word}")
            print(f"  Positive: {pos_count}/{len(self.pols) * 1000} ({pos_percentage * 100:.2f}%)")
            print(f"  Negative: {neg_count}/{len(self.pols) * 1000} ({neg_percentage * 100:.2f}%)")
            print("")

if __name__ == "__main__":
    sentiment_analysis = SentimentAnalysis()
    sentiment_analysis.read_data()
    sentiment_analysis.create_vectors()
    sentiment_analysis.train()
    sentiment_analysis.print_results()

    # Test the classifier
    test_review = "This movie was awful."
    prediction = sentiment_analysis.classify(test_review)
    print(f"Test Review: '{test_review}'")
    print(f"Prediction: {prediction}")
    
    test_review = "This movie was great."
    prediction = sentiment_analysis.classify(test_review)
    print(f"Test Review: '{test_review}'")
    print(f"Prediction: {prediction}")
    
    test_review = "This movie was hilarious."
    prediction = sentiment_analysis.classify(test_review)
    print(f"Test Review: '{test_review}'")
    print(f"Prediction: {prediction}")

        # Test the classifier with the entire datasets
    test_reviews = sentiment_analysis.data
    ground_truth_labels = sentiment_analysis.polarity

    accuracy = sentiment_analysis.calculate_accuracy(test_reviews, ground_truth_labels)
    print(f"Classification Accuracy: {accuracy * 100:.2f}%")

    confusion_matrix = sentiment_analysis.calculate_confusion_matrix(test_reviews, ground_truth_labels)

    # Create a PrettyTable for the confusion matrix
    confusion_table = PrettyTable()
    confusion_table.field_names = ["", "Predicted Negative", "Predicted Positive"]
    confusion_table.add_row(["Actual Negative", confusion_matrix[0][0], confusion_matrix[0][1]])
    confusion_table.add_row(["Actual Positive", confusion_matrix[1][0], confusion_matrix[1][1]])

    print("Confusion Matrix:")
    print(confusion_table)