#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <omp.h>
using namespace std;

#define EPOCHS 100
#define LEARNING_RATE 0.01

// Tokenize text
vector<string> tokenize(const string& text) {
    stringstream ss(text);
    string word;
    vector<string> tokens;
    while (ss >> word) {
        tokens.push_back(word);
    }
    return tokens;
}

// Create vocabulary
unordered_map<string, int> build_vocab(const vector<string>& texts) {
    unordered_map<string, int> vocab;
    int index = 0;
    for (const string& line : texts) {
        for (const string& word : tokenize(line)) {
            if (vocab.find(word) == vocab.end()) {
                vocab[word] = index++;
            }
        }
    }
    return vocab;
}

// Convert text to feature vector
vector<vector<float>> vectorize_texts(const vector<string>& texts, const unordered_map<string, int>& vocab) {
    vector<vector<float>> vectors(texts.size(), vector<float>(vocab.size(), 0.0));
    #pragma omp parallel for
    for (int i = 0; i < texts.size(); ++i) {
        for (const string& word : tokenize(texts[i])) {
            if (vocab.find(word) != vocab.end()) {
                vectors[i][vocab.at(word)] += 1;
            }
        }
    }
    return vectors;
}

// Sigmoid function
float sigmoid(float z) {
    return 1.0 / (1.0 + exp(-z));
}

// Predict
vector<float> predict(const vector<vector<float>>& X, const vector<float>& weights) {
    vector<float> predictions(X.size());
    #pragma omp parallel for
    for (int i = 0; i < X.size(); ++i) {
        float z = 0.0;
        for (int j = 0; j < X[0].size(); ++j) {
            z += X[i][j] * weights[j];
        }
        predictions[i] = sigmoid(z);
    }
    return predictions;
}

// Train using gradient descent and explicit matrix multiplication
void train(vector<vector<float>>& X, vector<int>& y, vector<float>& weights) {
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        vector<float> predictions = predict(X, weights);
        vector<float> gradient(weights.size(), 0.0);
        
        #pragma omp parallel for
        for (int j = 0; j < weights.size(); ++j) {
            float grad = 0.0;
            for (int i = 0; i < X.size(); ++i) {
                grad += (predictions[i] - y[i]) * X[i][j];
            }
            gradient[j] = grad / X.size();
        }

        #pragma omp parallel for
        for (int j = 0; j < weights.size(); ++j) {
            weights[j] -= LEARNING_RATE * gradient[j];
        }

        if (epoch % 10 == 0) {
            float loss = 0.0;
            for (int i = 0; i < y.size(); ++i) {
                loss += -y[i] * log(predictions[i] + 1e-9) - (1 - y[i]) * log(1 - predictions[i] + 1e-9);
            }
            cout << "Epoch " << epoch << ", Loss: " << loss / y.size() << endl;
        }
    }
}

// Evaluate accuracy
float evaluate(const vector<float>& predictions, const vector<int>& y) {
    int correct = 0;
    for (int i = 0; i < y.size(); ++i) {
        if ((predictions[i] >= 0.5 && y[i] == 1) || (predictions[i] < 0.5 && y[i] == 0)) {
            ++correct;
        }
    }
    return float(correct) / y.size();
}

// Load data
void load_data(const string& filepath, vector<string>& texts, vector<int>& labels) {
    ifstream file(filepath);
    string line;
    while (getline(file, line)) {
        size_t comma_pos = line.find_last_of(',');
        string text = line.substr(0, comma_pos);
        int label = stoi(line.substr(comma_pos + 1));
        texts.push_back(text);
        labels.push_back(label);
    }
}

int main() {
    vector<string> texts;
    vector<int> labels;

    load_data("movie_reviews.csv", texts, labels);

    unordered_map<string, int> vocab = build_vocab(texts);
    vector<vector<float>> X = vectorize_texts(texts, vocab);
    vector<float> weights(vocab.size(), 0.0);

    train(X, labels, weights);

    vector<float> predictions = predict(X, weights);
    float acc = evaluate(predictions, labels);
    cout << "Accuracy: " << acc * 100 << "%" << endl;

    return 0;
}
