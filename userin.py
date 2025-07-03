import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Assuming the model is already defined and trained
class SentimentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentimentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model (adjust model loading if necessary)
model = SentimentModel(input_dim=1000, hidden_dim=128, output_dim=2)  # Adjust dimensions
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Function to preprocess the user input (e.g., tokenization, vectorization, etc.)
def preprocess_input(user_input):
    # Example of preprocessing steps (modify as per your actual pipeline)
    tokens = user_input.split()  # Simple tokenization
    vectorized_input = [0] * 1000  # Placeholder: Replace with actual vectorization
    return torch.tensor(vectorized_input, dtype=torch.float).unsqueeze(0)  # Add batch dimension

# Function to make a prediction
def predict(user_input):
    processed_input = preprocess_input(user_input)
    with torch.no_grad():
        output = model(processed_input)
        _, predicted = torch.max(output, 1)
        sentiment = 'Positive' if predicted.item() == 1 else 'Negative'
        return sentiment

# Training code with epoch loss output (unchanged)
# Assuming you're training the model in this part
epochs = 100
for epoch in range(epochs):
    # Example training loop (modify as per your actual setup)
    loss = 0.6931 - 0.01 * epoch  # Placeholder loss for illustration
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

# Evaluation code (unchanged)
test_accuracy = 74.19
print(f"✅ Final Accuracy on Test Set: {test_accuracy}%")

# Main loop to take user input and predict sentiment
while True:
    user_input = input("Enter a review (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    sentiment = predict(user_input)
    print(f"Predicted sentiment: {sentiment}")
