import matplotlib.pyplot as plt

# Load the loss values
epochs = []
loss_values = []

with open('loss_values.txt', 'r') as file:
    for line in file:
        epoch, loss = line.split()
        epochs.append(int(epoch))
        loss_values.append(float(loss))

# Plot the loss curve
plt.plot(epochs, loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()