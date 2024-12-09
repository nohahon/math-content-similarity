import os
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the pickle file containing the training history (list of tuples)
pickle_folder_path = 'loss_acc/5e_6/'
pickle_files = [f for f in os.listdir(pickle_folder_path) if f.endswith('.pkl')]

print(pickle_files)

loss_values = []
accuracy_values = []
for pickle_file in pickle_files:
    pickle_file_path = os.path.join(pickle_folder_path, pickle_file)
    with open(pickle_file_path, 'rb') as f:
        training_history = pickle.load(f)
    loss_values.extend([item[1] for item in training_history])  # First element is loss
    accuracy_values.extend([item[0] for item in training_history])  # Second element is accuracy

# Ensure that any tensor is moved to the CPU, detached, and converted to numpy
loss_values = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in loss_values]
accuracy_values = [acc.detach().cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in accuracy_values]

# Group the data into batches of 6
batch_size = 6
num_batches = len(loss_values) // batch_size

# Compute the average loss and accuracy for each batch
batches_loss = []
batches_accuracy = []

for i in range(num_batches):
    batch_loss = np.mean(loss_values[i * batch_size: (i + 1) * batch_size])
    batch_accuracy = np.mean(accuracy_values[i * batch_size: (i + 1) * batch_size])

    batches_loss.append(batch_loss)
    batches_accuracy.append(batch_accuracy)

# Plotting Loss vs. Batch
plt.figure(figsize=(20, 6))
plt.plot(batches_loss, color='tab:red', label='Average Loss per Batch')
plt.xlabel('Batch')
plt.ylabel('Average Loss')
plt.title('Average Loss per Batch')
plt.tight_layout()
plt.savefig('plots_loss_acc/5e_6/average_loss_ep2_5e_6_compl.png')
plt.close()

# Plotting Accuracy vs. Batch
plt.figure(figsize=(20, 6))
plt.plot(batches_accuracy, color='tab:blue', label='Average Accuracy per Batch')
plt.xlabel('Batch')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy per Batch')
plt.tight_layout()
plt.savefig('plots_loss_acc/5e_6/average_acc_ep2_5e_6_compl.png')
plt.close()

# Optionally, display the plots
# plt.show()  # Uncomment if you want to display the plot as well

print("Plots saved: 'average_loss_per_batch.png' and 'average_accuracy_per_batch.png'")