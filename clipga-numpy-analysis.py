import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(folder):
    data = {}
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.npy'):
            filepath = os.path.join(folder, filename)
            try:
                # Attempt to load the numpy array
                arr = np.load(filepath)
                if arr.size == 0:
                    print(f"Warning: '{filename}' is empty and will be skipped.")
                    continue
                key = int(filename.split('_')[0].replace('layer', '').replace('gradient', ''))
                data.setdefault(key, []).append(arr)
            except Exception as e:
                # Handle exceptions so one bad file doesn't stop the whole process
                print(f"Error loading '{filename}': {e}")
                continue
    return data

# Load activations and gradients
activations = load_data('activations')
gradients = load_data('gradients')

# Create analysis folder if it doesn't exist
analysis_folder = 'analysis'
os.makedirs(analysis_folder, exist_ok=True)

# Save activation plots
for layer in activations.keys():
    means = [np.mean(a) for a in activations[layer]]
    plt.figure()
    plt.plot(means)
    plt.title(f'Mean Activation for Layer {layer} Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Activation')
    plt.savefig(os.path.join(analysis_folder, f'activation_layer_{layer}.png'))
    plt.close()

# Save gradient plot
means_grad = [np.mean(g) for g in gradients]
plt.figure()
plt.plot(means_grad)
plt.title('Mean Gradient Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Mean Gradient')
plt.savefig(os.path.join(analysis_folder, 'mean_gradient.png'))
plt.close()
