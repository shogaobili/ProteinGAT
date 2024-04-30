import re
import matplotlib.pyplot as plt

acc_train_values = []
acc_ss3_values = []
acc_ss8_values = []

with open('logfiles/3-9-3-8-layers.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        acc_train_match = re.search(r'loss_train: (\d+\.\d+)', line)
        acc_ss3_match = re.search(r'loss_val: (\d+\.\d+)', line)
        acc_ss8_match = re.search(r'acc_val: (\d+\.\d+)', line)
        if acc_train_match:
            acc_train = float(acc_train_match.group(1))
            acc_train_values.append(acc_train)
        if acc_ss3_match:
            acc_ss3 = float(acc_ss3_match.group(1))
            acc_ss3_values.append(acc_ss3)
        if acc_ss8_match:
            acc_ss8 = float(acc_ss8_match.group(1))
            acc_ss8_values.append(acc_ss8)

epochs = list(range(1, len(acc_train_values) + 1))

smooth_window = 100
acc_train_smooth = []
acc_ss3_smooth = []
acc_ss8_smooth = []

for i in range(len(epochs)):
    if i < smooth_window:
        acc_train_smooth.append(sum(acc_train_values[:i+1])/(i+1))
        acc_ss3_smooth.append(sum(acc_ss3_values[:i+1])/(i+1))
        acc_ss8_smooth.append(sum(acc_ss8_values[:i+1])/(i+1))
    else:
        acc_train_smooth.append(sum(acc_train_values[i-smooth_window:i+1])/smooth_window)
        acc_ss3_smooth.append(sum(acc_ss3_values[i-smooth_window:i+1])/smooth_window)
        acc_ss8_smooth.append(sum(acc_ss8_values[i-smooth_window:i+1])/smooth_window)

plt.plot(epochs, acc_train_smooth, label='loss_train')
plt.plot(epochs, acc_ss3_smooth, label='loss_val')
plt.plot(epochs, acc_ss8_smooth, label='acc_val')

plt.ylim((0, 2))
plt.yticks([i/10.0 for i in range(20)])

plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()

plt.show()