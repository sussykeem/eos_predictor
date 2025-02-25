import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from eos_dataloader import EOS_Dataloader
from cnn import CNN


eos_dataloader = EOS_Dataloader()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(cnn, epochs=100, learning_rate=0.0001, step_size=10, gamma=0.9):
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    cnn.to(device)  # Move the model to the device (GPU if available)

    loss_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(eos_dataloader.train_loader, 0):
            inputs, labels = data  # Get the inputs and labels from the data loader
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device

            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = cnn(inputs, predict=True)  # Forward pass
            loss = cnn.weighted_loss(outputs, labels)  # Calculate the loss
            loss_history.append(loss.item())
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the model's parameters

            running_loss += loss.item()  # Accumulate the loss for the epoch

        scheduler.step()

        # Print the average loss for the epoch
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {running_loss / len(eos_dataloader.train_loader):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    print("Training finished!")

    return cnn

# add hyperparameter search and loop for all models

def save_model(model, file_path="cnn_model.pth"):

    del model.output_layer

    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


cnn = CNN(300)

cnn = train_model(cnn)

save_model(cnn)