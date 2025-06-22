import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Handwritten Digit Image Generator")

# Definición completa del modelo (Autoencoder)
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Cargar el modelo completo
latent_dim = 20
model = Autoencoder(latent_dim)
model.load_state_dict(torch.load("mnist_autoencoder.pth", map_location=torch.device("cpu")))
model.eval()

decoder = model.decoder  # Usar solo el decoder

# Interfaz Streamlit
st.title("Handwritten Digit Image Generator")
st.markdown("Generate synthetic MNIST-like images using your trained model.")

digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))

if st.button("Generate Images"):
    torch.manual_seed(digit)  # Para generar diferentes imágenes por dígito
    z = torch.randn(5, latent_dim)
    with torch.no_grad():
        generated = decoder(z).numpy()

    st.subheader(f"Generated images of digit {digit}")
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(generated[i][0], cmap="gray")
        axs[i].axis("off")
        axs[i].set_title(f"Sample {i+1}")
    st.pyplot(fig)
