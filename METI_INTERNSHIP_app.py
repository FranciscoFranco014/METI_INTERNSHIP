
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Handwritten Digit Image Generator")

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

# Cargar modelo
latent_dim = 20
decoder = Decoder(latent_dim)
decoder.load_state_dict(torch.load("mnist_autoencoder.pth", map_location=torch.device("cpu")))
decoder.eval()

st.title("Handwritten Digit Image Generator")
st.markdown("Generate synthetic MNIST-like images using your trained model.")

digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))
if st.button("Generate Images"):
    torch.manual_seed(digit)  # Seed para generar imágenes distintas por dígito
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
