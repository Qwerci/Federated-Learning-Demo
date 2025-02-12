import streamlit as st
import matplotlib.pyplot as plt
from eval import evaluate_logs

st.title("Federated Learning Demo with Flower")

# Display training metrics
st.subheader("Training Metrics")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, 6), [0.85, 0.88, 0.90, 0.92, 0.93])
ax.set_title("Training Accuracy Over Rounds")
ax.set_xlabel("Rounds")
ax.set_ylabel("Accuracy")
st.pyplot(fig)

# Evaluate logs and display test evaluation
test_loss, test_accuracy = evaluate_logs()
st.subheader("Test Evaluation")
st.write(f"Test Loss: {test_loss:.4f}")
st.write(f"Test Accuracy: {test_accuracy:.4f}")
