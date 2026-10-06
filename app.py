import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np

# ---------------------------------
# Load trained model
# ---------------------------------
model = tf.keras.models.load_model("handwritten_digit_model.keras")

# ---------------------------------
# Page title
# ---------------------------------
st.title("✍️ Handwritten Digit Recognizer")
st.write("Draw a digit from 0 to 9")

# ---------------------------------
# Canvas key
# ---------------------------------
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# ---------------------------------
# Clear button
# ---------------------------------
if st.button("🗑️ Clear"):
    st.session_state.canvas_key += 1
    st.rerun()

# ---------------------------------
# Drawing canvas
# ---------------------------------
canvas = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    return_image_data=True,
    key=f"canvas_{st.session_state.canvas_key}",
)

# ---------------------------------
# Predict button
# ---------------------------------
if st.button("🎯 Predict"):

    if canvas.image_data is None:
        st.warning("Please draw a digit first.")

    else:

        # Get grayscale image
        image = canvas.image_data[:, :, 0].astype(np.uint8)

        # ---------------------------------
        # Find the handwritten digit
        # ---------------------------------
        coords = np.argwhere(image > 30)

        if coords.size == 0:
            st.warning("Please draw a digit first.")

        else:

            # ---------------------------------
            # Crop empty space
            # ---------------------------------
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            image = image[
                y_min:y_max + 1,
                x_min:x_max + 1
            ]

            # ---------------------------------
            # Make image square
            # ---------------------------------
            height, width = image.shape
            size = max(height, width)

            square = np.zeros(
                (size, size),
                dtype=np.uint8
            )

            y_offset = (size - height) // 2
            x_offset = (size - width) // 2

            square[
                y_offset:y_offset + height,
                x_offset:x_offset + width
            ] = image

            # ---------------------------------
            # Resize to MNIST 28x28
            # ---------------------------------
            image = tf.image.resize(
                square[..., np.newaxis],
                (28, 28)
            ).numpy()

            # ---------------------------------
            # Normalize pixels
            # ---------------------------------
            image = image / 255.0

            # ---------------------------------
            # Add batch dimension
            # ---------------------------------
            image = image.reshape(
                1, 28, 28, 1
            )

            # ---------------------------------
            # Make prediction
            # ---------------------------------
            prediction = model.predict(
                image,
                verbose=0
            )

            predicted_digit = np.argmax(
                prediction
            )

            confidence = np.max(
                prediction
            )

            # ---------------------------------
            # Display result
            # ---------------------------------
            st.success(
                f"🎯 Prediction: {predicted_digit}"
            )

            st.info(
                f"Confidence: {confidence * 100:.2f}%"
            )