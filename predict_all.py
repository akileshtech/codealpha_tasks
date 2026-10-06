import tensorflow as tf
import numpy as np
from PIL import Image
import os


# ==========================================
# LOAD SAVED MODEL
# ==========================================

model = tf.keras.models.load_model(
    "handwritten_digit_model.keras"
)

print("Model loaded successfully!\n")


# ==========================================
# PREPROCESS IMAGE
# ==========================================

def preprocess_image(image_path):

    # Open image as grayscale
    image = Image.open(image_path).convert("L")

    # Convert to NumPy
    image = np.array(image)

    # Black digit on white background
    # becomes white digit on black background
    image = 255 - image

    # Remove very faint pixels
    image[image < 30] = 0

    # Find digit pixels
    coords = np.argwhere(image > 0)

    if coords.size == 0:
        return None

    # Crop around digit
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    image = image[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # --------------------------------------
    # Make digit fit inside 20x20
    # --------------------------------------

    height, width = image.shape

    scale = 20.0 / max(height, width)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    image_pil = Image.fromarray(image)

    image_pil = image_pil.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    image = np.array(image_pil)

    # --------------------------------------
    # Put digit on 28x28 canvas
    # --------------------------------------

    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    height, width = image.shape

    x_offset = (28 - width) // 2
    y_offset = (28 - height) // 2

    canvas[
        y_offset:y_offset + height,
        x_offset:x_offset + width
    ] = image

    # --------------------------------------
    # Center using center of mass
    # --------------------------------------

    total = canvas.sum()

    if total > 0:

        y_indices, x_indices = np.indices(canvas.shape)

        center_y = (y_indices * canvas).sum() / total
        center_x = (x_indices * canvas).sum() / total

        shift_y = int(round(13.5 - center_y))
        shift_x = int(round(13.5 - center_x))

        centered = np.zeros_like(canvas)

        src_y1 = max(0, -shift_y)
        src_y2 = min(28, 28 - shift_y)

        src_x1 = max(0, -shift_x)
        src_x2 = min(28, 28 - shift_x)

        dst_y1 = max(0, shift_y)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        dst_x1 = max(0, shift_x)
        dst_x2 = dst_x1 + (src_x2 - src_x1)

        centered[
            dst_y1:dst_y2,
            dst_x1:dst_x2
        ] = canvas[
            src_y1:src_y2,
            src_x1:src_x2
        ]

        canvas = centered

    # Normalize
    canvas = canvas.astype("float32") / 255.0

    # CNN input shape
    canvas = canvas.reshape(
        1, 28, 28, 1
    )

    return canvas


# ==========================================
# TEST ALL 10 DIGITS
# ==========================================

correct = 0
total = 0


for actual_digit in range(10):

    image_path = f"images/{actual_digit}.png"

    if not os.path.exists(image_path):

        print(
            f"{actual_digit}.png not found!"
        )

        continue

    image = preprocess_image(
        image_path
    )

    if image is None:

        print(
            f"{actual_digit}: No digit detected"
        )

        continue
    debug_image = (image[0,:,:,0] *255).astype(np.uint8)
    Image.fromarray(debug_image).save(
        f"debug_{actual_digit}.png"
    )

    # Predict
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

    total += 1

    if predicted_digit == actual_digit:

        correct += 1
        result = "CORRECT"

    else:

        result = "WRONG"

    print(
        f"Actual: {actual_digit} | "
        f"Predicted: {predicted_digit} | "
        f"Confidence: {confidence:.2%} | "
        f"{result}"
    )


# ==========================================
# FINAL RESULT
# ==========================================

print("\n======================")
print("FINAL RESULT")
print("======================")

print(
    f"Correct: {correct}/{total}"
)

if total > 0:

    accuracy = (
        correct / total
    ) * 100

    print(
        f"Accuracy: {accuracy:.2f}%"
    )