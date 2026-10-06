import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ============================================================
# 1. LOAD MNIST DATASET
# ============================================================

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training images:", x_train.shape)
print("Test images:", x_test.shape)


# ============================================================
# 2. NORMALIZE PIXELS
# ============================================================

x_train = x_train / 255.0
x_test = x_test / 255.0


# ============================================================
# 3. ADD CHANNEL DIMENSION FOR CNN
# ============================================================

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print("Training shape:", x_train.shape)
print("Test shape:", x_test.shape)


# ============================================================
# 4. CREATE CNN MODEL
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(28, 28, 1)),

    # First convolution
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    # Reduce image size
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Second convolution
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    # Reduce image size again
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Convert feature maps into vector
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # 10 digit classes: 0-9
    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# ============================================================
# 5. DISPLAY MODEL ARCHITECTURE
# ============================================================

model.summary()


# ============================================================
# 6. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)
# Save trained model
model.save("handwritten_digit_model.keras")

print("✅ Model saved successfully!")

# ============================================================
# 8. TEST MODEL
# ============================================================

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test
)

print()
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================================
# 9. TEST ONE MNIST IMAGE
# ============================================================

index = 0

prediction = model.predict(
    x_test[index:index + 1],
    verbose=0
)

predicted_digit = np.argmax(prediction)
confidence = np.max(prediction)

print()
print("MNIST Test Prediction")
print("---------------------")
print("Predicted digit:", predicted_digit)
print("Actual digit:", y_test[index])
print("Confidence:", confidence)


# ============================================================
# 10. LOAD YOUR OWN HANDWRITTEN IMAGE
# ============================================================

print()
print("Processing your handwritten image...")


image = Image.open(
    "images/my_digit.png"
).convert("L")


# ============================================================
# 11. CONVERT IMAGE TO NUMPY
# ============================================================

image = np.array(image)


# ============================================================
# 12. INVERT COLORS
# ============================================================

# Paint:
# Black digit + white background
#
# MNIST:
# White digit + black background

image = 255 - image


# ============================================================
# 13. FIND THE DIGIT
# ============================================================

coords = np.argwhere(image > 30)


if coords.size == 0:

    print("No digit detected!")

else:

    # Find bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    print(
        "Detected digit area:",
        x_min,
        y_min,
        x_max,
        y_max
    )


    # ========================================================
    # 14. CROP THE DIGIT
    # ========================================================

    image = image[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]


    # ========================================================
    # 15. MAKE IMAGE SQUARE
    # ========================================================

    height, width = image.shape

    size = max(
        height,
        width
    )

    canvas = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    y_offset = (size - height) // 2
    x_offset = (size - width) // 2

    canvas[
        y_offset:y_offset + height,
        x_offset:x_offset + width
    ] = image


    # ========================================================
    # 16. RESIZE TO MNIST SIZE
    # ========================================================

    image = Image.fromarray(canvas)

    image = image.resize(
        (28, 28)
    )


    # ========================================================
    # 17. CONVERT BACK TO NUMPY
    # ========================================================

    image = np.array(image)


    # ========================================================
    # 18. DISPLAY PROCESSED IMAGE
    # ========================================================

    plt.imshow(image, cmap="gray")
    plt.title("Processed Handwritten Digit")
    plt.axis("off")
    plt.savefig("processed_digit.png")
    plt.close()

    print("Processed image saved as: processed_digit.png")

    # ========================================================
    # 19. NORMALIZE
    # ========================================================

    image = image / 255.0


    # ========================================================
    # 20. ADD BATCH + CHANNEL DIMENSIONS
    # ========================================================

    image = image.reshape(
        1,
        28,
        28,
        1
    )


    # ========================================================
    # 21. PREDICT YOUR HANDWRITTEN DIGIT
    # ========================================================

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


    # ========================================================
    # 22. DISPLAY RESULT
    # ========================================================

    print()
    print("YOUR HANDWRITTEN DIGIT")
    print("----------------------")

    print(
        "Predicted digit:",
        predicted_digit
    )

    print(
        "Confidence:",
        confidence
    )