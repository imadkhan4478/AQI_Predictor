"""A small feedforward neural network (TensorFlow/Keras) baseline."""

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Set via Keras' helper so Python's `random`, NumPy and TensorFlow are all
# covered; seeding tf.random alone leaves initialisers free to vary, and two runs
# then differ by more than the gaps the comparison is measuring.
RANDOM_SEED = 42


def build_model(n_features):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_and_predict(X_train, y_train, X_test):
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(n_features=X_train_scaled.shape[1])
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.15,
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0,
    )
    return model.predict(X_test_scaled, verbose=0).flatten()
