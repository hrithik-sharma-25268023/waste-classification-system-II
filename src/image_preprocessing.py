"""image preprocessing
   reads dataset from path
"""

import tensorflow as tf

IMG_SIZE = (96, 96)
BATCH_SIZE = 32

def load_dataset(dataset_path, img_size=96, batch_size=32, val_split=0.2):
    
    IMG_SIZE = (img_size, img_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=val_split,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=val_split,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size
    )

    # Normalize (0–1)
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds