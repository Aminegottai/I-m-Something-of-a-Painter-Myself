# train_cyclegan.py
import os
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers

# -------------------------
# 0. Configuration
# -------------------------
photo_dir = "/kaggle/input/gan-getting-started/photo_jpg"
monet_dir = "/kaggle/input/gan-getting-started/monet_jpg"
output_dir = Path("/kaggle/working/generated")
output_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir = Path("/kaggle/working/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 50
IMG_SIZE = 256
lambda_cycle = 15.0
AUTOTUNE = tf.data.AUTOTUNE

# Vérifier GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# -------------------------
# 1. Prétraitement
# -------------------------
def preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    # Normalisation [-1, 1]
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img

def make_dataset(dir_path, batch_size):
    files = sorted([str(p) for p in Path(dir_path).glob("*.jpg")] + [str(p) for p in Path(dir_path).glob("*.png")])
    if len(files) == 0:
        raise ValueError(f"Aucune image trouvée dans {dir_path}")
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.shuffle(buffer_size=len(files))
    ds = ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds

photo_ds = make_dataset(photo_dir, BATCH_SIZE)
monet_ds = make_dataset(monet_dir, BATCH_SIZE)

print(f"Total photos dataset batches: {len(list(photo_ds))}")
print(f"Total monet dataset batches: {len(list(monet_ds))}")

# -------------------------
# 2. Générateur (U-Net-like)
# -------------------------
def build_generator():
    inputs = layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])

    # Encoder
    e1 = layers.Conv2D(64, 4, strides=2, padding='same', use_bias=False)(inputs)
    e1 = layers.BatchNormalization()(e1)
    e1 = layers.LeakyReLU(0.2)(e1)

    e2 = layers.Conv2D(128, 4, strides=2, padding='same', use_bias=False)(e1)
    e2 = layers.BatchNormalization()(e2)
    e2 = layers.LeakyReLU(0.2)(e2)

    e3 = layers.Conv2D(256, 4, strides=2, padding='same', use_bias=False)(e2)
    e3 = layers.BatchNormalization()(e3)
    e3 = layers.LeakyReLU(0.2)(e3)

    b = layers.Conv2D(512, 4, strides=2, padding='same', use_bias=False)(e3)
    b = layers.BatchNormalization()(b)
    b = layers.LeakyReLU(0.2)(b)

    # Decoder (upsample)
    d1 = layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False)(b)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.ReLU()(d1)
    d1 = layers.Concatenate()([d1, e3])

    d2 = layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(d1)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.ReLU()(d2)
    d2 = layers.Concatenate()([d2, e2])

    d3 = layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(d2)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.ReLU()(d3)
    d3 = layers.Concatenate()([d3, e1])

    outputs = layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(d3)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="Generator")

# -------------------------
# 3. Discriminateur (PatchGAN)
# -------------------------
def build_discriminator():
    inputs = layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])
    x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)

    # Patch output
    x = layers.Conv2D(1, 4, strides=1, padding='same')(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name="Discriminator")

# -------------------------
# 4. Construire modèles
# -------------------------
G_photo2monet = build_generator()
G_monet2photo = build_generator()
D_photo = build_discriminator()
D_monet = build_discriminator()

# -------------------------
# 5. Losses & Optimizers
# -------------------------
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    return real_loss + generated_loss

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
    # L1 loss
    return tf.reduce_mean(tf.abs(real_image - cycled_image))

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

# Build optimizers with model variables to ensure consistency
generator_optimizer.build(G_photo2monet.trainable_variables + G_monet2photo.trainable_variables)
discriminator_optimizer.build(D_photo.trainable_variables + D_monet.trainable_variables)

# -------------------------
# 6. Checkpoint Manager
# -------------------------
checkpoint = tf.train.Checkpoint(
    G_photo2monet=G_photo2monet,
    G_monet2photo=G_monet2photo,
    D_photo=D_photo,
    D_monet=D_monet,
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer
)
manager = tf.train.CheckpointManager(checkpoint, directory=str(checkpoint_dir), max_to_keep=5)

# -------------------------
# 7. Train step
# -------------------------
@tf.function
def train_step(photo, monet):
    with tf.GradientTape(persistent=True) as tape:
        # Générations
        fake_monet = G_photo2monet(photo, training=True)
        cycled_photo = G_monet2photo(fake_monet, training=True)

        fake_photo = G_monet2photo(monet, training=True)
        cycled_monet = G_photo2monet(fake_photo, training=True)

        # Discriminateurs
        disc_real_monet = D_monet(monet, training=True)
        disc_fake_monet = D_monet(fake_monet, training=True)

        disc_real_photo = D_photo(photo, training=True)
        disc_fake_photo = D_photo(fake_photo, training=True)

        # Pertes générateurs (adversarial)
        gen_loss_monet = generator_loss(disc_fake_monet)
        gen_loss_photo = generator_loss(disc_fake_photo)

        # Cycle losses
        cycle_loss_photo = calc_cycle_loss(photo, cycled_photo)
        cycle_loss_monet = calc_cycle_loss(monet, cycled_monet)
        total_cycle_loss = lambda_cycle * (cycle_loss_photo + cycle_loss_monet)

        total_gen_loss = gen_loss_monet + gen_loss_photo + total_cycle_loss

        # Pertes discriminateurs
        disc_loss_monet = discriminator_loss(disc_real_monet, disc_fake_monet)
        disc_loss_photo = discriminator_loss(disc_real_photo, disc_fake_photo)

    # Gradients générateurs (tous les paramètres des 2 générateurs)
    gen_variables = G_photo2monet.trainable_variables + G_monet2photo.trainable_variables
    generator_grads = tape.gradient(total_gen_loss, gen_variables)
    generator_optimizer.apply_gradients(zip(generator_grads, gen_variables))

    # Gradients discriminateurs séparément (pour stabilité)
    disc_monet_grads = tape.gradient(disc_loss_monet, D_monet.trainable_variables)
    if disc_monet_grads is not None:
        discriminator_optimizer.apply_gradients(zip(disc_monet_grads, D_monet.trainable_variables))

    disc_photo_grads = tape.gradient(disc_loss_photo, D_photo.trainable_variables)
    if disc_photo_grads is not None:
        discriminator_optimizer.apply_gradients(zip(disc_photo_grads, D_photo.trainable_variables))

    del tape

# -------------------------
# 8. Entraînement
# -------------------------
print("Début de l'entraînement...")
for epoch in range(EPOCHS):
    print(f"Époque {epoch+1}/{EPOCHS}")
    # Utiliser zip_longest-like behavior: on s'arrête à la plus courte pour simplicité
    for i, (photo_batch, monet_batch) in enumerate(zip(photo_ds, monet_ds)):
        train_step(photo_batch, monet_batch)
        if (i + 1) % 50 == 0:
            print(f"  - batch {i+1}")

    # Sauvegarde checkpoint à chaque epoch
    save_path = manager.save()
    print(f"Checkpoint sauvegardé: {save_path}")

# -------------------------
# 9. Génération & sauvegarde des images
# -------------------------
print("Génération d'images en style Monet...")
count = 0
for i, photo_batch in enumerate(photo_ds):
    fake_monet_batch = G_photo2monet(photo_batch, training=False)
    fake_monet_batch = (fake_monet_batch + 1.0) * 127.5  # remettre en [0,255]
    fake_monet_batch = tf.cast(fake_monet_batch, tf.uint8).numpy()

    for im_arr in fake_monet_batch:
        im = Image.fromarray(im_arr)
        im.save(output_dir / f"image_{count}.png")
        count += 1

print(f"✅ {count} images générées.")

# Vérification exigence Kaggle (optionnel)
if 7000 <= count <= 10000:
    print(f"✅ Nombre d'images ({count}) conforme aux exigences de Kaggle.")
else:
    print(f"⚠️ Nombre d'images ({count}) non conforme. Kaggle exige 7000–10000 images (si nécessaire).")

# -------------------------
# 10. Compression .zip
# -------------------------
zip_path = "/kaggle/working/images.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for img_file in output_dir.glob("*.png"):
        zipf.write(img_file, arcname=img_file.name)

print("✅ images.zip prêt pour la soumission :", zip_path)