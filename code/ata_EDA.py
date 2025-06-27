# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as k
from tensorflow import keras
from tensorflow.keras import layers, Model
import os
import sys
import gc
import matplotlib.pyplot as plt
import polars as pl
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# %%

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("GPU:", gpu)
    tf.config.experimental.set_memory_growth(gpu, True)
# %%

# df = pd.read_parquet("/fast_scratch_1/TRISEP_data/AdvancedTutorial/small_dataset.parquet", engine='pyarrow')
df = pl.read_parquet("/fast_scratch_1/TRISEP_data/AdvancedTutorial/small_dataset.parquet")
# %%
def target_z(dataset, bins, min, max, normalize, alpha):
    z = pl.scan_parquet(dataset).select(z=pl.col("target").arr.get(2)).collect()["z"]
    min = z.min() if min is None else min
    max = z.max() if max is None else max

    plt.hist(
        z, bins=bins, range=(min, max), density=normalize, alpha=alpha, label=dataset
    )

def cloud_size(dataset, bins, min, max, normalize, alpha):
    size = (
        pl.scan_parquet(dataset)
        .select(size=pl.col("point_cloud").list.len())
        .collect()["size"]
    )
    min = size.min() if min is None else min
    max = size.max() if max is None else max

    plt.hist(
        size, bins=bins, range=(min, max), density=normalize, alpha=alpha, label=dataset
    )

# %%
# Hardcoded variables - modify these as needed
# subcommand = "target-z"  # Change to "cloud-size" or "target-z"
subcommand = "cloud-size"  # Change to "cloud-size" or "target-z"
data_files = ["/fast_scratch_1/TRISEP_data/AdvancedTutorial/small_dataset.parquet"]  # List of data files
output_file = None  # Set to a file path string to save, or None to show plot
bins = 100
max_val = None  # Set to float value or None for auto
min_val = None  # Set to float value or None for auto
normalize = False  # Set to True to normalize

alpha = 1.0 if len(data_files) == 1 else 0.5
for dataset in data_files:
    if subcommand == "target-z":
        target_z(dataset, bins, min_val, max_val, normalize, alpha)
        plt.xlabel("z [mm]")
        plt.ylabel("Count")
    elif subcommand == "cloud-size":
        cloud_size(dataset, bins, min_val, max_val, normalize, alpha)
        plt.xlabel("Number of points")
        plt.ylabel("Count")

plt.legend()

if output_file:
    plt.savefig(output_file)
    print(f"Created `{output_file}`")
else:
    plt.show()

# %%


from data.dataset import PointCloudDataset

config = {"cloud_size": 140}
dataset = PointCloudDataset(
    "/fast_scratch_1/TRISEP_data/AdvancedTutorial/small_dataset.parquet", config
)

index = 0  # First event
point_cloud, target = dataset[index]

# # %%
# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")

# ax.scatter(point_cloud[0], point_cloud[1], point_cloud[2])
# ax.scatter(0, 0, target.item(), color="red")

# %%

# for i in range(10):
#     point_cloud, target = dataset[i]
#     print(f"point_cloud shape: {point_cloud.shape}, target shape: {target.shape}")
#     print(f"target: {target}")
# %%

np_data = np.zeros((100_000, 3, 140))
np_labels = np.zeros((100_000, 1))

for i in range(100_000):
    point_cloud, target = dataset[i]
    np_data[i] = point_cloud.numpy()
    np_labels[i] = target.numpy()

# %%

data_transposed = np_data.transpose(0, 2, 1)  # Transpose to (B, L, C)
# data_transposed = np_data.transpose(0, 2, 1)  # Transpose to (B, L, C)

# We need to train test and validation split
# X_train, X_test, y_train, y_test = train_test_split(np_data, np_labels, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(data_transposed, np_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# %%




# # def create_custom_model(data, cfg, dense_units=256, num_of_layers=6, activation='elu', dropout=0.1):
# def create_custom_model(input_shape, cfg, dense_units=256, num_of_layers=6, activation='elu', dropout=0.1):
#     # normalization = k.layers.Normalization()
#     # normalization.adapt(data)
#     inputs = k.Input(shape=input_shape)
#     # x = normalization(inputs)
#     x = k.layers.Dense(dense_units, activation=activation)(inputs)

#     for _ in range(num_of_layers):
#         x = k.layers.Dense(dense_units)(x)
#         x = k.layers.BatchNormalization()(x)
#         x = k.layers.Activation(activation)(x)
#         x = k.layers.Dropout(dropout)(x)
#     if cfg.experiment.db:
#         x = k.layers.GlobalAveragePooling1D()(x)
#     outputs = k.layers.Dense(1, activation='sigmoid')(x)
#     model = k.Model(inputs=inputs, outputs=outputs)
#     return model
    

# %%

# class DeepSet(k.Model):
#     def __init__(self,
#                  phi_units=(64, 64),
#                  rho_units=(64, 64),
#                  output_units=1,
#                  aggregator='sum',  # one of 'sum','mean','max'
#                  use_mean=False,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.use_mean = use_mean
#         self.aggregator = aggregator

#         # φ network: maps each point → embedding
#         self.phi = tf.keras.Sequential([
#             k.layers.Dense(u, activation="relu") for u in phi_units
#         ])

#         # ρ network: maps aggregated embedding → output
#         self.rho = tf.keras.Sequential(
#             [k.layers.Dense(u, activation="relu") for u in rho_units]
#             + [k.layers.Dense(output_units, activation=None)]
#         )

#     def call(self, x, training=False):
#         # x: (batch, n_points, features)
#         h = self.phi(x, training=training)   # → (batch, n_points, φ_dim)
#         if self.aggregator == 'mean':
#             s = tf.reduce_mean(h, axis=2)
#         elif self.aggregator == 'max':
#             s = tf.reduce_max(h, axis=2)
#         else:  # 'sum'
#             s = tf.reduce_sum(h, axis=2)
#         y = self.rho(s, training=training)   # → (batch, output_units)
#         return tf.squeeze(y, axis=-1)        # → (batch,)

# class DeepSet(k.Model):
#     def __init__(self,
#                  phi_units=(64, 64),
#                  rho_units=(64, 64),
#                  output_units=1,
#                  aggregator='max',
#                  dropout=0.1,
#                  use_coords=True,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.aggregator = aggregator
#         self.dropout = dropout
#         self.use_coords = use_coords

#         # Enhanced φ network with geometric features and deeper architecture
#         phi_layers = []
#         input_dim = 3 if not use_coords else 6  # x,y,z + distance features
        
#         for i, units in enumerate(phi_units):
#             phi_layers.append(k.layers.Dense(units, activation=None))
#             phi_layers.append(k.layers.BatchNormalization())
#             phi_layers.append(k.layers.ReLU())
#             if dropout > 0:
#                 phi_layers.append(k.layers.Dropout(dropout))
        
#         self.phi = tf.keras.Sequential(phi_layers)

#         # Enhanced ρ network with residual connections
#         rho_layers = []
#         prev_units = phi_units[-1]
        
#         for i, units in enumerate(rho_units):
#             rho_layers.append(k.layers.Dense(units, activation=None))
#             rho_layers.append(k.layers.BatchNormalization())
#             rho_layers.append(k.layers.ReLU())
#             if dropout > 0:
#                 rho_layers.append(k.layers.Dropout(dropout))
#             prev_units = units
            
#         rho_layers.append(k.layers.Dense(output_units, activation=None))
#         self.rho = tf.keras.Sequential(rho_layers)

#     def add_geometric_features(self, x):
#         """Add geometric features to make the model more spatially aware."""
#         # x: (batch, n_points, 3) -> (batch, n_points, 6)
        
#         # Compute distances from origin
#         distances = tf.norm(x, axis=-1, keepdims=True)  # (batch, n_points, 1)
        
#         # Compute centroid and distances from centroid
#         centroid = tf.reduce_mean(x, axis=1, keepdims=True)  # (batch, 1, 3)
#         centroid_distances = tf.norm(x - centroid, axis=-1, keepdims=True)  # (batch, n_points, 1)
        
#         # Concatenate original coordinates with geometric features
#         enhanced_features = tf.concat([x, distances, centroid_distances], axis=-1)
#         return enhanced_features

#     def call(self, x, training=False):
#         # x: (batch, n_points, features)
        
#         # Add geometric features if enabled
#         if self.use_coords:
#             x = self.add_geometric_features(x)
        
#         # Apply φ network to each point
#         h = self.phi(x, training=training)   # → (batch, n_points, φ_dim)
        
#         # Multi-scale aggregation - combine different aggregation methods
#         if self.aggregator == 'multi':
#             s_max = tf.reduce_max(h, axis=1)     # (batch, φ_dim)
#             s_mean = tf.reduce_mean(h, axis=1)   # (batch, φ_dim)
#             s_sum = tf.reduce_sum(h, axis=1)     # (batch, φ_dim)
#             s = tf.concat([s_max, s_mean, s_sum], axis=-1)  # (batch, 3*φ_dim)
#         elif self.aggregator == 'mean':
#             s = tf.reduce_mean(h, axis=1)  # Fixed: was axis=2
#         elif self.aggregator == 'max':
#             s = tf.reduce_max(h, axis=1)   # Fixed: was axis=2
#         else:  # 'sum'
#             s = tf.reduce_sum(h, axis=1)   # Fixed: was axis=2
            
#         # Apply ρ network
#         y = self.rho(s, training=training)   # → (batch, output_units)
#         return tf.squeeze(y, axis=-1)        # → (batch,)

# # %%

# regularizer = k.regularizers.l2(1e-5)
# def build_set_transformer(data,
#                          num_heads: int = 8,
#                          embed_dim: int = 64,
#                          ff_dim: int = 256,
#                          num_tr_blocks: int = 4,
#                          dropout: float = 0.1,
#                          activation: str = 'gelu'):
#     if embed_dim % num_heads != 0:
#         raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
#     # Input: (batch_size, num_events, num_features)
#     inputs = k.Input(shape=data.shape[1:], name="collision_events")
#     # # Normalization
#     # norm_layer = k.layers.Normalization()
#     # norm_layer.adapt(data)
#     # x = norm_layer(inputs)
#     # Project to embedding dimension if needed
#     if data.shape[-1] != embed_dim:
#         x = k.layers.Dense(embed_dim, name="event_embedding")(inputs)
#     # Self-attention transformer blocks
#     for i in range(num_tr_blocks):
#         # Multi-head self-attention - events attend to each other
#         attention = k.layers.MultiHeadAttention(
#             num_heads=num_heads,
#             key_dim=embed_dim // num_heads,
#             dropout=dropout,
#             name=f"self_attention_{i}"
#         )(x, x)  # No masking - all events can attend to all events
#         # Add & Norm
#         x = k.layers.Add(name=f"add_attention_{i}")([x, attention])
#         x = k.layers.LayerNormalization(epsilon=1e-6, name=f"ln_attention_{i}")(x)
#         # Feed-forward network
#         ff = k.layers.Dense(ff_dim, activation=activation, name=f"ff1_{i}")(x)
#         ff = k.layers.Dense(embed_dim, name=f"ff2_{i}")(ff)
#         if dropout > 0:
#             ff = k.layers.Dropout(dropout, name=f"ff_dropout_{i}")(ff)
#         # Add & Norm
#         x = k.layers.Add(name=f"add_ff_{i}")([x, ff])
#         x = k.layers.LayerNormalization(epsilon=1e-6, name=f"ln_ff_{i}")(x)
#     # Permutation-invariant aggregation - bag-level representation
#     # GlobalAveragePooling1D gives same result regardless of event order
#     bag_representation = k.layers.GlobalAveragePooling1D(name="bag_pooling")(x)
#     # Classification head
#     x = k.layers.Dense(ff_dim // 2, activation=activation, name="classifier_dense")(bag_representation)
#     if dropout > 0:
#         x = k.layers.Dropout(dropout, name="classifier_dropout")(x)
#     # Binary classification output
#     outputs = k.layers.Dense(1, activation='sigmoid', name="bag_classification")(x)
#     #
#     return k.Model(inputs=inputs, outputs=outputs, name="SetTransformer")
# model = build_set_transformer(X_train, 
#                               num_heads = 4, 
#                               embed_dim = 16, 
#                               ff_dim    = 32, 
#                               num_tr_blocks = 2, 
#                               dropout  = 0.1)  



# %%

# class TNet(Model):
#     """Learns a (C×C) feature alignment matrix."""
#     def __init__(self, conv_dims, fc_dims, out_dim):
#         super().__init__()
#         # 1) shared‐MLP on each point
#         self.conv_blocks = []
#         for d in conv_dims:
#             self.conv_blocks += [
#                 layers.Conv1D(d, 1, activation=None),
#                 layers.BatchNormalization(),
#                 layers.ReLU()
#             ]
#         # 2) MLP on the global max‐pooled feature
#         self.fc_blocks = []
#         for d in fc_dims:
#             self.fc_blocks += [
#                 layers.Dense(d, activation=None),
#                 layers.BatchNormalization(),
#                 layers.ReLU()
#             ]
#         # final linear layer to predict C*C entries
#         self.out = layers.Dense(out_dim)

#     def call(self, x):
#         # x: (B, L, C)
#         for layer in self.conv_blocks:
#             x = layer(x)
#         x = tf.reduce_max(x, axis=1)   # (B, C_out)
#         for layer in self.fc_blocks:
#             x = layer(x)
#         return self.out(x)             # (B, C*C)


# class Regressor(Model):
#     """PointNet-style regressor predicting a single scalar per cloud."""
#     def __init__(self, config):
#         super().__init__()
#         # --- 1) Pre-TNet feature extractor
#         pre_layers = []
#         in_dim = 3
#         for d in config['pre_dims']:
#             pre_layers += [
#                 layers.Conv1D(d, 1, activation=None),
#                 layers.BatchNormalization(),
#                 layers.ReLU()
#             ]
#             in_dim = d
#         self.pre_net = tf.keras.Sequential(pre_layers)

#         # remember feature dim
#         self.num_feats = in_dim

#         # --- 2) Feature transform T-Net
#         self.tnet = TNet(
#             conv_dims  = config['tnet_conv'],
#             fc_dims    = config['tnet_fc'],
#             out_dim    = self.num_feats * self.num_feats
#         )

#         # --- 3) Post-TNet feature extractor
#         post_layers = []
#         in_dim = self.num_feats
#         for d in config['post_dims']:
#             post_layers += [
#                 layers.Conv1D(d, 1, activation=None),
#                 layers.BatchNormalization(),
#                 layers.ReLU()
#             ]
#             in_dim = d
#         self.post_net = tf.keras.Sequential(post_layers)

#         # --- 4) Final MLP to one scalar
#         fc = []
#         for d in config['fc_dims']:
#             fc += [layers.Dense(d, activation='relu')]
#         fc += [layers.Dense(1)]   # output
#         self.head = tf.keras.Sequential(fc)

#     def call(self, x):
#         # x: (B, L, 3)
#         x = self.pre_net(x)                     # (B, L, num_feats)
#         t = self.tnet(x)                        # (B, num_feats*num_feats)
#         t = tf.reshape(t, (-1, self.num_feats, self.num_feats))
#         # apply transform to each point: (B, L, C) @ (B, C, C) → (B, L, C)
#         x = tf.matmul(x, t)
#         x = self.post_net(x)                    # (B, L, out_feats)
#         x = tf.reduce_max(x, axis=1)            # global max‐pool → (B, out_feats)
#         y = self.head(x)                        # (B, 1)
#         return tf.squeeze(y, axis=-1)           # (B,)

# # === example instantiation & summary ===
# config = {
#     'pre_dims':    [64, 64],
#     'tnet_conv':   [64, 128, 1024],
#     'tnet_fc':     [512, 256],
#     'post_dims':   [64, 128, 1024],
#     'fc_dims':     [512, 256],
# }
# tf_regressor = Regressor(config)
# tf_regressor.build((None, 140, 3))
# tf_regressor.summary()

# optimizer = k.optimizers.Adam(learning_rate=1e-3)
# loss_fn = k.losses.MeanSquaredError()

# tf_regressor.compile(optimizer=optimizer, loss=loss_fn, metrics=["mae"])
# # %%
# # 4. Train the model
# print("\n--- Starting Training ---")
# history = tf_regressor.fit(
#     X_train,
#     y_train,
#     batch_size=128,
#     epochs=20,
#     validation_data=(X_val, y_val),
#     verbose=1
# )
# print("--- Training Finished ---\n")

# # 5. Evaluate the model on the test set
# print("--- Evaluating on Test Set ---")
# test_loss, test_mae = tf_regressor.evaluate(X_test, y_test, verbose=0)
# print(f"Test Loss: {test_loss:.4f}")
# print(f"Test MAE: {test_mae:.4f}")

# # 6. Plot training history
# print("\n--- Plotting Training History ---")
# history_data = history.history
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.plot(history_data['loss'], label='Training Loss')
# plt.plot(history_data['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss (MSE)')
# plt.title('Training and Validation Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history_data['mae'], label='Training MAE')
# plt.plot(history_data['val_mae'], label='Validation MAE')
# plt.xlabel('Epochs')
# plt.ylabel('Mean Absolute Error')
# plt.title('Training and Validation MAE')
# plt.legend()

# plt.tight_layout()
# plt.show()


# %%

class DeepSet(k.Model):
    def __init__(self,
                 phi_units=(64, 64),
                 rho_units=(64, 64),
                 output_units=1,
                 aggregator='max',  # 'max' often works better than 'sum'
                 dropout=0.3,       # Moderate dropout
                 l2_reg=1e-4,       # Light L2 regularization
                 **kwargs):
        super().__init__(**kwargs)
        self.aggregator = aggregator
        
        # L2 regularizer
        regularizer = k.regularizers.l2(l2_reg)

        # φ network: maps each point → embedding with regularization
        phi_layers = []
        for u in phi_units:
            phi_layers.append(k.layers.Dense(u, 
                                           activation=None,
                                           kernel_regularizer=regularizer))
            phi_layers.append(k.layers.BatchNormalization())
            phi_layers.append(k.layers.Activation('elu'))
            if dropout > 0:
                phi_layers.append(k.layers.Dropout(dropout))
        self.phi = tf.keras.Sequential(phi_layers)

        # ρ network: maps aggregated embedding → output with regularization
        rho_layers = []
        for u in rho_units:
            rho_layers.append(k.layers.Dense(u, 
                                           activation=None,
                                           kernel_regularizer=regularizer))
            rho_layers.append(k.layers.BatchNormalization())
            rho_layers.append(k.layers.Activation('elu'))
            if dropout > 0:
                rho_layers.append(k.layers.Dropout(dropout))
        rho_layers.append(k.layers.Dense(output_units, 
                                       activation=None,
                                       kernel_regularizer=regularizer))
        self.rho = tf.keras.Sequential(rho_layers)

    def call(self, x, training=False):
        # x: (batch, n_points, features)
        h = self.phi(x, training=training)   # → (batch, n_points, φ_dim)
        
        if self.aggregator == 'mean':
            s = tf.reduce_mean(h, axis=1)
        elif self.aggregator == 'max':
            s = tf.reduce_max(h, axis=1)
        else:  # 'sum'
            s = tf.reduce_sum(h, axis=1)
            
        y = self.rho(s, training=training)   # → (batch, output_units)
        return tf.squeeze(y, axis=-1)        # → (batch,)

# %%
# Create regularized vanilla DeepSet
model = DeepSet(
    phi_units=(1024, 1024, 1024, 1024),
    rho_units=(128, 128, 128, 128),
    output_units=1,
    aggregator='max',
    dropout=0.2,               # Moderate dropout
    l2_reg=1e-2                # Light L2 regularization
)

# Compile with reasonable learning rate
model.compile(
    optimizer=k.optimizers.Adam(learning_rate=1e-3),
    loss='mse',
    metrics=['mae']
)

# Simple callbacks
callbacks = [
    k.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True
    )
]

# Train
history = model.fit(
    data_transposed,
    np_labels,
    batch_size=128,
    epochs=50,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)
# %%

history = model.history.history
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('DeepSet Training and Validation Loss')
plt.legend()
plt.show() 
# %%
# from model.regressor import Regressor

# og_model = Regressor()
# %%

predictions = model.predict(X_test)

# %%

# 1. Prediction vs. True Value Scatter Plot
plt.figure(figsize=(8, 8))
plt.scatter(y_test, predictions, alpha=0.3, label='Predictions')
# Add the ideal y=x line for reference
lims = [min(plt.xlim()), max(plt.xlim())]
plt.plot(lims, lims, 'r--', alpha=0.75, zorder=5, label='Ideal (y=x)')
plt.xlabel("True Values (z-coordinate)")
plt.ylabel("Predicted Values")
plt.title("DeepSet Prediction vs. True Value")
plt.legend()
plt.grid(True)
plt.axis('equal') # Ensures the y=x line is at 45 degrees
plt.show()

# %%

residuals = y_test.flatten() - predictions.flatten()

plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=100, density=True, label='Error Distribution')
plt.axvline(0, color='r', linestyle='--', label='Zero Error')
plt.xlabel("Residual (True Value - Predicted Value)")
plt.ylabel("Density")
plt.title("DeepSet Histogram of Prediction Residuals")

# Add stats to the plot
mean_error = np.mean(residuals)
std_error = np.std(residuals)
plt.text(0.95, 0.95, f'Mean Error: {mean_error:.2f}\nStd Dev: {std_error:.2f}', 
         ha='right', va='top', transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

plt.legend()
plt.grid(True)
plt.show()

# %%

plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, alpha=0.3)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (True - Predicted)")
plt.title("Residuals vs. Predicted Values")
plt.grid(True)
plt.show()
# %%
