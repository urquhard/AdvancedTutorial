import tensorflow as tf
from tensorflow.keras import layers, Model

class TNet(Model):
    """Learns a (C×C) feature alignment matrix."""
    def __init__(self, conv_dims, fc_dims, out_dim):
        super().__init__()
        # 1) shared‐MLP on each point
        self.conv_blocks = []
        for d in conv_dims:
            self.conv_blocks += [
                layers.Conv1D(d, 1, activation=None),
                layers.BatchNormalization(),
                layers.ReLU()
            ]
        # 2) MLP on the global max‐pooled feature
        self.fc_blocks = []
        for d in fc_dims:
            self.fc_blocks += [
                layers.Dense(d, activation=None),
                layers.BatchNormalization(),
                layers.ReLU()
            ]
        # final linear layer to predict C*C entries
        self.out = layers.Dense(out_dim)

    def call(self, x):
        # x: (B, L, C)
        for layer in self.conv_blocks:
            x = layer(x)
        x = tf.reduce_max(x, axis=1)   # (B, C_out)
        for layer in self.fc_blocks:
            x = layer(x)
        return self.out(x)             # (B, C*C)


class Regressor(Model):
    """PointNet-style regressor predicting a single scalar per cloud."""
    def __init__(self, config):
        super().__init__()
        # --- 1) Pre-TNet feature extractor
        pre_layers = []
        in_dim = 3
        for d in config['pre_dims']:
            pre_layers += [
                layers.Conv1D(d, 1, activation=None),
                layers.BatchNormalization(),
                layers.ReLU()
            ]
            in_dim = d
        self.pre_net = tf.keras.Sequential(pre_layers)

        # remember feature dim
        self.num_feats = in_dim

        # --- 2) Feature transform T-Net
        self.tnet = TNet(
            conv_dims  = config['tnet_conv'],
            fc_dims    = config['tnet_fc'],
            out_dim    = self.num_feats * self.num_feats
        )

        # --- 3) Post-TNet feature extractor
        post_layers = []
        in_dim = self.num_feats
        for d in config['post_dims']:
            post_layers += [
                layers.Conv1D(d, 1, activation=None),
                layers.BatchNormalization(),
                layers.ReLU()
            ]
            in_dim = d
        self.post_net = tf.keras.Sequential(post_layers)

        # --- 4) Final MLP to one scalar
        fc = []
        for d in config['fc_dims']:
            fc += [layers.Dense(d, activation='relu')]
        fc += [layers.Dense(1)]   # output
        self.head = tf.keras.Sequential(fc)

    def call(self, x):
        # x: (B, L, 3)
        x = self.pre_net(x)                     # (B, L, num_feats)
        t = self.tnet(x)                        # (B, num_feats*num_feats)
        t = tf.reshape(t, (-1, self.num_feats, self.num_feats))
        # apply transform to each point: (B, L, C) @ (B, C, C) → (B, L, C)
        x = tf.matmul(x, t)
        x = self.post_net(x)                    # (B, L, out_feats)
        x = tf.reduce_max(x, axis=1)            # global max‐pool → (B, out_feats)
        y = self.head(x)                        # (B, 1)
        return tf.squeeze(y, axis=-1)           # (B,)

# === example instantiation & summary ===
config = {
    'pre_dims':    [64, 64],
    'tnet_conv':   [64, 128, 1024],
    'tnet_fc':     [512, 256],
    'post_dims':   [64, 128, 1024],
    'fc_dims':     [512, 256],
}
model = Regressor(config)
model.build((None, 140, 3))
model.summary()