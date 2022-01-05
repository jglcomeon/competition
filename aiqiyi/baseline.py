import pandas as pd
import numpy as np
import json
import math
import tensorflow as tf

data_dir = "./data/wsdm_model_data/"


class DataGenerator:
    def __init__(self, df, batch_size):
        self.data = df
        self.num = df.shape[0]
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.num / self.batch_size)

    def __iter__(self):
        while True:
            input_1, input_2, output = [], [], []
            for row in self.data.itertuples():
                idx = row.Index
                seq = [row.launch_seq, row.playtime_seq]
                fea = row.duration_prefer + row.interact_prefer + list(row[7:18])
                input_1.append(np.array(seq))
                input_2.append(np.array(fea))
                output.append(row.label)
                if len(input_1) == self.batch_size or idx == self.num - 1:
                    input_1 = np.array(input_1).transpose([0, 2, 1])
                    input_2 = np.array(input_2)
                    output = np.array(output)
                    yield (input_1, input_2), output
                    input_1, input_2, output = [], [], []


def build_model(seq_feature_num, seq_len, feature_num):
    input_1 = tf.keras.Input(shape=(seq_len, seq_feature_num))
    output_1 = tf.keras.layers.GRU(64)(input_1)

    input_2 = tf.keras.Input(shape=(feature_num, ))
    layer = tf.keras.layers.Dense(256, activation="elu")(input_2)
    layer = tf.keras.layers.Dense(128, activation="elu")(layer)
    output_2 = tf.keras.layers.Dense(64, activation="elu")(layer)

    output = tf.concat([output_1, output_2], -1)
    output = tf.keras.layers.Dense(1, activation="relu")(output)

    model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)

    return model

# train data
data = pd.read_csv(data_dir + "train_data.txt", sep="\t")
data["launch_seq"] = data.launch_seq.apply(lambda x: json.loads(x))
data["playtime_seq"] = data.playtime_seq.apply(lambda x: json.loads(x))
data["duration_prefer"] = data.duration_prefer.apply(lambda x: json.loads(x))
data["interact_prefer"] = data.interact_prefer.apply(lambda x: json.loads(x))

# shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# testing DataGenerator
generator_test = DataGenerator(data[:20], batch_size=8)
for i, item in enumerate(iter(generator_test)):
    if(i == len(generator_test)):
        break
    (input_1, input_2), output = item
    print(i, input_1.shape, input_2.shape)
    print(i, output.shape, output)

model = build_model(seq_feature_num=2, seq_len=32, feature_num=38)
model.summary()

train = DataGenerator(data.loc[30001:], 128)
dev = DataGenerator(data.loc[:30000], 64)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="mae",
    metrics=["mse"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model.fit(
    iter(train),
    steps_per_epoch=len(train),
    validation_data=iter(dev),
    validation_steps=len(dev),
    epochs=20,
    callbacks=[early_stopping]
)

data = pd.read_csv(data_dir + "test_data.txt", sep="\t")
data["launch_seq"] = data.launch_seq.apply(lambda x: json.loads(x))
data["playtime_seq"] = data.playtime_seq.apply(lambda x: json.loads(x))
data["duration_prefer"] = data.duration_prefer.apply(lambda x: json.loads(x))
data["interact_prefer"] = data.interact_prefer.apply(lambda x: json.loads(x))

test = DataGenerator(data, 64)
# can also load model from saved weights
# model = build_model(seq_feature_num=2, seq_len=32, feature_num=38)
# model.load_weights(data_dir + "best_weights.h5")
prediction = model.predict(iter(test), steps=len(test))
data["prediction"] = np.reshape(prediction, -1)
data = data[["user_id", "prediction"]]
# can clip outputs to [0, 7] or use other tricks

data.to_csv(data_dir + "baseline_submission.csv", index=False, header=False, float_format="%.2f")