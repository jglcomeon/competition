# coding=UTF-8
import pandas as pd
import tensorflow as tf
from deepfm import model_fn
from criteo_data_load import input_fn


def get_hparams():
    vocab_sizes = {
      'end_date': 123, 'father_id_score': 1, 'cast_id_score': 1, 'tag_score': 1, 'device_type': 1,
                      'device_ram': 1, 'device_rom': 1, 'sex': 1, 'age': 1, 'education': 1,
                      'occupation_status': 1, 'territory_score': 1, 'launch_seq': 32, 'playtime_seq': 32, 'duration_prefer': 16,
                      'interact_prefer': 11
    }
    # step = tf.Variable(0, trainable=False)
    # schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
    #     [10000, 15000], [1e-0, 1e-1, 1e-2])
    # # lr and wd can be a function or a tensor
    # lr = 1e-3 * schedule(step)
    # wd = lambda: 1e-4 * schedule(step)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        use_locking=False,
        name='Adam'
    )

    return {
        'embed_dim': 128,
        'vocab_sizes': vocab_sizes,
        'multi_embed_combiner': 'sum',
        # 在这个case中，没有多个field共享同一个vocab的情况，而且field_name和vocab_name相同
        'field_vocab_mapping': {'end_date': 'end_date', 'father_id_score': 'father_id_score', 'cast_id_score': 'cast_id_score', 'tag_score': 'tag_score', 'device_type': 'device_type',
                      'device_ram': 'device_ram', 'device_rom': 'device_rom', 'sex': 'sex', 'age': 'age', 'education': 'education',
                      'occupation_status': 'occupation_status', 'territory_score': 'territory_score', 'launch_seq': 'launch_seq', 'playtime_seq': 'playtime_seq', 'duration_prefer': 'duration_prefer',
                      'interact_prefer': 'interact_prefer'},
        'dropout_rate': 0.1,
        'batch_norm': False,
        'hidden_units': [128, 64],
        'optimizer': optimizer
    }


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.set_random_seed(999)
    hparams = get_hparams()
    deepfm = tf.estimator.Estimator(model_fn=model_fn,
                                    model_dir='models/criteo',
                                    params=hparams)


    deepfm.train(input_fn=lambda: input_fn(data_file='data/new_train.csv',
                                           n_repeat=5,
                                           batch_size=64,
                                           batches_per_shuffle=10), max_steps=500)

    pred_dict = deepfm.predict(input_fn=lambda: input_fn(data_file='data/new_test.csv',
                                                         n_repeat=1,
                                                         batch_size=64,
                                                         batches_per_shuffle=-1))
    print(pred_dict)
    res = []
    for pred_res in pred_dict:
        res.append(pred_res['probabilities'])
    res = pd.DataFrame(res)
    res.to_csv('res.csv')



