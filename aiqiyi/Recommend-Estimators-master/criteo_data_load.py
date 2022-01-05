import tensorflow as tf
import tf_utils


#COLUMNS = ['gender', 'age', 'tagid', 'province', 'city', 'make', 'model']
#COLUMNS = ['end_date', 'launch_seq', 'playtime_seq', 'duration_prefer', 'father_id_score', 'cast_id_score',
# 'tag_score', 'device_type', 'device_ram', 'device_rom', 'sex', 'age', 'education', 'occupation_status', 'territory_score', 'interact_prefer']
DEFAULT_VALUES = [[-1], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['']]
COLUMNS_MAX_TOKENS = [('end_date', 1), ('father_id_score', 1), ('cast_id_score', 1), ('tag_score', 1), ('device_type', 1),
                      ('device_ram', 1), ('device_rom', 1), ('sex', 1), ('age', 1), ('education', 1),
                      ('occupation_status', 1), ('territory_score', 1), ('launch_seq', 32), ('playtime_seq', 32), ('duration_prefer', 16), ('interact_prefer', 11)]


def _decode_tsv(line):
    columns = tf.io.decode_csv(line, record_defaults=DEFAULT_VALUES, field_delim='\t')
    print(len(columns))
    y = columns[0]
    # sess = tf.compat.v1.Session()
    #
    # print(sess.run(y))

    feat_columns = dict(zip((t[0] for t in COLUMNS_MAX_TOKENS), columns[1:]))
    print(len(feat_columns))
    X = {}
    for colname, max_tokens in COLUMNS_MAX_TOKENS:
        # 调用string_split时，第一个参数必须是一个list，所以要把columns[colname]放在[]中
        # 这时每个kv还是'k:v'这样的字符串
        kvpairs = tf.string_split([feat_columns[colname]], ',').values[:max_tokens]


        # k,v已经拆开, kvpairs是一个SparseTensor，因为每个kvpair格式相同，都是"k:v"
        # 既不会出现"k"，也不会出现"k:v1:v2:v3:..."
        # 所以，这时的kvpairs实际上是一个满阵
        kvpairs = tf.string_split(kvpairs, ':')



        # kvpairs是一个[n_valid_pairs,2]矩阵
        kvpairs = tf.reshape(kvpairs.values, kvpairs.dense_shape)



        feat_ids, feat_vals = tf.split(kvpairs, num_or_size_splits=2, axis=1)

        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)

        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)

        # 不能调用squeeze, squeeze的限制太多, 当原始矩阵有1行或0行时，squeeze都会报错
        X[colname + "_ids"] = tf.reshape(feat_ids, shape=[-1])
        X[colname + "_values"] = tf.reshape(feat_vals, shape=[-1])
    return X, y


def input_fn(data_file, n_repeat, batch_size, batches_per_shuffle):


    # ----------- define reading ops
    dataset = tf.data.TextLineDataset(data_file).skip(1)  # skip the header
    dataset = dataset.map(_decode_tsv, num_parallel_calls=4)

    if batches_per_shuffle > 0:
        dataset = dataset.shuffle(batches_per_shuffle * batch_size)

    dataset = dataset.repeat(n_repeat)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.padded_batch(batch_size=batch_size,
    #                                padded_shapes=pad_shapes,
    #                                padding_values=pad_values)

    iterator = dataset.make_one_shot_iterator()
    dense_Xs, ys = iterator.get_next()

    # ----------- convert dense to sparse
    sparse_Xs = {}
    for c, _ in COLUMNS_MAX_TOKENS:
        for suffix in ["ids", "values"]:
            k = "{}_{}".format(c, suffix)
            # 个人理解就是将one-hot向量转换为查表的形式，即indices与values一一对应，节省内存加快计算
            sparse_Xs[k] = tf_utils.to_sparse_input_and_drop_ignore_values(dense_Xs[k])

    # ----------- return
    return sparse_Xs, ys
