import os
import pickle
import string
from typing import List

import numpy as np
import pandas as pd
import pymorphy2
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve


def train_test_split_data(converted=False, sample=None, **kwargs):
    # get train data
    if converted:
        train_path = 'data/train_tokens.csv'
    else:
        train_path = 'data/train.csv'

    print('load train data {}'.format(train_path))
    train = pd.read_csv(train_path, engine='python')

    if converted:
        val_path = 'data/val_tokens.csv'
    else:
        val_path = 'data/val.csv'

    print('load val data {}'.format(val_path))
    val = pd.read_csv(val_path, engine='python')

    if sample is not None:
        print('get sample of train data {}'.format(sample))
        train = train.head(sample)

    if sample is not None:
        print('get sample of val data {}'.format(sample))
        val = val.head(sample)

    x_train = train['text'].fillna('')
    y_train = train['label'].fillna(0).astype(int)

    x_test = val['text'].fillna('')
    y_test = val['label'].fillna(0).astype(int)

    return x_train, x_test, y_train, y_test


def preprocess(x, vector):
    x_transform = pd.DataFrame(vector.transform(x).todense(), columns=vector.get_feature_names_out())
    return x_transform


def tokenizer(x):
    x = x.lower()
    for item in string.punctuation:
        x = x.replace(item, ' ')
    x = x.replace('\n', ' ')
    result = [morph.parse(item)[0].normal_form for item in x.split(' ')
              if item and not item.isspace() and len(item) > 1]

    return " ".join(result)


def fit_model(model, x, y, max_features=1000, max_df=1.0, min_df=1,
              ngram_range=(1, 1), vocabulary=None, verbose=True, **kwargs):

    # create vocabulary again based on top words
    vector = TfidfVectorizer(vocabulary=vocabulary, max_features=max_features,
                             max_df=max_df, min_df=min_df, ngram_range=ngram_range)

    if verbose:
        print('fit vector')
        print(vector)

    vector.fit(x.fillna(''))
    x_train = preprocess(x, vector)

    if verbose:
        print('fit model')
        print(model)

    model.fit(x_train, y)

    predict = model.predict_proba(x_train)[:, 1]
    calib = IsotonicRegression(y_min=0., y_max=1., out_of_bounds='clip')

    if verbose:
        print('fit calib')
        print(calib)

    calib.fit(predict, y)

    return vector, model, calib


def save_model(vector, model, calib):
    # save model and vector
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/vector.pkl", "wb") as f:
        pickle.dump(vector, f)
    with open("model/calib.pkl", "wb") as f:
        pickle.dump(calib, f)


def load_vector_model_and_calib():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/vector.pkl", "rb") as f:
        vector = pickle.load(f)
    with open("model/calib.pkl", "rb") as f:
        calib = pickle.load(f)

    return vector, model, calib


def model_predict(vector, model, calib, x):
    x_transform = preprocess(x, vector)
    predict_raw = model.predict_proba(x_transform)[:, 1]
    predict = calib.predict(predict_raw)
    return predict


def load_data_fit_score_and_save(C, penalty, solver, random_state, max_iter, **kwargs):
    # create dataset with tokens if needed
    dataset_convert_and_save()
    # load data
    x_train, x_test, y_train, y_test = train_test_split_data(**kwargs)
    # set the model
    model = LogisticRegression(C=C, penalty=penalty, solver=solver, random_state=random_state, max_iter=max_iter)
    # fit models
    vector, model, calib = fit_model(model, x_train, y_train, **kwargs)
    # save fitted model
    save_model(vector, model, calib)
    # calc score
    # and score
    score_train = roc_auc_score(y_train, model_predict(vector, model, calib, x_train))
    score_test = roc_auc_score(y_test, model_predict(vector, model, calib, x_test))
    print("model: {}, num features: {}, train: {:.3f} test {:.3f}".format(
        model.__class__.__name__, model.coef_.shape, score_train, score_test))
    return score_test, score_train


def select_max_features(args, list_max_features, n_jobs):

    def eval_one_iter(model, max_features):
        args['max_features'] = max_features
        vector, model_local, calib = fit_model(clone(model), x_train, y_train, **args)
        score_train = roc_auc_score(y_train, model_predict(vector, model_local, calib, x_train))
        score_test = roc_auc_score(y_test, model_predict(vector, model_local, calib, x_test))
        return {'score_test': round(score_test, 4),
                'score_train': round(score_train, 4),
                'max_features': model_local.coef_[0].shape[0]}

    x_train, x_test, y_train, y_test = train_test_split_data(converted=True, sample=args['sample'])
    _, model, _ = load_vector_model_and_calib()

    score_and_max_features = [Parallel(n_jobs=n_jobs)(delayed(eval_one_iter)(model, max_features)
                                                      for max_features in list_max_features)][0]
    results = pd.DataFrame(score_and_max_features)
    return results


def get_model_coef():
    vector, model, _ = load_vector_model_and_calib()
    s = pd.DataFrame(model.coef_, columns=vector.get_feature_names_out()).T
    s['abs'] = s[0].abs()
    s = s.sort_values('abs', ascending=False)
    s.drop('abs', axis=1, inplace=True)
    s[0] = s[0].round(3)
    return s


def dataset_convert_and_save():

    def convert_text_to_tokens(x, y):
        data = y.copy()
        data = data.to_frame()
        data['text'] = x.apply(tokenizer)
        return data

    train_path = 'data/train_tokens.csv'
    val_path = 'data/val_tokens.csv'

    if not os.path.isfile(train_path) or not os.path.isfile(val_path):

        x_train, x_test, y_train, y_test = train_test_split_data(converted=False)

        print('saved train tokenizer data {}'.format(train_path))
        tokens_train = convert_text_to_tokens(x_train, y_train)
        tokens_train.to_csv(train_path)

        print('saved val tokenizer data {}'.format(train_path))
        tokens_test = convert_text_to_tokens(x_test, y_test)
        tokens_test.to_csv(val_path)


def save_precision_recall(**kwargs):
    # load data
    x_train, x_test, y_train, y_test = train_test_split_data(**kwargs)
    vector, model, calib = load_vector_model_and_calib()

    data_list = []
    for x, y, name in [(x_train, y_train, 'train'), (x_test, y_test, 'val')]:
        predict = model_predict(vector, model, calib, x)
        precision, recall, thresholds = precision_recall_curve(y, predict)
        roc_auc = roc_auc_score(y, predict)

        data = pd.DataFrame({'precision': precision,
                             'recall': recall,
                             'thresholds': [0] + thresholds.tolist(),
                             'roc_auc': roc_auc})

        count = [(predict >= x).sum() for x in data.thresholds.tolist()]


        data['count'] = count
        data['count_ratio'] = data['count'] / precision.shape[0]
        data = data.round(3)
        data['type'] = name
        data_list.append(data)

    results = pd.concat(data_list, axis=0)
    results.to_csv('results/precision_recall.csv', index=False)

    roc_train = results[results['type'] == 'train']['roc_auc'].values[0]
    roc_val = results[results['type'] == 'val']['roc_auc'].values[0]

    df = results.set_index('recall')
    ax = df[df['type'] == 'train']['precision'].plot(label='train roc auc: {}'.format(roc_train))
    ax = df[df['type'] == 'val']['precision'].plot(ax=ax, label='val roc auc: {}'.format(roc_val))
    fig = ax.figure
    fig.legend()
    fig.savefig('results/precision_recall.png')


def backward_features_selection(num_trials, min_score_loss, n_jobs, **kwargs):

    def eval_one_trial(random_seed, model):
        vocabulary = x_transform_train.columns
        model_local = clone(model)
        model_local.fit(x_transform_train[vocabulary], y_train)
        best_score = roc_auc_score(y_test, model_local.predict_proba(x_transform_test[vocabulary])[:, 1])

        drop_columns = []
        vocab = np.random.RandomState(random_seed).choice(vocabulary, size=len(vocabulary), replace=False)
        for col in vocab:
            current_vocab = list(set(vocabulary) - set(col) - set(drop_columns))
            model_local.fit(x_transform_train[current_vocab], y_train)
            score = roc_auc_score(y_test, model_local.predict_proba(x_transform_test[current_vocab])[:, 1])

            if score - best_score < min_score_loss:
                drop_columns.append(col)
                best_score = score

        final_vocab = list(set(vocabulary) - set(drop_columns))
        return {'features': final_vocab,
                'num_features': len(final_vocab),
                'scores': round(best_score, 4)}

    x_train, x_test, y_train, y_test = train_test_split_data(**kwargs)
    vector, model, _ = load_vector_model_and_calib()
    x_transform_train = preprocess(x_train, vector)
    x_transform_test = preprocess(x_test, vector)
    print('transform data done')

    vocab_and_score = [Parallel(n_jobs=n_jobs)(delayed(eval_one_trial)(i, model)
                                               for i in range(num_trials))][0]
    result = pd.DataFrame(vocab_and_score)

    backward_score = result[['scores', 'num_features']].describe(percentiles=np.linspace(0, 1., 11))

    backward_features = (
        result['features']
        .explode('features')
        .value_counts()
    )

    return backward_score, backward_features


def hyper_parameters_selection(c_list, n_jobs, **kwargs):

    def eval_one_model(c, model):
        model_local = clone(model)
        model_local.C = c
        model_local.fit(x_transform_train, y_train)
        score_train = roc_auc_score(y_train, model_local.predict_proba(x_transform_train)[:, 1])
        score_test = roc_auc_score(y_test, model_local.predict_proba(x_transform_test)[:, 1])

        return {'score_train': score_train,
                'score_test': score_test,
                'C': model_local.C}

    x_train, x_test, y_train, y_test = train_test_split_data(**kwargs)
    vector, model, _ = load_vector_model_and_calib()
    x_transform_train = preprocess(x_train, vector)
    x_transform_test = preprocess(x_test, vector)

    print('transform data done shape {}'.format(x_transform_train.shape))

    c_and_score = [Parallel(n_jobs=n_jobs)(delayed(eval_one_model)(i, model) for i in c_list)][0]
    result = pd.DataFrame(c_and_score)

    return result


def get_vocab() -> List[str]:
    """
    function return final vocabulary

    :return:
    """
    vocabulary = ['состояние удовлетворительный', 'удовлетворительный сознание', 'самостоятельно', 'определяться орган',
                  'напряжение', 'номер', 'норма', 'кровь', 'патология', 'жёсткий', 'приём', 'лёгочный', 'узел',
                  'высокий', 'слабость', 'покой', 'хсн', 'решение', 'пациент', 'полость', 'положение', 'позвоночник',
                  'выраженный', 'ка', 'нижний', 'одышка', 'общий', 'снижение', 'препарат', 'экг', '60', 'для',
                  'плановый', 'мсек', 'не увеличить', 'сердце не', 'головной', 'фон', 'справа', 'ст', 'исследование',
                  'хронический гастрит', 'регургитация', '130', '70 мм', 'аортальный', 'план', 'обследование', 'лёгкий',
                  'место', 'средний', 'эхокг', 'свободный', '70', 'диагноз', 'по данные', 'чсс', 'ритм', 'поступление',
                  'мл', 'треть', 'аорта', 'поступить', 'около', '36 град', 'акк', 'удовлетворительный наполнение',
                  'недостаточность', 'тактика', 'лж', 'мг', 'получать', 'отрицать', 'признак', 'слева', 'боль',
                  'диабет', 'систолический', 'сустав', 'розовый костный', 'онмк', 'доза', 'уплотнить', 'риск',
                  'коронарный', 'сутки', 'степень', 'левый', 'митральный', 'результат', 'синдром', 'осложнение',
                  'ожирение', 'нарушить', '90 мм', 'клинический', '140', 'терапия', 'сердце', 'лаб', 'гипертензия',
                  'масса', 'курс', 'мина', 'ст риск', 'клетка', 'анамнез', 'пка', 'вид', 'желудочковый', 'после',
                  'кт', 'сердцебиение', 'вено', 'стеноз', 'сердечный', 'желудочек', 'смотреть', 'отдел']

    return vocabulary


def advanced(args: dict) -> None:
    """
    Function for select all important model parameters
    It takes time !!!
    It depends on initial vocabulary size (1 - 24 hours on ordinary machine)

    :param args:
    :return:
    """
    args['vocabulary'] = args['vocabulary'] = get_vocab()
    args['verbose'] = False
    n_jobs = 8

    max_features_path = 'results/select_max_features.csv'
    list_max_features = [100, 200, 400, 800, 1200, 1600, 2400, 3200]
    if not os.path.isfile(max_features_path):
        max_features_results = select_max_features(args=args, list_max_features=list_max_features, n_jobs=n_jobs)
        max_features_results.to_csv(max_features_path)

    num_trials = 8
    min_score_loss = 0.002
    backward_features_path = 'results/backward_features.csv'
    if not os.path.isfile(backward_features_path):
        load_data_fit_score_and_save(**args)
        backward_score, backward_features = backward_features_selection(
            num_trials=num_trials, min_score_loss=min_score_loss, n_jobs=n_jobs, **args)
        backward_score.to_csv('results/backward_score.csv')
        backward_features.to_csv(backward_features_path)

    hyper_parameters_path = 'results/hyper_parameters.csv'
    c_list = [0.01, 0.1, 0.2, 0.4, 1., 2., 4., 10., 20., 40., 100., 200., 400.]
    if not os.path.isfile(hyper_parameters_path):
        load_data_fit_score_and_save(**args)
        hyper_score = hyper_parameters_selection(c_list=c_list, n_jobs=n_jobs, **args)
        hyper_score.to_csv(hyper_parameters_path)


def main(args: dict) -> None:
    """
    Main function for fit model and inference on valid dataset

    :param args:
    :return:
    """
    args['vocabulary'] = get_vocab()
    # fit model and save
    load_data_fit_score_and_save(**args)
    # model coef
    model_coef = get_model_coef()
    model_coef.to_csv('results/coef.csv')
    model_coef[model_coef[0] > 0].head(20).to_csv('results/coef_top_negative.csv')
    model_coef[model_coef[0] < 0].head(20).to_csv('results/coef_top_positive.csv')
    # precision recall curve
    save_precision_recall(**args)


if __name__ == '__main__':
    morph = pymorphy2.MorphAnalyzer()
    ARGS = {'C': 1, 'max_features': 800, 'converted': True, 'max_df': 0.99, 'min_df': 0.005, 'max_iter': 1000,
            'ngram_range': (1, 2), 'random_state': 47, 'penalty': 'l1', 'solver': 'saga', 'sample': None,
            'vocabulary': None}
    main(ARGS)
