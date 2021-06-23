Module ktrain.text.data
=======================

Functions
---------

    
`check_unsupported_lang(lang, preprocess_mode)`
:   check for unsupported language (e.g., nospace langs not supported by Jieba)

    
`texts_from_array(x_train, y_train, x_test=None, y_test=None, class_names=[], max_features=20000, maxlen=400, val_pct=0.1, ngram_range=1, preprocess_mode='standard', lang=None, random_state=None, verbose=1)`
:   Loads and preprocesses text data from arrays.
    texts_from_array can handle data for both text classification
    and text regression.  If class_names is empty, a regression task is assumed.
    Args:
        x_train(list): list of training texts 
        y_train(list): labels in one of the following forms:
                       1. list of integers representing classes (class_names is required)
                       2. list of strings representing classes (class_names is not needed and ignored.)
                       3. a one or multi hot encoded array representing classes (class_names is required)
                       4. numerical values for text regresssion (class_names should be left empty)
        x_test(list): list of training texts 
        y_test(list): labels in one of the following forms:
                       1. list of integers representing classes (class_names is required)
                       2. list of strings representing classes (class_names is not needed and ignored.)
                       3. a one or multi hot encoded array representing classes (class_names is required)
                       4. numerical values for text regresssion (class_names should be left empty)
        class_names (list): list of strings representing class labels
                            shape should be (num_examples,1) or (num_examples,)
        max_features(int): max num of words to consider in vocabulary
                           Note: This is only used for preprocess_mode='standard'.
        maxlen(int): each document can be of most <maxlen> words. 0 is used as padding ID.
        ngram_range(int): size of multi-word phrases to consider
                          e.g., 2 will consider both 1-word phrases and 2-word phrases
                               limited by max_features
        val_pct(float): Proportion of training to use for validation.
                        Has no effect if x_val and  y_val is supplied.
        preprocess_mode (str):  Either 'standard' (normal tokenization) or one of {'bert', 'distilbert'}
                                tokenization and preprocessing for use with 
                                BERT/DistilBert text classification model.
        lang (str):            language.  Auto-detected if None.
        random_state(int):      If integer is supplied, train/test split is reproducible.
                                If None, train/test split will be random.
        verbose (boolean): verbosity

    
`texts_from_csv(train_filepath, text_column, label_columns=[], val_filepath=None, max_features=20000, maxlen=400, val_pct=0.1, ngram_range=1, preprocess_mode='standard', encoding=None, lang=None, sep=',', is_regression=False, random_state=None, verbose=1)`
:   Loads text data from CSV or TSV file. Class labels are assumed to be
    one of the following formats:
        1. one-hot-encoded or multi-hot-encoded arrays representing classes:
              Example with label_columns=['positive', 'negative'] and text_column='text':
                text|positive|negative
                I like this movie.|1|0
                I hated this movie.|0|1
            Classification will have a single one in each row: [[1,0,0], [0,1,0]]]
            Multi-label classification will have one more ones in each row: [[1,1,0], [0,1,1]]
        2. labels are in a single column of string or integer values representing classs labels
               Example with label_columns=['label'] and text_column='text':
                 text|label
                 I like this movie.|positive
                 I hated this movie.|negative
       3. labels are a single column of numerical values for text regression
          NOTE: Must supply is_regression=True for labels to be treated as numerical targets
                 wine_description|wine_price
                 Exquisite wine!|100
                 Wine for budget shoppers|8
    
    Args:
        train_filepath(str): file path to training CSV
        text_column(str): name of column containing the text
        label_column(list): list of columns that are to be treated as labels
        val_filepath(string): file path to test CSV.  If not supplied,
                               10% of documents in training CSV will be
                               used for testing/validation.
        max_features(int): max num of words to consider in vocabulary
                           Note: This is only used for preprocess_mode='standard'.
        maxlen(int): each document can be of most <maxlen> words. 0 is used as padding ID.
        ngram_range(int): size of multi-word phrases to consider
                          e.g., 2 will consider both 1-word phrases and 2-word phrases
                               limited by max_features
        val_pct(float): Proportion of training to use for validation.
                        Has no effect if val_filepath is supplied.
        preprocess_mode (str):  Either 'standard' (normal tokenization) or one of {'bert', 'distilbert'}
                                tokenization and preprocessing for use with 
                                BERT/DistilBert text classification model.
        encoding (str):        character encoding to use. Auto-detected if None
        lang (str):            language.  Auto-detected if None.
        sep(str):              delimiter for CSV (comma is default)
        is_regression(bool):  If True, integer targets will be treated as numerical targets instead of class IDs
        random_state(int):      If integer is supplied, train/test split is reproducible.
                                If None, train/test split will be random
        verbose (boolean): verbosity

    
`texts_from_df(train_df, text_column, label_columns=[], val_df=None, max_features=20000, maxlen=400, val_pct=0.1, ngram_range=1, preprocess_mode='standard', lang=None, is_regression=False, random_state=None, verbose=1)`
:   Loads text data from Pandas dataframe file. Class labels are assumed to be
    one of the following formats:
        1. one-hot-encoded or multi-hot-encoded arrays representing classes:
              Example with label_columns=['positive', 'negative'] and text_column='text':
                text|positive|negative
                I like this movie.|1|0
                I hated this movie.|0|1
            Classification will have a single one in each row: [[1,0,0], [0,1,0]]]
            Multi-label classification will have one more ones in each row: [[1,1,0], [0,1,1]]
        2. labels are in a single column of string or integer values representing class labels
               Example with label_columns=['label'] and text_column='text':
                 text|label
                 I like this movie.|positive
                 I hated this movie.|negative
       3. labels are a single column of numerical values for text regression
          NOTE: Must supply is_regression=True for integer labels to be treated as numerical targets
                 wine_description|wine_price
                 Exquisite wine!|100
                 Wine for budget shoppers|8
    
    Args:
        train_df(dataframe): Pandas dataframe
        text_column(str): name of column containing the text
        label_columns(list): list of columns that are to be treated as labels
        val_df(dataframe): file path to test dataframe.  If not supplied,
                               10% of documents in training df will be
                               used for testing/validation.
        max_features(int): max num of words to consider in vocabulary.
                           Note: This is only used for preprocess_mode='standard'.
        maxlen(int): each document can be of most <maxlen> words. 0 is used as padding ID.
        ngram_range(int): size of multi-word phrases to consider
                          e.g., 2 will consider both 1-word phrases and 2-word phrases
                               limited by max_features
        val_pct(float): Proportion of training to use for validation.
                        Has no effect if val_filepath is supplied.
        preprocess_mode (str):  Either 'standard' (normal tokenization) or one of {'bert', 'distilbert'}
                                tokenization and preprocessing for use with 
                                BERT/DistilBert text classification model.
        lang (str):            language.  Auto-detected if None.
        is_regression(bool):  If True, integer targets will be treated as numerical targets instead of class IDs
        random_state(int):      If integer is supplied, train/test split is reproducible.
                                If None, train/test split will be random
        verbose (boolean): verbosity

    
`texts_from_folder(datadir, classes=None, max_features=20000, maxlen=400, ngram_range=1, train_test_names=['train', 'test'], preprocess_mode='standard', encoding=None, lang=None, val_pct=0.1, random_state=None, verbose=1)`
:   Returns corpus as sequence of word IDs.
    Assumes corpus is in the following folder structure:
    ├── datadir
    │   ├── train
    │   │   ├── class0       # folder containing documents of class 0
    │   │   ├── class1       # folder containing documents of class 1
    │   │   ├── class2       # folder containing documents of class 2
    │   │   └── classN       # folder containing documents of class N
    │   └── test 
    │       ├── class0       # folder containing documents of class 0
    │       ├── class1       # folder containing documents of class 1
    │       ├── class2       # folder containing documents of class 2
    │       └── classN       # folder containing documents of class N
    
    Each subfolder should contain documents in plain text format.
    If train and test contain additional subfolders that do not represent
    classes, they can be ignored by explicitly listing the subfolders of
    interest using the classes argument.
    Args:
        datadir (str): path to folder
        classes (list): list of classes (subfolders to consider).
                        This is simply supplied as the categories argument
                        to sklearn's load_files function.
        max_features (int):  maximum number of unigrams to consider
                             Note: This is only used for preprocess_mode='standard'.
        maxlen (int):  maximum length of tokens in document
        ngram_range (int):  If > 1, will include 2=bigrams, 3=trigrams and bigrams
        train_test_names (list):  list of strings represnting the subfolder
                                 name for train and validation sets
                                 if test name is missing, <val_pct> of training
                                 will be used for validation
        preprocess_mode (str):  Either 'standard' (normal tokenization) or one of {'bert', 'distilbert'}
                                tokenization and preprocessing for use with 
                                BERT/DistilBert text classification model.
        encoding (str):        character encoding to use. Auto-detected if None
        lang (str):            language.  Auto-detected if None.
        val_pct(float):        Onlyl used if train_test_names  has 1 and not 2 names
        random_state(int):      If integer is supplied, train/test split is reproducible.
                                IF None, train/test split will be random
        verbose (bool):         verbosity