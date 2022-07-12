def load_datasets():

    base_data_dir = os.path.join('data')
    training_data_name = "train_1"
    train_data_dir = os.path.join('data', training_data_name)
    train_label_file = 'trainLabels.csv'

    # Load image mapping
    retina_df = pd.read_csv(os.path.join(base_data_dir, train_label_file))
    # Get patient ID
    retina_df['PatientId'] = retina_df['image'].map(lambda x: x.split('_')[0])
    # Get image path
    retina_df['path'] = retina_df['image'].map(lambda x: os.path.join(train_data_dir,'{}.jpeg'.format(x)))
    # See if data exists in training data set
    retina_df['exists'] = retina_df['path'].map(os.path.exists)

    print(retina_df['exists'].sum(), 'images found of', retina_df.shape[0], 'total')

    # Left right eye categorical variable
    # 1 is left eye, 0 is right eye
    retina_df['eye'] = retina_df['image'].map(lambda x: 1 if x.split('_')[-1]=='left' else 0)

    # # Output variable to categorical
    # retina_df['level_cat'] = retina_df['level'].map(lambda x: to_categorical(x, 1+retina_df['level'].max()))
    # Remove NA 
    retina_df.dropna(inplace = True)
    retina_df = retina_df[retina_df['exists']]

    # Split traing and valid sets
    rr_df = retina_df[['PatientId', 'level']].drop_duplicates()
    train_ids, valid_ids = train_test_split(rr_df['PatientId'], 
                                    test_size = 0.25, 
                                    random_state = 2018,
                                    stratify = rr_df['level'])
                                    
    raw_train_df = retina_df[retina_df['PatientId'].isin(train_ids)]
    valid_df = retina_df[retina_df['PatientId'].isin(valid_ids)]
    print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])
    
    # balance size variance in each class
    train_df = raw_train_df.groupby(['level', 'eye']).apply(lambda x: x.sample(75, replace = True)).reset_index(drop = True)                                                   

    return train_df, valid_df
    