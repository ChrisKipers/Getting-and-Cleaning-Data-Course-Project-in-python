import pandas as pd

base_dir = "/Users/ckipers/Downloads/UCI HAR Dataset"
feature_names = pd.read_csv(base_dir + "/features.txt", header=None, sep=" ").ix[:, 1].values
activity_labels = pd.read_csv(base_dir + "/activity_labels.txt", sep=" ", header=None, names=["id", "name"])

X_test = pd.read_fwf(base_dir + "/test/X_test.txt", header=None, names=feature_names)
y_test = pd.read_csv(base_dir + "/test/y_test.txt", header=None, squeeze=True, names=["activity"])

X_train = pd.read_fwf(base_dir + "/train/X_train.txt", header=None, names=feature_names)
y_train = pd.read_csv(base_dir + "/train/y_train.txt", header=None, squeeze=True, names=["activity"])

# Objective 1) Merge the training and test sets
X_all = pd.concat([X_test, X_train])
y_all = pd.concat([y_test, y_train])

# Objective 2) Extract only mean and std measurements
target_feature_name_regex = re.compile("(mean\(\)|std\(\))")
target_feature_names = [c_name for c_name in feature_names if target_feature_name_regex.search(c_name)]
X_all_target_features = X_all[target_feature_names]

# Objective 3) Use descriptive activity names
activity_labels_by_id = {row.id:row.name for row in activity_labels.itertuples()}
y_all_as_labels = y_all.map(activity_labels_by_id)

# Objective 4) Label the rows with the activities
all_data = pd.concat([X_all_target_features, y_all_as_labels], axis=1)

# Objective 5) Average of each variable for each activity
avg_column_val_for_activity = all_data.groupby("activity").mean()
