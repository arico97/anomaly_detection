from transactions import *

data_path = "./transactions/data/creditcard.csv"

model_path = "./transactions/models/model"

data = load_data(data_path)

train, test = split_data(data)

class_model = ClassicationModel(train)

class_model._train_model(model = 'lr')

class_model._save_model()

predictions = class_model._predict(test)

class_model._make_dashboard()
