import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import googleapiclient.discovery
import os

np.random.seed(42)
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

from flask_bootstrap import Bootstrap

Bootstrap(app)

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired


class LabForm(FlaskForm):
    latitude = StringField('latitude.', validators=[DataRequired()])
    longitude = StringField('longitude', validators=[DataRequired()])
    month = StringField('month', validators=[DataRequired()])
    day = StringField('day', validators=[DataRequired()])
    avg = StringField('avg_temp', validators=[DataRequired()])
    max = StringField('max_temp', validators=[DataRequired()])
    wind_s = StringField('max_wind_speed', validators=[DataRequired()])
    wind_avg = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('Submit')


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()

    if form.validate_on_submit():
        X_test = np.array([[float(form.latitude.data),
                            float(form.longitude.data),
                            str(form.month.data),
                            str(form.day.data),
                            float(form.avg.data),
                            float(form.max.data),
                            float(form.wind_s.data),
                            float(form.wind_avg.data)]])
        print(X_test.shape)
        fires = pd.read_csv('datasets/sanbul-5.csv', sep=',')
        X_test = pd.DataFrame(X_test, columns=['latitude','longitude','month','day','avg_temp','max_temp','max_wind_speed','avg_wind'])
        print(X_test)

        from sklearn.model_selection import train_test_split
        train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)
        from sklearn.model_selection import StratifiedShuffleSplit
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(fires, fires["month"]):
            strat_train_set = fires.loc[train_index]
            strat_test_set = fires.loc[test_index]

        fires = strat_train_set.drop(["burned_area"], axis=1)  # drop labels for training set
        fires_labels = strat_train_set["burned_area"].copy()
        fires_num = fires.drop(["month", "day"], axis=1)

        from sklearn.preprocessing import OneHotEncoder
        cat_encoder = OneHotEncoder()
        fires_cat = fires[["month"]]
        fires_cat_1hot = cat_encoder.fit_transform(fires_cat)
        cat_encoder = OneHotEncoder(sparse=False)
        fires_cat_1hot = cat_encoder.fit_transform(fires_cat)

        cat_encoder2 = OneHotEncoder()
        fires_cat = fires[["day"]]
        fires_cat_1hot_2 = cat_encoder2.fit_transform(fires_cat)
        cat_encoder2 = OneHotEncoder(sparse=False)
        fires_cat_1hot_2 = cat_encoder2.fit_transform(fires_cat)

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        num_pipeline = Pipeline([
            ('std_scaler', StandardScaler()),
        ])
        fires_num_tr = num_pipeline.fit_transform(fires_num)

        from sklearn.compose import ColumnTransformer
        num_attribs = list(fires_num)
        cat_attribs = ["month", "day"]
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])
        fires_prepared = full_pipeline.fit_transform(fires)
        X_test = full_pipeline.transform(X_test)

        MODEL_NAME = "my_sanbul_model"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "term-224506-9bc8286b5d7b.json"
        project_id = 'term-224506'
        model_id = MODEL_NAME
        model_path = "projects/{}/models/{}".format(project_id, model_id)
        model_path += "/versions/v0001/"
        ml_resource = googleapiclient.discovery.build("ml", "v1").projects()

        input_data_json = {"signature_name": "serving_default",
                           "instances": X_test.tolist()}
        request = ml_resource.predict(name=model_path, body=input_data_json)
        response = request.execute()
        print("\nresponse:\n", response)

        if "error" in response:
            raise RuntimeError(response["error"])

        predD = np.array([pred['dense_1'] for pred in response["predictions"]])
        print(predD[0][0])
        res = predD[0][0]
        return render_template('result.html', res=res)

    return render_template('prediction.html', form=form)


if __name__ == '__main__':
    app.run()