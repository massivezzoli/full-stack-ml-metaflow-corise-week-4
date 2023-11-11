from metaflow import FlowSpec, step, card, conda_base, current, Parameter, Flow, trigger, retry, catch, timeout, project
from metaflow.cards import Markdown, Table, Image, Artifact

URL = "https://outerbounds-datasets.s3.us-west-2.amazonaws.com/taxi/latest.parquet"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

@project(name="taxi_fare_prediction")
@trigger(events=["s3"])
@conda_base(
    libraries={
        "pandas": "1.4.2",
        "pyarrow": "11.0.0",
        "scikit-learn": "1.1.2",
    }
)
class TaxiFarePrediction(FlowSpec):
    data_url = Parameter("data_url", default=URL)

    def transform_features(self, df):
        # TODO:
        # Try to complete tasks 2 and 3 with this function doing nothing like it currently is.
        # Understand what is happening.
        # Revisit task 1 and think about what might go in this function.
        obviously_bad_data_filters = [
            "fare_amount > 0",  # fare_amount in US Dollars
            "trip_distance <= 100",  # trip_distance in miles
            "trip_distance > 0",
            "passenger_count > 0",
            "tpep_pickup_datetime <= tpep_dropoff_datetime",
            "total_amount > 0",
        ]
        df = df.query(" & ".join(obviously_bad_data_filters))
        
        print(df.shape)
        return df

    @retry(times=3)
    @card(type="corise")
    @step
    def start(self):
        import pandas as pd
        from sklearn.model_selection import train_test_split

        df_ = pd.read_parquet(self.data_url)
        missing = df_.isna().sum(axis=0).to_frame()

        self.df = self.transform_features(df_)
        current.card.append(Markdown("# Taxi Fare Incoming Data"))
        current.card.append(Markdown(f" DataFrame incoming shape:{df_.shape}"))
        current.card.append(
            Table.from_dataframe(missing)
        )
        current.card.append(Markdown(f" Clean DataFrame shape:{self.df.shape}"))
        # NOTE: we are split into training and validation set in the validation step which uses cross_val_score.
        # This is a simple/naive way to do this, and is meant to keep this example simple, to focus learning on deploying Metaflow flows.
        # In practice, you want split time series data in more sophisticated ways and run backtests.
        self.X = self.df["trip_distance"].values.reshape(-1, 1)
        self.y = self.df["total_amount"].values
        self.next(self.linear_model)

    @step
    def linear_model(self):
        "Fit a single variable, linear model to the data."
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # TODO: Play around with the model if you are feeling it.
        reg = LinearRegression()
        self.model_name = type(reg).__name__
        
        self.model = make_pipeline(
            StandardScaler(),
            reg
        )

        self.next(self.validate)

    def gather_sibling_flow_run_results(self):
        # storage to populate and feed to a Table in a Metaflow card
        rows = []

        # loop through runs of this flow
        for run in Flow(self.__class__.__name__):
            if run.id != current.run_id:
                if run.successful:
                    icon = "✅"
                    msg = "OK"
                    score = str(run.data.scores.mean())
                else:
                    icon = "❌"
                    msg = "Error"
                    score = "NA"
                    for step in run:
                        for task in step:
                            if not task.successful:
                                msg = task.stderr
                row = [
                    Markdown(icon),
                    Artifact(run.id),
                    Artifact(run.created_at.strftime(DATETIME_FORMAT)),
                    Artifact(score),
                    Markdown(msg),
                ]
                rows.append(row)
            else:
                rows.append(
                    [
                        Markdown("✅"),
                        Artifact(run.id),
                        Artifact(run.created_at.strftime(DATETIME_FORMAT)),
                        Artifact(str(self.scores.mean())),
                        Markdown("This run..."),
                    ]
                )
        return rows

    @timeout(minutes=10)
    @card(type="corise")
    @step
    def validate(self):
        from sklearn.model_selection import cross_val_score

        self.scores = cross_val_score(self.model, self.X, self.y, cv=5)
        current.card.append(Markdown("# Taxi Fare Prediction Results"))
        current.card.append(
            Table(
                self.gather_sibling_flow_run_results(),
                headers=["Pass/fail", "Run ID", "Created At", "R^2 score", "Stderr"],
            )
        )
        self.next(self.end)

    @step
    def end(self):
        print("Success!")


if __name__ == "__main__":
    TaxiFarePrediction()
