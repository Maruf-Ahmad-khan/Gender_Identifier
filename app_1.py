import streamlit as st
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


class GenderPredictionApp:
    def __init__(self):
        self.predicted_gender = None
        st.set_page_config(page_title="Gender Prediction", layout="centered")

    def render_title(self):
        st.title(" Gender Prediction App")
        st.subheader("Please enter your preferences:")

    def get_user_input(self):
        with st.form("prediction_form"):
            self.Favorite_Color = st.text_input("Favorite Color")
            self.Favorite_Music_Genre = st.text_input("Favorite Music Genre")
            self.Favorite_Beverage = st.text_input("Favorite Beverage")
            self.Favorite_Soft_Drink = st.text_input("Favorite Soft Drink")

            submitted = st.form_submit_button("Predict Gender")
            return submitted

    def predict(self):
        try:
            user_data = CustomData(
                Favorite_Color=self.Favorite_Color,
                Favorite_Music_Genre=self.Favorite_Music_Genre,
                Favorite_Beverage=self.Favorite_Beverage,
                Favorite_Soft_Drink=self.Favorite_Soft_Drink
            )
            final_df = user_data.get_data_as_dataframe()

            prediction_pipeline = PredictPipeline()
            prediction = prediction_pipeline.predict(final_df)

            label_map = {0: "Female", 1: "Male"}
            self.predicted_gender = label_map.get(prediction[0], "Unknown")

        except Exception as e:
            st.error(f" Error during prediction: {e}")
            return

    def display_result(self):
        if self.predicted_gender:
            st.success(f" Predicted Gender: **{self.predicted_gender}**")

    def run(self):
        self.render_title()
        if self.get_user_input():
            self.predict()
            self.display_result()


if __name__ == "__main__":
    app = GenderPredictionApp()
    app.run()
