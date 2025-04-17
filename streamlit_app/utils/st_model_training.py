import streamlit as st
import sys
from .preprocess import  ml_training_features
from .exploratory import plot_actual_vs_predicted
from .model_training import data_split, train, model_evaluation, save_model

def show(df):
    """
    this is for model training page
    """
    st.header("Model Training")

    # prepare model training features
    X, y = ml_training_features(df)

    # Train test split
    test_size = st.slider("Data test size (%)", 0.1, 0.4, 20.0 /100.0)
    X_train, X_test, y_train, y_test  = data_split(X, y, test_size = test_size)

    st.write(f"Training Data:, {len(X_train)} samples")
    st.write(f"Testing Data:, {len(X_test)} samples")

    # model selection
    model_type = st.selectbox("Select model type", ["Linear Regression", "Random Forest"])

    # button for model training
    if st.button('Train Model'):
        
        with st.spinner("Training in progress..."):
            # train model
            model = train(X_train, y_train, model_type = "Linear Regression")

            # Evaluate the model
            metrics = model_evaluation(model, X_train, X_test, y_train, y_test)

            # Display the metrics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Training Metrics")
                st.write(f"RMSE: {metrics['training_rmse']:.2f} C")
                st.write(f"R2: {metrics['training_r2']:.2f}")

            with col2:
                st.subheader("Testing Metrics")
                st.write(f"RMSE: {metrics['test_rmse']:.2f} C")
                st.write(f"R2: {metrics['test_r2']:.2f}")

            # the scatter plot of actual vs predicted values
            st.subheader("Actual vs Predicted ")
            fig = plot_actual_vs_predicted(metrics['y_test'], metrics['y_pred'])
            st.pyplot(fig)

        # save the model
        save_model(model)

        st.success("Model training was successful, and model is stored")
        st.session_state['model'] = model
        st.session_state['model_type'] = model_type




    