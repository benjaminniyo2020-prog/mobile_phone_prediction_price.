import os
import pandas as pd
import streamlit as st

# Path to the model file (must be in the same folder as app.py)
model_file = os.path.join(os.path.dirname(__file__), "phone_sales_data.pkl")

# Load the trained model safely
if not os.path.exists(model_file):
    st.error("❌ Model file not found! Please upload 'phone_sales_data.pkl' to the repo.")
    loaded_model = None
else:
    try:
        loaded_model = joblib.load(model_file)
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}")
        loaded_model = None


# Function to predict phone price
def phone_price_prediction(Screen_Size_inches, RAM_GB, Storage_GB, Battery_Capacity_mAh, Camera_Quality_MP, loaded_model):
    new_phone = pd.DataFrame([{
        'Screen Size (inches)': Screen_Size_inches,
        'RAM (GB)': RAM_GB,
        'Storage (GB)': Storage_GB,
        'Battery Capacity (mAh)': Battery_Capacity_mAh,
        'Camera Quality (MP)': Camera_Quality_MP
    }])
    
    predicted_price = loaded_model.predict(new_phone)
    return predicted_price[0]


# Main Streamlit app
def main():
    st.set_page_config(page_title="Phone Price Predictor", page_icon="📱", layout="centered")
    st.title('📱 mobile Phone Price Prediction System')

    # Input fields
    screen_size = st.text_input('Enter the Screen Size (e.g., 6.2)')
    ram_gb = st.text_input('Enter the RAM (GB) (e.g., 4)')
    storage_gb = st.text_input('Enter the Storage (GB) (e.g., 64)')
    battery_capacity = st.text_input('Enter the Battery Capacity (mAh) (e.g., 4000)')
    camera_quality = st.text_input('Enter the Camera Quality (MP) (e.g., 48)')

    if st.button('📱 Predict Price'):
        if loaded_model is None:
            st.error("⚠️ Model not loaded. Please upload the model file.")
            return

        try:
            # Convert inputs to numeric
            screen_size = float(screen_size)
            ram_gb = int(ram_gb)
            storage_gb = int(storage_gb)
            battery_capacity = int(battery_capacity)
            camera_quality = int(camera_quality)

            # Make prediction
            price = phone_price_prediction(screen_size, ram_gb, storage_gb, battery_capacity, camera_quality, loaded_model)
            st.success(f'The predicted price for the phone is: **${price:.2f}**')
        except ValueError:
            st.error("⚠️ Please enter valid numeric values.")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {str(e)}")


# Run the app
if __name__ == '__main__':
    main()
