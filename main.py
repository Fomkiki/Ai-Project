import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import MobileNetV2


# สร้าง sidebar
page = st.sidebar.radio("Select Page", ["Development - Machine Learning", "Model (Machine Learning)","Development - Neuron Network","Model (Neuron Network)"])

# โหลดข้อมูล
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/car_price.csv")
    df = df.sample(n=1500, random_state=42)  
    return df

df = load_data()

# เตรียมข้อมูล
selected_features = ["Brand", "Year", "Engine_Size", "Fuel_Type", "Mileage", "Transmission", "Owner_Count", "Price"]
df = df[selected_features]
df.dropna(inplace=True)

# แปลงข้อมูลหมวดหมู่ให้เป็นตัวเลข
label_encoders = {}
categorical_columns = ["Brand", "Fuel_Type", "Transmission"]
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# แยก Features และ Target
X = df.drop(columns=["Price"])
y = df["Price"]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# แบ่งข้อมูล Train-Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

if page == "Development - Machine Learning":
    st.title('Development - Machine Learning')
    st.write('วิธีการพัฒนา Model - Machine Learning ')
    
    st.write("\n### คำอธิบายเกี่ยวกับ Dataset\n")
    st.write("\nDataset ที่ใช้คือ Car Price Dataset ที่มาคือ https://www.kaggle.com/  โดยใช้สำหรับการทำนายราคาของรถยนต์ โดยข้อมูลประกอบไปด้วย 10,000 รายการ ซึ่งแต่ละแถวแสดงข้อมูลของรถยนต์หนึ่งคัน รวมถึงคุณสมบัติต่างๆ ที่มีผลต่อราคา เช่น ยี่ห้อรถยนต์ ปีที่ผลิต ขนาดเครื่องยนต์ ประเภทเชื้อเพลิง ประเภทเกียร์ และระยะทางที่ใช้งาน เป็นต้น\n")
    
    st.write("\n### คำอธิบายของแต่ละ Column\n")
    st.write("- **Brand:** ยี่ห้อของรถยนต์ เช่น Toyota, BMW, Ford\n")
    st.write("- **Year:** ปีที่ผลิต ยิ่งใหม่ยิ่งมีราคาสูง\n")
    st.write("- **Engine_Size:** ขนาดเครื่องยนต์ (ลิตร)\n")
    st.write("- **Fuel_Type:** ประเภทเชื้อเพลิง (Petrol, Diesel, Hybrid, Electric)\n")
    st.write("- **Transmission:** ประเภทเกียร์ (Manual, Automatic, Semi-Automatic)\n")
    st.write("- **Mileage:** ระยะทางที่ใช้งาน (กิโลเมตร) ยิ่งมาก ราคามักจะต่ำลง\n")
    st.write("- **Owner_Count:** จำนวนเจ้าของเดิมของรถ ยิ่งน้อย ยิ่งมีมูลค่าสูง\n")
    st.write("- **Price:** ราคาของรถยนต์ (เป้าหมายที่ต้องการทำนาย)\n")
    
    st.write("\n### การทำงานของโค้ด\n")
    st.write("1. **โหลดข้อมูล:** ใช้ฟังก์ชัน `load_data()` เพื่อโหลดข้อมูลจากไฟล์ CSV และสุ่มตัวอย่าง 1,500 รายการเพื่อใช้ในการวิเคราะห์\n")
    st.write("2. **เตรียมข้อมูล:** เลือกเฉพาะคอลัมน์ที่เกี่ยวข้อง และจัดการค่าที่หายไป (`dropna()`)\n")
    st.write("3. **แปลงข้อมูลหมวดหมู่เป็นตัวเลข:** ใช้ `LabelEncoder()` แปลงค่าของ `Brand`, `Fuel_Type`, และ `Transmission` ให้อยู่ในรูปตัวเลข\n")
    st.write("4. **แยก Features และ Target:** แยก `X` (ตัวแปรต้น) และ `y` (ราคาของรถยนต์)\n")
    st.write("5. **Standardization:** ใช้ `StandardScaler()` เพื่อปรับข้อมูลให้อยู่ในช่วงมาตรฐาน\n")
    st.write("6. **แบ่งข้อมูล Train-Test:** ใช้ `train_test_split()` แบ่งข้อมูลเป็น 80% สำหรับฝึกโมเดล และ 20% สำหรับทดสอบ\n")
    st.write("7. **เลือกและฝึกโมเดล:**\n   - **Random Forest:** ใช้ `RandomForestRegressor` ทำนายราคา และแสดงผล R² Score\n   - **K-Means:** ใช้ `KMeans` จัดกลุ่มรถยนต์เป็น 5 กลุ่ม และแสดงการกระจายตัวของข้อมูล\n")
    st.write("8. **ทำนายค่าจากข้อมูลใหม่:** รับข้อมูลจากผู้ใช้ แปลงข้อมูลให้ตรงกับรูปแบบของโมเดล แล้วทำการพยากรณ์ราคา หรือกลุ่มของรถยนต์\n")
    st.write("\n### ทฤษฎีการทำงานของโมเดล\n")
    st.write("- **Random Forest Regression:** เป็นเทคนิคของการเรียนรู้แบบ Supervised Learning ที่ใช้ Decision Trees หลายต้นในการทำงานร่วมกันเพื่อสร้างแบบจำลองที่มีความแม่นยำสูง โดยใช้วิธีการ Bagging ในการรวมผลลัพธ์ของต้นไม้แต่ละต้น\n")
    st.write("- **K-Means Clustering:** เป็นเทคนิคของ Unsupervised Learning ที่ใช้ในการจัดกลุ่มข้อมูลโดยการแบ่งข้อมูลออกเป็น K กลุ่มตามลักษณะของข้อมูลที่คล้ายกัน ซึ่งเหมาะกับการค้นหารูปแบบของข้อมูลที่ไม่ทราบล่วงหน้า\n")




elif page == "Model (Machine Learning)":
    st.title('Model (Machine Learning)')
    model_type = st.selectbox("เลือกโมเดลที่ต้องการ", ["Random Forest", "K-Means"])

    if model_type == "Random Forest":
        # สร้างและฝึก Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # คำนวณค่า R²
        r2 = r2_score(y_test, y_pred)

        st.write("### Model Performance: (Random Forest - Regression)")
        st.write(f"**R²**: {r2:.4f}")  # แสดงค่า R²

        # แสดงกราฟ Regression
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5, label="Predictions")
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Ideal Prediction")
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.legend()
        st.pyplot(fig)

        # รับข้อมูลใหม่จากผู้ใช้
        st.write("### ทำนายราคาของรถใหม่")
        brand = st.selectbox("เลือกยี่ห้อ", label_encoders["Brand"].classes_)
        year = st.number_input("เลือกปีที่ผลิต", min_value=2000, max_value=2023, value=2015)
        engine_size = st.number_input("ขนาดเครื่องยนต์ (ลิตร)", min_value=0.8, max_value=5.0, value=1.8)
        fuel_type = st.selectbox("เลือกประเภทเชื้อเพลิง", label_encoders["Fuel_Type"].classes_)
        mileage = st.number_input("ระยะทางที่ใช้งาน (กิโลเมตร)", min_value=0, value=9000)
        transmission = st.selectbox("เลือกประเภทเกียร์", label_encoders["Transmission"].classes_)
        owner_count = st.number_input("จำนวนเจ้าของ", min_value=1, max_value=5, value=2)

        # แปลงข้อมูลที่รับเข้ามา
        brand_encoded = label_encoders["Brand"].transform([brand])[0]
        fuel_type_encoded = label_encoders["Fuel_Type"].transform([fuel_type])[0]
        transmission_encoded = label_encoders["Transmission"].transform([transmission])[0]

        new_data = np.array([[brand_encoded, year, engine_size, fuel_type_encoded, mileage, transmission_encoded, owner_count]])
        new_data_scaled = scaler.transform(new_data)

        # ทำนายราคาด้วยโมเดล Random Forest
        predicted_price = model.predict(new_data_scaled)
        st.write(f"**ราคาที่ทำนายได้**: {predicted_price[0]:,.2f} dollar")

        # แสดงจุดข้อมูลใหม่ในกราฟ (จุดใหม่จะมีสีแดง)
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5, label="Predictions")
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Ideal Prediction")

        # แสดงจุดข้อมูลใหม่ที่ทำนายไว้
        ax.scatter(predicted_price[0], predicted_price[0], color='red', marker='o', s=100, label="New Data (Predicted)")
        
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.legend()
        st.pyplot(fig)

    elif model_type == "K-Means":
        # ทำการฝึก K-Means
        kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X_scaled)

        st.write("### Clustering Results: (K-Means)")

        # แสดงข้อมูลแต่ละกลุ่ม
        for i in range(5):
            st.write(f"#### Cluster {i}")
            cluster_data = df[df["Cluster"] == i]
            st.write(f"Number of data points: {len(cluster_data)}")
            st.write(cluster_data.describe())  # แสดงสถิติเบื้องต้นของแต่ละกลุ่ม

        # กราฟแสดงการกระจายของ Cluster
        fig, ax = plt.subplots()
        scatter = ax.scatter(df["Engine_Size"], df["Price"], c=df["Cluster"], cmap="viridis", alpha=0.5)
        ax.set_xlabel("Engine Size")
        ax.set_ylabel("Mileage")
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        st.pyplot(fig)

        # รับข้อมูลใหม่จากผู้ใช้
        st.write("### ทำนายกลุ่มของข้อมูลใหม่")
        brand = st.selectbox("เลือกยี่ห้อ", label_encoders["Brand"].classes_)
        year = st.number_input("เลือกปีที่ผลิต", min_value=2000, max_value=2023, value=2015)
        engine_size = st.number_input("ขนาดเครื่องยนต์ (ลิตร)", min_value=0.8, max_value=5.0, value=1.8)
        fuel_type = st.selectbox("เลือกประเภทเชื้อเพลิง", label_encoders["Fuel_Type"].classes_)
        mileage = st.number_input("ระยะทางที่ใช้งาน (กิโลเมตร)", min_value=0, value=9000)
        transmission = st.selectbox("เลือกประเภทเกียร์", label_encoders["Transmission"].classes_)
        owner_count = st.number_input("จำนวนเจ้าของ", min_value=1, max_value=5, value=2)

        # แปลงข้อมูลที่รับเข้ามา
        brand_encoded = label_encoders["Brand"].transform([brand])[0]
        fuel_type_encoded = label_encoders["Fuel_Type"].transform([fuel_type])[0]
        transmission_encoded = label_encoders["Transmission"].transform([transmission])[0]

        new_data = np.array([[brand_encoded, year, engine_size, fuel_type_encoded, mileage, transmission_encoded, owner_count]])
        new_data_scaled = scaler.transform(new_data)

        # ทำนายกลุ่มของข้อมูลใหม่
        predicted_cluster = kmeans.predict(new_data_scaled)
        st.write(f"### Predicted Cluster: {predicted_cluster[0]}")

        # แสดงจุดข้อมูลใหม่ในกราฟเดิม
        fig, ax = plt.subplots()
        scatter = ax.scatter(df["Engine_Size"], df["Price"], c=df["Cluster"], cmap="viridis", alpha=0.5)
        ax.set_xlabel("Engine Size")
        ax.set_ylabel("Mileage")
        
        # แสดงจุดข้อมูลใหม่ที่ทำนายไว้ในกราฟเดิม (จุดใหม่จะมีสีแดง)
        ax.scatter(engine_size, mileage, color='red', marker='o', s=100, label="New Data (Predicted)")

        # เพิ่มแสดง Legend
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        ax.legend()
        st.pyplot(fig)

elif page == "Development - Neuron Network":
    st.title('Development - Neuron Network')
    st.write('วิธีการพัฒนา Model - CNN ')
    
    st.write("\n### คำอธิบายเกี่ยวกับ Dataset\n")
    st.write("\nDataset ที่ใช้คือ Dog vs Cat Dataset แหล่งที่มาคือ https://www.kaggle.com/ โดยใช้สำหรับการจำแนกภาพสุนัขและแมว โดยมีข้อมูลรูปภาพที่ถูกแบ่งเป็นหมวดหมู่สำหรับฝึก (train) และทดสอบ (test)\n")
    
    st.write("\n### การทำงานของโค้ด\n")
    st.write("1. **ตั้งค่า TensorFlow Policy:** กำหนดให้ใช้งาน precision แบบ `mixed_float16` เพื่อเพิ่มประสิทธิภาพการประมวลผลบน GPU\n")
    st.write("2. **Data Augmentation:** ปรับแต่งภาพเพื่อเพิ่มความหลากหลายของข้อมูล โดยใช้ `ImageDataGenerator()`\n")
    st.write("3. **โหลดและปรับแต่งโมเดล MobileNetV2:** ใช้โมเดลพื้นฐานที่ถูกฝึกมาแล้ว และเพิ่มชั้น Fully Connected Layer\n")
    st.write("4. **Train Model:** ใช้ Optimizer `Adam` และ Loss Function `binary_crossentropy` เทรนโมเดลเพื่อเรียนรู้การจำแนกภาพ\n")
    st.write("5. **Prediction:** รับภาพจากผู้ใช้และทำนายว่าเป็นสุนัขหรือแมว\n")
    
    st.write("\n### ทฤษฎีการทำงานของโมเดล\n")
    st.write("- **Convolutional Neural Network (CNN):** เป็นโครงข่ายประสาทเทียมที่ใช้กับข้อมูลภาพ โดยมีชั้น Convolution, Pooling และ Fully Connected เพื่อเรียนรู้คุณลักษณะของภาพ\n")
    st.write("- **MobileNetV2:** เป็นโมเดลที่ถูกออกแบบมาให้เบาและเร็วขึ้นสำหรับงาน Image Classification โดยใช้ Depthwise Separable Convolutions เพื่อลดจำนวนพารามิเตอร์\n")


elif page == "Model (Neuron Network)":
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("GPUs:", tf.config.list_physical_devices('GPU'))

    # ตั้งค่า Dataset และ Parameters
    dataset_path = "Dataset/"
    img_size = (64, 64)
    batch_size = 32
    epochs = 15

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_path + 'train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        dataset_path + 'train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        dataset_path + 'test',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    # โหลดโมเดลพื้นฐาน MobileNetV2
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
    from tensorflow.keras import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

    # โหลดโมเดลพื้นฐาน MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

    # Freeze Layers
    for layer in base_model.layers:
        layer.trainable = False

    # เพิ่มชั้น Dense และ BatchNormalization
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)  
    x = Dense(16, activation='relu')(x)   
    predictions = Dense(1, activation='sigmoid')(x)

    # สร้างโมเดลใหม่
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # เทรนโมเดล
    st.write('### Training CNN Model')
    history = model.fit(
        train_generator, 
        validation_data=val_generator, 
        epochs=epochs,
        callbacks=[reduce_lr, early_stopping]
    )

    # แสดงกราฟ Accuracy
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Train Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)

    st.write("### Model Training Completed!")

    # รับรูปภาพจากผู้ใช้
    st.write("### Upload an image for prediction:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # แสดงภาพที่อัปโหลด
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # ปรับขนาดภาพให้ตรงกับขนาดที่โมเดลต้องการ (64x64)
        image = image.resize((64, 64))
        image_array = img_to_array(image)  # เปลี่ยนเป็นอาเรย์
        image_array = np.expand_dims(image_array, axis=0)  # เพิ่มมิติให้ตรงกับ input ของโมเดล
        image_array = image_array / 255.0  # Normalize ค่าให้ในช่วง [0,1]

        # ทำนายผลลัพธ์
        prediction = model.predict(image_array)

        # แสดงผลลัพธ์การทำนาย
        if prediction[0] > 0.5:
            st.write("This is a Dog! 🐶")
        else:
            st.write("This is a Cat! 🐱")


