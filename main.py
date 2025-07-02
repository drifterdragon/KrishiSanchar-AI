from fastapi import FastAPI, UploadFile, File, Body
from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import re
import io
#Loading the model
model_crop = joblib.load("./ML Models/cropclassification.pkl")
model_fertilizer = load_model("./Ml Models/fertilizer_prediction.keras")
scaler = joblib.load("./Ml Models/scaler.pkl")
input_columns = joblib.load("./Ml Models/input_columns_names.pkl")
model_tomato = load_model("./Ml Models/Tomato_disease_model.keras")
model_potato=load_model("./Ml Models/Potato_disease_model.keras")
model_maize=load_model("./Ml Models/maize_disease_model.keras")
# Class labels
class_names_potatos = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
class_names_tomatos = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]
class_names_maize=['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']


# Utility: image prediction
def predict_image(image: Image.Image, model):
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

def recommendation(predicted_class):
    with open(f"./Recommendation/{predicted_class}.txt","r",encoding="utf-8") as file:
        recomm=file.read()
    return recomm

def parse_recommendation(text: str) -> dict:
    # Define the headings (can be in Nepali if consistent)
    sections = {
        "disease": r"रोग के हो\?",
        "causes": r"रोग लाग्ने कारणहरू",
        "symptoms": r"लक्ष्‍यणहरू \(Symptoms\)",
        "management": r"रोकथाम र नियन्त्रणका उपायहरू \(Prevention & Management\)"
    }

    # Find each section and split
    result = {}
    pattern = '|'.join(f"(?P<{k}>{v})" for k, v in sections.items())
    matches = list(re.finditer(pattern, text))

    for i, match in enumerate(matches):
        section_name = match.lastgroup
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        # Optional: Split each section into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        result[section_name] = paragraphs

    return result


# FastAPI app
app = FastAPI()

#Api Endpoint for crop and fertilizer prediction

@app.post("/predict/crop")
async def predict_agriculture_inputs(data: dict = Body(...)):
    try:
        # Validate Area_Harvested
        area_raw = data.get("Area_Harvested")
        production_raw = data.get("Production")

        if area_raw in [None, "", 0, "0"] or production_raw in [None, "", 0, "0"]:
            raise HTTPException(
                status_code=422,
                detail="Area_Harvested and Production cannot be zero or empty."
            )

        area = float(area_raw)
        production = float(production_raw)

        # Crop prediction
        input_df = pd.DataFrame([[area, production]], columns=["Area Harvested (ha)", "Production (t)"])
        prediction_crop = model_crop.predict(input_df)[0]

        # Fertilizer input creation
        new_input = {
            "Item": prediction_crop,
            "Area Harvested (ha)": area,
            "Production (t)": production
        }
        input_df = pd.DataFrame([new_input])
        input_df = pd.get_dummies(input_df, columns=["Item"], dtype='float64')
        input_df.columns = input_df.columns.str.replace("Item_", "", regex=False)

        for col in input_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[input_columns]
        input_df[['Area Harvested (ha)', 'Production (t)']] = scaler.transform(
            input_df[['Area Harvested (ha)', 'Production (t)']]
        )

        prediction_fertilizer = model_fertilizer.predict(input_df)[0]

        return JSONResponse(content={
            "Crops": str(prediction_crop),
            "Urea (Nitrogen)": float(round(prediction_fertilizer[0], 4)),
            "DAP (Phosphate)": float(round(prediction_fertilizer[1], 4)),
            "Potash (K₂O)": float(round(prediction_fertilizer[2], 4))
        })

    except HTTPException as e:
        raise e  # Re-raise to let FastAPI handle it

    except Exception as e:
        return JSONResponse(
            content={"error": f"Invalid input or prediction failed: {str(e)}"},
            status_code=400
        )

#api endpoint for crop disease detection
#Tomato Disease Prediction

# Tomato disease prediction
@app.post("/predict/Tomato")
async def predict_tomato(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        prediction = predict_image(image, model_tomato)
        prediction_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names_tomatos[prediction_index]
        prediction_confidence = float(prediction[0][prediction_index] * 100)
        if(predicted_class=='Tomato_healthy'):
            return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": round(prediction_confidence, 2)})
            
            

        # Get raw recommendation text
        raw_recommendation = recommendation(predicted_class)

        # Convert to structured sections
        structured_recommendation = parse_recommendation(raw_recommendation)
        return JSONResponse(content={
        "predicted_class": predicted_class,
        "confidence": round(prediction_confidence, 2),
        "recommendation": {
        section: [para.replace('\n', ' ') for para in paragraphs]
        for section, paragraphs in structured_recommendation.items()
        }
        })

       
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
    

# Potato disease prediction
@app.post("/predict/Potato")
async def predict_tomato(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        prediction = predict_image(image, model_potato)
        prediction_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names_potatos[prediction_index]
        prediction_confidence = float(prediction[0][prediction_index] * 100)
        if(predicted_class=='Potato___healthy'):
            return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": round(prediction_confidence, 2)})
            
            

        # Get raw recommendation text
        raw_recommendation = recommendation(predicted_class)

        # Convert to structured sections
        structured_recommendation = parse_recommendation(raw_recommendation)
        return JSONResponse(content={
        "predicted_class": predicted_class,
        "confidence": round(prediction_confidence, 2),
        "recommendation": {
        section: [para.replace('\n', ' ') for para in paragraphs]
        for section, paragraphs in structured_recommendation.items()
        }
        })

       
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# Potato disease prediction
@app.post("/predict/Maize")
async def predict_tomato(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        prediction = predict_image(image, model_maize)
        prediction_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names_maize[prediction_index]
        prediction_confidence = float(prediction[0][prediction_index] * 100)
        if(predicted_class=='Healthy'):
            return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": round(prediction_confidence, 2)})
            
            

        # Get raw recommendation text
        raw_recommendation = recommendation(predicted_class)

        # Convert to structured sections
        structured_recommendation = parse_recommendation(raw_recommendation)
        return JSONResponse(content={
        "predicted_class": predicted_class,
        "confidence": round(prediction_confidence, 2),
        "recommendation": {
        section: [para.replace('\n', ' ') for para in paragraphs]
        for section, paragraphs in structured_recommendation.items()
        }
        })

       
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})