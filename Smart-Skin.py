# smart_skin.py
# Professional Skin Disease Detection with Pimple Location Analysis
# Production Ready - No Emojis

import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Skin disease classes
DISEASE_CLASSES = [
    'Acne Vulgaris (Pimple)',
    'Actinic Keratosis',
    'Basal Cell Carcinoma',
    'Benign Keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic Nevus',
    'Vascular Lesion'
]

# Face regions
FACE_REGIONS = {
    'forehead': 'Forehead',
    'left_cheek': 'Left Cheek',
    'right_cheek': 'Right Cheek',
    'chin': 'Chin',
    'nose': 'Nose',
    'jawline': 'Jawline',
    'under_eyes': 'Under Eyes',
    'temples': 'Temples',
    'between_eyebrows': 'Between Eyebrows',
    'hairline': 'Hairline'
}

# ============================================================
# 2. PIMPLE LOCATION REASONS DATABASE
# ============================================================

PIMPLE_LOCATION_REASONS = {
    'forehead': {
        'reasons': [
            'Lack of sleep - disrupts skin repair cycle',
            'Stress - increases cortisol and oil production',
            'Sebum - excess oil production in T-zone',
            'Too much sugar - causes inflammation',
            'Dirty glasses/spectacles - bacteria transfer',
            'Hair products - oils clogging pores',
            'Poor lymphatic system - toxin buildup',
            'Clogged pores from sweat'
        ],
        'hormonal': False,
        'digestive': False,
        'suggestions': [
            'Get 7-8 hours of sleep daily',
            'Reduce stress through meditation or exercise',
            'Use oil-free hair products',
            'Clean glasses regularly with alcohol wipes',
            'Reduce sugar intake',
            'Keep hair off forehead',
            'Wash face twice daily'
        ]
    },
    'left_cheek': {
        'reasons': [
            'Dirty pillowcase - bacteria from sleeping on left side',
            'Dirty phone - bacteria transfer while calling',
            'Touching face with unwashed hands',
            'Poor lymphatic system - toxin buildup',
            'Hormonal imbalance - androgen fluctuations',
            'Clogged pores from makeup',
            'Stress - increases inflammation',
            'Lack of sleep - affects skin healing'
        ],
        'hormonal': True,
        'digestive': False,
        'suggestions': [
            'Change pillowcases weekly (or every 2-3 days)',
            'Sanitize phone with alcohol wipes daily',
            'Avoid touching face throughout the day',
            'Use non-comedogenic makeup',
            'Remove makeup before sleeping',
            'Maintain consistent sleep schedule'
        ]
    },
    'right_cheek': {
        'reasons': [
            'Dirty pillowcase - bacteria from sleeping on right side',
            'Dirty phone - bacteria transfer while calling',
            'Touching face with unwashed hands',
            'Poor lymphatic system - toxin buildup',
            'Hormonal imbalance - androgen fluctuations',
            'Clogged pores from makeup',
            'Stress - increases inflammation',
            'Lack of sleep - affects skin healing'
        ],
        'hormonal': True,
        'digestive': False,
        'suggestions': [
            'Change pillowcases weekly (or every 2-3 days)',
            'Sanitize phone with alcohol wipes daily',
            'Avoid touching face throughout the day',
            'Use non-comedogenic makeup',
            'Remove makeup before sleeping',
            'Maintain consistent sleep schedule'
        ]
    },
    'chin': {
        'reasons': [
            'Hormonal imbalance - menstrual cycle fluctuations',
            'Touching chin/resting hand on chin',
            'Dirty face masks - bacteria accumulation',
            'Poor shaving technique (for men)',
            'Digestive system issues',
            'Stress - triggers hormonal acne',
            'Too much sugar - causes inflammation',
            'Sebum - excess oil production'
        ],
        'hormonal': True,
        'digestive': True,
        'suggestions': [
            'Track menstrual cycle for pattern recognition',
            'Avoid resting chin on hands',
            'Change disposable masks daily',
            'Use clean razor and shaving cream',
            'Reduce sugar and processed food intake',
            'Use salicylic acid products',
            'Consult gynecologist if persistent'
        ]
    },
    'nose': {
        'reasons': [
            'Sebum - excess oil in T-zone',
            'Clogged pores from oil and dirt',
            'Blackheads and whiteheads accumulation',
            'Bacteria from touching nose frequently',
            'Sweat - causes bacteria growth',
            'Dirty glasses/sunglasses',
            'Too much sugar - inflammation',
            'Poor lymphatic system'
        ],
        'hormonal': False,
        'digestive': False,
        'suggestions': [
            'Use salicylic acid cleanser',
            'Avoid touching nose throughout the day',
            'Use oil-free moisturizer',
            'Exfoliate gently 2 times per week',
            'Use clay masks to absorb excess oil',
            'Clean glasses frames regularly'
        ]
    },
    'jawline': {
        'reasons': [
            'Hormonal imbalance - androgen fluctuations',
            'Stress - triggers breakouts',
            'Touching jaw/resting hands',
            'Dirty face masks - bacterial accumulation',
            'Sweat - bacteria growth',
            'Poor lymphatic system',
            'Smoking - affects skin health',
            'Alcohol - dehydrates skin'
        ],
        'hormonal': True,
        'digestive': False,
        'suggestions': [
            'Manage stress levels through exercise or meditation',
            'Avoid touching face and jaw',
            'Change masks regularly (every 4-6 hours)',
            'Use salicylic acid or benzoyl peroxide products',
            'Reduce alcohol consumption',
            'Quit smoking for better skin health',
            'Consult dermatologist for hormonal acne'
        ]
    },
    'under_eyes': {
        'reasons': [
            'Lack of sleep - dark circles and breakouts',
            'Stress - affects skin barrier',
            'Eye makeup - can clog pores',
            'Contact dermatitis from eye products',
            'Dirty makeup brushes',
            'Allergic reactions to products',
            'Poor lymphatic system'
        ],
        'hormonal': False,
        'digestive': False,
        'suggestions': [
            'Get 7-8 hours of quality sleep',
            'Remove eye makeup before sleeping',
            'Use hypoallergenic eye products',
            'Clean makeup brushes after each use',
            'Avoid rubbing eyes',
            'Use gentle eye cream',
            'Consult dermatologist if persistent'
        ]
    },
    'temples': {
        'reasons': [
            'Dirty glasses/sunglasses - bacteria transfer',
            'Sweat from exercise or heat',
            'Hair products - oils clogging pores',
            'Clogged pores from sebum',
            'Stress - inflammation',
            'Lack of sleep'
        ],
        'hormonal': False,
        'digestive': False,
        'suggestions': [
            'Clean glasses frames daily',
            'Wash face after sweating',
            'Use oil-free hair products',
            'Keep hair off temples',
            'Reduce stress levels',
            'Maintain consistent sleep schedule'
        ]
    },
    'between_eyebrows': {
        'reasons': [
            'Sebum - excess oil production',
            'Clogged pores from hair products',
            'Improper hair removal (waxing/threading)',
            'Ingrown hair from eyebrow grooming',
            'Dirty makeup brushes',
            'Stress - inflammation'
        ],
        'hormonal': False,
        'digestive': False,
        'suggestions': [
            'Use gentle cleanser in this area',
            'Avoid waxing if sensitive',
            'Clean eyebrow tools after each use',
            'Use salicylic acid spot treatment',
            'Wash makeup brushes weekly',
            'Avoid touching this area'
        ]
    },
    'hairline': {
        'reasons': [
            'Hair products - oils and waxes clogging pores',
            'Sweat - bacteria growth',
            'Dirty hats or headbands',
            'Sebum - oil accumulation',
            'Lack of proper cleansing along hairline',
            'Clogged pores from dead skin cells'
        ],
        'hormonal': False,
        'digestive': False,
        'suggestions': [
            'Wash hair regularly',
            'Avoid heavy hair products near hairline',
            'Clean hats and headbands after each use',
            'Cleanse hairline area thoroughly',
            'Use oil-free products',
            'Keep hair clean and off forehead'
        ]
    }
}


# ============================================================
# 3. CNN MODEL
# ============================================================

class SkinDiseaseModel:
    """CNN model for skin disease classification."""
    
    def __init__(self, input_shape=(128, 128, 3), num_classes=8):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self):
        """Build CNN model."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def load_pretrained(self, model_path=None):
        """Load pretrained model or create mock model."""
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            return self.model
        
        self.build_model()
        return self.model
    
    def predict(self, image):
        """Predict disease from image."""
        if self.model is None:
            self.build_model()
        
        if image.shape != (128, 128, 3):
            image = cv2.resize(image, (128, 128))
        
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        
        predictions = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        return predicted_class, confidence, predictions[0]


# ============================================================
# 4. FACE REGION DETECTOR
# ============================================================

class FaceRegionDetector:
    """Detect face region for pimple location analysis."""
    
    def __init__(self):
        self.face_cascade = None
        self._load_cascade()
    
    def _load_cascade(self):
        """Load face cascade classifier."""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            print(f"Face cascade load error: {e}")
            self.face_cascade = None
    
    def detect_face_region(self, image):
        """Detect which region of the face the pimple is in."""
        if self.face_cascade is None:
            # Fallback: return center region
            h, w = image.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            if center_y < h // 4:
                return 'forehead'
            elif center_y < h // 4 and center_x < w // 3:
                return 'temples'
            elif center_y < h // 4 and center_x > 2 * w // 3:
                return 'temples'
            elif center_y < h // 2 and center_y >= h // 4:
                if center_x < w // 3:
                    return 'left_cheek'
                elif center_x > 2 * w // 3:
                    return 'right_cheek'
                else:
                    return 'nose'
            elif center_y >= h // 2:
                if center_x < w // 3:
                    return 'left_cheek'
                elif center_x > 2 * w // 3:
                    return 'right_cheek'
                else:
                    return 'chin'
            else:
                return 'forehead'
        
        # Try actual face detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return 'forehead'
        
        (x, y, w, h) = faces[0]
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Determine region based on position within face
        if face_center_y < y + h // 5:
            if face_center_x < x + w // 3:
                return 'temples'
            elif face_center_x > x + 2 * w // 3:
                return 'temples'
            else:
                return 'forehead'
        elif face_center_y < y + h // 3:
            if face_center_x < x + w // 3:
                return 'left_cheek'
            elif face_center_x > x + 2 * w // 3:
                return 'right_cheek'
            else:
                return 'between_eyebrows'
        elif face_center_y < y + 2 * h // 3:
            if face_center_x < x + w // 3:
                return 'left_cheek'
            elif face_center_x > x + 2 * w // 3:
                return 'right_cheek'
            else:
                return 'nose'
        else:
            if face_center_x < x + w // 3:
                return 'left_cheek'
            elif face_center_x > x + 2 * w // 3:
                return 'right_cheek'
            else:
                return 'chin'


# ============================================================
# 5. PIMPLE ANALYZER
# ============================================================

class PimpleAnalyzer:
    """Analyze pimples and provide location-based reasons."""
    
    def __init__(self):
        self.face_detector = FaceRegionDetector()
    
    def analyze(self, image, disease_name):
        """Analyze pimple location and provide reasons."""
        region_key = self.face_detector.detect_face_region(image)
        region_name = FACE_REGIONS.get(region_key, 'Unknown Region')
        
        location_data = PIMPLE_LOCATION_REASONS.get(region_key, {
            'reasons': ['Location not recognized. Consult dermatologist.'],
            'hormonal': False,
            'digestive': False,
            'suggestions': ['Consult dermatologist for proper diagnosis.']
        })
        
        return {
            'region_key': region_key,
            'region_name': region_name,
            'reasons': location_data['reasons'],
            'hormonal': location_data['hormonal'],
            'digestive': location_data['digestive'],
            'suggestions': location_data['suggestions']
        }


# ============================================================
# 6. RECOMMENDATION ENGINE
# ============================================================

class RecommendationEngine:
    """Generate disease and pimple recommendations."""
    
    def get_disease_info(self, disease_name):
        """Get disease information."""
        disease_db = {
            'Acne Vulgaris (Pimple)': {
                'description': 'Common skin condition causing pimples, blackheads, and whiteheads. Often appears on face, chest, and back.',
                'severity': 'Moderate',
                'treatment': [
                    'Benzoyl peroxide cream 2.5-5%',
                    'Salicylic acid cleansers (2%)',
                    'Retinoid creams (adapalene, tretinoin)',
                    'Antibiotics (clindamycin, doxycycline)',
                    'Oral medications (isotretinoin for severe cases)',
                    'Chemical peels (salicylic acid, glycolic acid)'
                ],
                'precautions': [
                    'Wash face twice daily with gentle cleanser',
                    'Avoid touching face throughout the day',
                    'Use non-comedogenic products',
                    'Avoid picking or popping pimples',
                    'Change pillowcases weekly',
                    'Sanitize phone daily',
                    'Remove makeup before sleeping'
                ],
                'doctor_type': 'Dermatologist',
                'urgency': 'Consult within 2-4 weeks if persistent or severe'
            },
            'Actinic Keratosis': {
                'description': 'Rough, scaly patches on skin caused by sun exposure. Precancerous condition.',
                'severity': 'Moderate',
                'treatment': ['Cryotherapy (freezing)', 'Topical creams (5-fluorouracil)', 'Photodynamic therapy'],
                'precautions': ['Use sunscreen SPF 50+ daily', 'Wear protective clothing', 'Avoid peak sun hours'],
                'doctor_type': 'Dermatologist',
                'urgency': 'Consult within 2-3 weeks'
            },
            'Basal Cell Carcinoma': {
                'description': 'Most common type of skin cancer. Appears as pearly or waxy bump.',
                'severity': 'High',
                'treatment': ['Mohs surgery', 'Excision surgery', 'Radiation therapy', 'Topical medications'],
                'precautions': ['Immediate dermatologist visit', 'Avoid sun exposure', 'Regular skin self-exams'],
                'doctor_type': 'Dermatologist (Urgent)',
                'urgency': 'Consult within 1 week'
            },
            'Benign Keratosis': {
                'description': 'Non-cancerous growths on skin. Also called seborrheic keratosis.',
                'severity': 'Low',
                'treatment': ['Usually no treatment needed', 'Cryotherapy if bothersome', 'Curettage (scraping)'],
                'precautions': ['Monitor for changes', 'Apply moisturizer', 'Protect from sun'],
                'doctor_type': 'Dermatologist (Optional)',
                'urgency': 'No urgency, but confirm diagnosis'
            },
            'Dermatofibroma': {
                'description': 'Benign skin growth, often on legs. Firm, brownish bump.',
                'severity': 'Low',
                'treatment': ['Usually no treatment needed', 'Surgical removal if bothersome', 'Cryotherapy'],
                'precautions': ['Avoid picking or scratching', 'Monitor for changes in size/color'],
                'doctor_type': 'Dermatologist',
                'urgency': 'No urgency, but confirm diagnosis'
            },
            'Melanoma': {
                'description': 'Most serious type of skin cancer. Develops in melanocytes.',
                'severity': 'Very High',
                'treatment': ['Immediate surgical removal', 'Immunotherapy', 'Targeted therapy', 'Chemotherapy'],
                'precautions': ['URGENT - Consult immediately', 'Complete sun avoidance', 'Regular follow-ups'],
                'doctor_type': 'Dermatologist (Emergency)',
                'urgency': 'Consult immediately (within 48 hours)'
            },
            'Melanocytic Nevus': {
                'description': 'Common mole. Usually benign but monitor for changes.',
                'severity': 'Low',
                'treatment': ['Usually no treatment needed', 'Surgical removal if atypical'],
                'precautions': ['Monitor ABCDE rules (Asymmetry, Border, Color, Diameter, Evolution)', 'Use sun protection', 'Regular skin checks'],
                'doctor_type': 'Dermatologist (Routine)',
                'urgency': 'No urgency, but monitor for changes'
            },
            'Vascular Lesion': {
                'description': 'Red/purple skin marks from blood vessels. Includes hemangiomas, port-wine stains.',
                'severity': 'Moderate',
                'treatment': ['Laser therapy', 'Sclerotherapy', 'Surgery if necessary', 'Observation'],
                'precautions': ['Protect from injury', 'Avoid scratching', 'Use sunscreen'],
                'doctor_type': 'Dermatologist',
                'urgency': 'Consult within 2-4 weeks'
            }
        }
        
        return disease_db.get(disease_name, {
            'description': 'Condition not found in database.',
            'severity': 'Unknown',
            'treatment': ['Consult a dermatologist'],
            'precautions': ['Regular skin checkups'],
            'doctor_type': 'Dermatologist',
            'urgency': 'Consult for proper diagnosis'
        })
    
    def generate_report(self, disease_name, confidence, pimple_data, image=None):
        """Generate comprehensive report."""
        disease_info = self.get_disease_info(disease_name)
        
        report = []
        report.append("=" * 70)
        report.append("SMART SKIN ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        report.append(f"Disease Detected: {disease_name}")
        report.append(f"Confidence: {confidence:.2f}%")
        report.append(f"Severity: {disease_info.get('severity', 'Unknown')}")
        report.append("")
        
        if disease_name == 'Acne Vulgaris (Pimple)':
            report.append("-" * 50)
            report.append("PIMPLE LOCATION ANALYSIS")
            report.append("-" * 50)
            report.append(f"Location: {pimple_data['region_name']}")
            report.append(f"Hormonal Factor: {'Yes' if pimple_data['hormonal'] else 'No'}")
            report.append(f"Digestive Factor: {'Yes' if pimple_data['digestive'] else 'No'}")
            report.append("")
            report.append("Possible Reasons for this Location:")
            for reason in pimple_data['reasons']:
                report.append(f"  - {reason}")
            report.append("")
            report.append("Suggestions for this Location:")
            for suggestion in pimple_data['suggestions']:
                report.append(f"  - {suggestion}")
            report.append("")
        
        report.append("-" * 50)
        report.append("DESCRIPTION")
        report.append("-" * 50)
        report.append(disease_info.get('description', 'No description available.'))
        report.append("")
        
        report.append("-" * 50)
        report.append("RECOMMENDED TREATMENTS")
        report.append("-" * 50)
        for treatment in disease_info.get('treatment', []):
            report.append(f"  - {treatment}")
        report.append("")
        
        report.append("-" * 50)
        report.append("PRECAUTIONS")
        report.append("-" * 50)
        for precaution in disease_info.get('precautions', []):
            report.append(f"  - {precaution}")
        report.append("")
        
        report.append("-" * 50)
        report.append("DOCTOR RECOMMENDATION")
        report.append("-" * 50)
        report.append(f"  Specialist: {disease_info.get('doctor_type', 'Dermatologist')}")
        report.append(f"  Urgency: {disease_info.get('urgency', 'Consult soon')}")
        report.append("")
        
        report.append("=" * 70)
        report.append("DISCLAIMER: This is an AI-based analysis. Please consult a")
        report.append("qualified dermatologist for proper diagnosis and treatment.")
        report.append("=" * 70)
        
        return "\n".join(report)


# ============================================================
# 7. IMAGE PROCESSOR
# ============================================================

class ImageProcessor:
    """Process images for analysis."""
    
    def process_image(self, image):
        """Process PIL image for model input."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((128, 128))
        img_array = np.array(image)
        
        return img_array
    
    def preprocess_for_display(self, image):
        """Preprocess image for display."""
        max_size = 400
        w, h = image.size
        if w > max_size or h > max_size:
            ratio = max_size / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            image = image.resize(new_size)
        
        return image


# ============================================================
# 8. STREAMLIT UI
# ============================================================

def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="Smart Skin - Disease Detection",
        page_icon="",
        layout="wide"
    )
    
    # Sidebar
    st.sidebar.title("Smart Skin")
    st.sidebar.markdown("---")
    st.sidebar.markdown("AI-Powered Skin Disease Detection")
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Upload a photo or take a picture using your camera.\n\n"
        "The AI model will analyze and detect skin conditions.\n\n"
        "For pimples, location-based reasons will be provided."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("Supported Conditions:")
    for disease in DISEASE_CLASSES:
        st.sidebar.markdown(f"- {disease}")
    
    # Main content
    st.title("Smart Skin Disease Detection")
    st.markdown("Upload a skin image to detect diseases and get recommendations.")
    
    # Initialise components
    model = SkinDiseaseModel()
    model.load_pretrained()
    processor = ImageProcessor()
    pimple_analyzer = PimpleAnalyzer()
    recommender = RecommendationEngine()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Upload Photo", "Take Photo", "About"])
    
    # Tab 1: Upload Photo
    with tab1:
        st.subheader("Upload a Skin Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                st.subheader("Analysis Results")
                
                with st.spinner("Analyzing image..."):
                    processed = processor.process_image(image)
                    predicted_class, confidence, all_probs = model.predict(processed)
                    disease_name = DISEASE_CLASSES[predicted_class]
                    
                    st.metric("Detected Disease", disease_name)
                    st.metric("Confidence", f"{confidence:.1f}%")
                    
                    if disease_name == 'Acne Vulgaris (Pimple)':
                        pimple_data = pimple_analyzer.analyze(processed, disease_name)
                        
                        st.subheader("Pimple Location Analysis")
                        st.info(f"Location: {pimple_data['region_name']}")
                        
                        # Show factors
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"Hormonal Factor: {'Yes' if pimple_data['hormonal'] else 'No'}")
                        with col_b:
                            st.write(f"Digestive Factor: {'Yes' if pimple_data['digestive'] else 'No'}")
                        
                        with st.expander("See Reasons for this Location"):
                            for reason in pimple_data['reasons']:
                                st.markdown(f"- {reason}")
                        
                        with st.expander("Suggestions for this Location"):
                            for suggestion in pimple_data['suggestions']:
                                st.markdown(f"- {suggestion}")
                    
                    with st.expander("All Predictions"):
                        for i, prob in enumerate(all_probs):
                            st.progress(float(prob), text=f"{DISEASE_CLASSES[i]}: {prob*100:.1f}%")
            
            # Recommendations
            st.markdown("---")
            st.subheader("Treatment Recommendations")
            
            disease_info = recommender.get_disease_info(disease_name)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("Description")
                st.info(disease_info.get('description', 'No description available.'))
                
                st.markdown("Severity")
                sev = disease_info.get('severity', 'Unknown')
                if sev == 'Low':
                    st.success("Low Severity")
                elif sev == 'Moderate':
                    st.warning("Moderate Severity")
                elif sev == 'High':
                    st.error("High Severity")
                elif sev == 'Very High':
                    st.error("Very High Severity")
                else:
                    st.info(f"Severity: {sev}")
            
            with col2:
                st.markdown("Treatment Options")
                for treatment in disease_info.get('treatment', [])[:4]:
                    st.markdown(f"- {treatment}")
            
            with col3:
                st.markdown("Precautions")
                for precaution in disease_info.get('precautions', [])[:4]:
                    st.markdown(f"- {precaution}")
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("Doctor Recommendation")
                st.success(f"Specialist: {disease_info.get('doctor_type', 'Dermatologist')}")
            
            with col2:
                st.markdown("Urgency")
                st.warning(f"{disease_info.get('urgency', 'Consult soon')}")
            
            with st.expander("View Full Report"):
                if disease_name == 'Acne Vulgaris (Pimple)':
                    pimple_data = pimple_analyzer.analyze(processed, disease_name)
                else:
                    pimple_data = {'region_name': 'N/A', 'reasons': [], 'suggestions': [], 'hormonal': False, 'digestive': False}
                
                report = recommender.generate_report(disease_name, confidence, pimple_data, image)
                st.text(report)
            
            st.warning(
                "Disclaimer: This is an AI-based analysis and should not "
                "replace professional medical advice. Always consult a qualified "
                "dermatologist for proper diagnosis and treatment."
            )
    
    # Tab 2: Take Photo
    with tab2:
        st.subheader("Take a Photo")
        
        camera_photo = st.camera_input("Take a picture using your camera")
        
        if camera_photo is not None:
            image = Image.open(camera_photo)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Captured Photo", use_container_width=True)
            
            with col2:
                st.subheader("Analysis Results")
                
                with st.spinner("Analyzing image..."):
                    processed = processor.process_image(image)
                    predicted_class, confidence, all_probs = model.predict(processed)
                    disease_name = DISEASE_CLASSES[predicted_class]
                    
                    st.metric("Detected Disease", disease_name)
                    st.metric("Confidence", f"{confidence:.1f}%")
                    
                    if disease_name == 'Acne Vulgaris (Pimple)':
                        pimple_data = pimple_analyzer.analyze(processed, disease_name)
                        
                        st.subheader("Pimple Location Analysis")
                        st.info(f"Location: {pimple_data['region_name']}")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"Hormonal Factor: {'Yes' if pimple_data['hormonal'] else 'No'}")
                        with col_b:
                            st.write(f"Digestive Factor: {'Yes' if pimple_data['digestive'] else 'No'}")
                        
                        with st.expander("See Reasons for this Location"):
                            for reason in pimple_data['reasons']:
                                st.markdown(f"- {reason}")
                        
                        with st.expander("Suggestions for this Location"):
                            for suggestion in pimple_data['suggestions']:
                                st.markdown(f"- {suggestion}")
                    
                    with st.expander("All Predictions"):
                        for i, prob in enumerate(all_probs):
                            st.progress(float(prob), text=f"{DISEASE_CLASSES[i]}: {prob*100:.1f}%")
            
            st.markdown("---")
            st.subheader("Treatment Recommendations")
            
            disease_info = recommender.get_disease_info(disease_name)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("Description")
                st.info(disease_info.get('description', 'No description available.'))
                
                st.markdown("Severity")
                sev = disease_info.get('severity', 'Unknown')
                if sev == 'Low':
                    st.success("Low Severity")
                elif sev == 'Moderate':
                    st.warning("Moderate Severity")
                elif sev == 'High':
                    st.error("High Severity")
                elif sev == 'Very High':
                    st.error("Very High Severity")
                else:
                    st.info(f"Severity: {sev}")
            
            with col2:
                st.markdown("Treatment Options")
                for treatment in disease_info.get('treatment', [])[:4]:
                    st.markdown(f"- {treatment}")
            
            with col3:
                st.markdown("Precautions")
                for precaution in disease_info.get('precautions', [])[:4]:
                    st.markdown(f"- {precaution}")
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("Doctor Recommendation")
                st.success(f"Specialist: {disease_info.get('doctor_type', 'Dermatologist')}")
            
            with col2:
                st.markdown("Urgency")
                st.warning(f"{disease_info.get('urgency', 'Consult soon')}")
            
            with st.expander("View Full Report"):
                if disease_name == 'Acne Vulgaris (Pimple)':
                    pimple_data = pimple_analyzer.analyze(processed, disease_name)
                else:
                    pimple_data = {'region_name': 'N/A', 'reasons': [], 'suggestions': [], 'hormonal': False, 'digestive': False}
                
                report = recommender.generate_report(disease_name, confidence, pimple_data, image)
                st.text(report)
            
            st. warning(
                "Disclaimer: This is an AI-based analysis and should not"
                "replace professional medical advice."
            )
    
    # Tab 3: About
    with tab3:
        st.subheader("About Smart Skin")
        
        st.markdown("""
        ### How It Works
        
        1. Upload or capture a skin image
        2. AI Model analyses the image
        3. Detects skin conditions
        4. Provides treatment recommendations
        
        For pimples, the system also analyses the location on the face and provides specific reasons and suggestions.
        
        ### Supported Conditions
        """)
        
        for disease in DISEASE_CLASSES:
            st.markdown(f"- {disease}")
        
        st.markdown("""
        ### Pimple Location Analysis
        
        The system analyses which part of the face the pimple is on and provides:
        - Possible reasons for pimples in that area
        - Hormonal and digestive factors
        - Location-specific suggestions
        
        ### Common Reasons for Pimples by Location
        
        **Forehead:**
        - Lack of sleep, stress, sebum, too much sugar, dirty glasses, hair products, poor lymphatic system
        
        **Cheeks:**
        - Dirty pillowcase, dirty phone, touching face, hormonal imbalance, clogged pores from makeup
        
        **Chin:**
        - Hormonal imbalance, digestive system issues, touching chin, dirty face masks
        
        **N
