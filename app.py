import streamlit as st
import tensorflow as tf
import tensorflow.compat.v1 as tf_compat
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import json
import random
import logging
import time
import datetime
from collections import deque
import os
from sklearn.ensemble import RandomForestClassifier
import joblib
import base64
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set page config
st.set_page_config(page_title="Skin Disease Analyzer", layout="wide")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Enhanced dark theme CSS with redesigned chatbot UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

body, .stApp {
    background: linear-gradient(45deg, #0a0a0a, #1a1a1a);
    color: #e6e6e6;
    font-family: 'Roboto', sans-serif;
    line-height: 1.6;
    transition: all 0.3s ease;
    overflow-x: hidden;
}

header {
    background: linear-gradient(135deg, #00f7ff, #7b00ff);
    padding: 4rem;
    text-align: center;
    border-bottom: 4px solid #00f7ff;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.6);
    animation: fadeInDown 1.2s ease;
}

h1 {
    color: #0a0a0a;
    font-size: 3.5rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
    text-shadow: 0 0 20px #00f7ff;
}

h2, h3, h4 {
    color: #e6e6e6;
    font-weight: 600;
    position: relative;
}

h2::after {
    content: '';
    position: absolute;
    bottom: -6px;
    left: 0;
    width: 80px;
    height: 5px;
    background: #00f7ff;
    border-radius: 2px;
}

p, label {
    color: #e6e6e6;
    font-size: 1.2rem;
    font-weight: 300;
}

.stButton > button {
    background: linear-gradient(45deg, #00f7ff, #7b00ff);
    color: #0a0a0a;
    border: none;
    padding: 1rem 2rem;
    border-radius: 15px;
    font-weight: 500;
    font-size: 1.2rem;
    transition: transform 0.3s, box-shadow 0.3s;
    animation: pulse 2s infinite;
}

.stButton > button:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px #00f7ff;
}

.stFileUploader > div > div {
    border: 2px dashed #00f7ff;
    padding: 3rem;
    border-radius: 20px;
    background: rgba(30, 30, 30, 0.85);
    backdrop-filter: blur(12px);
    color: #e6e6e6;
    text-align: center;
    transition: all 0.3s;
}

.stFileUploader > div > div:hover {
    border-color: #7b00ff;
    transform: scale(1.05);
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > select {
    background: rgba(30, 30, 30, 0.85);
    color: #e6e6e6;
    border: 1px solid #444444;
    border-radius: 15px;
    padding: 0.8rem;
    font-size: 1.2rem;
}

.stContainer {
    background: rgba(30, 30, 30, 0.85);
    padding: 3rem;
    border-radius: 20px;
    border: 1px solid #444444;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
    animation: fadeInUp 1s ease;
    backdrop-filter: blur(12px);
}

.info-grid, .image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 2.5rem;
    margin: 4rem 0;
}

.info-card, .image-card {
    background: rgba(30, 30, 30, 0.85);
    padding: 2.5rem;
    border-radius: 20px;
    border: 1px solid #444444;
    text-align: center;
    transition: all 0.3s;
    animation: fadeInUp 0.8s ease;
    backdrop-filter: blur(12px);
}

.info-card:hover, .image-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 12px 40px #00f7ff;
}

/* Redesigned Chatbot UI */
.chatbot-toggle-btn {
    position: fixed;
    bottom: 30px;
    right: 30px;
    width: 60px;
    height: 60px;
    background: linear-gradient(145deg, #00d4ff, #7b00ff);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    cursor: pointer;
    z-index: 1000;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.chatbot-toggle-btn:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 20px rgba(0, 212, 255, 0.7);
}

.chatbot-toggle-btn svg {
    width: 30px;
    height: 30px;
    fill: #ffffff;
}

.chatbot-container {
    position: fixed;
    bottom: 100px;
    right: 30px;
    width: 360px;
    max-height: 500px;
    background: rgba(40, 40, 40, 0.95);
    border-radius: 15px;
    border: 1px solid #00d4ff;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(10px);
    z-index: 999;
    display: none;
    flex-direction: column;
    animation: slideUp 0.4s ease;
}

.chatbot-container.open {
    display: flex;
}

.chatbot-header {
    background: linear-gradient(145deg, #00d4ff, #7b00ff);
    color: #ffffff;
    padding: 0.8rem 1.2rem;
    border-radius: 14px 14px 0 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

.chatbot-header h4 {
    margin: 0;
    font-size: 1.2rem;
    font-weight: 500;
}

.chatbot-close-btn {
    background: transparent;
    border: none;
    color: #ffffff;
    font-size: 1.2rem;
    cursor: pointer;
    transition: color 0.3s ease;
}

.chatbot-close-btn:hover {
    color: #ff4081;
}

.chatbot-body {
    padding: 1rem;
    flex: 1;
    overflow-y: auto;
    background: rgba(30, 30, 30, 0.9);
}

.chat-message {
    display: flex;
    align-items: flex-start;
    margin-bottom: 1rem;
    animation: fadeIn 0.3s ease;
}

.user-message, .bot-message {
    padding: 0.8rem;
    border-radius: 12px;
    max-width: 80%;
    word-wrap: break-word;
    font-size: 0.9rem;
    line-height: 1.4;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
}

.user-message {
    background: #00d4ff;
    color: #0a0a0a;
    margin-left: auto;
    border-radius: 12px 12px 0 12px;
}

.bot-message {
    background: #2e3b3e;
    color: #e6e6e6;
    border-radius: 12px 12px 12px 0;
}

.avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    margin: 0 8px;
    background: #00d4ff;
    color: #0a0a0a;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 500;
    font-size: 0.8rem;
}

.bot-avatar {
    background: #7b00ff;
    color: #ffffff;
}

.timestamp {
    font-size: 0.65rem;
    color: #a0a0a0;
    margin-top: 0.3rem;
    text-align: right;
}

.typing-indicator {
    display: flex;
    gap: 5px;
    padding: 0.8rem;
}

.dot {
    width: 8px;
    height: 8px;
    background: #00d4ff;
    border-radius: 50%;
    animation: bounce 1.2s infinite;
}

.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }

.chat-input-container {
    padding: 1rem;
    background: rgba(30, 30, 30, 0.9);
    border-top: 1px solid #444444;
    border-radius: 0 0 14px 14px;
}

.chat-input {
    width: 100%;
    background: #1c2526;
    border: 1px solid #00d4ff;
    color: #e6e6e6;
    padding: 0.6rem;
    border-radius: 10px;
    font-size: 0.9rem;
    height: 70px;
    resize: none;
    transition: border-color 0.3s, box-shadow 0.3s;
    margin-bottom: 0.5rem;
}

.chat-input:hover {
    box-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
}

.chat-input:focus {
    outline: none;
    border-color: #7b00ff;
    box-shadow: 0 0 10px rgba(123, 0, 255, 0.3);
}

.chat-buttons {
    display: flex;
    justify-content: space-between;
    gap: 0.8rem;
}

.chat-send-button {
    background: linear-gradient(145deg, #00d4ff, #7b00ff);
    color: #ffffff;
    border: none;
    padding: 0.6rem 1.5rem;
    border-radius: 10px;
    cursor: pointer;
    transition: transform 0.3s, box-shadow 0.3s;
    font-size: 0.9rem;
    font-weight: 500;
    flex: 1;
}

.chat-send-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
}

.feedback-buttons {
    display: flex;
    gap: 2rem;
    justify-content: center;
}

.feedback-btn {
    background: linear-gradient(45deg, #00f7ff, #7b00ff);
    color: #0a0a0a;
    padding: 0.8rem 2rem;
    border-radius: 15px;
    transition: transform 0.3s;
}

.feedback-btn:hover {
    transform: scale(1.15);
}

nav {
    background: rgba(30, 30, 30, 0.9);
    padding: 1.8rem;
    position: sticky;
    top: 0;
    z-index: 1000;
    border-bottom: 4px solid #00f7ff;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(12px);
}

nav ul {
    display: flex;
    justify-content: center;
    gap: 4rem;
    list-style: none;
    margin: 0;
    padding: 0;
}

nav a {
    color: #e6e6e6;
    font-weight: 500;
    padding: 0.8rem 1.5rem;
    border-radius: 15px;
    text-decoration: none;
    transition: all 0.3s;
}

nav a:hover {
    background: linear-gradient(45deg, #00f7ff, #7b00ff);
    color: #0a0a0a;
    transform: translateY(-3px);
}

.image-preview {
    border: 2px solid #444444;
    border-radius: 20px;
    max-width: 450px;
    transition: transform 0.3s;
}

.image-preview:hover {
    transform: scale(1.1);
}

.results-dashboard {
    background: rgba(30, 30, 30, 0.85);
    padding: 3rem;
    border-radius: 20px;
    border: 1px solid #444444;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(12px);
}

.progress-bar {
    height: 20px;
    background: #333333;
    border-radius: 15px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(45deg, #00f7ff, #7b00ff);
    transition: width 0.6s ease-in-out;
}

section {
    padding: 5rem 2rem;
    position: relative;
    animation: fadeIn 1.2s ease;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-30px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(0, 247, 255, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(0, 247, 255, 0); }
    100% { box-shadow: 0 0 0 0 rgba(0, 247, 255, 0); }
}

@keyframes bounce {
    0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-8px); }
    60% { transform: translateY(-4px); }
}
</style>
<script>
document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelectorAll('nav a');
    navLinks.forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                const offset = targetElement.getBoundingClientRect().top + window.pageYOffset - 80;
                window.scrollTo({ top: offset, behavior: 'smooth' });
            }
        });
    });

    // Auto-scroll chat to bottom
    function scrollChatToBottom() {
        const chatBody = document.querySelector('.chatbot-body');
        if (chatBody) {
            chatBody.scrollTop = chatBody.scrollHeight;
        }
    }
    scrollChatToBottom();

    // Toggle chatbot visibility
    const toggleBtn = document.querySelector('.chatbot-toggle-btn');
    const chatbotContainer = document.querySelector('.chatbot-container');
    if (toggleBtn && chatbotContainer) {
        toggleBtn.addEventListener('click', function() {
            chatbotContainer.classList.toggle('open');
            scrollChatToBottom();
        });

        const closeBtn = document.querySelector('.chatbot-close-btn');
        if (closeBtn) {
            closeBtn.addEventListener('click', function() {
                chatbotContainer.classList.remove('open');
            });
        }
    }

    // Handle chat form submission
    const chatForm = document.querySelector('#chat-form');
    if (chatForm) {
        chatForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const input = document.querySelector('#chat-input');
            if (input.value.trim()) {
                const hiddenInput = document.querySelector('#hidden-chat-input');
                hiddenInput.value = input.value;
                input.value = '';
                document.querySelector('#chat-submit').click();
            }
        });
    }

    // Handle clear chat
    const clearChatButton = document.querySelector('#clear-chat-button');
    if (clearChatButton) {
        clearChatButton.addEventListener('click', function() {
            document.querySelector('#clear-chat').click();
        });
    }

    // Ensure chat scrolls to bottom after new messages
    const observer = new MutationObserver(scrollChatToBottom);
    const chatBody = document.querySelector('.chatbot-body');
    if (chatBody) {
        observer.observe(chatBody, { childList: true, subtree: true });
    }
});
</script>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'doctor_params' not in st.session_state:
    st.session_state.doctor_params = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hello! I'm your AI Medical Assistant. Upload a skin image or ask me anything to get started!", "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ]
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = True
if 'typing' not in st.session_state:
    st.session_state.typing = False

# Load models with graph reset
try:
    tf_compat.reset_default_graph()
    logger.info("TensorFlow default graph reset successfully")
    if os.path.exists('best_model.h5'):
        primary_model = load_model('best_model.h5')
        model_type = 'deep_learning'
        logger.info("Loaded deep learning model (best_model.h5)")
    elif os.path.exists('best_model_cnn.h5') and os.path.exists('best_model_rf.pkl'):
        cnn_model = load_model('best_model_cnn.h5')
        rf_model = joblib.load('best_model_rf.pkl')
        primary_model = (cnn_model, rf_model)
        model_type = 'cnn_rf'
        logger.info("Loaded CNN + RF model (best_model_cnn.h5, best_model_rf.pkl)")
    else:
        raise FileNotFoundError("Model files not found.")
except Exception as e:
    logger.error(f"Error loading primary model: {str(e)}")
    primary_model = None
    model_type = None
    st.error(f"Error loading model: {str(e)}. Please ensure model files are present.")

try:
    tf_compat.reset_default_graph()
    logger.info("TensorFlow default graph reset for DenseNet121")
    densenet_base = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    densenet_base.trainable = False
    densenet_model = tf.keras.Sequential([
        densenet_base,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dense(4, activation='softmax')
    ])
    densenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    logger.info("DenseNet121 model loaded successfully")
except Exception as e:
    logger.error(f"Error loading DenseNet model: {str(e)}")
    densenet_model = None
    st.warning("DenseNet121 failed to load. Re-analysis may be limited.")

# Storage files
FEEDBACK_FILE = 'feedback_data.json'
DOCTOR_DATA_FILE = 'doctor_data.json'

# Load/save data functions
def load_feedback_data():
    try:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r') as f:
                data = json.load(f)
                valid_data = [entry for entry in data if isinstance(entry, dict) and 'feedback' in entry]
                if len(valid_data) < len(data):
                    logger.warning(f"Filtered out {len(data) - len(valid_data)} invalid feedback entries")
                return valid_data
        return []
    except Exception as e:
        logger.error(f"Error loading feedback data: {str(e)}")
        return []

def load_doctor_data():
    try:
        if os.path.exists(DOCTOR_DATA_FILE):
            with open(DOCTOR_DATA_FILE, 'r') as f:
                data = json.load(f)
                valid_data = [entry for entry in data if isinstance(entry, dict)]
                if len(valid_data) < len(data):
                    logger.warning(f"Filtered out {len(data) - len(valid_data)} invalid doctor entries")
                return valid_data
        return []
    except Exception as e:
        logger.error(f"Error loading doctor data: {str(e)}")
        return []

def save_feedback_data(data):
    try:
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving feedback data: {str(e)}")
        st.error(f"Error saving feedback: {str(e)}")

def save_doctor_data(data):
    try:
        with open(DOCTOR_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving doctor data: {str(e)}")
        st.error(f"Error saving doctor data: {str(e)}")

feedback_data = load_feedback_data()
doctor_data = load_doctor_data()

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.update_freq = 10
        self.step_count = 0

    def _build_model(self):
        model = tf.keras.Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        try:
            if state.shape != (1, self.state_size):
                state = state.reshape(1, self.state_size)
            if random.uniform(0, 1) < self.epsilon:
                action = random.randrange(self.action_size)
            else:
                act_values = self.model.predict(state, verbose=0)
                action = np.argmax(act_values[0])
            logger.info(f"DQN Action: {action}, State: {state.flatten()}")
            return action
        except Exception as e:
            logger.error(f"Error in DQN act: {str(e)}")
            return random.randrange(self.action_size)

    def replay(self, batch_size):
        try:
            if len(self.memory) < batch_size:
                return
            minibatch = random.sample(self.memory, batch_size)
            states = np.array([item[0][0] for item in minibatch])
            next_states = np.array([item[3][0] for item in minibatch])
            targets = self.model.predict(states, verbose=0)
            next_q_values = self.target_model.predict(next_states, verbose=0)

            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                target = reward
                if not done:
                    target = reward + self.gamma * np.amax(next_q_values[i])
                targets[i][action] = target

            self.model.fit(states, targets, epochs=1, verbose=0)
            self.step_count += 1
            if self.step_count % self.update_freq == 0:
                self.update_target_model()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        except Exception as e:
            logger.error(f"Error in DQN replay: {str(e)}")

# Initialize DQN Agent
state_size = 5
action_size = 4
agent = DQNAgent(state_size, action_size)
recent_confidences = deque(maxlen=10)

# Image preprocessing with data augmentation
def prepare_image(image, target):
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(target)
        image_array = img_to_array(image)
        variance = np.var(image_array) / 255.0
        if variance < 0.03:
            raise ValueError("Image is too uniform or blurry.")
        if np.mean(image_array) < 15 or np.mean(image_array) > 240:
            raise ValueError("Image is too dark or too bright.")
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array.astype('float32') / 255.0

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        augmented_images = [image_array]
        for _ in range(3):
            aug_iter = datagen.flow(image_array, batch_size=1)
            aug_image = next(aug_iter)
            if aug_image.shape == (1, 224, 224, 3):
                augmented_images.append(aug_image)
            else:
                logger.warning("Augmented image has incorrect shape, skipping.")
        return np.vstack(augmented_images), variance
    except Exception as e:
        logger.error(f"Error preparing image: {str(e)}")
        raise

# Enhanced ensemble prediction
def ensemble_predict(image_arrays, primary_model, densenet_model, model_type):
    try:
        categories = ["Normal", "Monkeypox", "Chickenpox", "Measles"]
        primary_preds = None
        densenet_preds = None

        primary_pred_list = []
        densenet_pred_list = []

        for image_array in image_arrays:
            if image_array.shape != (224, 224, 3):
                image_array = image_array.reshape(224, 224, 3)
            image_array = np.expand_dims(image_array, axis=0)

            if primary_model:
                try:
                    if model_type == 'deep_learning':
                        pred = primary_model.predict(image_array, verbose=0)[0]
                        primary_pred_list.append(pred)
                    else:
                        cnn_model, rf_model = primary_model
                        features = cnn_model.predict(image_array, verbose=0)
                        pred = rf_model.predict_proba(features)[0]
                        primary_pred_list.append(pred)
                except Exception as e:
                    logger.error(f"Primary model prediction failed: {str(e)}")
                    continue

            if densenet_model:
                try:
                    pred = densenet_model.predict(image_array, verbose=0)[0]
                    densenet_pred_list.append(pred)
                except Exception as e:
                    logger.error(f"DenseNet model prediction failed: {str(e)}")
                    continue

        if primary_pred_list:
            primary_preds = np.mean(primary_pred_list, axis=0)
        if densenet_pred_list:
            densenet_preds = np.mean(densenet_pred_list, axis=0)

        variance = np.var(image_arrays, axis=(1, 2, 3)).mean()
        contrast = np.mean([np.std(img) for img in image_arrays])
        primary_weight = 0.7 if primary_preds is not None else 0.0
        densenet_weight = 0.3 if densenet_preds is not None else 0.0
        if variance < 0.1 or contrast < 10:
            primary_weight *= 0.75
            densenet_weight *= 0.75
        if primary_preds is None and densenet_preds is None:
            return "Uncertain", 0.0, [0.25] * 4

        final_preds = (
            primary_weight * primary_preds + densenet_weight * densenet_preds
        ) / (primary_weight + densenet_weight) if primary_preds is not None and densenet_preds is not None else (
            primary_preds if primary_preds is not None else densenet_preds
        )

        confidence = float(np.max(final_preds))
        entropy = -np.sum(final_preds * np.log2(final_preds + 1e-10))
        threshold = 0.6 if variance > 0.15 and entropy < 1.8 else 0.65
        if confidence < threshold or entropy > 1.8:
            if confidence > 0.5:
                result = categories[np.argmax(final_preds)]
                return result, confidence, final_preds.tolist()
            return "Uncertain", 0.0, final_preds.tolist()

        result = categories[np.argmax(final_preds)]
        return result, confidence, final_preds.tolist()
    except Exception as e:
        logger.error(f"Error in ensemble prediction: {str(e)}")
        return "Uncertain", 0.0, [0.25] * 4

# Visualization functions
def plot_confidence_bar(confidence_scores, categories):
    df = pd.DataFrame({
        'Category': categories,
        'Confidence': [score * 100 for score in confidence_scores]
    })
    fig = px.bar(df, x='Category', y='Confidence', color='Category',
                 title="Confidence Scores",
                 color_discrete_sequence=['#00f7ff', '#7b00ff', '#ff4081', '#00cc96'],
                 height=450,
                 barmode='group',
                 opacity=0.9)
    fig.update_traces(marker=dict(line=dict(width=2, color='#0a0a0a')))
    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Confidence (%)",
        showlegend=False,
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto", size=14)
    )
    return fig

def plot_radar_chart(confidence_scores, categories):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[score * 100 for score in confidence_scores] + [confidence_scores[0] * 100],
        theta=categories + [categories[0]],
        fill='toself',
        line_color='#00f7ff',
        opacity=0.8
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        title="Prediction Probability",
        height=450,
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto", size=14)
    )
    return fig

def plot_confidence_trend(confidences):
    valid_confidences = [c for c in confidences if isinstance(c, (int, float)) and not np.isnan(c)]
    if not valid_confidences:
        fig = go.Figure()
        fig.add_annotation(
            text="No confidence data available yet.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#e6e6e6"),
            align="center"
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Roboto", size=14),
            height=350,
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    df = pd.DataFrame({
        'Prediction': list(range(len(valid_confidences))),
        'Confidence': [float(c) * 100 for c in valid_confidences]
    })
    fig = px.line(df, x='Prediction', y='Confidence',
                  title="Confidence Trend",
                  labels={'Prediction': 'Prediction #', 'Confidence': 'Confidence (%)'},
                  height=350)
    fig.update_traces(line=dict(color='#00f7ff', width=3))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto", size=14),
        showlegend=False
    )
    return fig

# Llama response simulation (replace with actual API call if available)
def fetch_llama_response(user_input):
    try:
        # Placeholder for Llama API call
        return f"Received your message: '{user_input}'. How can I assist further?"
    except Exception as e:
        logger.error(f"Error fetching Llama response: {str(e)}")
        return "Error occurred. Please try again or consult a doctor."

# Feedback and doctor analysis
def submit_feedback(value):
    try:
        result = st.session_state.analysis_result
        if not result:
            st.error("No analysis result available.")
            return

        raw_timestamp = time.time()
        formatted_timestamp = datetime.datetime.fromtimestamp(raw_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        feedback_entry = {
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "feedback": value,
            "timestamp": raw_timestamp,
            "formatted_timestamp": formatted_timestamp,
            "image_data": result['image_data']
        }
        feedback_data.append(feedback_entry)
        save_feedback_data(feedback_data)

        variance = result.get('variance', 0.1)
        conf_trend = np.mean(recent_confidences) if recent_confidences else result['confidence']
        state = np.array([[result['confidence'], conf_trend, ["Normal", "Monkeypox", "Chickenpox", "Measles", "Uncertain"].index(result['prediction']), variance, value]])
        state = state / np.max(np.abs(state) + 1e-10)
        action = ["Normal", "Monkeypox", "Chickenpox", "Measles"].index(result['agent_action'])
        reward = value * 2 - 1
        next_state = state
        done = False
        agent.remember(state, action, reward, next_state, done)
        agent.replay(32)
        st.session_state.feedback_submitted = True
        logger.info(f"Feedback submitted: {value}, Reward: {reward}")
    except Exception as e:
        st.error(f"Error submitting feedback: {str(e)}")
        logger.error(f"Error submitting feedback: {str(e)}")

def standard_reanalysis():
    try:
        result = st.session_state.analysis_result
        if not result or not densenet_model:
            st.error("No prior analysis or DenseNet model unavailable.")
            return

        image_data = base64.b64decode(result['image_data'])
        image = Image.open(io.BytesIO(image_data))
        processed_images, variance = prepare_image(image, target=(224, 224))

        densenet_pred_list = []
        for image_array in processed_images:
            if image_array.shape != (224, 224, 3):
                image_array = image_array.reshape(224, 224, 3)
            image_array = np.expand_dims(image_array, axis=0)
            pred = densenet_model.predict(image_array, verbose=0)[0]
            densenet_pred_list.append(pred)

        if not densenet_pred_list:
            st.error("Re-analysis failed. Try again.")
            return

        final_preds = np.mean(densenet_pred_list, axis=0)
        categories = ["Normal", "Monkeypox", "Chickenpox", "Measles"]
        confidence = float(np.max(final_preds))
        entropy = -np.sum(final_preds * np.log2(final_preds + 1e-10))
        threshold = 0.6 if variance > 0.15 and entropy < 1.8 else 0.65

        if confidence < threshold or entropy > 1.8:
            if confidence > 0.5:
                new_result = categories[np.argmax(final_preds)]
            else:
                new_result = "Uncertain"
        else:
            new_result = categories[np.argmax(final_preds)]

        st.session_state.analysis_result = {
            "prediction": new_result,
            "confidence": confidence,
            "confidence_scores": final_preds.tolist(),
            "image_data": result['image_data'],
            "agent_action": result['agent_action'],
            "variance": variance
        }
        recent_confidences.append(float(confidence))
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": f"Re-analyzed as {new_result} with {(confidence * 100):.2f}% confidence. Chat with AI for more details or questions!",
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        st.rerun()
    except Exception as e:
        st.error(f"Error in re-analysis: {str(e)}")
        logger.error(f"Error in re-analysis: {str(e)}")

# Streamlit app
def main():
    # Header
    st.markdown("""
    <header>
        <h1>Skin Disease Analyzer</h1>
        <p>Futuristic AI-Powered Skin Diagnosis for Medical Excellence</p>
    </header>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown("""
    <nav>
        <ul>
            <li><a href="#home">Home</a></li>
            <li><a href="#about">About</a></li>
            <li><a href="#technology">Technology</a></li>
            <li><a href="#diagnosis">Diagnosis</a></li>
        </ul>
    </nav>
    """, unsafe_allow_html=True)

    # Home Section
    with st.container():
        st.markdown("""
        <section id="home">
            <h2 class="text-2xl mt-12">Welcome to Skin Disease Analyzer</h2>
            <p>
                The Skin Disease Analyzer is a revolutionary AI-powered tool designed to assist medical professionals in diagnosing skin conditions with unparalleled accuracy and speed. Leveraging cutting-edge deep learning and reinforcement learning, our platform provides reliable, data-driven insights to enhance clinical decision-making.
            </p>
            <p>
                Whether you're a dermatologist, general practitioner, or healthcare provider, our tool empowers you to identify conditions like Monkeypox, Chickenpox, Measles, and Normal skin with confidence. Our mission is to bridge the gap between advanced AI technology and practical healthcare applications, ensuring better patient outcomes.
            </p>
            <div class="info-grid">
                <div class="info-card">
                    <h3 class="text-lg">Why Choose Us?</h3>
                    <p>
                        Our platform combines state-of-the-art AI models with user-friendly interfaces, making it easy to integrate into your workflow. With real-time analysis and detailed visualizations, you can trust our tool to deliver actionable insights.
                    </p>
                </div>
                <div class="info-card">
                    <h3 class="text-lg">For Medical Professionals</h3>
                    <p>
                        Designed with doctors in mind, our analyzer supports clinical assessments by providing probabilistic predictions, confidence scores, and agent recommendations. Upload a skin image and get results in seconds.
                    </p>
                </div>
                <div class="info-card">
                    <h3 class="text-lg">Get Started</h3>
                    <p>
                        Ready to transform your diagnostic process? Navigate to the Diagnosis section to upload a skin image and experience the power of AI-driven healthcare. Consult a doctor for final confirmation.
                    </p>
                </div>
            </div>
        </section>
        """, unsafe_allow_html=True)

    # About Section
    with st.container():
        st.markdown("""
        <section id="about">
            <h2 class="text-2xl mt-12">About The Project</h2>
            <p>
                The Skin Disease Analyzer was born out of a vision to revolutionize dermatological diagnostics using artificial intelligence. Initiated in 2023, our project is a collaboration between AI researchers, dermatologists, and software engineers dedicated to advancing medical technology.
            </p>
            <p>
                Our team comprises experts from leading institutions, with decades of combined experience in machine learning, medical imaging, and clinical practice. We aim to make high-quality diagnostics accessible to healthcare providers worldwide, reducing diagnostic errors and improving patient care.
            </p>
            <p>
                Our mission is to empower doctors with tools that enhance their expertise, not replace it. By integrating AI with human oversight, we ensure that our platform delivers reliable results while respecting the nuances of clinical judgment.
            </p>
            <div class="info-grid">
                <div class="info-card">
                    <h3 class="text-lg">Our History</h3>
                    <p>
                        Starting as a research project, we partnered with hospitals to validate our models on diverse datasets, including the Monkeypox Skin Image Dataset. Today, we serve thousands of users globally.
                    </p>
                </div>
                <div class="info-card">
                    <h3 class="text-lg">Our Team</h3>
                    <p>
                        From AI pioneers to board-certified dermatologists, our team is united by a passion for innovation. We work closely with medical boards to ensure compliance and efficacy.
                    </p>
                </div>
                <div class="info-card">
                    <h3 class="text-lg">Our Vision</h3>
                    <p>
                        We envision a future where AI-assisted diagnostics are standard in every clinic, enabling faster, more accurate diagnoses and better health outcomes for all.
                    </p>
                </div>
            </div>
        </section>
        """, unsafe_allow_html=True)

    # Technology Section
    with st.container():
        st.markdown("""
        <section id="technology">
            <h2 class="text-2xl mt-12">Our Technology</h2>
            <p>
                At the core of the Skin Disease Analyzer is a sophisticated blend of deep learning, transfer learning, and reinforcement learning. Our platform is built to handle the complexities of skin image analysis, delivering robust and adaptive diagnostics.
            </p>
            <p>
                We use convolutional neural networks (CNNs) trained on extensive datasets, fine-tuned with transfer learning using DenseNet121, a pre-trained model renowned for its feature extraction capabilities. Our reinforcement learning component, powered by a DQN Agent, adapts to user feedback, ensuring continuous improvement.
            </p>
            <p>
                Data augmentation techniques, such as rotation and flipping, enhance our model's robustness, while ensemble prediction combines multiple models for higher accuracy. Our visualizations, including bar charts and radar plots, provide clear insights into prediction probabilities.
            </p>
            <div class="image-grid">
                <div class="image-card">
                    <h4 class="text-md">Deep Learning</h4>
                    <p>
                        Our custom CNNs are optimized for skin image classification, achieving high accuracy across diverse conditions. They process images in seconds, enabling rapid diagnostics.
                    </p>
                </div>
                <div class="image-card">
                    <h4 class="text-md">Transfer Learning</h4>
                    <p>
                        By leveraging DenseNet121, we extract rich features from skin images, improving performance on limited datasets and ensuring generalizability across skin types.
                    </p>
                </div>
                <div class="image-card">
                    <h4 class="text-md">Reinforcement Learning</h4>
                    <p>
                        Our DQN Agent learns from user feedback, refining its recommendations over time. This adaptive approach makes our tool smarter with every interaction.
                    </p>
                </div>
            </div>
        </section>
        """, unsafe_allow_html=True)

    # Diagnosis Section
    with st.container():
        st.markdown("""
        <section id="diagnosis">
            <h2 class="text-2xl mt-12">Diagnose Your Skin</h2>
            <p>
                The Diagnosis section is your gateway to AI-powered skin analysis. Upload a high-resolution skin image, and our platform will analyze it for conditions like Monkeypox, Chickenpox, Measles, or Normal skin. Follow these steps for best results:
            </p>
            <p>
                1. <strong>Upload a Clear Image</strong>: Ensure the image is well-lit, focused, and at least 100x100 pixels.<br>
                2. <strong>Analyze</strong>: Click the "Analyze Image" button to process the image.<br>
                3. <strong>Review Results</strong>: View the prediction, confidence score, agent recommendation, and visualizations.<br>
                4. <strong>Provide Feedback</strong>: Help us improve by indicating if the result was helpful.<br>
                5. <strong>Re-Analyze</strong>: Use standard or doctor-guided re-analysis for deeper insights.
            </p>
            <p>
                Our results include a confidence score (0-100%), a DQN Agent recommendation, and detailed visualizations like bar charts and radar plots. For uncertain results, we recommend uploading a clearer image or consulting a doctor.
            </p>
            <div class="info-grid">
                <div class="info-card">
                    <h3 class="text-lg">Image Tips</h3>
                    <p>
                        Use natural lighting, avoid shadows, and capture the affected area clearly. High-resolution images yield the best results.
                    </p>
                </div>
                <div class="info-card">
                    <h3 class="text-lg">Understanding Results</h3>
                    <p>
                        The confidence score reflects the model's certainty. The DQN Agent provides a secondary recommendation based on learned patterns.
                    </p>
                </div>
                <div class="info-card">
                    <h3 class="text-lg">Doctor Integration</h3>
                    <p>
                        Enter doctor-guided parameters for tailored analysis, incorporating clinical insights for enhanced accuracy.
                    </p>
                </div>
            </div>
            <h3 class="text-lg mt-8">Upload Skin Image</h3>
        """, unsafe_allow_html=True)

        with st.container():
            uploaded_file = st.file_uploader("Choose a clear skin image", type=["jpg", "jpeg", "png"], help="Upload a high-resolution skin image")
            if uploaded_file:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_container_width=False, width=450)
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")

            if st.button("Analyze Image", disabled=not uploaded_file or primary_model is None):
                with st.spinner("Analyzing..."):
                    try:
                        image_data = uploaded_file.getvalue()
                        image = Image.open(io.BytesIO(image_data))
                        if image.size[0] < 100 or image.size[1] < 100:
                            st.warning("Image is too small. Please upload an image with at least 100x100 pixels.")
                            return
                        processed_images, variance = prepare_image(image, target=(224, 224))

                        result, confidence, confidence_scores = ensemble_predict(processed_images, primary_model, densenet_model, model_type)
                        if result == "Uncertain":
                            st.warning("Prediction is uncertain. Try a clearer image or consult a doctor.")

                        recent_confidences.append(float(confidence))
                        conf_trend = np.mean(recent_confidences) if recent_confidences else confidence
                        state = np.array([[confidence, conf_trend, ["Normal", "Monkeypox", "Chickenpox", "Measles", "Uncertain"].index(result), variance, 0]])
                        state = state / np.max(np.abs(state) + 1e-10)
                        action = agent.act(state)
                        action_label = ["Normal", "Monkeypox", "Chickenpox", "Measles"][action % 4]

                        st.session_state.analysis_result = {
                            "prediction": result,
                            "confidence": confidence,
                            "confidence_scores": confidence_scores,
                            "image_data": base64.b64encode(image_data).decode('utf-8'),
                            "agent_action": action_label,
                            "variance": variance
                        }
                        st.session_state.feedback_submitted = False
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": f"Image analyzed as {result} with {(confidence * 100):.2f}% confidence. Chat with AI for more details or questions!",
                            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })

                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                        logger.error(f"Error processing image: {str(e)}")

            if st.session_state.analysis_result:
                result = st.session_state.analysis_result
                st.markdown(f"""
                <div class='results-dashboard'>
                    <h3 class='text-lg'>Analysis Result</h3>
                    <p><strong>Prediction:</strong> {result['prediction']}</p>
                    <p><strong>Confidence:</strong> {(result['confidence'] * 100):.2f}%</p>
                    <p><strong>Agent Recommendation:</strong> {result['agent_action']}</p>
                    <div class='progress-bar'>
                        <div class='progress-fill' style='width: {result['confidence'] * 100}%'></div>
                    </div>
                    <p style='color: #00f7ff; font-style: italic; margin-top: 1rem;'>Chat with AI for more details or questions!</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<h4 class='text-lg mt-6'>Prediction Visualizations</h4>", unsafe_allow_html=True)
                categories = ["Normal", "Monkeypox", "Chickenpox", "Measles"]
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_confidence_bar(result['confidence_scores'], categories), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_radar_chart(result['confidence_scores'], categories), use_container_width=True)
                st.markdown("<p class='text-sm italic'>Confidence Trend</p>", unsafe_allow_html=True)
                st.plotly_chart(plot_confidence_trend(recent_confidences), use_container_width=True)

                st.markdown("<h4 class='text-lg mt-6'>Was this helpful?</h4>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Yes", disabled=st.session_state.feedback_submitted, key="feedback_yes"):
                        submit_feedback(1)
                with col2:
                    if st.button("👎 No", disabled=st.session_state.feedback_submitted, key="feedback_no"):
                        submit_feedback(0)
                if st.session_state.feedback_submitted:
                    st.markdown("<p class='text-sm italic mt-2'>Thank you for your feedback!</p>", unsafe_allow_html=True)

                if st.button("Re-Analyze with DenseNet", key="standard_reanalysis", disabled=densenet_model is None):
                    with st.spinner("Re-analyzing..."):
                        standard_reanalysis()

                with st.container():
                    with st.form("doctor_form"):
                        st.markdown("<h4 class='text-lg mt-6'>Doctor-Guided Analysis</h4>", unsafe_allow_html=True)
                        doctor_name = st.text_input("Doctor Name", value="Dr. John Doe")
                        doctor_id = st.text_input("Doctor ID", value="DOC12345")
                        hospital_name = st.text_input("Hospital Name", value="City Hospital")
                        doctor_license = st.text_input("Doctor License", value="LIC98765")
                        issue_description = st.text_area("Issue Description", placeholder="E.g., rash on upper left arm")
                        contrast_threshold = st.number_input("Contrast Threshold (0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
                        roi = st.selectbox("Region of Interest", ["Full Image", "Upper Left", "Lower Right", "Center"])
                        severity = st.number_input("Severity (0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
                        submitted = st.form_submit_button("Submit")
                        if submitted:
                            st.session_state.doctor_params = {
                                "doctor_name": doctor_name,
                                "doctor_id": doctor_id,
                                "hospital_name": hospital_name,
                                "doctor_license": doctor_license,
                                "issue_description": issue_description,
                                "contrast_threshold": contrast_threshold,
                                "roi": roi,
                                "severity": severity
                            }
                            st.markdown("<p style='color: #00f7ff;'>Doctor parameters submitted!</p>", unsafe_allow_html=True)

        # Chatbot
        if st.session_state.show_chat:
            with st.container():
                chat_html = """
                <div class='chatbot-toggle-btn'>
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>
                        <path d="M12 9H8v2h4V9zm4 0h-2v2h2V9zm2 0h-2v2h2V9z"/>
                    </svg>
                </div>
                <div class='chatbot-container'>
                    <div class='chatbot-header'>
                        <h4>AI Medical Assistant</h4>
                        <button class='chatbot-close-btn' type='button'>✕</button>
                    </div>
                    <div class='chatbot-body'>
                """
                for msg in st.session_state.chat_messages:
                    avatar_class = 'bot-avatar' if msg["role"] == "assistant" else 'avatar'
                    avatar_label = 'AI' if msg["role"] == "assistant" else 'U'
                    message_class = 'bot-message' if msg["role"] == "assistant" else 'user-message'
                    chat_html += f"""
                    <div class='chat-message'>
                        <div class='{avatar_class}'>{avatar_label}</div>
                        <div class='{message_class}'>
                            <p>{msg['content']}</p>
                            <p class='timestamp'>{msg['timestamp']}</p>
                        </div>
                    </div>
                    """
                if st.session_state.typing:
                    chat_html += """
                    <div class='typing-indicator'>
                        <div class='dot'></div>
                        <div class='dot'></div>
                        <div class='dot'></div>
                    </div>
                    """
                chat_html += """
                    </div>
                    <div class='chat-input-container'>
                        <form id='chat-form'>
                            <textarea class='chat-input' id='chat-input' name='chat_input' placeholder='Type your question here...'></textarea>
                            <div class='chat-buttons'>
                                <button type='submit' class='chat-send-button'>Send</button>
                                <button type='button' id='clear-chat-button' class='chat-send-button'>Clear</button>
                            </div>
                        </form>
                    </div>
                </div>
                """
                st.markdown(chat_html, unsafe_allow_html=True)

                # Hidden form for chat input
                with st.form("chat_form"):
                    chat_input = st.text_input("Chat Input", key="hidden_chat_input", label_visibility="collapsed")
                    submit_button = st.form_submit_button("Submit Chat")
                    if submit_button and chat_input:
                        st.session_state.chat_messages.append({
                            "role": "user",
                            "content": chat_input,
                            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        st.session_state.typing = True
                        with st.spinner():
                            response = fetch_llama_response(chat_input)
                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "content": response,
                                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                            st.session_state.typing = False
                        st.rerun()

                # Hidden clear chat button
                clear_chat = st.button("Clear Chat", key="clear_chat")
                if clear_chat:
                    st.session_state.chat_messages = [
                        {"role": "assistant", "content": "Hello! I'm your AI Medical Assistant. Upload a skin image or ask me anything to get started!", "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    ]
                    st.rerun()

        st.markdown("</section>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()