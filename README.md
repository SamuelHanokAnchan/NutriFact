# FoodVisionAI

## Automated Nutritional Analysis with AI-Powered Food Recognition

FoodVisionAI is a comprehensive nutrition tracking application that combines deep learning-based food classification with personalized health recommendations. Developed as a Data Analytics-3 course project, this application demonstrates the practical integration of computer vision, nutrition APIs, and generative AI to create an end-to-end health monitoring solution.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Model Architecture](#model-architecture)
6. [Dataset](#dataset)
7. [Training Methodology](#training-methodology)
8. [API Integrations](#api-integrations)
9. [Installation](#installation)
10. [Usage Guide](#usage-guide)
11. [Project Structure](#project-structure)
12. [Performance Metrics](#performance-metrics)
13. [Future Enhancements](#future-enhancements)
14. [References](#references)
15. [License](#license)

---

## Project Overview

Manual calorie tracking is often tedious, inaccurate, and time-consuming. Users frequently struggle to estimate the nutritional content of their meals, leading to poor dietary decisions and difficulty in achieving health goals. FoodVisionAI addresses these challenges by providing:

- Instant food recognition from photographs using deep learning
- Automatic retrieval of comprehensive nutritional information
- Personalized daily calorie targets based on user goals
- AI-powered health recommendations considering individual profiles
- Portion-aware nutrition tracking with adjustable quantities

The application serves as a personal nutrition assistant, enabling users to make informed dietary choices aligned with their weight management objectives.

---

## Features

### Core Functionality

**Food Recognition System**
- Primary recognition using a trained MobileNetV2 model capable of classifying 101 food categories
- Fallback recognition using Google Gemini 2.0 Flash via OpenRouter API for enhanced accuracy
- Support for both camera capture and image upload

**Nutritional Analysis**
- Real-time nutrition data retrieval from USDA FoodData Central API
- Comprehensive macronutrient breakdown including calories, protein, carbohydrates, fat, fiber, and sugar
- Portion size adjustment with dynamic recalculation of nutritional values (10g to 1500g range)

**Personalized Goal Setting**
- Four-step onboarding process collecting user demographics and health objectives
- Support for weight loss, weight gain, and weight maintenance goals
- Automatic daily calorie target calculation using the Mifflin-St Jeor BMR equation
- Activity level consideration with four intensity categories

**AI Health Advisor**
- Generative AI integration using Google Gemini 2.0 Flash model
- Personalized food recommendations based on user BMI, goals, and activity level
- Health rating system categorizing food choices as healthy, moderate, or unhealthy
- Intelligent alternative food suggestions for suboptimal choices

**Progress Tracking**
- Daily calorie consumption monitoring with visual progress indicators
- Macronutrient tracking with recommended daily intake targets
- Water intake logging with eight-cup daily goal
- BMI calculation and categorization
- Meal history with timestamps and portion sizes

### User Interface

- Modern, responsive design with gradient color scheme
- Custom SVG icon system for visual consistency
- Progress bars with animated transitions
- Color-coded meal type tags (breakfast, lunch, dinner, snack)
- Mobile-friendly layout with collapsible sidebar

---

## System Architecture

```
+------------------+     +-------------------+     +------------------+
|                  |     |                   |     |                  |
|   User Device    +---->+  Streamlit App    +---->+  MobileNetV2     |
|   (Camera/Upload)|     |  (Frontend)       |     |  Model           |
|                  |     |                   |     |                  |
+------------------+     +--------+----------+     +------------------+
                                  |
                                  |
                    +-------------+-------------+
                    |             |             |
                    v             v             v
            +-------+---+ +-------+---+ +-------+--------+
            |           | |           | |                |
            | USDA API  | | OpenRouter| | Session State  |
            | (Nutrition)| | (Gemini) | | (User Data)    |
            |           | |           | |                |
            +-----------+ +-----------+ +----------------+
```

### Data Flow

1. User uploads or captures a food image through the Streamlit interface
2. Image is preprocessed and passed to the MobileNetV2 classifier
3. If confidence is low or user requests, fallback to Gemini vision model
4. Identified food name is sent to USDA API for nutritional data
5. User adjusts portion size via slider; nutrition values scale accordingly
6. Upon request, food details and user profile are sent to Gemini for health advice
7. All data is stored in session state and displayed on the dashboard

---

## Technology Stack

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| Frontend | Streamlit | 1.28+ | Web application framework |
| Deep Learning | TensorFlow | 2.17.1 | Model training and inference |
| Deep Learning | Keras | 3.10.0 | High-level neural network API |
| Base Model | MobileNetV2 | - | Pre-trained feature extractor |
| Dataset | Food-101 | - | Training data via TensorFlow Datasets |
| Nutrition API | USDA FoodData Central | v1 | Nutritional information retrieval |
| AI API | OpenRouter | - | Gemini 2.0 Flash access |
| Language | Python | 3.9+ | Primary development language |
| Image Processing | Pillow | 10.0+ | Image manipulation |
| HTTP Client | Requests | 2.31+ | API communication |
| Numerical Computing | NumPy | 1.24+ | Array operations |

---

## Model Architecture

### Base Model: MobileNetV2

MobileNetV2 was selected as the base architecture for the following reasons:

- **Efficiency**: Lightweight design suitable for deployment on resource-constrained devices
- **Performance**: Achieves strong accuracy through depthwise separable convolutions
- **Pre-training**: ImageNet weights provide robust feature extraction capabilities
- **Inference Speed**: Optimized for real-time predictions

### Custom Classification Head

The pre-trained MobileNetV2 base is extended with a custom classification head:

```
Input Layer (224 x 224 x 3)
         |
         v
MobileNetV2 Base (Frozen/Partially Frozen)
         |
         v
Global Average Pooling 2D
         |
         v
Batch Normalization
         |
         v
Dense Layer (256 units, ReLU activation)
         |
         v
Dropout (0.5)
         |
         v
Dense Layer (128 units, ReLU activation)
         |
         v
Dropout (0.3)
         |
         v
Output Layer (101 units, Softmax activation)
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Total Parameters | 2,422,949 |
| Trainable Parameters (Phase 1) | 163,685 |
| Trainable Parameters (Phase 2) | 2,670,821 |
| Input Shape | (224, 224, 3) |
| Output Classes | 101 |

---

## Dataset

### Food-101

The model is trained on the Food-101 dataset, a widely-used benchmark for food classification tasks.

| Attribute | Value |
|-----------|-------|
| Total Images | 101,000 |
| Categories | 101 |
| Images per Category | 1,000 |
| Image Format | JPEG |
| Resolution | Variable (resized to 224x224) |

### Sample Categories

The dataset includes diverse food categories such as:
- Apple Pie, Baby Back Ribs, Baklava, Beef Carpaccio
- Caesar Salad, Cannoli, Cheesecake, Chicken Curry
- Donuts, Dumplings, Edamame, Eggs Benedict
- Fish and Chips, French Fries, Fried Rice, Frozen Yogurt
- Hamburger, Hot Dog, Ice Cream, Lasagna
- Pizza, Ramen, Steak, Sushi, Waffles

### Data Split

| Split | Percentage | Samples |
|-------|------------|---------|
| Training | 80% | 60,600 |
| Validation | 20% | 15,150 |

### Preprocessing Pipeline

1. Resize images to 224 x 224 pixels
2. Normalize pixel values to [0, 1] range
3. Apply data augmentation during training:
   - Random horizontal flip
   - Random rotation (up to 20 degrees)
   - Random zoom (up to 20%)

---

## Training Methodology

### Two-Phase Transfer Learning Approach

Training was conducted in two phases to optimize feature extraction while preventing overfitting.

#### Phase 1: Feature Extraction

In this phase, the MobileNetV2 base model is frozen, and only the custom classification head is trained.

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size | 32 |
| Training Time | Approximately 90 minutes |
| Final Validation Accuracy | 52.53% |

#### Phase 2: Fine-Tuning

The top 50 layers of the base model are unfrozen and trained with a reduced learning rate.

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Unfrozen Layers | 50 |
| Training Time | Approximately 160 minutes |
| Final Validation Accuracy | 62.21% |

### Callbacks

- **Early Stopping**: Monitors validation loss with patience of 3 epochs
- **Model Checkpoint**: Saves best model based on validation accuracy
- **Reduce LR on Plateau**: Reduces learning rate by factor of 0.5 when validation loss stagnates

### Training Environment

| Specification | Value |
|---------------|-------|
| Hardware | MacBook Air M4 (Base Variant) |
| RAM | 16 GB |
| Training Mode | CPU (Apple Silicon optimized) |
| Total Training Time | Approximately 4 hours |

---

## API Integrations

### USDA FoodData Central API

Provides comprehensive nutritional information for identified foods.

**Endpoint**: `https://api.nal.usda.gov/fdc/v1/foods/search`

**Retrieved Nutrients**:
- Energy (calories)
- Protein
- Total Carbohydrates
- Total Fat
- Dietary Fiber
- Sugars

**Usage**: Nutrition values are returned per 100g serving and scaled based on user-specified portion size.

### OpenRouter API (Google Gemini 2.0 Flash)

Serves two purposes within the application:

**1. Fallback Food Recognition**

When the local model confidence is low or user requests verification, Gemini's vision capabilities identify the food item.

**2. AI Health Advisor**

Generates personalized health recommendations by analyzing:
- User profile (age, gender, weight, height, BMI)
- Health goals (lose/gain/maintain weight)
- Activity level
- Food nutritional content
- Portion size

**Response Format**:
```json
{
    "health_rating": "healthy | moderate | unhealthy",
    "advice": "Personalized recommendation text",
    "alternative": "Healthier food suggestion",
    "reason": "Brief explanation for the rating"
}
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git (optional, for cloning repository)

### Step 1: Clone or Download the Repository

```bash
git clone https://github.com/username/FoodVisionAI.git
cd FoodVisionAI
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install streamlit tensorflow tensorflow-datasets numpy pillow requests
```

### Step 4: Configure API Keys

Open `app.py` and replace the placeholder API keys with your own:

```python
OPENROUTER_API_KEY = "your_openrouter_api_key_here"
USDA_API_KEY = "your_usda_api_key_here"
```

**Obtaining API Keys**:
- USDA API: Register at https://fdc.nal.usda.gov/api-key-signup.html
- OpenRouter: Create account at https://openrouter.ai/

### Step 5: Train the Model (Optional)

If you wish to train the model from scratch:

```bash
jupyter notebook notebooks/01_train_model.ipynb
```

Alternatively, place a pre-trained model in the `models/` directory:
- `models/food_classifier.keras`
- `models/food_classes.json`

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## Usage Guide

### Initial Setup (Onboarding)

1. **Personal Information**: Enter your name, age, gender, weight, and height
2. **Goal Selection**: Choose between weight loss, weight gain, or maintenance
3. **Activity Level**: Select your typical physical activity frequency
4. **Target Setting**: Set target weight and timeline (for non-maintenance goals)

### Logging Food

**Method 1: Camera/Upload**
1. Navigate to the "Camera / Upload" tab
2. Either capture a photo using your device camera or upload an existing image
3. Select the meal type (breakfast, lunch, dinner, snack)
4. Click "Analyze Food" to identify the food item
5. Adjust portion size using the slider (10g - 1500g)
6. Click "Get AI Health Advice" for personalized recommendations
7. Review nutrition facts and click "Add to Today's Log"

**Method 2: Manual Entry**
1. Navigate to the "Manual Entry" tab
2. Type the food name in the text field
3. Select meal type and click "Get Nutrition Info"
4. Adjust portion and log as described above

### Dashboard Overview

- **Calories Remaining**: Shows daily calorie budget minus consumed calories
- **Water Intake**: Track glasses of water consumed
- **BMI Display**: Current BMI with category indicator
- **Macros Progress**: Visual bars for protein, carbs, and fat intake
- **Meal History**: Chronological list of logged foods with quantities

### Settings

Access the sidebar to:
- Reset daily data
- Edit profile information

---

## Project Structure

```
FoodVisionAI/
|
|-- app.py                      # Main Streamlit application
|
|-- models/
|   |-- food_classifier.keras   # Trained model file
|   |-- food_classes.json       # Class name mappings
|   |-- model_config.json       # Model configuration
|
|-- notebooks/
|   |-- 01_train_model.ipynb    # Model training notebook
|
|-- outputs/
|   |-- training_history.png    # Training metrics visualization
|
|-- README.md                   # Project documentation
|
|-- requirements.txt            # Python dependencies
```

---

## Performance Metrics

### Model Accuracy

| Metric | Phase 1 | Phase 2 (Final) |
|--------|---------|-----------------|
| Training Accuracy | 48.76% | 66.47% |
| Validation Accuracy | 52.53% | 62.21% |
| Validation Loss | 1.9191 | 1.5650 |

### Inference Performance

| Metric | Value |
|--------|-------|
| Average Inference Time | < 500ms |
| Model Size | ~14 MB |
| Memory Footprint | ~200 MB |

### Sample Predictions

| True Label | Predicted Label | Confidence |
|------------|-----------------|------------|
| Spaghetti Carbonara | Spaghetti Carbonara | 96.17% |
| Clam Chowder | Clam Chowder | 95.98% |
| Cheese Plate | Cheese Plate | 99.96% |
| Churros | Churros | 99.89% |
| Ice Cream | Ice Cream | 98.84% |

---

## Future Enhancements

### Short-Term Improvements

1. **Expanded Dataset**: Incorporate regional cuisines (Indian, Asian, Mediterranean) for broader food recognition
2. **Barcode Scanning**: Add support for packaged food identification via barcode
3. **Meal Planning**: Suggest daily meal plans based on remaining calorie budget
4. **Recipe Integration**: Provide healthy recipe suggestions based on available ingredients

### Long-Term Goals

1. **Mobile Application**: Develop native iOS and Android applications using React Native or Flutter
2. **Cloud Deployment**: Host application on cloud platforms (AWS, GCP, Azure) for wider accessibility
3. **User Accounts**: Implement authentication and persistent data storage
4. **Social Features**: Enable sharing of meal logs and achievements with friends
5. **Wearable Integration**: Sync with fitness trackers for activity data

### Technical Improvements

1. **Model Optimization**: Implement quantization and pruning for faster inference
2. **Offline Mode**: Enable food recognition without internet connectivity
3. **Multi-Food Detection**: Identify multiple food items in a single image
4. **Portion Estimation**: Use computer vision to estimate portion sizes automatically

---

## References

### Academic Resources

1. Bossard, L., Guillaumin, M., & Van Gool, L. (2014). Food-101 - Mining Discriminative Components with Random Forests. European Conference on Computer Vision.

2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. IEEE Conference on Computer Vision and Pattern Recognition.

3. Mifflin, M. D., St Jeor, S. T., Hill, L. A., Scott, B. J., Daugherty, S. A., & Koh, Y. O. (1990). A new predictive equation for resting energy expenditure in healthy individuals. The American Journal of Clinical Nutrition.

### Technical Documentation

- TensorFlow Documentation: https://www.tensorflow.org/
- Streamlit Documentation: https://docs.streamlit.io/
- USDA FoodData Central API: https://fdc.nal.usda.gov/api-guide.html
- OpenRouter API: https://openrouter.ai/docs

### Tools and Frameworks

- Python: https://www.python.org/
- Keras: https://keras.io/
- NumPy: https://numpy.org/
- Pillow: https://pillow.readthedocs.io/

---

## License

This project is developed for educational purposes as part of the Data Analytics-3 course curriculum. All rights reserved.

For academic use, please cite this project appropriately. For commercial applications, please contact the author for licensing information.

---

## Acknowledgments

- TensorFlow team for the comprehensive deep learning framework
- Streamlit team for the intuitive web application framework
- USDA for providing free access to nutritional data
- Food-101 dataset creators for the training data
- OpenRouter for API access to state-of-the-art language models

---

**Project Author**: [Your Name]  
**Course**: Data Analytics-3  
**Institution**: [Your Institution]  
**Date**: December 2024

---

For questions, issues, or contributions, please open an issue in the project repository or contact the author directly.
