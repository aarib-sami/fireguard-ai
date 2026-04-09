# FireGuard AI

A forest fire prediction application that uses machine learning to assess fire risk based on weather conditions and location data.

## Features

- Interactive map for location selection
- Real-time fire risk prediction using AI model
- Weather data integration (currently using mock data)
- Responsive web interface

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/aarib-sami/fireguard-ai.git
   cd fireguard-ai
   ```

2. Set up the backend:

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Set up the frontend:
   ```bash
   cd ../frontend
   npm install
   ```

## Running the Application

1. Start the backend server:

   ```bash
   cd backend
   python app.py
   ```

   The backend will run on http://localhost:8000

2. Start the frontend development server:

   ```bash
   cd frontend
   npm run dev
   ```

   The frontend will run on http://localhost:5173 (default Vite port)

3. Open your browser and navigate to http://localhost:5173

## API Configuration

The application uses OpenWeatherMap API for live weather data. To enable live weather data:

1. Get an API key from [OpenWeatherMap](https://openweathermap.org/api)
2. Create a `.env` file in the `backend` directory:
   ```
   API_KEY=your_api_key_here
   ```

**Note:** Currently, the live weather API is not in use, so predictions are based on mock data and may not be accurate.

## Hosting

### Frontend

You can host the frontend on Netlify, Vercel, or any static site host that supports Vite. Deploy the contents of the `frontend` build output.

### Backend

For the backend, use a service that supports Python/Flask apps, such as:

- Render
- Railway
- Fly.io
- Heroku
- DigitalOcean App Platform
- AWS Elastic Beanstalk / ECS

The backend must be reachable from your deployed frontend. For example, if your backend URL is `https://fireguard-backend.example.com`, that URL must be used by the frontend API call.

### Environment variable for backend API URL

The frontend is now configured to use a deployment-specific backend URL via `VITE_API_BASE_URL`.

1. In your Netlify or Vercel dashboard, add an environment variable:

   ```env
   VITE_API_BASE_URL=https://your-backend.example.com
   ```

2. Locally, you can add it to `frontend/.env` for testing:

   ```env
   VITE_API_BASE_URL=http://localhost:8000
   ```

3. The frontend code will use this variable and send requests to `${VITE_API_BASE_URL}/predict`.

If `VITE_API_BASE_URL` is not set, the app will fall back to the local development proxy path `/predict`.

## Project Structure

```
fireguard-ai/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── requirements.txt       # Python dependencies
│   ├── models/
│   │   └── forest_fire_model.pth  # Trained ML model
│   └── forestFire.csv         # Training data
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main React component
│   │   ├── components/        # React components
│   │   └── ...
│   ├── package.json           # Node dependencies
│   └── vite.config.js         # Vite configuration
└── README.md                  # This file
```

## Technologies Used

- **Backend:** Python, Flask, PyTorch
- **Frontend:** React, Vite, Mapbox GL JS
- **Machine Learning:** PyTorch for fire risk prediction
- **Mapping:** Mapbox for interactive maps
