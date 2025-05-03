# 📚 BookBuddy - Book Recommendation Engine

BookBuddy is an interactive web application built with Streamlit that provides personalized book recommendations using item-item collaborative filtering. The application allows users to search for books, get recommendations for similar titles, and explore book data visualizations.

## Features

- **Book Search with Autocomplete**: Easily find books by title, author, or ISBN with autocomplete suggestions
- **Advanced Filtering**: Filter books by minimum rating and sort them by various criteria
- **Item-Item Collaborative Filtering**: Get personalized book recommendations based on similar characteristics
- **Data Visualization**: Explore book data through various charts and graphs
- **Recently Viewed Books**: Keep track of books you've previously viewed
- **CSV Data Import**: Upload your own book data in CSV format

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone this repository or download the source code

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

## Using Docker

You can also run BookBuddy using Docker:

1. Build the Docker image:
   ```
   docker build -t bookbuddy .
   ```

2. Run the container:
   ```
   docker run -p 5000:5000 bookbuddy
   ```

## CSV Data Format

Your CSV file should contain the following columns:
- User-ID: Unique identifier for users
- ISBN: International Standard Book Number
- Book-Rating: Rating given by the user (0-10)
- Book-Title: Title of the book
- Book-Author: Author of the book
- Year-Of-Publication: Year the book was published
- Publisher: Publisher of the book
- Image-URL-S: URL to small book cover image
- Image-URL-M: URL to medium book cover image
- Image-URL-L: URL to large book cover image

## Local Storage

The application saves the following data locally:
- Uploaded CSV files are stored in the application directory
- Recently viewed books are saved in a `recent_searches.json` file for persistence

## How It Works

BookBuddy uses item-item collaborative filtering to find book recommendations:

1. It creates a user-item matrix from book ratings
2. Calculates similarity between books using cosine similarity
3. Recommends books similar to the ones you've selected

## Development

### Project Structure

- `app.py`: Main Streamlit application
- `data_processing.py`: Data loading, processing, and preparation functions
- `recommender.py`: Recommendation algorithm implementation
- `visualization.py`: Data visualization functions
- `.streamlit/config.toml`: Streamlit configuration
- `requirements.txt`: Project dependencies
- `Dockerfile`: Container configuration

