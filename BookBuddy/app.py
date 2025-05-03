import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json
from recommender import get_item_based_recommendations
from data_processing import load_and_preprocess_data
from visualization import display_rating_distribution, display_top_authors, display_publication_year_trend, display_publisher_distribution

# Set page configuration
st.set_page_config(
    page_title="BookBuddy - Book Recommendations",
    page_icon="📚",
    layout="wide",
)

# App title and description
st.title("📚 BookBuddy")
st.markdown("""
    *Discover your next favorite book with our recommendation engine!*
""")

# Initialize session state for storing data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    
if 'books_df' not in st.session_state:
    st.session_state.books_df = None
    
if 'ratings_df' not in st.session_state:
    st.session_state.ratings_df = None
    
if 'item_similarity_df' not in st.session_state:
    st.session_state.item_similarity_df = None

if 'recent_searches' not in st.session_state:
    st.session_state.recent_searches = []
    
if 'searched_books' not in st.session_state:
    st.session_state.searched_books = []

# Constants
RECENT_SEARCHES_FILE = "recent_searches.json"
MAX_RECENT_SEARCHES = 5

# Load recent searches if available
def load_recent_searches():
    if os.path.exists(RECENT_SEARCHES_FILE):
        try:
            with open(RECENT_SEARCHES_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error loading recent searches: {str(e)}")
            return []
    return []

# Save recent searches
def save_recent_searches():
    # Ensure all values are JSON serializable
    serializable_searches = []
    for book in st.session_state.recent_searches:
        # Create a new dict with all values as strings
        serializable_book = {}
        for key, value in book.items():
            serializable_book[key] = str(value) if value is not None else ""
        serializable_searches.append(serializable_book)
    
    # Save to file
    try:
        with open(RECENT_SEARCHES_FILE, 'w') as f:
            json.dump(serializable_searches, f)
    except Exception as e:
        st.warning(f"Error saving recent searches: {str(e)}")

# Add a book to recent searches
def add_to_recent_searches(isbn):
    if st.session_state.data_loaded and isbn:
        # Get book info
        book = st.session_state.books_df[st.session_state.books_df['ISBN'] == isbn]
        if not book.empty:
            # Convert numpy/pandas data types to Python native types
            book_info = {
                'ISBN': str(isbn),  # Ensure ISBN is a string
                'Title': str(book.iloc[0]['Book-Title']),
                'Author': str(book.iloc[0]['Book-Author']),
                'Image': str(book.iloc[0]['Image-URL-M']) if not pd.isna(book.iloc[0]['Image-URL-M']) else ""
            }
            
            # Remove if already exists
            st.session_state.recent_searches = [b for b in st.session_state.recent_searches if b['ISBN'] != book_info['ISBN']]
            
            # Add to the beginning
            st.session_state.recent_searches.insert(0, book_info)
            
            # Keep only the latest MAX_RECENT_SEARCHES
            st.session_state.recent_searches = st.session_state.recent_searches[:MAX_RECENT_SEARCHES]
            
            # Save to file
            try:
                save_recent_searches()
            except Exception as e:
                st.warning(f"Could not save recent searches: {str(e)}")

# Function to display a book with robust error handling
def display_book(book_row, is_recommendation=False, index=None):
    try:
        col1, col2 = st.columns([1, 3])
        
        # Ensure we have a valid book record
        if book_row is None:
            st.warning("Book information is not available.")
            return
        
        # Get book ISBN safely
        isbn = None
        if 'ISBN' in book_row:
            isbn = str(book_row['ISBN'])
        elif 'isbn' in book_row:  # Check for lowercase variant
            isbn = str(book_row['isbn'])
        
        # If we still don't have an ISBN, check for pandas index
        if isbn is None and hasattr(book_row, 'name'):
            try:
                isbn = str(book_row.name)
            except:
                pass
        
        # If we still don't have an ISBN, generate a placeholder
        if isbn is None:
            # Create a unique placeholder
            import hashlib
            isbn = hashlib.md5(str(book_row).encode()).hexdigest()[:10]
        
        with col1:
            # Display book cover image safely
            try:
                img_url = None
                if 'Image-URL-M' in book_row:
                    img_url = book_row['Image-URL-M']
                elif 'Image' in book_row:
                    img_url = book_row['Image']
                
                if img_url is None or pd.isna(img_url) or img_url == '':
                    # Use a placeholder image if no image URL is available
                    st.image("https://images.unsplash.com/photo-1513542789411-b6a5d4f31634", width=150)
                else:
                    st.image(img_url, width=150)
            except Exception as e:
                # Fallback to placeholder on any error
                st.image("https://images.unsplash.com/photo-1513542789411-b6a5d4f31634", width=150)
        
        with col2:
            # Get book title safely
            title = "Unknown Title"
            if 'Book-Title' in book_row:
                title = book_row['Book-Title']
            elif 'Title' in book_row:
                title = book_row['Title']
            
            # Display title with appropriate formatting
            if is_recommendation:
                st.markdown(f"#### {title}")
            else:
                st.markdown(f"### {title}")
            
            # Get and display author safely
            author = "Unknown Author"
            if 'Book-Author' in book_row:
                author = book_row['Book-Author']
            elif 'Author' in book_row:
                author = book_row['Author']
            st.markdown(f"**Author:** {author}")
            
            # Get and display publication info safely
            year = "Unknown"
            if 'Year-Of-Publication' in book_row:
                year = book_row['Year-Of-Publication']
            
            publisher = "Unknown Publisher"
            if 'Publisher' in book_row:
                publisher = book_row['Publisher']
            
            st.markdown(f"**Published:** {year} by {publisher}")
            
            # Display ISBN
            st.markdown(f"**ISBN:** {isbn}")
            
            # Display rating if available
            if 'Average-Rating' in book_row and not pd.isna(book_row['Average-Rating']):
                try:
                    rating = float(book_row['Average-Rating'])
                    st.markdown(f"**Average Rating:** {rating:.2f}/10")
                except:
                    # Skip rating display if conversion fails
                    pass
            
            # Add recommendation button if not already showing recommendations
            if not is_recommendation:
                # Create a unique key for each button
                btn_key = f"btn_{isbn}_{index if index is not None else 'main'}"
                if st.button("Show Similar Books", key=btn_key):
                    add_to_recent_searches(isbn)
                    st.session_state.selected_book = isbn
                    st.rerun()
    except Exception as e:
        st.error(f"Error displaying book information: {str(e)}")
        # Don't halt application on individual book display errors

# Sidebar content
with st.sidebar:
    st.markdown("## Data Loading")
    
    # Instructions for CSV format
    with st.expander("📋 CSV Format Instructions"):
        st.markdown("""
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
        
        Example CSV files can be downloaded from [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/).
        
        The CSV file will be stored in the same directory as the application.
        """)
    
    # Create tabs for loading options
    load_tab1, load_tab2 = st.tabs(["Upload CSV", "Use Existing CSV"])
    
    with load_tab1:
        # File uploader for CSV with increased size limit
        st.info("This app supports large CSV files up to 1000 MB")
        uploaded_file = st.file_uploader("Upload Books CSV Data", type="csv", help="Upload a CSV file with book data")
        
        # Save uploaded file
        if uploaded_file is not None:
            with st.spinner("Saving uploaded file..."):
                file_path = os.path.join('.', uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"File saved as {uploaded_file.name}")
                
                # Store the file path in session state
                if 'csv_file_path' not in st.session_state:
                    st.session_state.csv_file_path = file_path
    
    with load_tab2:
        # List existing CSV files
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        
        if csv_files:
            selected_file = st.selectbox(
                "Select an existing CSV file",
                options=[""] + csv_files,
                index=0,
                help="Choose a previously uploaded CSV file"
            )
            
            if selected_file:
                st.session_state.csv_file_path = os.path.join('.', selected_file)
                st.success(f"Selected {selected_file}")
        else:
            st.info("No CSV files found. Please upload a file first.")

    # Sample data option
    use_sample_data = st.checkbox("Use Sample Data", value=not bool(uploaded_file) and not('csv_file_path' in st.session_state and st.session_state.csv_file_path))
    
    # Additional options for large files
    if not use_sample_data:
        with st.expander("⚙️ Advanced Options"):
            chunk_size = st.slider(
                "Chunk size for processing (MB)",
                min_value=10,
                max_value=100,
                value=50,
                step=10,
                help="Processing large files in chunks can improve performance"
            )
            
            use_caching = st.checkbox(
                "Cache processed data",
                value=True,
                help="Save processed data to disk to speed up future loads"
            )
    
    # Load data button
    load_data_button = st.button("Load Data")
    
    if load_data_button or st.session_state.data_loaded:
        with st.spinner("Processing data..."):
            if use_sample_data:
                # Create sample data
                st.session_state.books_df, st.session_state.ratings_df, st.session_state.item_similarity_df = load_and_preprocess_data(sample=True)
            elif 'csv_file_path' in st.session_state and st.session_state.csv_file_path:
                # Load from saved file path
                st.session_state.books_df, st.session_state.ratings_df, st.session_state.item_similarity_df = load_and_preprocess_data(file_path=st.session_state.csv_file_path)
            elif uploaded_file is not None:
                # Load directly from uploaded file
                st.session_state.books_df, st.session_state.ratings_df, st.session_state.item_similarity_df = load_and_preprocess_data(file=uploaded_file)
            else:
                st.error("Please upload a CSV file, select an existing file, or use sample data")
                st.stop()
            
            st.session_state.data_loaded = True
            st.success("Data loaded successfully!")
            
            # Also load recent searches
            if not st.session_state.recent_searches:
                st.session_state.recent_searches = load_recent_searches()
    
    # Display data stats after loading
    if st.session_state.data_loaded:
        st.markdown("### Dataset Statistics")
        st.markdown(f"**Number of Books:** {len(st.session_state.books_df)}")
        st.markdown(f"**Number of Ratings:** {len(st.session_state.ratings_df)}")
        st.markdown(f"**Number of Users:** {st.session_state.ratings_df['User-ID'].nunique()}")

# Main content
if st.session_state.data_loaded:
    # Create tabs for different app sections
    tab1, tab2, tab3, tab4 = st.tabs(["Search Books", "Recommendations", "Recently Viewed", "Data Insights"])
    
    with tab1:
        st.header("Search Books")
        
        # Search options
        search_col1, search_col2 = st.columns(2)
        
        with search_col1:
            # Get all book titles for autocomplete
            all_book_titles = st.session_state.books_df['Book-Title'].unique().tolist()
            all_authors = st.session_state.books_df['Book-Author'].unique().tolist()
            
            # Autocomplete search
            search_type = st.radio("Search by:", ["Title", "Author", "ISBN"], horizontal=True)
            
            if search_type == "Title":
                search_term = st.selectbox(
                    "Start typing a book title:",
                    options=[""] + all_book_titles,
                    index=0
                )
            elif search_type == "Author":
                search_term = st.selectbox(
                    "Start typing an author name:",
                    options=[""] + all_authors,
                    index=0
                )
            else:
                search_term = st.text_input("Enter ISBN:")
        
        with search_col2:
            min_rating = st.slider("Minimum Average Rating", 0.0, 10.0, 0.0, 0.5)
            sort_by = st.selectbox(
                "Sort results by:",
                options=["Relevance", "Rating (High to Low)", "Year (Newest First)", "Year (Oldest First)"],
                index=0
            )
        
        # Filter books based on search term and rating
        filtered_books = st.session_state.books_df.copy()
        
        if search_term:
            if search_type == "Title":
                filtered_books = filtered_books[
                    filtered_books['Book-Title'].str.contains(search_term, case=False, na=False)
                ]
            elif search_type == "Author":
                filtered_books = filtered_books[
                    filtered_books['Book-Author'].str.contains(search_term, case=False, na=False)
                ]
            else:  # ISBN
                filtered_books = filtered_books[
                    filtered_books['ISBN'].str.contains(search_term, case=False, na=False)
                ]
                
            # Add to searched books if exact match
            if len(filtered_books) > 0 and search_type == "Title" and search_term in all_book_titles:
                exact_match = filtered_books[filtered_books['Book-Title'] == search_term]
                if not exact_match.empty:
                    add_to_recent_searches(exact_match.iloc[0]['ISBN'])
        
        if min_rating > 0:
            filtered_books = filtered_books[filtered_books['Average-Rating'] >= min_rating]
        
        # Sort results
        if sort_by == "Rating (High to Low)":
            filtered_books = filtered_books.sort_values('Average-Rating', ascending=False)
        elif sort_by == "Year (Newest First)":
            filtered_books = filtered_books.sort_values('Year-Of-Publication', ascending=False)
        elif sort_by == "Year (Oldest First)":
            filtered_books = filtered_books.sort_values('Year-Of-Publication', ascending=True)
        
        # Display search results
        st.subheader(f"Results: {len(filtered_books)} books found")
        
        if len(filtered_books) > 0:
            # Display top 10 books
            for i, (_, book) in enumerate(filtered_books.head(10).iterrows()):
                st.markdown("---")
                display_book(book, index=f"search_{i}")
        else:
            st.info("No books found matching your criteria.")
    
    with tab2:
        st.header("Book Recommendations")
        
        if 'selected_book' in st.session_state:
            selected_isbn = st.session_state.selected_book
            selected_book = st.session_state.books_df[st.session_state.books_df['ISBN'] == selected_isbn].iloc[0]
            
            st.subheader("Selected Book")
            display_book(selected_book, index="selected")
            
            st.markdown("### Similar Books You Might Like")
            
            # Get recommendations
            recommendations = get_item_based_recommendations(
                selected_isbn, 
                st.session_state.item_similarity_df, 
                st.session_state.books_df
            )
            
            if len(recommendations) > 0:
                for i, (_, book) in enumerate(recommendations.iterrows()):
                    st.markdown("---")
                    display_book(book, is_recommendation=True, index=f"rec_{i}")
            else:
                st.info("No recommendations available for this book.")
        else:
            # Select a random book for demo purposes
            sample_book = st.session_state.books_df.sample(1).iloc[0]
            
            st.info("Select a book from the Search tab to see recommendations, or check out this random selection:")
            display_book(sample_book, index="random_sample")
            
            if st.button("Get Recommendations for This Book", key="btn_random_sample"):
                add_to_recent_searches(sample_book['ISBN'])
                st.session_state.selected_book = sample_book['ISBN']
                st.rerun()
    
    with tab3:
        st.header("Recently Viewed Books")
        
        if st.session_state.recent_searches and len(st.session_state.recent_searches) > 0:
            for book in st.session_state.recent_searches:
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    if book['Image'] and not pd.isna(book['Image']):
                        st.image(book['Image'], width=100)
                    else:
                        st.image("https://images.unsplash.com/photo-1513542789411-b6a5d4f31634", width=100)
                
                with col2:
                    st.markdown(f"**{book['Title']}**")
                    st.text(f"By: {book['Author']}")
                    st.text(f"ISBN: {book['ISBN']}")
                
                with col3:
                    if st.button("View Similar", key=f"recent_{book['ISBN']}"):
                        st.session_state.selected_book = book['ISBN']
                        st.rerun()
                
                st.markdown("---")
        else:
            st.info("No recently viewed books. Start searching or browsing to build your history!")
    
    with tab4:
        st.header("Data Insights")
        
        # Create tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["Rating Distribution", "Top Authors", "Publication Trend", "Publishers"])
        
        with viz_tab1:
            # Display rating distribution
            fig = display_rating_distribution(st.session_state.ratings_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            # Display top authors
            fig = display_top_authors(st.session_state.books_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab3:
            # Display publication year trend
            fig = display_publication_year_trend(st.session_state.books_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab4:
            # Display publisher distribution
            fig = display_publisher_distribution(st.session_state.books_df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top books
        st.subheader("Top Rated Books")
        top_books = st.session_state.books_df.sort_values('Average-Rating', ascending=False).head(10)
        
        for i, (_, book) in enumerate(top_books.iterrows()):
            st.markdown(f"**{i+1}.** {book['Book-Title']} by {book['Book-Author']} - Rating: {book['Average-Rating']:.2f}/10")

# Footer
st.markdown("---")
st.markdown("""
    ### About BookBuddy
    BookBuddy uses item-item collaborative filtering to recommend books based on similar characteristics.
    Upload your book data in CSV format to get personalized recommendations.
    
    Your CSV file will be saved in the application's directory for future use. Recent searches are also saved locally.
""")
