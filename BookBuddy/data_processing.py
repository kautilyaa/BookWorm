import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io
import streamlit as st

def generate_sample_data():
    """Generate sample book data for demonstration purposes."""
    # Sample data based on the given format
    sample_books = [
        {"User-ID": 276725, "ISBN": "034545104X", "Book-Rating": 0, "Book-Title": "Flesh Tones: A Novel", "Book-Author": "M. J. Rose", "Year-Of-Publication": 2002, "Publisher": "Ballantine Books", "Image-URL-S": "http://images.amazon.com/images/P/034545104X.01.THUMBZZZ.jpg", "Image-URL-M": "https://images.unsplash.com/photo-1513542789411-b6a5d4f31634", "Image-URL-L": "http://images.amazon.com/images/P/034545104X.01.LZZZZZZZ.jpg"},
        {"User-ID": 276726, "ISBN": "0155061224", "Book-Rating": 5, "Book-Title": "Rites of Passage", "Book-Author": "Judith Rae", "Year-Of-Publication": 2001, "Publisher": "Harcourt Brace", "Image-URL-S": "http://images.amazon.com/images/P/0155061224.01.THUMBZZZ.jpg", "Image-URL-M": "https://images.unsplash.com/photo-1643585148061-67ed5aa78797", "Image-URL-L": "http://images.amazon.com/images/P/0155061224.01.LZZZZZZZ.jpg"},
        {"User-ID": 276727, "ISBN": "0446520802", "Book-Rating": 0, "Book-Title": "The Notebook", "Book-Author": "Nicholas Sparks", "Year-Of-Publication": 1996, "Publisher": "Warner Books", "Image-URL-S": "http://images.amazon.com/images/P/0446520802.01.THUMBZZZ.jpg", "Image-URL-M": "https://images.unsplash.com/photo-1497633762265-9d179a990aa6", "Image-URL-L": "http://images.amazon.com/images/P/0446520802.01.LZZZZZZZ.jpg"},
        {"User-ID": 276729, "ISBN": "052165615X", "Book-Rating": 3, "Book-Title": "Help!: Level 1", "Book-Author": "Philip Prowse", "Year-Of-Publication": 1999, "Publisher": "Cambridge University Press", "Image-URL-S": "http://images.amazon.com/images/P/052165615X.01.THUMBZZZ.jpg", "Image-URL-M": "https://images.unsplash.com/photo-1491841573634-28140fc7ced7", "Image-URL-L": "http://images.amazon.com/images/P/052165615X.01.LZZZZZZZ.jpg"},
        {"User-ID": 276729, "ISBN": "0521795028", "Book-Rating": 6, "Book-Title": "The Amsterdam Connection", "Book-Author": "Sue Leather", "Year-Of-Publication": 2001, "Publisher": "Cambridge University Press", "Image-URL-S": "http://images.amazon.com/images/P/0521795028.01.THUMBZZZ.jpg", "Image-URL-M": "https://images.unsplash.com/photo-1484415063229-3d6335668531", "Image-URL-L": "http://images.amazon.com/images/P/0521795028.01.LZZZZZZZ.jpg"},
        {"User-ID": 276731, "ISBN": "0971880107", "Book-Rating": 0, "Book-Title": "The Adventures of Super Diaper Baby", "Book-Author": "Dav Pilkey", "Year-Of-Publication": 2002, "Publisher": "Blue Sky Press", "Image-URL-S": "http://images.amazon.com/images/P/0971880107.01.THUMBZZZ.jpg", "Image-URL-M": "https://images.unsplash.com/photo-1517770413964-df8ca61194a6", "Image-URL-L": "http://images.amazon.com/images/P/0971880107.01.LZZZZZZZ.jpg"},
        {"User-ID": 276736, "ISBN": "0385504209", "Book-Rating": 0, "Book-Title": "The Da Vinci Code", "Book-Author": "Dan Brown", "Year-Of-Publication": 2003, "Publisher": "Doubleday", "Image-URL-S": "http://images.amazon.com/images/P/0385504209.01.THUMBZZZ.jpg", "Image-URL-M": "https://images.unsplash.com/photo-1659999604440-bec7ace198d7", "Image-URL-L": "http://images.amazon.com/images/P/0385504209.01.LZZZZZZZ.jpg"},
        {"User-ID": 276737, "ISBN": "0345914813", "Book-Rating": 0, "Book-Title": "Harry Potter and the Sorcerer's Stone", "Book-Author": "J.K. Rowling", "Year-Of-Publication": 1997, "Publisher": "Scholastic", "Image-URL-S": "http://images.amazon.com/images/P/0345914813.01.THUMBZZZ.jpg", "Image-URL-M": "https://images.unsplash.com/photo-1456315138460-858d1089ddba", "Image-URL-L": "http://images.amazon.com/images/P/0345914813.01.LZZZZZZZ.jpg"},
        {"User-ID": 276744, "ISBN": "0375759778", "Book-Rating": 0, "Book-Title": "The Great Gatsby", "Book-Author": "F. Scott Fitzgerald", "Year-Of-Publication": 1925, "Publisher": "Scribner", "Image-URL-S": "http://images.amazon.com/images/P/0375759778.01.THUMBZZZ.jpg", "Image-URL-M": "https://images.unsplash.com/photo-1509114859430-5f2c74177f4b", "Image-URL-L": "http://images.amazon.com/images/P/0375759778.01.LZZZZZZZ.jpg"},
        {"User-ID": 276747, "ISBN": "0553277456", "Book-Rating": 8, "Book-Title": "To Kill a Mockingbird", "Book-Author": "Harper Lee", "Year-Of-Publication": 1960, "Publisher": "HarperCollins", "Image-URL-S": "http://images.amazon.com/images/P/0553277456.01.THUMBZZZ.jpg", "Image-URL-M": "https://images.unsplash.com/photo-1610500795224-fb86b02926d7", "Image-URL-L": "http://images.amazon.com/images/P/0553277456.01.LZZZZZZZ.jpg"}
    ]
    
    # Add more ratings to create a better similarity matrix
    additional_ratings = []
    for i in range(20):
        user_id = 276800 + i
        for book in sample_books:
            rating = np.random.randint(0, 11)  # Ratings from 0-10
            additional_ratings.append({
                "User-ID": user_id,
                "ISBN": book["ISBN"],
                "Book-Rating": rating
            })
    
    # Create DataFrames
    books_df = pd.DataFrame(sample_books)
    ratings_df = pd.DataFrame([
        {"User-ID": book["User-ID"], "ISBN": book["ISBN"], "Book-Rating": book["Book-Rating"]}
        for book in sample_books
    ] + additional_ratings)
    
    return books_df, ratings_df

def preprocess_data(books_df, ratings_df):
    """Preprocess the book and ratings data."""
    # Clean up column types
    books_df['Year-Of-Publication'] = pd.to_numeric(books_df['Year-Of-Publication'], errors='coerce')
    books_df['Year-Of-Publication'] = books_df['Year-Of-Publication'].fillna(0).astype(int)
    
    # Handle missing values
    books_df['Book-Title'] = books_df['Book-Title'].fillna('Unknown Title')
    books_df['Book-Author'] = books_df['Book-Author'].fillna('Unknown Author')
    books_df['Publisher'] = books_df['Publisher'].fillna('Unknown Publisher')
    
    # Create average rating for each book
    avg_ratings = ratings_df.groupby('ISBN')['Book-Rating'].mean().reset_index()
    avg_ratings.rename(columns={'Book-Rating': 'Average-Rating'}, inplace=True)
    
    # Merge average ratings with books data
    books_df = pd.merge(books_df, avg_ratings, on='ISBN', how='left')
    books_df['Average-Rating'] = books_df['Average-Rating'].fillna(0)
    
    return books_df, ratings_df

def create_similarity_matrix(ratings_df):
    """Create item-item similarity matrix using collaborative filtering with robust handling
    of large datasets and numerical issues."""
    # Check the size of the dataset
    n_ratings = len(ratings_df)
    n_users = ratings_df['User-ID'].nunique()
    n_items = ratings_df['ISBN'].nunique()
    
    st.info(f"Creating similarity matrix for {n_users} users and {n_items} books based on {n_ratings} ratings.")
    
    # For large datasets, focus only on items with a minimum number of ratings
    # This improves performance and reduces numerical issues
    if n_items > 1000:
        item_counts = ratings_df['ISBN'].value_counts()
        min_ratings = 3  # Minimum ratings to consider an item
        filtered_items = item_counts[item_counts >= min_ratings].index
        ratings_df = ratings_df[ratings_df['ISBN'].isin(filtered_items)]
        st.info(f"Filtered to {len(filtered_items)} books with at least {min_ratings} ratings.")
    
    # For very large datasets, additionally sample users for better performance
    if n_ratings > 500000:
        st.warning("Large dataset detected. Sampling data for performance optimization.")
        # Sample users with a minimum number of ratings
        user_counts = ratings_df['User-ID'].value_counts()
        active_users = user_counts[user_counts >= 5].index[:10000]  # Take up to 10k active users
        ratings_df = ratings_df[ratings_df['User-ID'].isin(active_users)]
        st.info(f"Sampled to {len(ratings_df)} ratings from {len(active_users)} active users.")
    
    # Create user-item matrix with improved error handling
    with st.spinner("Creating user-item matrix..."):
        try:
            # For datasets beyond a certain size, use a sparse matrix approach
            if n_ratings > 100000:
                from scipy.sparse import csr_matrix
                
                # Create mappings for user IDs and ISBNs to matrix indices
                user_ids = sorted(ratings_df['User-ID'].unique())
                isbn_ids = sorted(ratings_df['ISBN'].unique())
                
                user_map = {id: i for i, id in enumerate(user_ids)}
                isbn_map = {id: i for i, id in enumerate(isbn_ids)}
                
                # Map the ratings to matrix indices
                user_idx = ratings_df['User-ID'].map(user_map).values
                isbn_idx = ratings_df['ISBN'].map(isbn_map).values
                
                # Ensure ratings are valid numbers
                valid_ratings = pd.to_numeric(ratings_df['Book-Rating'], errors='coerce').fillna(0).values
                
                # Create a sparse matrix
                matrix = csr_matrix(
                    (valid_ratings, (user_idx, isbn_idx)), 
                    shape=(len(user_ids), len(isbn_ids))
                )
                
                # Keep matrix sparse for calculations
                user_item_sparse = matrix
                # Store column mapping for later
                column_mapping = {i: isbn for isbn, i in isbn_map.items()}
            else:
                # For smaller datasets, use pivot table but handle errors
                # First ensure ratings are valid numbers
                ratings_df['Book-Rating'] = pd.to_numeric(ratings_df['Book-Rating'], errors='coerce').fillna(0)
                
                user_item_matrix = ratings_df.pivot_table(
                    index='User-ID', 
                    columns='ISBN', 
                    values='Book-Rating',
                    fill_value=0
                )
                user_item_sparse = None  # Not using sparse format for small datasets
        except Exception as e:
            st.error(f"Error creating user-item matrix: {str(e)}")
            # Fallback to a more aggressive sampling approach
            st.warning("Falling back to a more limited dataset for calculation.")
            
            # Focus on books with most ratings
            top_books = ratings_df['ISBN'].value_counts().head(500).index
            limited_ratings = ratings_df[ratings_df['ISBN'].isin(top_books)]
            
            # Ensure ratings are valid numbers
            limited_ratings['Book-Rating'] = pd.to_numeric(limited_ratings['Book-Rating'], errors='coerce').fillna(0)
            
            user_item_matrix = limited_ratings.pivot_table(
                index='User-ID', 
                columns='ISBN', 
                values='Book-Rating',
                fill_value=0
            )
            user_item_sparse = None  # Not using sparse format in fallback
    
    # Calculate similarity matrix with robust error handling
    with st.spinner("Calculating item similarity... This might take a while."):
        try:
            # Handle calculation differently based on whether we're using sparse matrices
            if user_item_sparse is not None:
                # Using sparse matrices for large datasets
                from scipy.sparse import linalg as splinalg
                from sklearn.preprocessing import normalize
                
                # Normalize each item vector to prevent division by zero
                item_vectors = user_item_sparse.T  # Transpose to get items as rows
                
                # Only calculate similarities for most popular items if there are too many
                n_items_actual = item_vectors.shape[0]
                max_items_to_process = 5000  # Maximum items to calculate similarities for
                
                if n_items_actual > max_items_to_process:
                    # Calculate item popularities
                    item_sums = np.array(item_vectors.sum(axis=1)).flatten()
                    top_indices = np.argsort(-item_sums)[:max_items_to_process]  # Negative for descending order
                    
                    # Select only the top items
                    selected_items = item_vectors[top_indices]
                    selected_items = normalize(selected_items, norm='l2', axis=1, copy=True)
                    
                    # Create mapping from new indices to original ISBNs
                    selected_isbns = [column_mapping[idx] for idx in top_indices]
                    
                    # Calculate similarity
                    similarity_matrix = selected_items.dot(selected_items.T).toarray()
                    
                    # Create DataFrame
                    item_similarity_df = pd.DataFrame(
                        similarity_matrix,
                        index=selected_isbns,
                        columns=selected_isbns
                    )
                    
                    st.info(f"Calculated similarities for the top {max_items_to_process} most popular books.")
                else:
                    # For smaller item sets, calculate all similarities
                    normalized_items = normalize(item_vectors, norm='l2', axis=1, copy=True)
                    similarity_matrix = normalized_items.dot(normalized_items.T).toarray()
                    
                    # Create DataFrame
                    item_similarity_df = pd.DataFrame(
                        similarity_matrix,
                        index=[column_mapping[i] for i in range(n_items_actual)],
                        columns=[column_mapping[i] for i in range(n_items_actual)]
                    )
            else:
                # For smaller datasets, use standard approach but with error handling
                # First check for zero-variance columns (all same value)
                item_variances = user_item_matrix.var().fillna(0)
                valid_columns = item_variances[item_variances > 0].index
                
                if len(valid_columns) < len(user_item_matrix.columns):
                    st.warning(f"Removed {len(user_item_matrix.columns) - len(valid_columns)} books with no rating variance.")
                    filtered_matrix = user_item_matrix[valid_columns]
                else:
                    filtered_matrix = user_item_matrix
                
                # Calculate similarity with numpy to explicitly handle numerical issues
                matrix_values = filtered_matrix.values
                # Transpose for item-item similarity
                matrix_t = matrix_values.T
                
                # Calculate similarities manually to handle edge cases
                from numpy.linalg import norm
                n_items = matrix_t.shape[0]
                sim_matrix = np.zeros((n_items, n_items))
                
                # Calculate pairwise cosine similarities with explicit error handling
                for i in range(n_items):
                    for j in range(i, n_items):  # Only calculate upper triangle
                        vec1 = matrix_t[i]
                        vec2 = matrix_t[j]
                        
                        # Calculate norms
                        norm1 = norm(vec1)
                        norm2 = norm(vec2)
                        
                        # Handle zero norms to avoid division by zero
                        if norm1 == 0 or norm2 == 0:
                            sim = 0  # Define similarity as 0 if either vector has zero norm
                        else:
                            # Dot product with protection against overflow
                            dot_product = np.sum(vec1 * vec2)
                            # Safe division
                            sim = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
                        
                        # Fill symmetric matrix
                        sim_matrix[i, j] = sim
                        sim_matrix[j, i] = sim  # Symmetric
                
                # Convert to DataFrame for easier lookup
                item_similarity_df = pd.DataFrame(
                    sim_matrix,
                    index=filtered_matrix.columns,
                    columns=filtered_matrix.columns
                )
        except Exception as e:
            st.error(f"Error calculating similarity matrix: {str(e)}")
            # Create a minimal similarity matrix for demonstration
            st.warning("Creating a simplified similarity matrix...")
            
            # Get the most popular books (limited number)
            popular_books = ratings_df['ISBN'].value_counts().head(100).index.tolist()
            
            # Filter ratings to just these books
            small_ratings = ratings_df[ratings_df['ISBN'].isin(popular_books)]
            small_ratings['Book-Rating'] = pd.to_numeric(small_ratings['Book-Rating'], errors='coerce').fillna(0)
            
            # Create a small pivot table
            small_matrix = small_ratings.pivot_table(
                index='User-ID', 
                columns='ISBN', 
                values='Book-Rating',
                fill_value=0
            )
            
            # Simple correlation instead of cosine similarity for robustness
            item_similarity_df = small_matrix.corr(method='pearson').fillna(0)
    
    return item_similarity_df

def load_and_preprocess_data(file=None, sample=False, file_path=None):
    """Load and preprocess data from a CSV file or use sample data, with optimized handling
    for large datasets by only loading the necessary data."""
    if sample:
        # Use sample data
        books_df, ratings_df = generate_sample_data()
    else:
        try:
            # Define required columns
            required_columns = ['User-ID', 'ISBN', 'Book-Rating', 'Book-Title', 'Book-Author', 
                            'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
            
            # Show loading message
            with st.spinner("Loading data file... This may take a few minutes for large files."):
                # Check file size first for large files
                file_size_mb = None
                if file_path is not None:
                    import os
                    try:
                        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                    except:
                        pass
                
                # For very large files, use chunking to load only what we need
                if file_size_mb and file_size_mb > 200:  # Files larger than 200MB
                    st.info(f"Large file detected ({file_size_mb:.1f} MB). Using optimized loading approach.")
                    
                    # First, load only book metadata to get unique books
                    book_columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 
                                'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
                    
                    # Read in chunks to find all unique books
                    chunk_size = 100000  # Adjust based on memory constraints
                    books_list = []
                    
                    # Define a chunk iterator
                    chunks = pd.read_csv(
                        file_path, 
                        usecols=book_columns,
                        chunksize=chunk_size, 
                        low_memory=True
                    )
                    
                    # Process each chunk
                    for i, chunk in enumerate(chunks):
                        # Extract unique books from this chunk
                        chunk_books = chunk.drop_duplicates(subset=['ISBN'])
                        books_list.append(chunk_books)
                        
                        # Progress indicator
                        if i % 5 == 0:
                            st.write(f"Processed {(i+1) * chunk_size} rows...")
                    
                    # Combine all chunks and remove duplicates again
                    books_df = pd.concat(books_list).drop_duplicates(subset=['ISBN'])
                    
                    # Now load the ratings data, potentially sampling for very large files
                    st.write(f"Loading ratings data...")
                    rating_chunks = pd.read_csv(
                        file_path,
                        usecols=['User-ID', 'ISBN', 'Book-Rating'],
                        chunksize=chunk_size,
                        low_memory=True
                    )
                    
                    # Choose a sensible limit for ratings
                    max_ratings = 1000000  # 1 million ratings should be plenty
                    ratings_list = []
                    total_ratings = 0
                    
                    for chunk in rating_chunks:
                        # Keep only ratings for books we already have
                        chunk = chunk[chunk['ISBN'].isin(books_df['ISBN'])]
                        ratings_list.append(chunk)
                        
                        total_ratings += len(chunk)
                        if total_ratings >= max_ratings:
                            st.warning(f"Limiting to {max_ratings} ratings for performance.")
                            break
                    
                    # Combine chunks
                    ratings_df = pd.concat(ratings_list)
                    
                    # If still too large, sample
                    if len(ratings_df) > max_ratings:
                        ratings_df = ratings_df.sample(n=max_ratings, random_state=42)
                    
                    st.success(f"Successfully loaded {len(books_df)} unique books and {len(ratings_df)} ratings.")
                else:
                    # For smaller files, use the standard approach
                    if file is not None:
                        # Load from uploaded file
                        df = pd.read_csv(file, low_memory=True)
                    elif file_path is not None:
                        # Load from saved file path
                        df = pd.read_csv(file_path, low_memory=True)
                    else:
                        st.error("No file provided.")
                        return None, None, None
                    
                    # Check if all required columns exist
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        st.error(f"Missing columns in the CSV: {', '.join(missing_columns)}")
                        return None, None, None
                    
                    # Check data size and potentially sample if too large
                    if len(df) > 500000:
                        st.warning(f"Large dataset ({len(df)} rows). Processing a sample of 500,000 rows for better performance.")
                        df = df.sample(n=500000, random_state=42)
                    
                    # Split into books and ratings dataframes
                    books_df = df[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 
                                    'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']].drop_duplicates(subset=['ISBN'])
                    ratings_df = df[['User-ID', 'ISBN', 'Book-Rating']]
                    
                    st.success(f"Successfully loaded {len(df)} rows with {books_df['ISBN'].nunique()} unique books.")
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.exception(e)  # Show the full exception for debugging
            return None, None, None
    
    # Preprocess the data
    with st.spinner("Preprocessing data..."):
        books_df, ratings_df = preprocess_data(books_df, ratings_df)
    
    # Clean memory before heavy computation
    import gc
    gc.collect()
    
    # Create similarity matrix for item-based collaborative filtering
    with st.spinner("Creating recommendation model... This might take a while for large datasets."):
        item_similarity_df = create_similarity_matrix(ratings_df)
    
    return books_df, ratings_df, item_similarity_df
